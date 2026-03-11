from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, Subset


def _resolve_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _is_cuda(device) -> bool:
    d = _resolve_device(device)
    return d.type == "cuda"


def _extract_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    return output


def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            _, data, target = batch
            return data, target
        if len(batch) == 2:
            data, target = batch
            return data, target
    raise ValueError(f"Unsupported batch format for Fisher estimation: type={type(batch)}")


class FisherEstimator(ABC):
    """Abstract base class for Fisher information estimation."""

    @abstractmethod
    def estimate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Estimate Fisher information. If batch_size is specified, use that many samples."""
        pass


class DiagonalFisherEstimator(FisherEstimator):
    """
    Empirical diagonal Fisher: average of per-sample squared gradients.

    Args:
        use_vmap:
            True  -> vmap per-sample grads (faster, more memory).
            False -> sequential per-sample grads (slower, safer).
    """

    def __init__(self, use_vmap: bool = False):
        self.use_vmap = use_vmap

    def estimate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        model.eval()
        device_obj = _resolve_device(device)

        if _is_cuda(device_obj):
            torch.cuda.empty_cache()

        if batch_size is not None:
            dataset = dataloader.dataset
            n = min(int(batch_size), len(dataset))
            limited_dataset = Subset(dataset, range(n))
            fisher_loader = DataLoader(limited_dataset, batch_size=max(1, n), shuffle=False)
        else:
            fisher_loader = dataloader

        if self.use_vmap:
            return self._estimate_vmap(model, fisher_loader, criterion, device_obj)
        return self._estimate_sequential(model, fisher_loader, criterion, device_obj)

    def _estimate_sequential(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Memory-efficient sequential per-sample gradient computation."""
        param_names = [n for n, _ in model.named_parameters()]
        fisher = {n: torch.zeros_like(p, device="cpu") for n, p in model.named_parameters()}
        total_samples = 0

        from tqdm import tqdm

        iterator = tqdm(dataloader, desc="Estimating Fisher", leave=False)
        for batch in iterator:
            data, target = _unpack_batch(batch)
            data, target = data.to(device), target.to(device)

            for i in range(data.size(0)):
                model.zero_grad(set_to_none=True)
                logits = _extract_logits(model(data[i : i + 1]))
                y_i = target[i : i + 1]
                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss = F.cross_entropy(logits, y_i)
                else:
                    loss = criterion(logits, y_i)
                loss.backward()

                for n, p in model.named_parameters():
                    if p.grad is not None:
                        fisher[n].add_(p.grad.detach().cpu().pow(2))

                total_samples += 1

        if total_samples > 0:
            for n in fisher:
                fisher[n] /= float(total_samples)

        flat = [fisher[n].reshape(-1) for n in param_names]
        return torch.cat(flat) if flat else torch.tensor([], dtype=torch.float32)

    def _estimate_vmap(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """vmap-based per-sample gradient computation."""
        params = {name: p for name, p in model.named_parameters() if p.requires_grad}
        buffers = dict(model.named_buffers())

        fisher_req = {name: torch.zeros_like(p, device="cpu") for name, p in params.items()}
        total_samples = 0

        def compute_loss_stateless(params_in, buffers_in, x, y):
            out = functional_call(model, (params_in, buffers_in), (x.unsqueeze(0),))
            logits = _extract_logits(out)
            y_batch = y.unsqueeze(0) if y.dim() == 0 else y.unsqueeze(0)
            if isinstance(criterion, nn.CrossEntropyLoss):
                return F.cross_entropy(logits, y_batch)
            return criterion(logits, y_batch)

        grad_fn = grad(compute_loss_stateless)

        from tqdm import tqdm

        iterator = tqdm(dataloader, desc="Estimating Fisher (vmap)", leave=False)
        for batch in iterator:
            x, y = _unpack_batch(batch)
            x, y = x.to(device), y.to(device)
            bs = x.size(0)
            if bs == 0:
                continue
            total_samples += bs

            batch_grads = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, x, y)
            for name, g in batch_grads.items():
                fisher_req[name].add_((g.detach().cpu().pow(2)).sum(dim=0))

        param_names = [n for n, _ in model.named_parameters()]
        fisher_all = {}
        for n, p in model.named_parameters():
            if n in fisher_req:
                if total_samples > 0:
                    fisher_all[n] = fisher_req[n] / float(total_samples)
                else:
                    fisher_all[n] = fisher_req[n]
            else:
                fisher_all[n] = torch.zeros_like(p, device="cpu")

        flat = [fisher_all[n].reshape(-1) for n in param_names]
        return torch.cat(flat) if flat else torch.tensor([], dtype=torch.float32)


class FullFisherEstimator(FisherEstimator):
    """Full empirical Fisher matrix estimation."""

    def estimate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        model.eval()
        device_obj = _resolve_device(device)

        if _is_cuda(device_obj):
            torch.cuda.empty_cache()

        if batch_size is not None:
            dataset = dataloader.dataset
            n = min(int(batch_size), len(dataset))
            limited_dataset = Subset(dataset, range(n))
            fisher_loader = DataLoader(limited_dataset, batch_size=max(1, n), shuffle=False)
        else:
            fisher_loader = dataloader

        p = sum(param.numel() for param in model.parameters())
        fisher = torch.zeros(p, p, device=device_obj)
        n_samples = 0

        for batch in fisher_loader:
            data, target = _unpack_batch(batch)
            data, target = data.to(device_obj), target.to(device_obj)
            for i in range(data.size(0)):
                model.zero_grad(set_to_none=True)
                logits = _extract_logits(model(data[i : i + 1]))
                y_i = target[i : i + 1]
                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss = F.cross_entropy(logits, y_i)
                else:
                    loss = criterion(logits, y_i)
                loss.backward()

                grad_vec = torch.cat(
                    [
                        p_.grad.view(-1)
                        if p_.grad is not None
                        else torch.zeros(p_.numel(), device=device_obj, dtype=p_.dtype)
                        for p_ in model.parameters()
                    ]
                )
                fisher += torch.outer(grad_vec, grad_vec)
                n_samples += 1

        return fisher / float(n_samples) if n_samples > 0 else fisher


def fisher_norm_distance(
    model: nn.Module,
    old_params: torch.Tensor,
    new_params: torch.Tensor,
    dataloader: DataLoader,
    criterion: nn.Module,
    device,
) -> float:
    """
    Compute Fisher-weighted distance sqrt(d^T F d) without materializing full F.
    Uses identity: d^T F d = (1/N) * sum_i (d^T g_i)^2.
    """
    device_obj = _resolve_device(device)

    saved_params = torch.cat([p.data.view(-1).clone() for p in model.parameters()])

    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(old_params[idx : idx + n].view_as(p))
            idx += n

    diff = (new_params - old_params).to(device_obj)

    model.eval()
    sum_sq_dots = 0.0
    n_samples = 0

    for batch in dataloader:
        data, target = _unpack_batch(batch)
        data, target = data.to(device_obj), target.to(device_obj)

        for i in range(data.size(0)):
            model.zero_grad(set_to_none=True)
            logits = _extract_logits(model(data[i : i + 1]))
            y_i = target[i : i + 1]
            if isinstance(criterion, nn.CrossEntropyLoss):
                loss = F.cross_entropy(logits, y_i)
            else:
                loss = criterion(logits, y_i)
            loss.backward()

            grad_vec = torch.cat(
                [
                    p.grad.view(-1)
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=device_obj, dtype=p.dtype)
                    for p in model.parameters()
                ]
            )

            dot = torch.dot(diff, grad_vec)
            sum_sq_dots += dot.item() ** 2
            n_samples += 1

    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(saved_params[idx : idx + n].view_as(p))
            idx += n

    return float(np.sqrt(sum_sq_dots / n_samples)) if n_samples > 0 else 0.0
