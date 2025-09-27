import argparse
import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from contextlib import contextmanager, nullcontext


def _sdp_disable_context():
    """Disable flash/efficient attention kernels during higher-order autograd.

    On Ampere+ GPUs timm's ViT defaults to Flash Attention. The forward and
    first-order backward pass are supported, but double backward (needed for
    Hessian-vector products) is not. Wrapping the relevant calls in this context
    forces PyTorch to fall back to math kernels, restoring higher-order
    differentiation at the expense of a modest slowdown.
    """
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=True
        )
    return nullcontext()

# ---------------------------------------------------------------------------
# Helpers for flattening perturbations
# ---------------------------------------------------------------------------


@dataclass
class FlatnessConfig:
    """Configuration knobs for the flatness/sharpness estimators."""

    rho: float = 0.05
    num_random_samples: int = 10
    gaussian_std: Optional[float] = None
    max_batches: Optional[int] = 1
    power_iters: int = 5
    trace_samples: int = 5
    grad_batches: Optional[int] = 1


def _unwrap_batch(batch):
    """Convert a ``(idx, inputs, targets)`` batch into tensors only."""
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            _, inputs, targets = batch
            return inputs, targets
        if len(batch) == 2:
            return batch
    raise ValueError("Unexpected batch format")


def _clone_params(params: Iterable[torch.nn.Parameter]) -> List[torch.Tensor]:
    """Detach and clone parameter tensors for later restoration."""
    return [p.data.detach().clone() for p in params]


def _restore_params(params: Iterable[torch.nn.Parameter], copies: List[torch.Tensor]):
    for p, saved in zip(params, copies):
        p.data.copy_(saved)


def _add_vector_to_params(params: List[torch.nn.Parameter], vec: torch.Tensor):
    """Add a flattened vector ``vec`` onto a list of parameters in-place."""
    pointer = 0
    for p in params:
        numel = p.numel()
        slice_vec = vec[pointer : pointer + numel].view_as(p)
        p.data.add_(slice_vec)
        pointer += numel


def _compute_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    known_classes: Optional[int] = None,
) -> float:
    """Average cross-entropy loss on ``loader`` (used for base/E–Sh estimates)."""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs, targets = _unwrap_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
            if known_classes is not None and known_classes > 0:
                loss = criterion(logits[:, known_classes:], targets - known_classes)
            else:
                loss = criterion(logits, targets)
            total_loss += loss.item()
            total_samples += targets.size(0)

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


def _compute_grad_vector(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    max_batches: Optional[int] = None,
    known_classes: Optional[int] = None,
) -> torch.Tensor:
    """Return the flattened gradient of the empirical loss w.r.t. ``params``."""
    model.train()
    for p in params:
        if p.grad is not None:
            p.grad = None

    criterion = nn.CrossEntropyLoss(reduction="mean")
    batches_processed = 0
    with _sdp_disable_context():
        for batch_idx, batch in enumerate(loader):
            inputs, targets = _unwrap_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            if known_classes is not None and known_classes > 0:
                loss = criterion(logits[:, known_classes:], targets - known_classes)
            else:
                loss = criterion(logits, targets)
            loss.backward()
            batches_processed += 1

            if max_batches is not None and batches_processed >= max_batches:
                break

    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.detach().clone().view(-1))

    grad_vector = torch.cat(grads)
    return grad_vector


def _hessian_vector_product(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    vec: torch.Tensor,
    max_batches: Optional[int] = 1,
    known_classes: Optional[int] = None,
) -> torch.Tensor:
    """Monte-Carlo estimate of ``H v`` using double-backprop."""
    vec = vec.to(device)
    pointer = 0
    vec_splits = []
    for p in params:
        numel = p.numel()
        vec_splits.append(vec[pointer : pointer + numel].view_as(p))
        pointer += numel

    hvp_accumulator = torch.zeros_like(vec)
    batches_processed = 0
    criterion = nn.CrossEntropyLoss(reduction="mean")

    with _sdp_disable_context():
        for batch_idx, batch in enumerate(loader):
            inputs, targets = _unwrap_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)
            outputs = model(inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            if known_classes is not None and known_classes > 0:
                loss = criterion(logits[:, known_classes:], targets - known_classes)
            else:
                loss = criterion(logits, targets)

            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=True,
                allow_unused=True,
            )
            grad_terms = []
            for p, g in zip(params, grads):
                if g is None:
                    grad_terms.append(torch.zeros_like(p).view(-1))
                else:
                    grad_terms.append(g.contiguous().view(-1))
            grad_vec = torch.cat(grad_terms)
            grad_v = torch.dot(grad_vec, vec)
            hv = torch.autograd.grad(
                grad_v,
                params,
                retain_graph=False,
                allow_unused=True,
            )
            hv_terms = []
            for p, h in zip(params, hv):
                if h is None:
                    hv_terms.append(torch.zeros_like(p).reshape(-1))
                else:
                    hv_terms.append(h.detach().reshape(-1))
            hv_flat = torch.cat(hv_terms)
            hvp_accumulator += hv_flat

            batches_processed += 1
            if max_batches is not None and batches_processed >= max_batches:
                break

    if batches_processed == 0:
        return hvp_accumulator
    return hvp_accumulator / batches_processed


def _power_iteration_lambda_max(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    dim: int,
    num_iters: int,
    max_batches: Optional[int],
    known_classes: Optional[int],
) -> float:
    """Estimate the dominant Hessian eigenvalue ``lambda_max`` via power iteration."""
    if num_iters <= 0:
        return 0.0

    vec = torch.randn(dim, device=device)
    vec = vec / (vec.norm() + 1e-12)
    eigenvalue = 0.0

    for _ in range(num_iters):
        hv = _hessian_vector_product(
            model, loader, device, params, vec, max_batches=max_batches, known_classes=known_classes
        )
        norm = hv.norm()
        if norm.item() == 0:
            eigenvalue = 0.0
            break
        vec = hv / (norm + 1e-12)
        eigenvalue = norm.item()

    return float(eigenvalue)


def _hutchinson_trace(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    dim: int,
    num_samples: int,
    max_batches: Optional[int],
    known_classes: Optional[int],
) -> float:
    """Estimate ``tr(H)`` using the Hutchinson trace estimator with Rademacher noise."""
    if num_samples <= 0:
        return 0.0

    trace_estimate = 0.0
    for _ in range(num_samples):
        vec = torch.randint_like(torch.zeros(dim, device=device), low=0, high=2, dtype=torch.float32)
        vec = vec * 2 - 1  # Rademacher
        hv = _hessian_vector_product(
            model, loader, device, params, vec, max_batches=max_batches, known_classes=known_classes
        )
        trace_estimate += torch.dot(vec, hv).item()

    return trace_estimate / num_samples


def evaluate_flatness_metrics(
    network: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Optional[FlatnessConfig] = None,
    known_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Compute a suite of sharpness/flatness proxies for the current model.

    ``base_loss`` and ``sh0_max`` map onto zeroth-order sharpness definitions,
    ``grad_norm`` / ``first_order_sharpness`` correspond to ``Sh^{(1)}``, while
    ``lambda_max`` / ``hessian_trace`` approximate the curvature of the second
    order Taylor expansion. The Monte-Carlo term provides the distributional
    sharpness ``E-Sh``.

    Pass a model exposing only LoRA parameters to restrict the analysis to that
    subspace.
    """
    config = config or FlatnessConfig()
    wrapped_model = network.module if isinstance(network, nn.DataParallel) else network
    params = [p for p in wrapped_model.parameters() if p.requires_grad]
    if not params:
        return {"base_loss": 0.0}

    flat_metrics: Dict[str, float] = {}

    base_loss = _compute_loss(
        wrapped_model, loader, device, max_batches=config.max_batches, known_classes=known_classes
    )
    flat_metrics["base_loss"] = float(base_loss)

    grad_vector = _compute_grad_vector(
        wrapped_model,
        loader,
        device,
        params,
        max_batches=config.grad_batches,
        known_classes=known_classes,
    )
    grad_norm = grad_vector.norm().item()
    flat_metrics["grad_norm"] = grad_norm
    flat_metrics["first_order_sharpness"] = config.rho * grad_norm

    total_dim = grad_vector.numel()
    param_backup = _clone_params(params)

    if grad_norm > 0:
        direction = grad_vector / (grad_norm + 1e-12)
        perturb = direction * config.rho
        _add_vector_to_params(params, perturb)
        perturbed_loss = _compute_loss(
            wrapped_model, loader, device, max_batches=config.max_batches, known_classes=known_classes
        )
        sh0 = perturbed_loss - base_loss
        flat_metrics["sh0_max"] = float(sh0)
        _restore_params(params, param_backup)
    else:
        flat_metrics["sh0_max"] = 0.0

    # Random expectation sharpness
    gaussian_std = config.gaussian_std or (config.rho / (total_dim ** 0.5))
    rand_losses: List[float] = []
    for _ in range(config.num_random_samples):
        noise = torch.randn(total_dim, device=device) * gaussian_std
        _add_vector_to_params(params, noise)
        loss = _compute_loss(
            wrapped_model, loader, device, max_batches=config.max_batches, known_classes=known_classes
        )
        rand_losses.append(loss - base_loss)
        _restore_params(params, param_backup)

    if rand_losses:
        rand_tensor = torch.tensor(rand_losses)
        flat_metrics["esh_mean"] = float(rand_tensor.mean().item())
        flat_metrics["esh_std"] = float(rand_tensor.std(unbiased=False).item())
    else:
        flat_metrics["esh_mean"] = 0.0
        flat_metrics["esh_std"] = 0.0

    # Hessian spectral proxies
    lambda_max = _power_iteration_lambda_max(
        wrapped_model,
        loader,
        device,
        params,
        dim=total_dim,
        num_iters=config.power_iters,
        max_batches=config.max_batches,
        known_classes=known_classes,
    )
    flat_metrics["lambda_max"] = lambda_max

    trace_est = _hutchinson_trace(
        wrapped_model,
        loader,
        device,
        params,
        dim=total_dim,
        num_samples=config.trace_samples,
        max_batches=config.max_batches,
        known_classes=known_classes,
    )
    flat_metrics["hessian_trace"] = trace_est

    _restore_params(params, param_backup)
    wrapped_model.zero_grad(set_to_none=True)
    return flat_metrics


# ---------------------------------------------------------------------------
# Optional CLI to evaluate flatness from a stored config/checkpoint
# ---------------------------------------------------------------------------


def _load_args(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser(description="Evaluate flatness metrics for a trained model")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_batches", type=int, default=1)
    parser.add_argument("--power_iters", type=int, default=5)
    parser.add_argument("--trace_samples", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Deferred imports to avoid circular deps when this module is imported during training
    from utils import factory
    from utils.data_manager import DataManager

    config = _load_args(args.config)
    config.setdefault("device", [args.device])

    data_manager = DataManager(
        config["dataset"],
        config["shuffle"],
        config["seed"][0] if isinstance(config["seed"], list) else config["seed"],
        config["init_cls"],
        config["increment"],
        config,
    )

    model = factory.get_model(config["model_name"], config)
    # Assume the learner can rebuild the backbone for evaluation (implementation dependent)
    device = torch.device(args.device)
    model._network.to(device)

    # Evaluate using the last task's training loader
    train_dataset = data_manager.get_dataset(
        np.arange(model._known_classes, model._total_classes) if model._total_classes > model._known_classes else np.arange(config["init_cls"]),
        source="train",
        mode="train",
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 128),
        shuffle=False,
        num_workers=2,
    )

    flat_config = FlatnessConfig(
        rho=args.rho,
        num_random_samples=args.num_samples,
        max_batches=args.max_batches,
        power_iters=args.power_iters,
        trace_samples=args.trace_samples,
    )

    metrics = evaluate_flatness_metrics(model._network, loader, device, flat_config)
    logging.info("Flatness metrics: %s", metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
