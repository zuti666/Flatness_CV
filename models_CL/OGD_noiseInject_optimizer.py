"""FO_SO_NI_OPT: OGD + EWC with optimizer-aligned structured noise injection."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_CL.baseLearner import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


def _gram_schmidt(vectors: List[torch.Tensor], eps: float = 1e-10) -> List[torch.Tensor]:
    ortho: List[torch.Tensor] = []
    for v in vectors:
        w = v.clone()
        for u in ortho:
            w -= torch.dot(u, v) * u
        norm = w.norm()
        if norm > eps:
            ortho.append(w / norm)
    return ortho


class _ParamBasis:
    def __init__(self, max_rank: int, eps: float = 1e-8):
        self.max_rank = int(max_rank)
        self.eps = float(eps)
        self._basis: torch.Tensor | None = None  # [k, D] on CPU

    def _reorthonormalize(self) -> None:
        if self._basis is None or self._basis.numel() == 0:
            return
        rows = []
        B = self._basis.detach().float().cpu()
        for i in range(B.shape[0]):
            v = B[i].clone()
            for u in rows:
                v -= torch.dot(u, v) * u
            n = v.norm()
            if n > self.eps:
                rows.append(v / n)
        if not rows:
            self._basis = None
            return
        out = torch.stack(rows, dim=0)
        if out.shape[0] > self.max_rank:
            out = out[-self.max_rank :]
        self._basis = out

    def _align_dim(self, dim: int) -> None:
        if self._basis is None:
            return
        cur = int(self._basis.shape[1])
        if cur == dim:
            return
        if cur < dim:
            pad = torch.zeros((self._basis.shape[0], dim - cur), dtype=self._basis.dtype)
            self._basis = torch.cat([self._basis, pad], dim=1)
            return
        self._basis = self._basis[:, :dim]
        self._reorthonormalize()

    def add(self, vec: torch.Tensor) -> None:
        v = vec.detach().float().cpu().reshape(-1)
        if v.numel() == 0:
            return
        self._align_dim(v.numel())
        n = v.norm()
        if n < self.eps:
            return
        v = v / n
        if self._basis is None:
            self._basis = v.unsqueeze(0)
            return
        B = self._basis
        proj = torch.mv(B, v)
        v = v - torch.mv(B.t(), proj)
        n = v.norm()
        if n < self.eps:
            return
        v = v / n
        if B.shape[0] >= self.max_rank:
            B = torch.cat([B[1:], v.unsqueeze(0)], dim=0)
        else:
            B = torch.cat([B, v.unsqueeze(0)], dim=0)
        self._basis = B

    def project(self, grad: torch.Tensor) -> torch.Tensor:
        g = grad.reshape(-1)
        if self._basis is None or self._basis.numel() == 0:
            return grad
        self._align_dim(g.numel())
        if self._basis is None or self._basis.numel() == 0:
            return grad
        B = self._basis.to(device=g.device, dtype=g.dtype)
        coeff = torch.mv(B, g)
        g_proj = g - torch.mv(B.t(), coeff)
        return g_proj.reshape_as(grad)

    @property
    def rank(self) -> int:
        return 0 if self._basis is None else int(self._basis.shape[0])


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

        # OGD hyperparams
        self._ogd_max_dirs = int(args.get("ogd_max_dirs", 60))
        self._ogd_rank_per_param = int(args.get("ogd_rank_per_param", 20))
        self._ogd_store_batches = int(args.get("ogd_store_batches", 50))
        self._ogd_eps = float(args.get("ogd_eps", 1e-8))
        self._noise_std = float(args.get("noise_std", 0.01))
        self._noise_start_task = int(args.get("noise_start_task", 1))
        self._noise_cov_mode = str(args.get("noise_cov_mode", "fisher_inv")).lower()
        self._noise_h_eps = float(args.get("noise_h_eps", 1e-8))
        self._noise_max_precond = float(args.get("noise_max_precond", 1e3))
        self._noise_reproject = self._parse_bool(args.get("noise_reproject", True))
        self._noise_only_with_grad = self._parse_bool(args.get("noise_only_with_grad", True))

        # EWC hyperparams
        self._ewc_lambda = float(args.get("ewc_lambda", 20.0))
        self._ewc_gamma = float(args.get("ewc_gamma", 1.0))
        self._ewc_max_batches = int(args.get("ewc_max_batches", 100))
        self._ewc_eps = float(args.get("ewc_eps", 1e-5))

        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # buffers (CPU)
        self._bases: Dict[str, _ParamBasis] = {}
        self._directions: List[torch.Tensor] = []
        self._fisher: Dict[str, torch.Tensor] | None = None
        self._checkpoint: Dict[str, torch.Tensor] | None = None
        self._shape_mismatch_warned: set[str] = set()
        self._unknown_cov_mode_warned = False

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._align_directions_to_model(self._network)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self._align_buffers_to_model(self._network)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args.get("train_num_workers", 8),
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args.get("train_num_workers", 8),
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[FO_SO] Failed to compute class means: %s", exc)

    def _train(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)

        stage = "init" if self._cur_task == 0 else "update"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage=stage)

        if stage == "init":
            epochs = int(self.args.get("init_epoch", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("init_milestones", []),
                gamma=float(self.args.get("init_lrate_decay", 1.0)),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )
        else:
            epochs = int(self.args.get("epochs", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )

        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            model.train()
            total_loss = 0.0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                optimizer.zero_grad()
                outputs = model(inputs)
                logits = outputs["logits"]

                if stage == "init":
                    loss_task = F.cross_entropy(logits, targets)
                    eval_logits, eval_targets = logits, targets
                else:
                    fake_targets = targets - self._known_classes
                    loss_task = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets

                loss = loss_task + self._ewc_penalty(model)
                loss.backward()
                self._project_gradients(model)
                optimizer.step()
                self._inject_optimizer_structured_noise(model, optimizer)

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss.detach().item())

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => Loss {total_loss/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            if (epoch % 5 == 4):
                test_acc = self._compute_accuracy(model, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            prog_bar.set_description(info)
        logging.info(info)

        # After-task: checkpoint, fisher, bases
        self._checkpoint = {name: p.detach().cpu().clone() for name, p in self._iter_trainable(model)}
        self._fisher = self._compute_fisher(train_loader, model)
        self._update_bases(train_loader)

    # helpers
    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    def _project_gradients(self, model: nn.Module):
        if not self._directions:
            return
        params = [p for _, p in self._iter_trainable(model)]
        flat_grad = self._flatten_current_grads(params)
        if flat_grad.numel() == 0:
            return
        flat_grad = self._project(flat_grad)
        self._assign_flat_grad(params, flat_grad)

    def _flatten_current_grads(self, params: List[torch.nn.Parameter]) -> torch.Tensor:
        flats = []
        for p in params:
            if p.grad is not None:
                flats.append(p.grad.reshape(-1))
        return torch.cat(flats) if flats else torch.tensor([], device=self._device)

    def _assign_flat_grad(self, params: List[torch.nn.Parameter], flat: torch.Tensor) -> None:
        offset = 0
        for p in params:
            if p.grad is None:
                continue
            numel = p.numel()
            p.grad.copy_(flat[offset : offset + numel].reshape_as(p))
            offset += numel

    def _project(self, g: torch.Tensor) -> torch.Tensor:
        if not self._directions or g.numel() == 0:
            return g
        g_proj = g.clone()
        for s in self._directions:
            if g_proj.numel() == s.numel():
                s_use = s
                if s_use.device != g_proj.device or s_use.dtype != g_proj.dtype:
                    s_use = s_use.to(device=g_proj.device, dtype=g_proj.dtype)
                dot = torch.dot(g_proj, s_use)
                g_proj = g_proj - dot * s_use
            else:
                n = min(g_proj.numel(), s.numel())
                if n == 0:
                    continue
                s_part = s[:n]
                if s_part.device != g_proj.device or s_part.dtype != g_proj.dtype:
                    s_part = s_part.to(device=g_proj.device, dtype=g_proj.dtype)
                dot = torch.dot(g_proj[:n], s_part)
                g_proj[:n] = g_proj[:n] - dot * s_part
        return g_proj

    def _align_directions_to_model(self, model: nn.Module) -> None:
        target_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._align_directions_to_dim(target_dim)

    def _align_directions_to_dim(self, target_dim: int) -> None:
        if not self._directions:
            return
        changed = False
        aligned: List[torch.Tensor] = []
        for v in self._directions:
            flat = v.detach().clone().reshape(-1).cpu()
            if flat.numel() == target_dim:
                aligned.append(flat)
                continue
            changed = True
            out = torch.zeros(target_dim, dtype=flat.dtype, device="cpu")
            n = min(target_dim, flat.numel())
            out[:n] = flat[:n]
            aligned.append(out)
        if not changed:
            return
        self._directions = _gram_schmidt(aligned, eps=self._ogd_eps)
        if len(self._directions) > self._ogd_max_dirs:
            self._directions = self._directions[: self._ogd_max_dirs]
        logger.info("[FO_SO_NI_OPT] Aligned %d directions to dim=%d", len(self._directions), target_dim)

    @staticmethod
    def _parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _project_vector(self, name: str, vec: torch.Tensor) -> torch.Tensor:
        basis = self._bases.get(name)
        if basis is None:
            return vec
        return basis.project(vec)

    def _get_fisher_diag_flat(self, name: str, p: torch.Tensor) -> torch.Tensor | None:
        if self._fisher is None:
            return None
        fisher = self._fisher.get(name)
        if fisher is None:
            return None
        if fisher.numel() != p.numel():
            if name not in self._shape_mismatch_warned:
                logger.warning(
                    "[FO_SO_NI_OPT] Fisher shape mismatch for %s: fisher=%s param=%s. Skip precondition.",
                    name,
                    tuple(fisher.shape),
                    tuple(p.shape),
                )
                self._shape_mismatch_warned.add(name)
            return None
        return fisher.reshape(-1).to(device=p.device, dtype=p.dtype)

    def _sample_structured_noise(self, name: str, p: torch.Tensor) -> torch.Tensor:
        z = torch.randn_like(p).reshape(-1)
        z = self._project_vector(name, z)

        if self._noise_cov_mode in {"fisher_inv", "h_inv_d", "h-1d", "hinvd"}:
            fisher_diag = self._get_fisher_diag_flat(name, p)
            if fisher_diag is not None:
                precond = torch.rsqrt(fisher_diag.clamp_min(self._noise_h_eps))
                if self._noise_max_precond > 0.0:
                    precond = precond.clamp_max(self._noise_max_precond)
                z = z * precond
                if self._noise_reproject:
                    z = self._project_vector(name, z)
        elif self._noise_cov_mode not in {"identity", "isotropic"} and not self._unknown_cov_mode_warned:
            logger.warning(
                "[FO_SO_NI_OPT] Unknown noise_cov_mode=%s, fallback to isotropic projected noise.",
                self._noise_cov_mode,
            )
            self._unknown_cov_mode_warned = True

        return (self._noise_std * z).reshape_as(p)

    def _inject_optimizer_structured_noise(self, model: nn.Module, optimizer: Optimizer) -> None:
        # Keep task-0 identical to plain SGD/OGD: no history-driven diffusion yet.
        if self._cur_task < self._noise_start_task or not self._bases:
            return
        if self._noise_std <= 0.0:
            return
        name_by_pid = {id(p): name for name, p in self._iter_trainable(model)}
        with torch.no_grad():
            for group in optimizer.param_groups:
                lr = float(group.get("lr", 0.0))
                if lr <= 0.0:
                    continue
                sqrt_lr = lr ** 0.5
                for p in group.get("params", []):
                    if p is None:
                        continue
                    if self._noise_only_with_grad and p.grad is None:
                        continue
                    name = name_by_pid.get(id(p))
                    if name is None:
                        continue
                    noise = self._sample_structured_noise(name, p)
                    p.add_(sqrt_lr * noise)

    def _ewc_penalty(self, model: nn.Module):
        if self._fisher is None or self._checkpoint is None:
            return torch.tensor(0.0, device=self._device)
        pen = None
        for name, p in self._iter_trainable(model):
            ref = self._checkpoint.get(name)
            fisher = self._fisher.get(name)
            if ref is None or fisher is None:
                continue
            diff = p - ref.to(device=p.device, dtype=p.dtype)
            fisher_d = fisher.to(device=p.device, dtype=p.dtype)
            term = (fisher_d * diff.pow(2)).sum()
            pen = term if pen is None else pen + term
        if pen is None:
            return torch.tensor(0.0, device=self._device)
        return 0.5 * self._ewc_lambda * pen

    def _match_tensor(self, stored: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if stored.shape == param.shape:
            return stored
        new = torch.zeros_like(param.detach().cpu())
        slices = tuple(slice(0, min(a, b)) for a, b in zip(stored.shape, param.shape))
        new[slices] = stored[slices]
        return new

    def _align_buffers_to_model(self, model: nn.Module):
        if self._fisher is not None:
            for name, p in model.named_parameters():
                if name in self._fisher:
                    self._fisher[name] = self._match_tensor(self._fisher[name], p)
        if self._checkpoint is not None:
            for name, p in model.named_parameters():
                if name in self._checkpoint:
                    self._checkpoint[name] = self._match_tensor(self._checkpoint[name], p)

    def _compute_fisher(self, loader, model: nn.Module) -> Dict[str, torch.Tensor]:
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        total = 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= self._ewc_max_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            model.zero_grad()
            with torch.enable_grad():
                logits = model(inputs)["logits"]
                logp = F.log_softmax(logits, dim=1)
                nll = -F.nll_loss(logp, targets, reduction="none")
                exp_cp = torch.mean(torch.exp(nll.detach()))
                loss = nll.mean()
                loss.backward()

            for name, p in self._iter_trainable(model):
                if p.grad is None:
                    continue
                g2 = (p.grad.detach().cpu() ** 2) * exp_cp.cpu()
                fisher[name] = g2 if name not in fisher else fisher[name] + g2
            total += 1

        if total > 0:
            for name in fisher:
                fisher[name] = (fisher[name] / float(total)).clamp_min(self._ewc_eps)

        if self._fisher is not None:
            for name, new_f in fisher.items():
                old = self._fisher.get(name)
                self._fisher[name] = new_f if old is None else self._ewc_gamma * old + new_f
            for name, old in self._fisher.items():
                if name not in fisher:
                    self._fisher[name] = self._ewc_gamma * old
        else:
            self._fisher = fisher

        logger.info(" Fisher estimated over %d batches", total)
        return self._fisher

    def _update_bases(self, loader) -> None:
        model = self._unwrap_network()
        model.eval()
        params = [p for _, p in self._iter_trainable(model)]
        collected_global: List[torch.Tensor] = []
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= self._ogd_store_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            model.zero_grad()
            with torch.enable_grad():
                logits = model(inputs)["logits"]
                idx = torch.arange(len(targets), device=targets.device)
                # Match OGD: use correct-class logit gradients to build direction bank.
                anchor = logits[idx, targets].sum()
                anchor.backward()
            flat = self._flatten_current_grads(params).detach().cpu()
            if flat.numel() > 0:
                collected_global.append(flat)

            for name, p in self._iter_trainable(model):
                if p.grad is None:
                    continue
                vec = p.grad.detach().view(-1).cpu()
                basis = self._bases.get(name)
                if basis is None:
                    basis = _ParamBasis(self._ogd_rank_per_param, eps=self._ogd_eps)
                    self._bases[name] = basis
                basis.add(vec)
        if collected_global:
            new_dirs = _gram_schmidt(collected_global, eps=self._ogd_eps)
            merged = _gram_schmidt(self._directions + new_dirs, eps=self._ogd_eps)
            if len(merged) > self._ogd_max_dirs:
                merged = merged[: self._ogd_max_dirs]
            self._directions = merged
        avg_rank = 0.0 if not self._bases else sum(b.rank for b in self._bases.values()) / len(self._bases)
        logger.info(
            "[FO_SO] Stored bases for %d params (avg rank %.2f), global dirs %d",
            len(self._bases),
            avg_rank,
            len(self._directions),
        )

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
