"""LoRA-space OGD + geometry-controlled diffusion.

This learner keeps the OGD drift-control basis in the *global flattened LoRA
A/B parameter space*, following `OGD_LoRA`, and adds the `q + epsilon`
perturbation correction from `models_CL/OGD_Fisher3.py`.

Design choice:
- Drift control and diffusion live in the same coordinate system: the trainable
  LoRA A/B parameter space.
- This is different from DeltaW-space EWC-style penalties, which operate on the
  composed LoRA weight update `B @ A`.
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_Project2.models_Full.OGD_utils.gradients import GradientMemory
from models_LoRAbasedCL.OGD_LoRA2 import Learner as OGDLoRALearner
from models_LoRAbasedCL.OGD_LoRA2 import _as_bool
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(OGDLoRALearner):
    """OGD + Gaussian/Fisher diffusion in the LoRA A/B global parameter space."""

    def __init__(self, args):
        super().__init__(args)

        # Disable the legacy diagonal second-order path from OGD_LoRA.
        self._ogd_second_order = False
        self._ogd_so_lambda = 0.0
        self._ogd_fisher_batches = 0

        # RWP / diffusion settings
        self._rwp_std = float(args.get("rwp_std", args.get("noise_std", 0.01)))
        self._rwp_start_task = int(args.get("rwp_start_task", args.get("noise_start_task", 1)))
        self._rwp_noise_clip = float(args.get("rwp_noise_clip", 0.0))
        noise_geometry = str(args.get("noise_geometry", "fisher")).lower()
        if noise_geometry in {"none", "off", "disabled"}:
            self._noise_geometry = "none"
        elif noise_geometry in {"gaussian", "isotropic"}:
            self._noise_geometry = "gaussian"
        elif noise_geometry in {"fisher", "structured"}:
            self._noise_geometry = "fisher"
        elif noise_geometry in {"fisher_shuffled", "structured_shuffled"}:
            self._noise_geometry = "fisher_shuffled"
        else:
            raise ValueError(f"Unsupported noise_geometry={noise_geometry}")

        # Fisher low-rank settings
        self._fisher_gamma = float(args.get("fisher_gamma", args.get("ewc_gamma", 1.0)))
        self._fisher_max_batches = int(args.get("fisher_max_batches", args.get("ewc_max_batches", 100)))
        self._fisher_eps = float(args.get("fisher_eps", args.get("ewc_eps", 1e-5)))
        self._fisher_rank = int(args.get("fisher_rank", 20))
        self._fisher_samples = int(args.get("fisher_samples", 4))

        # q-orthogonalization / epsilon projection
        self._q_eps = float(args.get("q_eps", 1e-12))
        self._use_q_orth = _as_bool(args.get("use_q_orth", True), default=True)
        self._use_epsilon_proj = _as_bool(args.get("use_epsilon_proj", True), default=True)

        # Low-rank Fisher representation in the flattened LoRA A/B space.
        self._fisher_basis: List[torch.Tensor] = []
        self._fisher_eigvals: torch.Tensor = torch.tensor([], dtype=torch.float32)

        # Reinitialize memory explicitly to make the storage intent obvious.
        self._ab_memory = GradientMemory(mode="orthonormal", max_directions=self._ogd_max_dirs)

    def incremental_train(self, data_manager):
        self._refresh_distributed_context()

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._log(f"Learning on {self._known_classes}-{self._total_classes}")
        self._log(
            f"[OGD_FISHER_LoRA] noise_geometry={self._noise_geometry}, "
            f"rwp_std={self._rwp_std:.4e}, rwp_start_task={self._rwp_start_task}, "
            f"fisher_rank={self._fisher_rank}"
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args.get("train_num_workers", 8),
        )

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args.get("train_num_workers", 8),
        )

        network = self._unwrap_network()
        if not self._lora_initialized:
            network.backbone = self.build_lora_backbone()
            network.backbone.to(self._device)
            self._lora_initialized = True
        self._network = network

        self._align_ab_memory_to_model(self._unwrap_network())
        self._align_fisher_to_model(self._unwrap_network())

        self._prepare_network()
        self._train_with_diffusion(self.train_loader, self.test_loader)

        save_dir = self.args.get("filepath", "./")
        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_dir, self._cur_task)

        model = self._unwrap_network()
        self._update_fisher_lowrank(self.train_loader, model)
        self._update_bases(self.train_loader)

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._unwrap_network()

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            self._log(f"[OGD_FISHER_LoRA][NME] Failed to compute class means: {exc}")

    def _train_with_diffusion(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)

        stage = "init" if self._cur_task == 0 else "update"
        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage=stage)
        if self._optimizer_type not in {"sgd", "adam", "adamw"}:
            raise ValueError(
                f"OGD_Fisher_LoRA supports optimizer_type in {{sgd, adam, adamw}}, got {self._optimizer_type}"
            )

        if stage == "init":
            epochs = int(self.args.get("init_epoch", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("init_milestones", self.args.get("milestones", [])),
                gamma=float(self.args.get("init_lrate_decay", self.args.get("lrate_decay", 1.0))),
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

        named_trainable = list(self._iter_trainable(model))
        named_lora_ab = list(self._iter_lora_ab_trainable(model))
        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)

        for epoch in prog_bar:
            self._network.train()
            total_loss = 0.0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                optimizer.zero_grad()
                outputs_clean = self._network(inputs)
                logits_clean = outputs_clean["logits"]
                loss_clean, eval_logits, eval_targets = self._task_loss(logits_clean, targets, stage)
                loss_clean.backward()

                if not named_lora_ab:
                    optimizer.step()
                    with torch.no_grad():
                        _, preds = torch.max(eval_logits, dim=1)
                        correct += preds.eq(eval_targets).cpu().sum()
                        total += len(eval_targets)
                        total_loss += float(loss_clean.detach().item())
                    continue

                clean_snapshot = self._snapshot_current_grads(named_trainable)
                clean_grad = self._get_flat_grad_vector(named_lora_ab).detach()
                q = self._project(clean_grad)

                perturbations = self._sample_and_apply_rwp(named_lora_ab)
                if perturbations:
                    try:
                        optimizer.zero_grad()
                        outputs_pert = self._network(inputs)
                        logits_pert = outputs_pert["logits"]
                        loss_pert, _, _ = self._task_loss(logits_pert, targets, stage)
                        loss_pert.backward()
                        pert_grad = self._get_flat_grad_vector(named_lora_ab).detach()
                    finally:
                        self._restore_rwp_perturbation(perturbations)

                    epsilon = pert_grad - clean_grad
                    if self._use_epsilon_proj:
                        epsilon = self._project(epsilon)
                    if self._use_q_orth:
                        q_norm2 = torch.dot(q, q)
                        if q_norm2 > self._q_eps:
                            epsilon = epsilon - (torch.dot(q, epsilon) / (q_norm2 + self._q_eps)) * q
                        else:
                            epsilon = torch.zeros_like(epsilon)
                    final_grad = q + epsilon
                    self._restore_grad_snapshot(named_trainable, clean_snapshot)
                else:
                    final_grad = q

                self._set_flat_grad_vector(named_lora_ab, final_grad)
                optimizer.step()

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss_clean.detach().item())

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => "
                f"Loss {total_loss/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            )
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)

    def _task_loss(self, logits: torch.Tensor, targets: torch.Tensor, stage: str):
        if stage == "init":
            loss = F.cross_entropy(logits, targets)
            return loss, logits, targets

        fake_targets = targets - self._known_classes
        eval_logits = logits[:, self._known_classes :]
        loss = F.cross_entropy(eval_logits, fake_targets)
        return loss, eval_logits, fake_targets

    def _snapshot_current_grads(
        self, named_params: List[Tuple[str, torch.nn.Parameter]]
    ) -> List[torch.Tensor | None]:
        snapshot: List[torch.Tensor | None] = []
        for _, p in named_params:
            snapshot.append(None if p.grad is None else p.grad.detach().clone())
        return snapshot

    def _restore_grad_snapshot(
        self,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        snapshot: List[torch.Tensor | None],
    ) -> None:
        for (_, p), grad in zip(named_params, snapshot):
            if grad is None:
                p.grad = None
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.copy_(grad)

    def _project(self, g: torch.Tensor) -> torch.Tensor:
        if g.numel() == 0 or len(self._ab_memory) == 0:
            return g
        return self._ab_memory.project_orthogonal(g)

    @torch.no_grad()
    def _sample_and_apply_rwp(
        self, named_params: List[Tuple[str, torch.nn.Parameter]]
    ) -> List[Tuple[torch.nn.Parameter, torch.Tensor]]:
        if self._cur_task < self._rwp_start_task:
            return []
        if self._rwp_std <= 0.0:
            return []
        if self._noise_geometry == "none":
            return []
        if not named_params:
            return []

        total_dim = sum(p.numel() for _, p in named_params)
        if total_dim <= 0:
            return []

        ref_param = named_params[0][1]
        z = torch.randn(total_dim, device=ref_param.device, dtype=ref_param.dtype)
        flat_noise = self._noise_transform(z)

        if len(self._ab_memory) > 0:
            flat_noise = self._project(flat_noise)

        if self._rwp_noise_clip > 0.0:
            norm = flat_noise.norm()
            if norm > self._rwp_noise_clip:
                flat_noise = flat_noise * (self._rwp_noise_clip / (norm + 1e-12))

        flat_noise = float(self._rwp_std) * flat_noise

        perturbations: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        offset = 0
        for _, p in named_params:
            numel = p.numel()
            delta = flat_noise[offset : offset + numel].reshape_as(p)
            offset += numel
            p.add_(delta)
            perturbations.append((p, delta))
        return perturbations

    @torch.no_grad()
    def _restore_rwp_perturbation(
        self, perturbations: List[Tuple[torch.nn.Parameter, torch.Tensor]]
    ) -> None:
        for p, delta in perturbations:
            p.sub_(delta)

    def _noise_transform(self, z: torch.Tensor) -> torch.Tensor:
        if self._noise_geometry == "gaussian":
            return z
        if self._noise_geometry == "fisher":
            return self._fisher_inv_sqrt_transform(z)
        if self._noise_geometry == "fisher_shuffled":
            return self._fisher_inv_sqrt_transform(z, shuffle_eigvals=True)
        return torch.zeros_like(z)

    def _fisher_inv_sqrt_transform(
        self, z: torch.Tensor, shuffle_eigvals: bool = False
    ) -> torch.Tensor:
        if not self._fisher_basis or self._fisher_eigvals.numel() == 0:
            return z

        proj = torch.zeros_like(z)
        parallel = torch.zeros_like(z)
        rank = min(len(self._fisher_basis), int(self._fisher_eigvals.numel()))
        eigvals = self._fisher_eigvals[:rank]
        if shuffle_eigvals:
            perm = torch.randperm(rank)
            eigvals = eigvals[perm]

        for i in range(rank):
            lam = float(eigvals[i].item())
            if lam <= 0.0:
                continue
            u = self._fisher_basis[i]
            if u.device != z.device or u.dtype != z.dtype:
                u = u.to(device=z.device, dtype=z.dtype)
            coeff = torch.dot(u, z)
            proj = proj + coeff * u
            parallel = parallel + (coeff / math.sqrt(lam + self._fisher_eps)) * u

        residual = z - proj
        return parallel + residual

    def _current_lora_dim(self, model: nn.Module) -> int:
        return sum(p.numel() for _, p in self._iter_lora_ab_trainable(model))

    def _align_ab_memory_to_model(self, model: nn.Module) -> None:
        target_dim = self._current_lora_dim(model)
        if target_dim <= 0 or len(self._ab_memory) == 0:
            return

        changed = False
        aligned: List[torch.Tensor] = []
        for v in self._ab_memory.vectors:
            flat = v.detach().clone().reshape(-1).cpu()
            if flat.numel() != target_dim:
                changed = True
                out = torch.zeros(target_dim, dtype=flat.dtype, device="cpu")
                n = min(target_dim, flat.numel())
                out[:n] = flat[:n]
                flat = out
            aligned.append(flat)

        if not changed:
            return

        new_memory = GradientMemory(mode="orthonormal", max_directions=self._ogd_max_dirs)
        for v in aligned:
            new_memory.add(v)
        self._ab_memory = new_memory
        self._log(f"[OGD_FISHER_LoRA] Aligned {len(self._ab_memory)} OGD directions to dim={target_dim}")

    def _align_fisher_to_model(self, model: nn.Module) -> None:
        if not self._fisher_basis:
            return

        target_dim = self._current_lora_dim(model)
        if target_dim <= 0:
            self._fisher_basis = []
            self._fisher_eigvals = torch.tensor([], dtype=torch.float32)
            return

        aligned_basis: List[torch.Tensor] = []
        aligned_vals: List[float] = []
        rank = min(len(self._fisher_basis), int(self._fisher_eigvals.numel()))

        for i in range(rank):
            lam = float(self._fisher_eigvals[i].item())
            if lam <= 0.0:
                continue
            flat = self._fisher_basis[i].detach().clone().reshape(-1).cpu()
            if flat.numel() != target_dim:
                out = torch.zeros(target_dim, dtype=flat.dtype, device="cpu")
                n = min(target_dim, flat.numel())
                out[:n] = flat[:n]
                flat = out
            aligned_basis.append(flat)
            aligned_vals.append(lam)

        if not aligned_basis:
            self._fisher_basis = []
            self._fisher_eigvals = torch.tensor([], dtype=torch.float32)
            return

        basis: List[torch.Tensor] = []
        vals: List[float] = []
        for vec, lam in sorted(zip(aligned_basis, aligned_vals), key=lambda x: x[1], reverse=True):
            w = vec.clone()
            for u in basis:
                w -= torch.dot(u, w) * u
            norm = w.norm()
            if norm > self._ogd_eps:
                basis.append((w / norm).cpu())
                vals.append(max(lam, self._fisher_eps))
            if len(basis) >= self._fisher_rank:
                break

        self._fisher_basis = basis
        self._fisher_eigvals = torch.tensor(vals, dtype=torch.float32)
        self._log(f"[OGD_FISHER_LoRA] Aligned Fisher rank={len(self._fisher_basis)} to dim={target_dim}")

    @torch.no_grad()
    def _update_fisher_lowrank(self, loader, model: nn.Module) -> None:
        if self._noise_geometry == "none":
            self._log(f"[OGD_FISHER_LoRA] Fisher update disabled for noise_geometry={self._noise_geometry}")
            return
        if self._fisher_rank <= 0:
            self._log(f"[OGD_FISHER_LoRA] Fisher update disabled because fisher_rank={self._fisher_rank}")
            return

        new_basis, new_vals, total = self._compute_task_fisher_lowrank(loader, model)
        if total <= 0 or not new_basis:
            self._log(f"[OGD_FISHER_LoRA] Fisher update skipped (batches={total})")
            return

        if self._fisher_basis and self._fisher_eigvals.numel() > 0:
            merged_basis, merged_vals = self._merge_lowrank_fisher(
                self._fisher_basis,
                self._fisher_eigvals,
                new_basis,
                new_vals,
                gamma=self._fisher_gamma,
                rank=self._fisher_rank,
            )
            self._fisher_basis = merged_basis
            self._fisher_eigvals = merged_vals
        else:
            self._fisher_basis = new_basis[: self._fisher_rank]
            self._fisher_eigvals = new_vals[: self._fisher_rank].clone()

        self._log(
            f"[OGD_FISHER_LoRA] Fisher updated over {total} batches, rank={len(self._fisher_basis)}"
        )

    def _compute_task_fisher_lowrank(
        self, loader, model: nn.Module
    ) -> Tuple[List[torch.Tensor], torch.Tensor, int]:
        was_training = model.training
        model.eval()
        named_lora_ab = list(self._iter_lora_ab_trainable(model))
        if not named_lora_ab:
            return [], torch.tensor([], dtype=torch.float32), 0

        samples: List[torch.Tensor] = []
        weights: List[float] = []
        total = 0
        stage = "init" if self._cur_task == 0 else "update"

        try:
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= self._fisher_max_batches:
                    break

                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                model.zero_grad()
                with torch.enable_grad():
                    logits = model(inputs)["logits"]
                    fisher_logits, fisher_targets = self._fisher_logits_and_targets(logits, targets, stage)
                    logp = F.log_softmax(fisher_logits, dim=1)
                    loglik = -F.nll_loss(logp, fisher_targets, reduction="none")
                    exp_cp = torch.mean(torch.exp(loglik.detach()))
                    loss = loglik.mean()
                    loss.backward()

                flat_grad = self._get_flat_grad_vector(named_lora_ab).detach().cpu()
                if flat_grad.numel() == 0:
                    continue

                total += 1
                weight = float(exp_cp.item())
                if self._fisher_samples <= 0:
                    continue
                if len(samples) < self._fisher_samples:
                    samples.append(flat_grad.to(dtype=torch.float16))
                    weights.append(weight)
                else:
                    j = int(np.random.randint(0, total))
                    if j < self._fisher_samples:
                        samples[j] = flat_grad.to(dtype=torch.float16)
                        weights[j] = weight
        finally:
            if was_training:
                model.train()

        if total == 0 or not samples or self._fisher_rank <= 0:
            return [], torch.tensor([], dtype=torch.float32), total

        basis, vals = self._fit_lowrank_from_samples(samples, weights, total, self._fisher_rank)
        return basis, vals, total

    def _fisher_logits_and_targets(self, logits: torch.Tensor, targets: torch.Tensor, stage: str):
        if stage == "init":
            return logits, targets
        return logits[:, self._known_classes :], targets - self._known_classes

    def _fit_lowrank_from_samples(
        self,
        samples: List[torch.Tensor],
        weights: List[float],
        total_batches: int,
        rank: int,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        m = len(samples)
        if m == 0 or rank <= 0:
            return [], torch.tensor([], dtype=torch.float32)

        vecs = [s.to(dtype=torch.float32) for s in samples]
        scales = [math.sqrt(max(w, 0.0) / max(float(total_batches), 1.0)) for w in weights]

        gram = torch.zeros((m, m), dtype=torch.float64)
        for i in range(m):
            vi = vecs[i]
            for j in range(i, m):
                vj = vecs[j]
                val = float(torch.dot(vi, vj).item()) * scales[i] * scales[j]
                gram[i, j] = val
                gram[j, i] = val

        evals, evecs = torch.linalg.eigh(gram)
        order = torch.argsort(evals, descending=True)

        basis: List[torch.Tensor] = []
        vals: List[float] = []
        for idx in order.tolist():
            lam = float(evals[idx].item())
            if lam <= self._fisher_eps:
                continue
            coeffs = evecs[:, idx]
            v = torch.zeros_like(vecs[0])
            for i in range(m):
                alpha = float(coeffs[i].item()) * scales[i]
                if alpha != 0.0:
                    v.add_(vecs[i], alpha=alpha)
            norm = v.norm()
            if norm <= self._ogd_eps:
                continue
            basis.append((v / norm).cpu())
            vals.append(max(lam, self._fisher_eps))
            if len(basis) >= rank:
                break

        if not basis:
            return [], torch.tensor([], dtype=torch.float32)
        return basis, torch.tensor(vals, dtype=torch.float32)

    def _merge_lowrank_fisher(
        self,
        old_basis: List[torch.Tensor],
        old_vals: torch.Tensor,
        new_basis: List[torch.Tensor],
        new_vals: torch.Tensor,
        gamma: float,
        rank: int,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        cand_basis: List[torch.Tensor] = []
        cand_scales: List[float] = []

        old_rank = min(len(old_basis), int(old_vals.numel()))
        for i in range(old_rank):
            lam = max(gamma * float(old_vals[i].item()), 0.0)
            if lam <= self._fisher_eps:
                continue
            cand_basis.append(old_basis[i])
            cand_scales.append(math.sqrt(lam))

        new_rank = min(len(new_basis), int(new_vals.numel()))
        for i in range(new_rank):
            lam = float(new_vals[i].item())
            if lam <= self._fisher_eps:
                continue
            cand_basis.append(new_basis[i])
            cand_scales.append(math.sqrt(lam))

        n = len(cand_basis)
        if n == 0:
            return [], torch.tensor([], dtype=torch.float32)

        gram = torch.zeros((n, n), dtype=torch.float64)
        for i in range(n):
            ui = cand_basis[i]
            si = cand_scales[i]
            for j in range(i, n):
                uj = cand_basis[j]
                sj = cand_scales[j]
                val = float(torch.dot(ui, uj).item()) * si * sj
                gram[i, j] = val
                gram[j, i] = val

        evals, evecs = torch.linalg.eigh(gram)
        order = torch.argsort(evals, descending=True)

        merged_basis: List[torch.Tensor] = []
        merged_vals: List[float] = []
        for idx in order.tolist():
            lam = float(evals[idx].item())
            if lam <= self._fisher_eps:
                continue
            coeffs = evecs[:, idx]
            v = torch.zeros_like(cand_basis[0])
            for i in range(n):
                alpha = float(coeffs[i].item()) * cand_scales[i]
                if alpha != 0.0:
                    v.add_(cand_basis[i], alpha=alpha)
            norm = v.norm()
            if norm <= self._ogd_eps:
                continue
            merged_basis.append((v / norm).cpu())
            merged_vals.append(max(lam, self._fisher_eps))
            if len(merged_basis) >= rank:
                break

        return merged_basis, torch.tensor(merged_vals, dtype=torch.float32)
