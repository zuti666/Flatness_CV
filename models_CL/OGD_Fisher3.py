"""OGD + low-rank Fisher RWP with q-orthogonal perturbation correction.

Core update implemented per mini-batch:
1) clean gradient g at theta,
2) OGD drift projection q = P_S_perp g,
3) Fisher-structured perturbation delta ~ F_old^{-1/2} and optional OGD noise projection,
4) perturbed gradient g' at theta + delta,
5) epsilon = g' - g, then epsilon_perp_to_q,
6) final gradient = q + epsilon_perp_to_q.

This keeps the first-order descent component along q while controlling diffusion
with old-task Fisher geometry.
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

from models_CL.baseLearner import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # OGD
        self._ogd_max_dirs = int(args.get("ogd_max_dirs", 60))
        self._ogd_store_batches = int(args.get("ogd_store_batches", 50))
        self._ogd_eps = float(args.get("ogd_eps", 1e-8))

        # RWP
        self._rwp_std = float(args.get("rwp_std", args.get("noise_std", 0.01)))
        self._rwp_start_task = int(args.get("rwp_start_task", args.get("noise_start_task", 1)))
        self._rwp_noise_clip = float(args.get("rwp_noise_clip", 0.0))

        # Fisher low-rank
        self._fisher_gamma = float(args.get("fisher_gamma", args.get("ewc_gamma", 1.0)))
        self._fisher_max_batches = int(args.get("fisher_max_batches", args.get("ewc_max_batches", 100)))
        self._fisher_eps = float(args.get("fisher_eps", args.get("ewc_eps", 1e-5)))
        self._fisher_rank = int(args.get("fisher_rank", 20))
        self._fisher_samples = int(args.get("fisher_samples", 4))

        # q-orthogonalization stability
        self._q_eps = float(args.get("q_eps", 1e-12))
        # whether to enforce q^T epsilon = 0 (ablation: 5.3.3)
        self._use_q_orth = bool(args.get("use_q_orth", True))
        # whether to project epsilon back to OGD-safe subspace (ablation: priority-4)
        # True = original behavior (double projection), False = skip redundant projection
        self._use_epsilon_proj = bool(args.get("use_epsilon_proj", True))

        # OGD basis on CPU
        self._directions: List[torch.Tensor] = []

        # Low-rank Fisher representation on CPU
        self._fisher_basis: List[torch.Tensor] = []
        self._fisher_eigvals: torch.Tensor = torch.tensor([], dtype=torch.float32)

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        self._align_directions_to_model(self._network)
        self._align_fisher_to_model(self._network)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

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

        model = self._unwrap_network()
        self._update_fisher_lowrank(self.train_loader, model)
        self._update_ogd_directions(self.train_loader)

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[OGD_FISHER3] Failed to compute class means: %s", exc)

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

        named_params = list(self._iter_trainable(model))
        trainable_params = [p for _, p in named_params]

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

                # 1) clean gradient at theta
                optimizer.zero_grad()
                outputs_clean = model(inputs)
                logits_clean = outputs_clean["logits"]
                if stage == "init":
                    loss_clean = F.cross_entropy(logits_clean, targets)
                    eval_logits, eval_targets = logits_clean, targets
                else:
                    fake_targets = targets - self._known_classes
                    loss_clean = F.cross_entropy(logits_clean[:, self._known_classes :], fake_targets)
                    eval_logits, eval_targets = logits_clean[:, self._known_classes :], fake_targets
                loss_clean.backward()
                clean_grad = self._flatten_current_grads_full(trainable_params).detach()

                # OGD drift control
                q = self._project(clean_grad)

                # 2) perturbed gradient at theta + delta
                perturbations = self._sample_and_apply_fisher_rwp(trainable_params)
                if perturbations:
                    try:
                        optimizer.zero_grad()
                        outputs_pert = model(inputs)
                        logits_pert = outputs_pert["logits"]
                        if stage == "init":
                            loss_pert = F.cross_entropy(logits_pert, targets)
                        else:
                            fake_targets = targets - self._known_classes
                            loss_pert = F.cross_entropy(logits_pert[:, self._known_classes :], fake_targets)
                        loss_pert.backward()
                        pert_grad = self._flatten_current_grads_full(trainable_params).detach()
                    finally:
                        self._restore_rwp_perturbation(perturbations)

                    epsilon = pert_grad - clean_grad
                    # Keep correction in OGD-safe subspace (ablation: use_epsilon_proj).
                    # True (default) = original double-projection behavior.
                    # False = skip; delta already lives in OGD-safe subspace so
                    #         epsilon ≈ F*delta is approximately safe too.
                    if self._use_epsilon_proj:
                        epsilon = self._project(epsilon)

                    # Enforce q^T epsilon_perp = 0  (ablation flag: use_q_orth).
                    if self._use_q_orth:
                        q_norm2 = torch.dot(q, q)
                        if q_norm2 > self._q_eps:
                            epsilon = epsilon - (torch.dot(q, epsilon) / (q_norm2 + self._q_eps)) * q
                        else:
                            epsilon = torch.zeros_like(epsilon)

                    final_grad = q + epsilon
                else:
                    final_grad = q

                # 3) apply final gradient
                self._assign_flat_grad_full(trainable_params, final_grad)
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
            if epoch % 5 == 4:
                test_acc = self._compute_accuracy(model, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            prog_bar.set_description(info)
        logging.info(info)

    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    @torch.no_grad()
    def _sample_and_apply_fisher_rwp(
        self, params: List[torch.nn.Parameter]
    ) -> List[Tuple[torch.nn.Parameter, torch.Tensor]]:
        if self._cur_task < self._rwp_start_task:
            return []
        if self._rwp_std <= 0.0:
            return []
        if not params:
            return []

        total_dim = sum(p.numel() for p in params)
        if total_dim <= 0:
            return []

        sigma = float(self._rwp_std)
        device = params[0].device
        dtype = params[0].dtype

        z = torch.randn(total_dim, device=device, dtype=dtype)
        flat_noise = self._fisher_inv_sqrt_transform(z)

        # Keep diffusion in OGD-safe subspace.
        if self._directions:
            flat_noise = self._project(flat_noise)

        if self._rwp_noise_clip > 0.0:
            norm = flat_noise.norm()
            if norm > self._rwp_noise_clip:
                flat_noise = flat_noise * (self._rwp_noise_clip / (norm + 1e-12))

        flat_noise = sigma * flat_noise

        perturbations: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        offset = 0
        for p in params:
            numel = p.numel()
            delta = flat_noise[offset : offset + numel].reshape_as(p)
            offset += numel
            p.add_(delta)
            perturbations.append((p, delta))

        return perturbations

    def _fisher_inv_sqrt_transform(self, z: torch.Tensor) -> torch.Tensor:
        if not self._fisher_basis or self._fisher_eigvals.numel() == 0:
            return z

        proj = torch.zeros_like(z)
        parallel = torch.zeros_like(z)

        rank = min(len(self._fisher_basis), int(self._fisher_eigvals.numel()))
        for i in range(rank):
            lam = float(self._fisher_eigvals[i].item())
            if lam <= 0.0:
                continue
            u = self._fisher_basis[i]
            if u.device != z.device or u.dtype != z.dtype:
                u = u.to(device=z.device, dtype=z.dtype)
            coeff = torch.dot(u, z)
            proj = proj + coeff * u
            parallel = parallel + (coeff / math.sqrt(lam + self._fisher_eps)) * u

        residual = z - proj
        # Residual (directions outside Fisher span) kept at unit scale.
        # Previously divided by sqrt(fisher_eps) which amplified ~316x — BUG FIXED.
        return parallel + residual

    @torch.no_grad()
    def _restore_rwp_perturbation(
        self, perturbations: List[Tuple[torch.nn.Parameter, torch.Tensor]]
    ) -> None:
        for p, delta in perturbations:
            p.sub_(delta)

    def _flatten_current_grads_full(self, params: List[torch.nn.Parameter]) -> torch.Tensor:
        flats = []
        for p in params:
            if p.grad is None:
                flats.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            else:
                flats.append(p.grad.reshape(-1))
        return torch.cat(flats) if flats else torch.tensor([], device=self._device)

    def _assign_flat_grad_full(self, params: List[torch.nn.Parameter], flat: torch.Tensor) -> None:
        offset = 0
        for p in params:
            numel = p.numel()
            view = flat[offset : offset + numel].reshape_as(p)
            offset += numel
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.copy_(view)

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
        self._directions = self._gram_schmidt(aligned, eps=self._ogd_eps)
        if len(self._directions) > self._ogd_max_dirs:
            self._directions = self._directions[: self._ogd_max_dirs]
        logger.info("[OGD_FISHER3] Aligned %d directions to dim=%d", len(self._directions), target_dim)

    def _align_fisher_to_model(self, model: nn.Module) -> None:
        if not self._fisher_basis:
            return
        target_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
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

        # Re-orthonormalize after dimension alignment.
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
        logger.info("[OGD_FISHER3] Aligned Fisher rank=%d to dim=%d", len(self._fisher_basis), target_dim)

    @torch.no_grad()
    def _update_fisher_lowrank(self, loader, model: nn.Module) -> None:
        new_basis, new_vals, total = self._compute_task_fisher_lowrank(loader, model)
        if total <= 0 or not new_basis:
            logger.info("[OGD_FISHER3] Fisher update skipped (batches=%d)", total)
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

        logger.info(
            "[OGD_FISHER3] Fisher updated over %d batches, rank=%d",
            total,
            len(self._fisher_basis),
        )

    def _compute_task_fisher_lowrank(
        self, loader, model: nn.Module
    ) -> Tuple[List[torch.Tensor], torch.Tensor, int]:
        model.eval()
        params = [p for _, p in self._iter_trainable(model)]

        samples: List[torch.Tensor] = []
        weights: List[float] = []
        total = 0

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
                logp = F.log_softmax(logits, dim=1)
                nll = -F.nll_loss(logp, targets, reduction="none")
                exp_cp = torch.mean(torch.exp(nll.detach()))
                loss = nll.mean()
                loss.backward()

            flat_grad = self._flatten_current_grads_full(params).detach().cpu()
            if flat_grad.numel() == 0:
                continue

            total += 1
            w = float(exp_cp.item())

            if self._fisher_samples <= 0:
                continue
            if len(samples) < self._fisher_samples:
                samples.append(flat_grad.to(dtype=torch.float16))
                weights.append(w)
            else:
                # Reservoir sampling for bounded memory.
                j = int(np.random.randint(0, total))
                if j < self._fisher_samples:
                    samples[j] = flat_grad.to(dtype=torch.float16)
                    weights[j] = w

        if total == 0 or not samples or self._fisher_rank <= 0:
            return [], torch.tensor([], dtype=torch.float32), total

        basis, vals = self._fit_lowrank_from_samples(samples, weights, total, self._fisher_rank)
        return basis, vals, total

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
            lam = float(old_vals[i].item())
            lam = max(gamma * lam, 0.0)
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

    @torch.no_grad()
    def _update_ogd_directions(self, loader) -> None:
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
                anchor = logits[idx, targets].sum()
                anchor.backward()

            flat = self._flatten_current_grads_full(params).detach().cpu()
            if flat.numel() > 0:
                collected_global.append(flat)

        if collected_global:
            new_dirs = self._gram_schmidt(collected_global, eps=self._ogd_eps)
            merged = self._gram_schmidt(self._directions + new_dirs, eps=self._ogd_eps)
            if len(merged) > self._ogd_max_dirs:
                merged = merged[: self._ogd_max_dirs]
            self._directions = merged

        logger.info("[OGD_FISHER3] Stored global dirs %d", len(self._directions))

    def _gram_schmidt(self, vectors: List[torch.Tensor], eps: float = 1e-10) -> List[torch.Tensor]:
        ortho: List[torch.Tensor] = []
        for v in vectors:
            w = v.clone()
            for u in ortho:
                w -= torch.dot(u, w) * u
            norm = w.norm()
            if norm > eps:
                ortho.append((w / norm).cpu())
        return ortho

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
