"""GEM_noise_pert: drift-diffusion continual learning in perturbation form.

Per-iteration update (shared backbone space):
1) Compute clean gradient g_t on current minibatch.
2) Drift control: project g_t against stored old-task direction v_t -> q_t.
3) Build structured covariance Sigma_t from (F1 - beta * F2 + eps I)^(-1).
4) Sample perturbation delta_t, then orthogonalize wrt q_t: delta_t^perp.
5) Recompute gradient at perturbed parameters q'_t = grad(theta + delta_t^perp).
6) Diffusion correction: g_corr = q_t + lambda * (q'_t - q_t).
7) Optimizer step with corrected shared gradient (head gradient stays clean).
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


def _flatten_grads(
    params: List[torch.nn.Parameter],
    device: torch.device,
    *,
    fill_zeros: bool = False,
) -> torch.Tensor:
    flats = []
    for p in params:
        if p.grad is not None:
            flats.append(p.grad.view(-1))
        elif fill_zeros:
            flats.append(torch.zeros_like(p).view(-1))
    if not flats:
        return torch.tensor([], device=device)
    return torch.cat(flats)


def _overwrite_grads_full(params: List[torch.nn.Parameter], new_flat: torch.Tensor) -> None:
    """Overwrite gradients using the full parameter layout.

    `new_flat` is assumed to be flattened over all `params` in order.
    Parameters without gradients keep `grad=None` but still advance offset.
    """
    expected = sum(p.numel() for p in params)
    if int(new_flat.numel()) != int(expected):
        return

    offset = 0
    for p in params:
        numel = p.numel()
        if p.grad is not None:
            p.grad.copy_(new_flat[offset : offset + numel].view_as(p))
        offset += numel


def _project_against_direction(
    g: torch.Tensor,
    v: torch.Tensor | None,
    *,
    min_denom: float = 1e-12,
    only_if_interfere: bool = True,
) -> torch.Tensor:
    """Drift control used in Algorithm DDCL perturbation form.

    q = g - <g, v> / ||v||^2 * v
    If `only_if_interfere=True`, apply only when <g, v> < 0.
    """
    if v is None or v.numel() == 0 or g.numel() == 0:
        return g
    if int(v.numel()) != int(g.numel()):
        return g
    v_norm2 = torch.dot(v, v)
    if v_norm2 <= min_denom:
        return g
    dot = torch.dot(g, v)
    if only_if_interfere and dot >= 0:
        return g
    return g - (dot / (v_norm2 + min_denom)) * v


def _flatten_grads_from_list(
    grads: List[torch.Tensor | None],
    params: List[torch.nn.Parameter],
    device: torch.device,
) -> torch.Tensor:
    """Flatten `torch.autograd.grad` output aligned to `params`."""
    flats: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            flats.append(torch.zeros_like(p, device=device).view(-1))
        else:
            flats.append(g.contiguous().view(-1))
    if not flats:
        return torch.tensor([], device=device)
    return torch.cat(flats)


def _orthonormalize_cols(mat: torch.Tensor, max_cols: int | None = None, eps: float = 1e-10) -> torch.Tensor:
    """Lightweight Gram-Schmidt for small column count.

    Avoids heavy `torch.linalg.qr` on very high-dimensional matrices.
    """
    if mat.numel() == 0:
        return mat
    d, k = mat.shape
    if d == 0 or k == 0:
        return mat[:, :0]
    if max_cols is not None:
        k = min(k, int(max_cols))

    cols: List[torch.Tensor] = []
    for i in range(k):
        v = mat[:, i]
        for u in cols:
            v = v - torch.dot(u, v) * u
        n = v.norm()
        if n > eps:
            cols.append(v / (n + 1e-12))
    if not cols:
        return mat[:, :0]
    return torch.stack(cols, dim=1)


def _safe_eigh_symmetric(
    mat: torch.Tensor,
    *,
    eig_floor: float,
    jitter_base: float,
    max_tries: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robust eigendecomposition for small symmetric matrices.

    Strategy:
    1) Explicit symmetrization.
    2) Retry with increasing diagonal jitter.
    3) Final fallback to diagonal-only eigenvalues.
    """
    # Enforce symmetry and finite values before decomposition.
    sym = 0.5 * (mat + mat.t())
    if not torch.isfinite(sym).all():
        sym = torch.nan_to_num(sym, nan=0.0, posinf=0.0, neginf=0.0)

    n = int(sym.shape[0])
    if n == 0:
        return torch.tensor([], device=sym.device, dtype=sym.dtype), torch.empty_like(sym)

    eye = torch.eye(n, device=sym.device, dtype=sym.dtype)
    base = max(float(jitter_base), float(eig_floor), 1e-8)

    for i in range(max_tries):
        jitter = 0.0 if i == 0 else base * (10.0 ** (i - 1))
        try:
            evals, evecs = torch.linalg.eigh(sym + jitter * eye)
            evals = torch.clamp(evals, min=float(eig_floor))
            return evals, evecs
        except RuntimeError:
            continue

    # Last-resort fallback: diagonal approximation.
    diag = torch.diagonal(sym)
    evals = torch.clamp(diag, min=float(eig_floor))
    evecs = eye
    return evals, evecs


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # Gradient snapshot controls
        self._gem_grad_batches = int(args.get("gem_grad_batches", 1))

        # Drift control (single old-task direction).
        self._drift_only_if_interfere = self._parse_bool(
            args.get("gem_noise_pert_only_if_interfere", True)
        )

        # Structured noise controls
        self._use_structured_noise = self._parse_bool(args.get("use_structured_noise", True))
        self._noise_start_task = int(args.get("noise_start_task", 1))
        self._noise_alpha = float(args.get("gem_noise_alpha", 1e-3))
        self._noise_sigma = float(args.get("gem_noise_sigma", 1.0))
        self._noise_beta = float(args.get("gem_noise_beta", 0.2))
        self._noise_eps = float(args.get("gem_noise_eps", 1e-4))
        self._noise_rank_cap = int(args.get("gem_noise_rank_cap", 32))
        self._noise_rank_requested = int(args.get("gem_noise_rank", 20))
        self._noise_rank = max(1, min(self._noise_rank_requested, self._noise_rank_cap))
        self._noise_min_denom = float(args.get("gem_noise_min_denom", 1e-6))
        self._noise_eig_clip = float(args.get("gem_noise_eig_clip", 1e-3))
        self._noise_orth_scale = float(args.get("gem_noise_orth_scale", 1.0))
        self._noise_clip_ratio = float(args.get("gem_noise_clip_ratio", 1.0))
        self._noise_trace_normalize = self._parse_bool(args.get("gem_noise_trace_normalize", True))
        self._use_adaptive_sigma = self._parse_bool(args.get("gem_noise_pert_adaptive_sigma", True))

        # Diffusion correction strength lambda in g_corr = q + lambda * (q' - q).
        self._pert_lambda = float(args.get("gem_noise_pert_lambda", 0.5))
        self._use_perturb_correction = self._parse_bool(args.get("gem_noise_pert_enable", True))

        # Shared-space past task gradients (CPU; one vector per task).
        self._task_grads_shared: List[torch.Tensor] = []

        # Shared-space low-rank Fisher cache (CPU).
        self._fisher_basis: torch.Tensor | None = None  # [P_shared, K]
        self._fisher_eigvals: torch.Tensor | None = None  # [K]
        self._fisher_dim_warned = False
        self._mem_dim_warned = False

    @staticmethod
    def _parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        if self._noise_rank_requested > self._noise_rank_cap:
            logger.warning(
                "[GEM_noise_pert] gem_noise_rank=%d capped to %d for stability.",
                self._noise_rank_requested,
                self._noise_rank_cap,
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

        stage = "init" if self._cur_task == 0 else "update"
        self._train(self.train_loader, self.test_loader, stage=stage)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Store one shared-gradient snapshot for the just-finished task.
        g_task = self._compute_task_gradient(self.train_loader, self._unwrap_network(), stage=stage)
        if g_task is not None:
            self._task_grads_shared.append(g_task.cpu())

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[GEM_noise_pert] Failed to compute class means: %s", exc)

    def _train(self, train_loader, test_loader, *, stage: str):
        model = self._unwrap_network()
        model.to(self._device)

        params = [p for p in model.parameters() if p.requires_grad]
        shared_params = [p for _, p in self._iter_shared_trainable(model)]
        optimizer = self._build_optimizer(params, stage=stage)

        self._refresh_fisher_cache(shared_params)

        epochs = int(self.args.get("init_epoch" if self._cur_task == 0 else "epochs", 1))
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("init_milestones" if self._cur_task == 0 else "milestones", []),
            gamma=float(self.args.get("init_lrate_decay" if self._cur_task == 0 else "lrate_decay", 1.0)),
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
                loss, eval_logits, eval_targets = self._compute_stage_loss(logits, targets, stage)
                loss.backward()

                g_shared = _flatten_grads(shared_params, self._device, fill_zeros=True)
                v_shared = self._build_old_direction(
                    target_dim=int(g_shared.numel()),
                    device=g_shared.device,
                    dtype=g_shared.dtype,
                )
                q_shared = _project_against_direction(
                    g_shared,
                    v_shared,
                    min_denom=max(float(self._noise_min_denom), 1e-12),
                    only_if_interfere=self._drift_only_if_interfere,
                )
                if q_shared.numel() > 0:
                    _overwrite_grads_full(shared_params, q_shared)

                if self._use_perturb_correction and self._use_structured_noise:
                    q_shared = self._apply_perturbation_correction(
                        model=model,
                        shared_params=shared_params,
                        inputs=inputs,
                        targets=targets,
                        stage=stage,
                        q_shared=q_shared,
                        g_shared=g_shared,
                    )

                optimizer.step()

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss.detach().item())

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

    def _apply_perturbation_correction(
        self,
        *,
        model: nn.Module,
        shared_params: List[torch.nn.Parameter],
        inputs: torch.Tensor,
        targets: torch.Tensor,
        stage: str,
        q_shared: torch.Tensor,
        g_shared: torch.Tensor,
    ) -> torch.Tensor:
        if self._cur_task < self._noise_start_task:
            return q_shared
        if q_shared is None or q_shared.numel() == 0:
            return q_shared
        if self._noise_sigma <= 0.0 or self._noise_alpha <= 0.0:
            return q_shared

        basis_old, eigvals_old = self._get_fisher_basis_for_dim(
            int(q_shared.numel()), device=q_shared.device, dtype=q_shared.dtype
        )
        basis_cur, eigvals_cur = self._build_current_fisher_from_grad_rank1(g_shared)
        delta = self._sample_structured_noise(
            ref_grad=q_shared,
            basis_old=basis_old,
            eigvals_old=eigvals_old,
            basis_cur=basis_cur,
            eigvals_cur=eigvals_cur,
        )
        if delta.numel() != q_shared.numel():
            return q_shared

        # Orthogonalize perturbation to preserve descent direction.
        q_norm2 = torch.dot(q_shared, q_shared)
        if q_norm2 > 0:
            coef = torch.dot(delta, q_shared) / (q_norm2 + 1e-12)
            delta = delta - coef * q_shared

        # Optional clipping relative to projected gradient norm.
        if self._noise_clip_ratio > 0.0:
            max_norm = self._noise_clip_ratio * q_shared.norm()
            n_norm = delta.norm()
            if n_norm > max_norm and max_norm > 0:
                delta = delta * (max_norm / (n_norm + 1e-12))

        q_pert = self._compute_perturbed_shared_grad(
            model=model,
            shared_params=shared_params,
            inputs=inputs,
            targets=targets,
            stage=stage,
            delta=delta,
        )
        if q_pert is None or q_pert.numel() != q_shared.numel():
            return q_shared

        lam = max(float(self._pert_lambda), 0.0)
        g_corr = q_shared + lam * (q_pert - q_shared)
        _overwrite_grads_full(shared_params, g_corr)
        return g_corr

    def _compute_perturbed_shared_grad(
        self,
        *,
        model: nn.Module,
        shared_params: List[torch.nn.Parameter],
        inputs: torch.Tensor,
        targets: torch.Tensor,
        stage: str,
        delta: torch.Tensor,
    ) -> torch.Tensor | None:
        if delta is None or delta.numel() == 0:
            return None
        total_dim = sum(p.numel() for p in shared_params)
        if int(delta.numel()) != int(total_dim):
            return None

        offset = 0
        with torch.no_grad():
            for p in shared_params:
                numel = p.numel()
                p.add_(delta[offset : offset + numel].view_as(p))
                offset += numel

        grad_list = None
        try:
            logits = model(inputs)["logits"]
            loss_pert, _, _ = self._compute_stage_loss(logits, targets, stage)
            grad_list = torch.autograd.grad(
                loss_pert,
                shared_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        finally:
            offset = 0
            with torch.no_grad():
                for p in shared_params:
                    numel = p.numel()
                    p.sub_(delta[offset : offset + numel].view_as(p))
                    offset += numel

        if grad_list is None:
            return None
        return _flatten_grads_from_list(list(grad_list), shared_params, inputs.device)

    def _build_old_direction(
        self,
        *,
        target_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self._cur_task <= 0 or not self._task_grads_shared or target_dim <= 0:
            return None
        memories = self._stack_task_memories_shared(target_dim=target_dim, device=device, dtype=dtype)
        if memories is None or memories.numel() == 0:
            return None
        v_t = torch.mean(memories, dim=0)
        if v_t.norm() <= max(float(self._noise_min_denom), 1e-12):
            return None
        return v_t

    def _sample_structured_noise(
        self,
        *,
        ref_grad: torch.Tensor,
        basis_old: torch.Tensor | None,
        eigvals_old: torch.Tensor | None,
        basis_cur: torch.Tensor | None,
        eigvals_cur: torch.Tensor | None,
    ) -> torch.Tensor:
        d = int(ref_grad.numel())
        if d == 0:
            return torch.tensor([], device=ref_grad.device, dtype=ref_grad.dtype)

        if self._use_adaptive_sigma:
            ref_norm2 = torch.dot(ref_grad, ref_grad)
            sigma2 = self._noise_sigma * self._noise_alpha * float(ref_norm2.item()) / float(max(d, 1))
        else:
            sigma2 = float(self._noise_sigma) * float(self._noise_sigma)
        if sigma2 <= 0.0:
            return torch.zeros_like(ref_grad)

        device = ref_grad.device
        dtype = ref_grad.dtype
        eps = max(float(self._noise_eps), float(self._noise_min_denom))
        eig_floor = max(float(self._noise_eig_clip), float(self._noise_min_denom))

        # Hard cap rank to keep runtime/memory bounded.
        if basis_old is not None and eigvals_old is not None:
            k_old = min(int(basis_old.shape[1]), int(eigvals_old.numel()), int(self._noise_rank_cap))
            basis_old = basis_old[:, :k_old]
            eigvals_old = eigvals_old[:k_old]
        if basis_cur is not None and eigvals_cur is not None:
            k_cur = min(int(basis_cur.shape[1]), int(eigvals_cur.numel()), int(self._noise_rank_cap))
            basis_cur = basis_cur[:, :k_cur]
            eigvals_cur = eigvals_cur[:k_cur]

        basis_list = []
        if basis_old is not None and basis_old.numel() > 0:
            basis_list.append(basis_old)
        if basis_cur is not None and basis_cur.numel() > 0:
            basis_list.append(basis_cur)

        if not basis_list:
            raw = torch.randn_like(ref_grad) / math.sqrt(eps)
            if self._noise_trace_normalize:
                raw = raw / math.sqrt(float(d) / eps)
            return math.sqrt(sigma2) * raw

        w_mat = torch.cat(basis_list, dim=1)  # [d, k1+k2]
        q_basis = _orthonormalize_cols(w_mat, max_cols=self._noise_rank_cap, eps=1e-10)
        r = int(q_basis.shape[1])
        if r == 0:
            raw = torch.randn_like(ref_grad) / math.sqrt(eps)
            if self._noise_trace_normalize:
                raw = raw / math.sqrt(float(d) / eps)
            return math.sqrt(sigma2) * raw

        s_mat = torch.zeros((r, r), device=device, dtype=dtype)
        if basis_old is not None and eigvals_old is not None and basis_old.numel() > 0 and eigvals_old.numel() > 0:
            a1 = q_basis.t() @ basis_old  # [r, k1]
            s_mat = s_mat + (a1 * eigvals_old.unsqueeze(0)) @ a1.t()
        if basis_cur is not None and eigvals_cur is not None and basis_cur.numel() > 0 and eigvals_cur.numel() > 0:
            a2 = q_basis.t() @ basis_cur  # [r, k2]
            s_mat = s_mat - float(self._noise_beta) * ((a2 * eigvals_cur.unsqueeze(0)) @ a2.t())

        # K = (M_t + eps I)|span(Q), where M_t = F1 - beta * F2
        eye_r = torch.eye(r, device=device, dtype=dtype)
        k_mat = s_mat + eps * eye_r
        evals_k, evecs_k = _safe_eigh_symmetric(
            k_mat,
            eig_floor=eig_floor,
            jitter_base=eps,
        )

        # Subspace component: N(0, K^{-1})
        z_sub = torch.randn(r, device=device, dtype=dtype)
        coeff_sub = evecs_k @ (z_sub / torch.sqrt(evals_k))
        sub = q_basis @ coeff_sub

        # Orthogonal complement component: N(0, eps^{-1} I_perp)
        z_full = torch.randn(d, device=device, dtype=dtype)
        z_perp = z_full - q_basis @ (q_basis.t() @ z_full)
        perp = self._noise_orth_scale * (z_perp / math.sqrt(eps))

        raw = sub + perp

        if self._noise_trace_normalize:
            tr_sub = torch.sum(1.0 / evals_k)
            tr_perp = (self._noise_orth_scale ** 2) * float(max(d - r, 0)) / eps
            raw = raw / torch.sqrt(tr_sub + tr_perp + 1e-12)

        return math.sqrt(sigma2) * raw

    def _build_current_fisher_from_grad_rank1(
        self, g_shared: torch.Tensor
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """Rank-1 Fisher approximation for current batch: F2 ~= g g^T."""
        if g_shared is None or g_shared.numel() == 0:
            return None, None

        g_norm = g_shared.norm()
        if g_norm <= self._noise_min_denom:
            return None, None
        g_hat = g_shared / (g_norm + 1e-12)
        lam = max(float((g_norm * g_norm).item()), self._noise_min_denom)
        u_out = g_hat.unsqueeze(1)
        lam_out = torch.tensor([lam], device=g_shared.device, dtype=g_shared.dtype)
        return u_out, lam_out

    def _refresh_fisher_cache(self, shared_params: List[torch.nn.Parameter]) -> None:
        target_dim = sum(p.numel() for p in shared_params)
        self._fisher_basis = None
        self._fisher_eigvals = None
        self._fisher_dim_warned = False
        self._mem_dim_warned = False

        if target_dim <= 0 or not self._task_grads_shared:
            return

        valid = [g for g in self._task_grads_shared if int(g.numel()) == int(target_dim)]
        if not valid:
            return
        if len(valid) != len(self._task_grads_shared):
            logger.warning(
                "[GEM_noise_pert] Ignored %d stale shared gradients with mismatched dim (expected %d).",
                len(self._task_grads_shared) - len(valid),
                target_dim,
            )

        g_mat = torch.stack(valid, dim=0).float()  # [N, P_shared]
        keep = g_mat.norm(dim=1) > 1e-12
        g_mat = g_mat[keep]
        n = int(g_mat.shape[0])
        if n == 0:
            return

        gram = (g_mat @ g_mat.t()) / float(n)  # [N, N]
        evals, evecs = torch.linalg.eigh(gram)
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        evecs = evecs[:, order]

        rank = int(min(self._noise_rank, self._noise_rank_cap, n, target_dim))
        if rank <= 0:
            return

        keep = evals > self._noise_min_denom
        if torch.count_nonzero(keep) == 0:
            return
        evals = evals[keep][:rank]
        evecs = evecs[:, keep][:, :rank]
        if evals.numel() == 0:
            return

        # Lift to parameter space: U = G^T V / sqrt(N * lambda).
        basis = g_mat.t() @ evecs  # [P_shared, K]
        basis = basis / torch.sqrt((float(n) * evals).unsqueeze(0) + 1e-12)
        basis = _orthonormalize_cols(basis, max_cols=self._noise_rank_cap, eps=1e-10)

        self._fisher_basis = basis.cpu()
        self._fisher_eigvals = evals.cpu()

    def _get_fisher_basis_for_dim(
        self, target_dim: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._fisher_basis is None or self._fisher_eigvals is None:
            return None, None
        if int(self._fisher_basis.shape[0]) != int(target_dim):
            if not self._fisher_dim_warned:
                logger.warning(
                    "[GEM_noise_pert] Fisher basis dim mismatch: cache=%d, need=%d. Fallback to isotropic noise.",
                    int(self._fisher_basis.shape[0]),
                    int(target_dim),
                )
                self._fisher_dim_warned = True
            return None, None

        k = min(int(self._fisher_basis.shape[1]), int(self._fisher_eigvals.numel()))
        if k <= 0:
            return None, None
        basis = self._fisher_basis[:, :k].to(device=device, dtype=dtype)
        eigvals = self._fisher_eigvals[:k].to(device=device, dtype=dtype)
        return basis, eigvals

    def _stack_task_memories_shared(
        self, target_dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        rows = [g for g in self._task_grads_shared if int(g.numel()) == int(target_dim)]
        if not rows:
            return None
        if len(rows) != len(self._task_grads_shared) and not self._mem_dim_warned:
            logger.warning(
                "[GEM_noise_pert] GEM memories contain mismatched dims. Using %d/%d shared vectors of dim=%d.",
                len(rows),
                len(self._task_grads_shared),
                int(target_dim),
            )
            self._mem_dim_warned = True
        return torch.stack(rows, dim=0).to(device=device, dtype=dtype)

    def _compute_stage_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, stage: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if stage == "init":
            loss = F.cross_entropy(logits, targets)
            return loss, logits, targets

        fake_targets = targets - self._known_classes
        logits_new = logits[:, self._known_classes :]
        loss = F.cross_entropy(logits_new, fake_targets)
        return loss, logits_new, fake_targets

    def _compute_task_gradient(self, loader, model: nn.Module, *, stage: str):
        model.eval()
        shared_params = [p for _, p in self._iter_shared_trainable(model)]
        if not shared_params:
            return None

        total = 0
        g_accum = None
        for b_idx, batch in enumerate(loader):
            if b_idx >= self._gem_grad_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            model.zero_grad()
            logits = model(inputs)["logits"]
            loss, _, _ = self._compute_stage_loss(logits, targets, stage)
            loss.backward()

            g = _flatten_grads(shared_params, self._device, fill_zeros=True).detach().cpu()
            if g.numel() == 0:
                continue
            g_accum = g if g_accum is None else g_accum + g
            total += 1

        if g_accum is None or total == 0:
            return None
        return g_accum / float(total)

    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    def _iter_shared_trainable(self, model: nn.Module):
        for name, p in self._iter_trainable(model):
            # Single-head classifier in IncrementalNet lives under "fc.*".
            if name.split(".", 1)[0] == "fc":
                continue
            yield name, p

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
