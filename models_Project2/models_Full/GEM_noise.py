"""GEM + structured diffusion noise in shared-parameter space.

Design choices for class-incremental stability:
1) GEM projection is applied on shared parameters only (exclude classifier head).
2) Fisher/low-rank curvature cache is built from shared-task gradients only.
3) Task-gradient snapshots use the same stage loss definition as training.
4) Structured noise is injected only into shared parameters and orthogonalized
   against the shared projected gradient.
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


def _project_multi(g: torch.Tensor, memories: torch.Tensor, iters: int = 50, lr: float = 1.0) -> torch.Tensor:
    """Approximate GEM dual PGD solver.

    Solve:
        min 0.5 ||g_tilde - g||^2
        s.t. <g_tilde, m_k> >= 0, for all memory gradients m_k.
    """
    if memories.numel() == 0:
        return g
    a_mat = memories  # [K, P]
    gram = a_mat @ a_mat.t()  # [K, K]
    b_vec = -(a_mat @ g)  # [K]
    dual = torch.zeros_like(b_vec)
    for _ in range(iters):
        grad = gram @ dual + b_vec
        dual = torch.clamp(dual - lr * grad, min=0.0)
    return g + a_mat.t() @ dual


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

        # GEM controls
        self._gem_mem_batch = int(args.get("gem_mem_batch", 64))
        self._gem_grad_batches = int(args.get("gem_grad_batches", 1))
        self._gem_pgditers = int(args.get("gem_pgditers", 50))
        self._gem_pgdlr = float(args.get("gem_pgdlr", 1.0))

        # Structured noise controls
        self._use_structured_noise = self._parse_bool(args.get("use_structured_noise", True))
        self._noise_start_task = int(args.get("noise_start_task", 1))
        self._noise_only_with_grad = self._parse_bool(args.get("noise_only_with_grad", True))
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
        self._noise_project_orth = self._parse_bool(args.get("gem_noise_project_orth", True))
        self._f2_ema_decay = float(args.get("gem_noise_f2_ema_decay", 0.95))

        # Shared-space past task gradients (CPU; one vector per task).
        self._task_grads_shared: List[torch.Tensor] = []

        # Shared-space low-rank Fisher cache (CPU).
        self._fisher_basis: torch.Tensor | None = None  # [P_shared, K]
        self._fisher_eigvals: torch.Tensor | None = None  # [K]
        self._fisher_dim_warned = False
        self._mem_dim_warned = False

        # EMA rank-1 F2 cache for current task in shared space.
        self._f2_ema_u_cpu: torch.Tensor | None = None
        self._f2_ema_lam: float = 0.0

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
                "[GEM_noise] gem_noise_rank=%d capped to %d for stability.",
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
            logging.exception("[GEM_noise] Failed to compute class means: %s", exc)

    def _train(self, train_loader, test_loader, *, stage: str):
        model = self._unwrap_network()
        model.to(self._device)

        params = [p for p in model.parameters() if p.requires_grad]
        shared_params = [p for _, p in self._iter_shared_trainable(model)]
        optimizer = self._build_optimizer(params, stage=stage)

        self._refresh_fisher_cache(shared_params)
        self._f2_ema_u_cpu = None
        self._f2_ema_lam = 0.0

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
                q_shared = g_shared.clone()
                if self._cur_task > 0 and self._task_grads_shared and q_shared.numel() > 0:
                    memories = self._stack_task_memories_shared(
                        target_dim=q_shared.numel(), device=q_shared.device, dtype=q_shared.dtype
                    )
                    if memories is not None and memories.numel() > 0:
                        dotprod = memories @ q_shared
                        if (dotprod < 0).any():
                            q_shared = _project_multi(
                                q_shared, memories, iters=self._gem_pgditers, lr=self._gem_pgdlr
                            )
                            _overwrite_grads_full(shared_params, q_shared)

                optimizer.step()

                if self._use_structured_noise:
                    self._inject_structured_noise(shared_params, optimizer, q_shared, g_shared)

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

    @torch.no_grad()
    def _inject_structured_noise(
        self,
        shared_params: List[torch.nn.Parameter],
        optimizer,
        q_shared: torch.Tensor,
        g_shared: torch.Tensor,
    ) -> None:
        if self._cur_task < self._noise_start_task:
            return
        if q_shared is None or q_shared.numel() == 0:
            return
        if self._noise_sigma <= 0.0 or self._noise_alpha <= 0.0:
            return

        if self._noise_only_with_grad and not any(p.grad is not None for p in shared_params):
            return

        basis_old, eigvals_old = self._get_fisher_basis_for_dim(
            int(q_shared.numel()), device=q_shared.device, dtype=q_shared.dtype
        )
        basis_cur, eigvals_cur = self._build_current_fisher_from_grad(g_shared)
        noise = self._sample_structured_noise(
            q_shared=q_shared,
            basis_old=basis_old,
            eigvals_old=eigvals_old,
            basis_cur=basis_cur,
            eigvals_cur=eigvals_cur,
        )
        if noise.numel() != q_shared.numel():
            return

        # Orthogonalize stochastic part in shared space.
        if self._noise_project_orth:
            q_norm2 = torch.dot(q_shared, q_shared)
            if q_norm2 > 0:
                coef = torch.dot(noise, q_shared) / (q_norm2 + 1e-12)
                noise = noise - coef * q_shared

        # Optional clipping relative to shared projected gradient norm.
        if self._noise_clip_ratio > 0.0:
            max_norm = self._noise_clip_ratio * q_shared.norm()
            n_norm = noise.norm()
            if n_norm > max_norm and max_norm > 0:
                noise = noise * (max_norm / (n_norm + 1e-12))

        lr_by_param = {}
        default_lr = 0.0
        for group in optimizer.param_groups:
            lr = float(group.get("lr", 0.0))
            default_lr = lr
            for p in group.get("params", []):
                lr_by_param[id(p)] = lr

        offset = 0
        for p in shared_params:
            numel = p.numel()
            should_apply = (not self._noise_only_with_grad) or (p.grad is not None)
            if should_apply:
                lr = float(lr_by_param.get(id(p), default_lr))
                if lr > 0.0:
                    step_noise = noise[offset : offset + numel].view_as(p)
                    p.add_(math.sqrt(lr) * step_noise)
            offset += numel

    def _sample_structured_noise(
        self,
        *,
        q_shared: torch.Tensor,
        basis_old: torch.Tensor | None,
        eigvals_old: torch.Tensor | None,
        basis_cur: torch.Tensor | None,
        eigvals_cur: torch.Tensor | None,
    ) -> torch.Tensor:
        d = int(q_shared.numel())
        if d == 0:
            return torch.tensor([], device=q_shared.device, dtype=q_shared.dtype)

        q_norm2 = torch.dot(q_shared, q_shared)
        sigma2 = self._noise_sigma * self._noise_alpha * float(q_norm2.item()) / float(max(d, 1))
        if sigma2 <= 0.0:
            return torch.zeros_like(q_shared)

        device = q_shared.device
        dtype = q_shared.dtype
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
            raw = torch.randn_like(q_shared) / math.sqrt(eps)
            if self._noise_trace_normalize:
                raw = raw / math.sqrt(float(d) / eps)
            return math.sqrt(sigma2) * raw

        w_mat = torch.cat(basis_list, dim=1)  # [d, k1+k2]
        q_basis = _orthonormalize_cols(w_mat, max_cols=self._noise_rank_cap, eps=1e-10)
        r = int(q_basis.shape[1])
        if r == 0:
            raw = torch.randn_like(q_shared) / math.sqrt(eps)
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

    def _build_current_fisher_from_grad(
        self, g_shared: torch.Tensor
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """Rank-1 EMA Fisher for F2 in shared space.

        Target:
            F2_t = decay * F2_{t-1} + (1-decay) * g g^T

        We keep a rank-1 approximation `F2_t ~= lam * u u^T` by updating the
        dominant eigenpair in span{u_prev, g_hat} (2D exact eigendecomposition).
        """
        if g_shared is None or g_shared.numel() == 0:
            return None, None

        decay = float(min(max(self._f2_ema_decay, 0.0), 0.9999))
        g_norm = g_shared.norm()
        if g_norm <= self._noise_min_denom:
            if self._f2_ema_u_cpu is None or self._f2_ema_lam <= self._noise_min_denom:
                return None, None
            u = self._f2_ema_u_cpu.to(device=g_shared.device, dtype=g_shared.dtype).unsqueeze(1)
            lam = torch.tensor(
                [max(float(self._f2_ema_lam), self._noise_min_denom)],
                device=g_shared.device,
                dtype=g_shared.dtype,
            )
            return u, lam

        g_hat = g_shared / (g_norm + 1e-12)
        b = float(1.0 - decay) * float((g_norm * g_norm).item())

        if self._f2_ema_u_cpu is None or self._f2_ema_lam <= self._noise_min_denom:
            new_u = g_hat
            new_lam = max(b, self._noise_min_denom)
        else:
            u_prev = self._f2_ema_u_cpu.to(device=g_shared.device, dtype=g_shared.dtype)
            u_prev = u_prev / (u_prev.norm() + 1e-12)
            a = decay * max(float(self._f2_ema_lam), self._noise_min_denom)
            c = torch.clamp(torch.dot(u_prev, g_hat), min=-1.0, max=1.0)
            c2 = float((c * c).item())
            s2 = max(0.0, 1.0 - c2)
            if s2 <= 1e-12:
                # Collinear directions -> closed form.
                new_u = u_prev
                new_lam = max(a + b, self._noise_min_denom)
            else:
                s = math.sqrt(s2)
                u_perp = (g_hat - c * u_prev) / (s + 1e-12)
                m11 = a + b * c2
                m12 = b * float(c.item()) * s
                m22 = b * s2
                m = torch.tensor([[m11, m12], [m12, m22]], device=g_shared.device, dtype=g_shared.dtype)
                vals, vecs = torch.linalg.eigh(m)
                top = int(torch.argmax(vals).item())
                coeff = vecs[:, top]
                new_u = coeff[0] * u_prev + coeff[1] * u_perp
                new_u = new_u / (new_u.norm() + 1e-12)
                new_lam = max(float(vals[top].item()), self._noise_min_denom)

        self._f2_ema_u_cpu = new_u.detach().cpu()
        self._f2_ema_lam = float(new_lam)

        u_out = new_u.unsqueeze(1)
        lam_out = torch.tensor([new_lam], device=g_shared.device, dtype=g_shared.dtype)
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
                "[GEM_noise] Ignored %d stale shared gradients with mismatched dim (expected %d).",
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
                    "[GEM_noise] Fisher basis dim mismatch: cache=%d, need=%d. Fallback to isotropic noise.",
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
                "[GEM_noise] GEM memories contain mismatched dims. Using %d/%d shared vectors of dim=%d.",
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
