import argparse
import json
import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

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
    attn = getattr(torch.nn, "attention", None)
    if attn is not None and hasattr(attn, "sdpa_kernel"):
        return attn.sdpa_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        )
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        )
    return nullcontext()

# ---------------------------------------------------------------------------
# Helpers for flattening perturbations
# ---------------------------------------------------------------------------


@dataclass
class FlatnessConfig:
    """Configuration knobs for the flatness/sharpness estimators."""

    rho: float = 0.05
    num_random_samples: int = 20
    gaussian_std: Optional[float] = None
    max_batches: Optional[int] = 1
    power_iters: int = 5
    trace_samples: int = 5
    grad_batches: Optional[int] = 1
    max_examples_per_batch: Optional[int] = 128
    # Optional persistence for metrics
    save_metrics_path: Optional[str] = None
    save_prefix: str = "flatness"


def _unwrap_batch(batch):
    """Convert a ``(idx, inputs, targets)`` batch into tensors only."""
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            _, inputs, targets = batch
            inputs, targets = _maybe_truncate_batch(inputs, targets)
            return inputs, targets
        if len(batch) == 2:
            inputs, targets = batch
            inputs, targets = _maybe_truncate_batch(inputs, targets)
            return inputs, targets
    raise ValueError("Unexpected batch format")


def _clone_params(params: Iterable[torch.nn.Parameter]) -> List[torch.Tensor]:
    """Detach and clone parameter tensors for later restoration."""
    return [p.detach().to("cpu").clone() for p in params]


def _restore_params(params: Iterable[torch.nn.Parameter], copies: List[torch.Tensor]):
    for p, saved in zip(params, copies):
        if saved.device == p.device and saved.dtype == p.dtype:
            p.data.copy_(saved)
        else:
            p.data.copy_(saved.to(device=p.device, dtype=p.dtype))


def _add_vector_to_params(params: List[torch.nn.Parameter], vec: torch.Tensor):
    """Add a flattened vector ``vec`` onto a list of parameters in-place."""
    pointer = 0
    for p in params:
        numel = p.numel()
        slice_vec = vec[pointer : pointer + numel].view_as(p)
        p.data.add_(slice_vec)
        pointer += numel


_GLOBAL_MAX_EXAMPLES_PER_BATCH: Optional[int] = None


def _truncate_first_dim(obj: Any, limit: int):
    if isinstance(obj, torch.Tensor):
        if obj.dim() > 0 and obj.size(0) > limit:
            return obj[:limit]
        return obj
    if isinstance(obj, np.ndarray):
        if obj.ndim > 0 and obj.shape[0] > limit:
            return obj[:limit]
        return obj
    if isinstance(obj, list):
        if len(obj) > limit:
            return obj[:limit]
        return obj
    if isinstance(obj, tuple):
        if len(obj) > limit:
            return obj[:limit]
        return obj
    return obj


def _maybe_truncate_batch(inputs, targets):
    limit = _GLOBAL_MAX_EXAMPLES_PER_BATCH
    if limit is None or limit <= 0:
        return inputs, targets
    inputs = _truncate_first_dim(inputs, limit)
    targets = _truncate_first_dim(targets, limit)
    return inputs, targets


@contextmanager
def _limit_batch_examples(limit: Optional[int]):
    global _GLOBAL_MAX_EXAMPLES_PER_BATCH
    prev = _GLOBAL_MAX_EXAMPLES_PER_BATCH
    _GLOBAL_MAX_EXAMPLES_PER_BATCH = limit
    try:
        yield
    finally:
        _GLOBAL_MAX_EXAMPLES_PER_BATCH = prev


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

    # 推荐：禁用 AMP，避免二阶梯度数值不稳
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


def _ggn_vector_product(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    vec: torch.Tensor,
    max_batches: Optional[int] = 1,
    known_classes: Optional[int] = None,
) -> torch.Tensor:
    """Compute Gv where G is the generalized Gauss–Newton matrix (CE case).

    Uses a finite-difference JVP for u = J v, then applies output-space
    Hessian of CE: HL = diag(p) - p p^T, and finally VJP: J^T s.
    """
    vec = vec.to(device)
    dim = vec.numel()
    out_accum = torch.zeros(dim, device=device)
    batches_processed = 0

    # Pre-split vec into param-shaped slices
    pointer = 0
    vec_slices: List[torch.Tensor] = []
    for p in params:
        n = p.numel()
        vec_slices.append(vec[pointer: pointer + n].view_as(p))
        pointer += n

    model.eval()
    with _sdp_disable_context():
        for batch_idx, batch in enumerate(loader):
            inputs, targets = _unwrap_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Base forward for logits at theta (used for p and VJP)
            outputs0 = model(inputs)
            logits0 = outputs0["logits"] if isinstance(outputs0, dict) else outputs0
            if known_classes is not None and known_classes > 0:
                logits_use = logits0[:, known_classes:]
                t_use = targets - known_classes
            else:
                logits_use = logits0
                t_use = targets

            # JVP via central differences around theta
            eps = 1e-3
            saved = _clone_params(params)
            # +eps
            _add_vector_to_params(params, eps * vec)
            logits_p = model(inputs)
            logits_p = logits_p["logits"] if isinstance(logits_p, dict) else logits_p
            # -eps
            _restore_params(params, saved)
            _add_vector_to_params(params, -eps * vec)
            logits_m = model(inputs)
            logits_m = logits_m["logits"] if isinstance(logits_m, dict) else logits_m
            # restore
            _restore_params(params, saved)

            if known_classes is not None and known_classes > 0:
                u = (logits_p[:, known_classes:] - logits_m[:, known_classes:]) / (2.0 * eps)
            else:
                u = (logits_p - logits_m) / (2.0 * eps)

            # Output-space CE Hessian HL = diag(p) - p p^T
            with torch.no_grad():
                p = torch.softmax(logits_use, dim=-1)
            up = u * p
            pu = up.sum(dim=-1, keepdim=True)
            s = up - p * pu

            # VJP: grad_theta <logits_use, s>
            scalar = (logits_use * s).sum()
            grads = torch.autograd.grad(scalar, params, retain_graph=False, allow_unused=True)
            flat = []
            for p_, g in zip(params, grads):
                if g is None:
                    flat.append(torch.zeros_like(p_).view(-1))
                else:
                    flat.append(g.detach().view(-1))
            out_accum += torch.cat(flat)

            batches_processed += 1
            if max_batches is not None and batches_processed >= max_batches:
                break

    return out_accum if batches_processed == 0 else out_accum / batches_processed


def _empirical_fisher_vector_product(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    vec: torch.Tensor,
    max_batches: Optional[int] = 1,
    known_classes: Optional[int] = None,
) -> torch.Tensor:
    """Compute EF v ≈ E[(g_i^T v) g_i] using per-sample gradients.

    Falls back to a biased batch-gradient approximation if per-sample
    gradients are too costly.
    """
    vec = vec.to(device)
    dim = vec.numel()
    out_accum = torch.zeros(dim, device=device)
    batches_processed = 0

    criterion = nn.CrossEntropyLoss(reduction="none")
    with _sdp_disable_context():
        for batch_idx, batch in enumerate(loader):
            inputs, targets = _unwrap_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            if known_classes is not None and known_classes > 0:
                logits = logits[:, known_classes:]
                targets_use = targets - known_classes
            else:
                targets_use = targets

            losses = criterion(logits, targets_use)  # [B]
            B = losses.shape[0]

            # Per-sample gradients (loop) — simple and robust
            G_rows: List[torch.Tensor] = []
            for i in range(B):
                model.zero_grad(set_to_none=True)
                losses[i].backward(retain_graph=True)
                gi = torch.cat([
                    (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)).view(-1)
                    for p in params
                ])
                G_rows.append(gi)
            G = torch.stack(G_rows, dim=0)  # [B, dim]

            s = G @ vec  # [B]
            out_accum += (G * s.unsqueeze(1)).mean(dim=0)

            batches_processed += 1
            if max_batches is not None and batches_processed >= max_batches:
                break

    return out_accum if batches_processed == 0 else out_accum / batches_processed


# ---------------------------------------------------------------------------
# Generic spectral/trace helpers for arbitrary matrix-vector products
# ---------------------------------------------------------------------------


def _power_iteration_generic(
    mv: callable,
    dim: int,
    num_iters: int,
    device: torch.device,
) -> float:
    if num_iters <= 0 or dim == 0:
        return 0.0
    v = torch.randn(dim, device=device)
    v = v / (v.norm() + 1e-12)
    eig = 0.0
    for _ in range(num_iters):
        w = mv(v)
        nrm = w.norm()
        if nrm.item() == 0:
            eig = 0.0
            break
        v = w / (nrm + 1e-12)
        eig = nrm.item()
    return float(eig)


def _lanczos_lambda_max_generic(
    mv: callable,
    dim: int,
    num_iters: int,
    device: torch.device,
    tol: float = 1e-3,
    reorth: bool = False,
) -> float:
    m = int(max(0, num_iters))
    if m == 0 or dim == 0:
        return 0.0
    v = torch.randn(dim, device=device)
    v = v / (v.norm() + 1e-12)
    v_prev = torch.zeros_like(v)
    beta_prev = 0.0
    alphas: List[float] = []
    betas: List[float] = []
    basis: List[torch.Tensor] = [v] if reorth else []
    prev_ritz: Optional[float] = None
    for it in range(m):
        hv = mv(v)
        alpha = torch.dot(v, hv).item()
        alphas.append(alpha)
        w = hv - alpha * v - beta_prev * v_prev
        if reorth:
            K = 5
            for q in basis[-K:]:
                coeff = torch.dot(w, q)
                w = w - coeff * q
        beta = w.norm().item()
        if it < m - 1:
            betas.append(beta)
        if beta <= 1e-12 or torch.isnan(torch.tensor(beta)):
            break
        v_prev = v
        v = w / (beta + 1e-12)
        beta_prev = beta
        if reorth:
            basis.append(v)
        if it >= 1 and tol is not None:
            k = len(alphas)
            T = torch.zeros((k, k), dtype=torch.float64, device=device)
            for i in range(k):
                T[i, i] = alphas[i]
                if i < len(betas):
                    T[i, i + 1] = betas[i]
                    T[i + 1, i] = betas[i]
            cur_ritz = float(torch.linalg.eigvalsh(T.cpu()).max().item())
            if prev_ritz is not None:
                rel = abs(cur_ritz - prev_ritz) / (abs(cur_ritz) + 1e-12)
                if rel < tol:
                    prev_ritz = cur_ritz
                    break
            prev_ritz = cur_ritz
    k = len(alphas)
    if k == 0:
        return 0.0
    T = torch.zeros((k, k), dtype=torch.float64, device=device)
    for i in range(k):
        T[i, i] = alphas[i]
        if i < len(betas):
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]
    return float(torch.linalg.eigvalsh(T.cpu()).max().item())


def _hutchinson_trace_generic(
    mv: callable,
    dim: int,
    num_samples: int,
    device: torch.device,
) -> float:
    if num_samples <= 0 or dim == 0:
        return 0.0
    est = 0.0
    for _ in range(num_samples):
        v = torch.empty(dim, device=device).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        mv_v = mv(v)
        est += torch.dot(v, mv_v).item()
    return est / num_samples


# ---------------------------------------------------------------------------
# Loss landscape: 1D/2D slices with random or filter-normalized directions
# ---------------------------------------------------------------------------


def _filter_normalize_direction(param: torch.Tensor, d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Filter-wise normalize direction d to match per-filter norms of param."""
    if param.ndim >= 2:
        p_flat = param.reshape(param.shape[0], -1)
        d_flat = d.reshape(d.shape[0], -1)
        p_norm = p_flat.norm(dim=1, keepdim=True)
        d_norm = d_flat.norm(dim=1, keepdim=True)
        scale = p_norm / (d_norm + eps)
        d_flat = d_flat * scale
        return d_flat.view_as(d)
    else:
        scale = (param.norm() / (d.norm() + eps)) if d.norm() > 0 else 1.0
        return d * scale


def _build_direction_list(
    params: List[torch.nn.Parameter],
    device: torch.device,
    filter_norm: bool = True,
    seed: Optional[int] = 42,
) -> List[torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
    dirs: List[torch.Tensor] = []
    for p in params:
        d = torch.randn_like(p, device=device)
        if filter_norm:
            d = _filter_normalize_direction(p.data, d)
        else:
            if d.norm() > 0:
                d = d / d.norm() * (p.data.norm() + 1e-12)
        dirs.append(d.to(device))
    return dirs


def _dir_inner(dirs_a: List[torch.Tensor], dirs_b: List[torch.Tensor]) -> torch.Tensor:
    s = 0.0
    for a, b in zip(dirs_a, dirs_b):
        s = s + torch.dot(a.view(-1), b.view(-1))
    return s


def _dir_norm(dirs: List[torch.Tensor]) -> float:
    total = 0.0
    for t in dirs:
        total += float(t.view(-1).dot(t.view(-1)))
    return float(total ** 0.5)


def _orthonormalize(dirs_v: List[torch.Tensor], dirs_u: List[torch.Tensor], eps: float = 1e-12) -> List[torch.Tensor]:
    proj = _dir_inner(dirs_v, dirs_u) / ( _dir_inner(dirs_u, dirs_u) + eps )
    out: List[torch.Tensor] = []
    for v, u in zip(dirs_v, dirs_u):
        out.append(v - proj * u)
    nrm = _dir_norm(out)
    if nrm > 0:
        out = [t / nrm for t in out]
    return out


def _apply_direction(params: List[torch.nn.Parameter], dirs: List[torch.Tensor], alpha: float) -> None:
    with torch.no_grad():
        for p, d in zip(params, dirs):
            p.add_(alpha * d.to(p.dtype))


def _loss_landscape_1d(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    radius: float = 0.5,
    num_points: int = 21,
    max_batches: int = 1,
    filter_norm: bool = True,
    known_classes: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    backup = _clone_params(params)
    dirs = _build_direction_list(params, device=device, filter_norm=filter_norm)
    concat_norm = _dir_norm(dirs)
    if concat_norm > 0:
        dirs = [d / concat_norm for d in dirs]

    xs = np.linspace(-radius, radius, num_points).astype(np.float64)
    losses = np.zeros_like(xs, dtype=np.float64)

    for i, x in enumerate(xs):
        _restore_params(params, backup)
        _apply_direction(params, dirs, float(x))
        losses[i] = _compute_loss(model, loader, device, max_batches=max_batches, known_classes=known_classes)

    _restore_params(params, backup)
    return {"x": xs, "loss": losses}


def _loss_landscape_2d(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    radius: float = 0.5,
    num_points: int = 21,
    max_batches: int = 1,
    filter_norm: bool = True,
    known_classes: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    backup = _clone_params(params)
    d1 = _build_direction_list(params, device=device, filter_norm=filter_norm, seed=123)
    d2 = _build_direction_list(params, device=device, filter_norm=filter_norm, seed=456)
    n1 = _dir_norm(d1)
    d1 = [t / n1 for t in d1] if n1 > 0 else d1
    d2 = _orthonormalize(d2, d1)

    xs = np.linspace(-radius, radius, num_points).astype(np.float64)
    ys = np.linspace(-radius, radius, num_points).astype(np.float64)
    Z = np.zeros((num_points, num_points), dtype=np.float64)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            _restore_params(params, backup)
            with torch.no_grad():
                for p, a, b in zip(params, d1, d2):
                    p.add_((float(x) * a + float(y) * b).to(p.dtype))
            Z[i, j] = _compute_loss(model, loader, device, max_batches=max_batches, known_classes=known_classes)

    _restore_params(params, backup)
    return {"x": xs, "y": ys, "loss": Z}


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


def _lanczos_lambda_max(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    dim: int,
    num_iters: int,
    max_batches: Optional[int],
    known_classes: Optional[int],
    tol: float = 1e-3,
    reorth: bool = False,
    seed: Optional[int] = None,
) -> float:
    """Estimate the dominant (algebraic) Hessian eigenvalue via Lanczos.

    Builds a size-``m`` tridiagonal approximation of the Hessian using the
    three-term Lanczos recurrence with Hessian–vector products and returns the
    largest eigenvalue of the tridiagonal matrix as an estimate of ``lambda_max``.

    Notes:
    - Shares the same HVP routine as power iteration (``_hessian_vector_product``).
    - Uses a single random start vector; no re-orthogonalization to keep memory
      and runtime small (reasonable for small ``m`` like 5–20).
    """
    m = int(max(0, num_iters))
    if m == 0 or dim == 0:
        return 0.0

    if seed is not None:
        torch.manual_seed(seed)

    was_training = model.training
    model.eval()

    v = torch.randn(dim, device=device)
    v = v / (v.norm() + 1e-12)
    v_prev = torch.zeros_like(v)
    beta_prev = 0.0

    alphas: List[float] = []
    betas: List[float] = []
    basis: List[torch.Tensor] = [v] if reorth else []
    prev_ritz: Optional[float] = None

    for it in range(m):
        hv = _hessian_vector_product(
            model,
            loader,
            device,
            params,
            v,
            max_batches=max_batches,
            known_classes=known_classes,
        )

        alpha = torch.dot(v, hv).item()
        alphas.append(alpha)

        w = hv - alpha * v - beta_prev * v_prev

        if reorth:
            K = 5
            for q in basis[-K:]:
                coeff = torch.dot(w, q)
                w = w - coeff * q

        beta = w.norm().item()
        if it < m - 1:
            betas.append(beta)

        if beta <= 1e-12 or torch.isnan(torch.tensor(beta)):
            break

        v_prev = v
        v = w / (beta + 1e-12)
        beta_prev = beta
        if reorth:
            basis.append(v)

        if it >= 1 and tol is not None:
            k = len(alphas)
            T = torch.zeros((k, k), dtype=torch.float64, device=device)
            for i in range(k):
                T[i, i] = alphas[i]
                if i < len(betas):
                    T[i, i + 1] = betas[i]
                    T[i + 1, i] = betas[i]
            cur_ritz = float(torch.linalg.eigvalsh(T.cpu()).max().item())
            if prev_ritz is not None:
                rel = abs(cur_ritz - prev_ritz) / (abs(cur_ritz) + 1e-12)
                if rel < tol:
                    prev_ritz = cur_ritz
                    break
            prev_ritz = cur_ritz

    k = len(alphas)
    if k == 0:
        if was_training:
            model.train()
        return 0.0

    T = torch.zeros((k, k), dtype=torch.float64, device=device)
    for i in range(k):
        T[i, i] = alphas[i]
        if i < len(betas):
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]

    lam_max = float(torch.linalg.eigvalsh(T.cpu()).max().item())

    if was_training:
        model.train()
    return lam_max

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
    global _GLOBAL_MAX_EXAMPLES_PER_BATCH
    if not params:
        return {"base_loss": 0.0}

    prev_max_examples = _GLOBAL_MAX_EXAMPLES_PER_BATCH
    _GLOBAL_MAX_EXAMPLES_PER_BATCH = config.max_examples_per_batch
    try:
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
    
        # max sharpness
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
    
        # Hessian spectral proxies (power iteration)
        lambda_max_power = _power_iteration_lambda_max(
            wrapped_model,
            loader,
            device,
            params,
            dim=total_dim,
            num_iters=config.power_iters,
            max_batches=config.max_batches,
            known_classes=known_classes,
        )
        # Backward‑compat key + explicit method key
        flat_metrics["lambda_max"] = lambda_max_power
        flat_metrics["lambda_max_power"] = lambda_max_power
    
        # Hessian spectral proxies (Lanczos)
        try:
            lambda_max_lanczos = _lanczos_lambda_max(
                wrapped_model,
                loader,
                device,
                params,
                dim=total_dim,
                num_iters=config.power_iters,
                max_batches=config.max_batches,
                known_classes=known_classes,
            )
            flat_metrics["lambda_max_lanczos"] = lambda_max_lanczos
        except Exception:
            # Keep evaluation robust even if Lanczos fails
            pass
    
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
    
        # Loss landscape slices (optional)
        try:
            do_1d = bool(getattr(config, "loss_land_1d", False))
            do_2d = bool(getattr(config, "loss_land_2d", False))
            loss_radius = float(getattr(config, "loss_land_radius", 0.5))
            loss_points = int(getattr(config, "loss_land_num_points", 21))
            loss_batches = int(getattr(config, "loss_land_max_batches", 1))
            loss_filter = bool(getattr(config, "loss_land_filter_norm", True))
            save_dir = getattr(config, "save_metrics_path", None)
            prefix = getattr(config, "save_prefix", "flatness")
    
            if do_1d:
                res1d = _loss_landscape_1d(
                    wrapped_model, loader, device, params,
                    radius=loss_radius, num_points=loss_points,
                    max_batches=loss_batches, filter_norm=loss_filter,
                    known_classes=known_classes,
                )
                flat_metrics["lossland_1d_min"] = float(np.min(res1d["loss"]))
                flat_metrics["lossland_1d_max"] = float(np.max(res1d["loss"]))
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    path1d = os.path.join(save_dir, f"{prefix}_lossland_1d.npz")
                    np.savez_compressed(path1d, x=res1d["x"], loss=res1d["loss"])
                    flat_metrics["lossland_1d_file"] = path1d
    
            if do_2d:
                res2d = _loss_landscape_2d(
                    wrapped_model, loader, device, params,
                    radius=loss_radius, num_points=loss_points,
                    max_batches=loss_batches, filter_norm=loss_filter,
                    known_classes=known_classes,
                )
                flat_metrics["lossland_2d_min"] = float(np.min(res2d["loss"]))
                flat_metrics["lossland_2d_max"] = float(np.max(res2d["loss"]))
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    path2d = os.path.join(save_dir, f"{prefix}_lossland_2d.npz")
                    np.savez_compressed(path2d, x=res2d["x"], y=res2d["y"], loss=res2d["loss"])
                    flat_metrics["lossland_2d_file"] = path2d
        except Exception as _err:
            logging.debug("Loss landscape evaluation failed: %s", _err)
        
        # ---------------- GGN / Fisher / Empirical Fisher (optional) ----------------
        # Define MVP closures that reuse the same data/batch budget
        def _mvp_ggn(v: torch.Tensor) -> torch.Tensor:
            return _ggn_vector_product(
                wrapped_model, loader, device, params, v,
                max_batches=config.max_batches, known_classes=known_classes
            )
    
        def _mvp_emp_fisher(v: torch.Tensor) -> torch.Tensor:
            return _empirical_fisher_vector_product(
                wrapped_model, loader, device, params, v,
                max_batches=config.max_batches, known_classes=known_classes
            )
    
        try:
            flat_metrics["ggn_lambda_max_power"] = _power_iteration_generic(
                _mvp_ggn, total_dim, config.power_iters, device
            )
            flat_metrics["ggn_lambda_max_lanczos"] = _lanczos_lambda_max_generic(
                _mvp_ggn, total_dim, config.power_iters, device
            )
            flat_metrics["ggn_trace"] = _hutchinson_trace_generic(
                _mvp_ggn, total_dim, config.trace_samples, device
            )
            # For CE/NLL, Fisher == GGN
            flat_metrics["fisher_trace"] = flat_metrics["ggn_trace"]
        except Exception:
            pass
    
        try:
            flat_metrics["emp_fisher_trace"] = _hutchinson_trace_generic(
                _mvp_emp_fisher, total_dim, max(1, config.trace_samples // 2), device
            )
        except Exception:
            pass
    
        _restore_params(params, param_backup)
        wrapped_model.zero_grad(set_to_none=True)
        # Optional persistence
        if getattr(config, "save_metrics_path", None):
            try:
                os.makedirs(config.save_metrics_path, exist_ok=True)
                save_file = os.path.join(
                    config.save_metrics_path,
                    f"{getattr(config, 'save_prefix', 'flatness')}_metrics.json",
                )
                with open(save_file, "w", encoding="utf-8") as fh:
                    json.dump(flat_metrics, fh, indent=2)
            except Exception as _err:
                logging.debug("Failed to save flatness metrics: %s", _err)
    
        return flat_metrics
    finally:
        _GLOBAL_MAX_EXAMPLES_PER_BATCH = prev_max_examples


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
