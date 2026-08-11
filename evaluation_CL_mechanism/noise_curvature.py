"""
noise_curvature.py
------------------
Compute the core theoretical quantity

    tr(H_old · Σ_noise)

and the noise–eigenvector alignment

    cos(ε, u_i^old)

for different noise geometries (Fisher-inverse vs isotropic Gaussian).

Two computation modes
---------------------
A. Analytic  (nearly free, ~milliseconds)
   Uses the stored Fisher eigvals/basis to derive the closed-form ratio.
   Valid under the approximation H_old ≈ Σ_i λ_i u_i u_i^T.

B. Monte Carlo  (~seconds, more accurate)
   Samples real noise vectors ε from each geometry, computes ε^T H_old ε
   using the existing MVP infrastructure, and averages.

The existing MVP builder ``build_mvp_fns`` from
``sharpness_evaluation_core.curvature`` is reused without modification.
"""
from __future__ import annotations

import math
import logging
import statistics
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from sharpness_evaluation_core.curvature import build_mvp_fns
from evaluation_weight_sharpness.loss_utils import _compute_loss
from evaluation_weight_sharpness.param_utils import _clone_params, _restore_params
from evaluation_weight_sharpness.power_iter import _power_iteration_lambda_max

logger = logging.getLogger(__name__)


def _sample_gaussian_noise(
    dim: int,
    sigma: float,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return sigma * torch.randn(dim, device=device, dtype=dtype)


def _project_onto_ogd_safe_subspace(
    vec: torch.Tensor,
    ogd_directions: Optional[Sequence[torch.Tensor]],
) -> torch.Tensor:
    """Mirror OGD_Fisher3._project for flat vectors."""
    if not ogd_directions:
        return vec
    out = vec.clone()
    for s in ogd_directions:
        s_use = s.reshape(-1)
        if s_use.device != out.device or s_use.dtype != out.dtype:
            s_use = s_use.to(device=out.device, dtype=out.dtype)
        n = min(out.numel(), s_use.numel())
        if n <= 0:
            continue
        dot = torch.dot(out[:n], s_use[:n])
        out[:n] = out[:n] - dot * s_use[:n]
    return out


def _flatten_current_grads(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    flats = []
    for p in params:
        if p.grad is not None:
            flats.append(p.grad.reshape(-1))
        else:
            flats.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
    return torch.cat(flats) if flats else torch.tensor([], dtype=torch.float32)


def _fit_lowrank_from_weighted_samples(
    samples: List[torch.Tensor],
    weights: List[float],
    total_batches: int,
    rank: int,
    fisher_eps: float,
) -> tuple[List[torch.Tensor], torch.Tensor]:
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
        if lam <= fisher_eps:
            continue
        coeffs = evecs[:, idx]
        v = torch.zeros_like(vecs[0])
        for i in range(m):
            alpha = float(coeffs[i].item()) * scales[i]
            if alpha != 0.0:
                v.add_(vecs[i], alpha=alpha)
        norm = v.norm()
        if norm <= 1e-12:
            continue
        basis.append((v / norm).cpu())
        vals.append(max(lam, fisher_eps))
        if len(basis) >= rank:
            break

    if not basis:
        return [], torch.tensor([], dtype=torch.float32)
    return basis, torch.tensor(vals, dtype=torch.float32)


@torch.no_grad()
def estimate_fisher_subspace(
    network: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    rank: int,
    max_batches: int,
    sample_cap: int,
    fisher_eps: float = 1e-5,
) -> Dict[str, object]:
    """Estimate an old-task low-rank Fisher proxy directly from a loader."""
    if loader is None or rank <= 0 or max_batches <= 0 or sample_cap <= 0:
        return {
            "basis": [],
            "eigvals": torch.tensor([], dtype=torch.float32),
            "batches": 0,
            "status": "skipped",
        }

    params = [p for p in network.parameters() if p.requires_grad]
    if not params:
        return {
            "basis": [],
            "eigvals": torch.tensor([], dtype=torch.float32),
            "batches": 0,
            "status": "skipped_no_params",
        }

    samples: List[torch.Tensor] = []
    weights: List[float] = []
    total = 0

    was_training = network.training
    network.eval()
    try:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= int(max_batches):
                break

            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            network.zero_grad()
            with torch.enable_grad():
                logits = network(inputs)["logits"]
                logp = F.log_softmax(logits, dim=1)
                log_prob = -F.nll_loss(logp, targets, reduction="none")
                loss = log_prob.mean()
                score_weight = torch.mean(torch.exp(log_prob.detach()))
                loss.backward()

            flat_grad = _flatten_current_grads(params).detach().cpu()
            if flat_grad.numel() == 0:
                continue

            total += 1
            if len(samples) < int(sample_cap):
                samples.append(flat_grad.to(dtype=torch.float16))
                weights.append(float(score_weight.item()))
                continue

            keep_idx = int(torch.randint(0, total, (1,)).item())
            if keep_idx < int(sample_cap):
                samples[keep_idx] = flat_grad.to(dtype=torch.float16)
                weights[keep_idx] = float(score_weight.item())
    finally:
        network.zero_grad()
        network.train(was_training)

    basis, eigvals = _fit_lowrank_from_weighted_samples(
        samples, weights, total, int(rank), float(fisher_eps)
    )
    status = "ok" if basis else "empty"
    return {
        "basis": basis,
        "eigvals": eigvals,
        "batches": total,
        "status": status,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  A. Analytic estimate
# ══════════════════════════════════════════════════════════════════════════════

def tr_h_sigma_analytic(
    fisher_eigvals: torch.Tensor,   # self._fisher_eigvals  (shape: [rank])
    fisher_basis: List[torch.Tensor],  # self._fisher_basis (list of unit vectors)
    sigma: float,                   # rwp_std
    fisher_eps: float = 1e-5,
) -> Dict[str, object]:
    """
    Closed-form ratio under the approximation H_old ≈ Σ_i λ_i u_i u_i^T.

    Σ_fisher  = σ² [Σ_i (1/(λ_i+ε)) u_i u_i^T  +  (I − U U^T)]
    Σ_gaussian = σ² I

    tr(H_old · Σ_fisher)   ≈ σ² · Σ_i λ_i / (λ_i + ε)
    tr(H_old · Σ_gaussian) ≈ σ² · tr(H_old) = σ² · Σ_i λ_i

    tr_ratio = tr(H_old · Σ_gaussian) / tr(H_old · Σ_fisher)
             = Σ λ_i / Σ [λ_i / (λ_i + ε)]

    This is exact for the implemented low-rank perturbation proxy restricted to
    the retained Fisher span, and still ignores OGD projection / q-orth effects.
    The MC estimate remains the more faithful quantity.
    """
    rank = len(fisher_basis)
    if rank == 0 or fisher_eigvals.numel() == 0:
        return {}

    lam = fisher_eigvals[:rank].float().cpu()
    tr_h_topk = float(lam.sum().item())
    tr_fisher = (sigma ** 2) * float(torch.sum(lam / (lam + float(fisher_eps))).item())
    tr_gaussian = (sigma ** 2) * tr_h_topk
    ratio = tr_gaussian / max(tr_fisher, 1e-30)

    return {
        "analytic_tr_h_topk":             tr_h_topk,
        "analytic_tr_h_sigma_fisher":     tr_fisher,
        "analytic_tr_h_sigma_gaussian":   tr_gaussian,
        "analytic_tr_ratio_gauss_fisher": ratio,
        "analytic_fisher_rank":           rank,
        "analytic_sigma":                 sigma,
        "analytic_eigvals_top5":          lam[:5].tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  B. Monte Carlo estimate
# ══════════════════════════════════════════════════════════════════════════════

def _sample_fisher_noise(
    dim: int,
    fisher_basis: List[torch.Tensor],
    fisher_eigvals: torch.Tensor,
    sigma: float,
    fisher_eps: float = 1e-5,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Draw one noise vector ε ~ N(0, σ² · F^{-1}) (projected to Fisher span).

    Mirrors the exact transform inside OGD_Fisher3._fisher_inv_sqrt_transform:
      parallel = Σ_i (c_i / √λ_i) u_i
      residual = z − Σ_i c_i u_i   (unchanged)
      ε = σ · (parallel + residual)
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    z = torch.randn(dim, device=device, dtype=dtype)
    proj = torch.zeros(dim, device=device, dtype=dtype)
    parallel = torch.zeros(dim, device=device, dtype=dtype)
    rank = min(len(fisher_basis), int(fisher_eigvals.numel()))
    for i in range(rank):
        lam = float(fisher_eigvals[i].item())
        if lam <= 0.0:
            continue
        u = fisher_basis[i].reshape(-1)
        if u.device != z.device or u.dtype != z.dtype:
            u = u.to(device=z.device, dtype=z.dtype)
        c = float(torch.dot(u, z).item())
        proj     = proj + c * u
        parallel = parallel + (c / math.sqrt(lam + fisher_eps)) * u
    residual = z - proj
    return sigma * (parallel + residual)

def sample_noise_by_geometry(
    dim: int,
    geometry: str,
    sigma: float,
    *,
    fisher_basis: Optional[List[torch.Tensor]] = None,
    fisher_eigvals: Optional[torch.Tensor] = None,
    ogd_directions: Optional[Sequence[torch.Tensor]] = None,
    fisher_eps: float = 1e-5,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    project_ogd: bool = False,
) -> torch.Tensor:
    """Sample flat perturbation under a named geometry."""
    geometry = str(geometry or "none").lower()
    if sigma <= 0.0 or geometry in {"none", "off", "disabled", "zero"}:
        return torch.zeros(dim, device=device, dtype=dtype or torch.float32)

    if geometry in {"fisher", "structured", "fisher_inv"}:
        if not fisher_basis or fisher_eigvals is None or fisher_eigvals.numel() == 0:
            eps = _sample_gaussian_noise(dim, sigma, device=device, dtype=dtype)
        else:
            eps = _sample_fisher_noise(
                dim,
                fisher_basis,
                fisher_eigvals,
                sigma,
                fisher_eps=fisher_eps,
                device=device,
                dtype=dtype,
            )
    elif geometry in {"gaussian", "isotropic", "iso"}:
        eps = _sample_gaussian_noise(dim, sigma, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported noise geometry: {geometry}")

    if project_ogd:
        eps = _project_onto_ogd_safe_subspace(eps, ogd_directions)
    return eps


def tr_h_sigma_mc(
    mvp_fn: Callable[[torch.Tensor], torch.Tensor],
    fisher_basis: List[torch.Tensor],
    fisher_eigvals: torch.Tensor,
    sigma: float,
    n_samples: int,
    device: torch.device,
    fisher_eps: float = 1e-5,
) -> Dict[str, float]:
    """
    Monte Carlo estimate of E[ε^T H_old ε] for Fisher and Gaussian noise.

    E[ε^T H_old ε] = tr(H_old · Σ_noise)

    Uses the existing MVP closure from build_mvp_fns — only reads the model,
    no gradient computation here.

    Returns
    -------
    dict with keys:
      mc_tr_h_sigma_fisher    : float
      mc_tr_h_sigma_gaussian  : float
      mc_tr_ratio             : float  (Gaussian / Fisher)
      mc_n_samples            : int
    """
    if not fisher_basis or fisher_eigvals.numel() == 0:
        return {}

    dim = fisher_basis[0].numel()
    vals_fisher   = []
    vals_gaussian = []

    for _ in range(n_samples):
        eps_f = _sample_fisher_noise(
            dim,
            fisher_basis,
            fisher_eigvals,
            sigma,
            fisher_eps,
            device=device,
            dtype=torch.float32,
        ).to(device)
        eps_g = _sample_gaussian_noise(
            dim,
            sigma,
            device=device,
            dtype=torch.float32,
        )

        hv_f = mvp_fn(eps_f)
        hv_g = mvp_fn(eps_g)

        vals_fisher.append(float(torch.dot(eps_f, hv_f).item()))
        vals_gaussian.append(float(torch.dot(eps_g, hv_g).item()))

    mean_f = statistics.mean(vals_fisher)
    mean_g = statistics.mean(vals_gaussian)
    ratio  = mean_g / max(abs(mean_f), 1e-30)

    logger.debug(
        "[noise_curvature] MC tr_h_sigma: fisher=%.4e  gaussian=%.4e  ratio=%.2f",
        mean_f, mean_g, ratio,
    )
    return {
        "mc_tr_h_sigma_fisher":   mean_f,
        "mc_tr_h_sigma_gaussian": mean_g,
        "mc_tr_ratio":            ratio,
        "mc_n_samples":           n_samples,
    }


def tr_h_sigma_mc_for_geometry(
    mvp_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    geometry: str,
    sigma: float,
    n_samples: int,
    device: torch.device,
    fisher_basis: Optional[List[torch.Tensor]] = None,
    fisher_eigvals: Optional[torch.Tensor] = None,
    ogd_directions: Optional[Sequence[torch.Tensor]] = None,
    fisher_eps: float = 1e-5,
    project_ogd: bool = False,
    dim: Optional[int] = None,
) -> Dict[str, float]:
    """Monte-Carlo estimate of tr(HΣ) for one concrete perturbation geometry."""
    if geometry in {"none", "off", "disabled", "zero"} or sigma <= 0.0:
        return {
            f"mc_tr_h_sigma_{geometry}": 0.0,
            f"mc_tr_h_sigma_{geometry}_status": "zero_noise",
            "mc_n_samples": int(n_samples),
        }

    if geometry in {"fisher", "structured", "fisher_inv"} and (
        not fisher_basis or fisher_eigvals is None or fisher_eigvals.numel() == 0
    ):
        return {
            f"mc_tr_h_sigma_{geometry}": float("nan"),
            f"mc_tr_h_sigma_{geometry}_status": "skipped_no_fisher",
            "mc_n_samples": int(n_samples),
        }

    if geometry in {"fisher", "structured", "fisher_inv"}:
        dim = fisher_basis[0].numel()
    elif dim is not None and dim > 0:
        dim = int(dim)
    elif fisher_basis:
        dim = fisher_basis[0].numel()
    else:
        raise ValueError("Need fisher_basis to infer flat noise dimension.")

    vals = []
    for _ in range(int(n_samples)):
        eps = sample_noise_by_geometry(
            dim,
            geometry,
            sigma,
            fisher_basis=fisher_basis,
            fisher_eigvals=fisher_eigvals,
            ogd_directions=ogd_directions,
            fisher_eps=fisher_eps,
            device=device,
            dtype=torch.float32,
            project_ogd=project_ogd,
        )
        hv = mvp_fn(eps)
        vals.append(float(torch.dot(eps, hv).item()))

    mean_val = statistics.mean(vals) if vals else float("nan")
    std_val = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        f"mc_tr_h_sigma_{geometry}": mean_val,
        f"mc_tr_h_sigma_{geometry}_std": std_val,
        f"mc_tr_h_sigma_{geometry}_status": "ok",
        "mc_n_samples": int(n_samples),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  C. Noise–eigenvector alignment  cos(ε, u_i)
# ══════════════════════════════════════════════════════════════════════════════

def noise_eigvec_alignment(
    fisher_basis: List[torch.Tensor],
    fisher_eigvals: torch.Tensor,
    sigma: float,
    n_samples: int = 200,
    topk: int = 3,
    fisher_eps: float = 1e-5,
) -> Dict[str, float]:
    """
    cos(ε, u_i^old) for top-k curvature eigenvectors.

    Expected result:
      Fisher noise → very low alignment  (designed to avoid high-curvature dirs)
      Gaussian noise → higher alignment (no geometric bias)

    All computation is pure linear algebra — no MVP / GPU needed.
    """
    if not fisher_basis or fisher_eigvals.numel() == 0:
        return {}

    dim  = fisher_basis[0].numel()
    out: Dict[str, float] = {}

    for k in range(min(topk, len(fisher_basis))):
        u = fisher_basis[k].reshape(-1).float()

        cos_f_vals: List[float] = []
        cos_g_vals: List[float] = []

        for _ in range(n_samples):
            eps_f = _sample_fisher_noise(dim, fisher_basis, fisher_eigvals, sigma, fisher_eps)
            eps_g = _sample_gaussian_noise(dim, sigma)

            norm_f = float(eps_f.norm().item()) + 1e-12
            norm_g = float(eps_g.norm().item()) + 1e-12

            cos_f_vals.append(abs(float(torch.dot(u, eps_f / norm_f).item())))
            cos_g_vals.append(abs(float(torch.dot(u, eps_g / norm_g).item())))

        mean_cos_f = statistics.mean(cos_f_vals)
        mean_cos_g = statistics.mean(cos_g_vals)

        out[f"cos_fisher_u{k}"]    = mean_cos_f
        out[f"cos_gaussian_u{k}"]  = mean_cos_g
        out[f"cos_ratio_u{k}"]     = mean_cos_f / max(mean_cos_g, 1e-12)
        out[f"eigval_u{k}"]        = float(fisher_eigvals[k].item())

        logger.debug(
            "[noise_curvature] cos u%d: fisher=%.4f  gaussian=%.4f  ratio=%.3f",
            k, mean_cos_f, mean_cos_g, out[f"cos_ratio_u{k}"],
        )

    return out


def noise_eigvec_alignment_for_geometry(
    fisher_basis: List[torch.Tensor],
    fisher_eigvals: torch.Tensor,
    *,
    geometry: str,
    sigma: float,
    ogd_directions: Optional[Sequence[torch.Tensor]] = None,
    n_samples: int = 200,
    topk: int = 3,
    fisher_eps: float = 1e-5,
    project_ogd: bool = False,
) -> Dict[str, float]:
    """Alignment between a concrete perturbation geometry and old-task eigvecs."""
    if not fisher_basis or fisher_eigvals.numel() == 0:
        return {}

    dim = fisher_basis[0].numel()
    out: Dict[str, float] = {}
    for k in range(min(topk, len(fisher_basis))):
        u = fisher_basis[k].reshape(-1).float()
        cos_vals: List[float] = []
        for _ in range(int(n_samples)):
            eps = sample_noise_by_geometry(
                dim,
                geometry,
                sigma,
                fisher_basis=fisher_basis,
                fisher_eigvals=fisher_eigvals,
                ogd_directions=ogd_directions,
                fisher_eps=fisher_eps,
                project_ogd=project_ogd,
            ).float()
            norm = float(eps.norm().item()) + 1e-12
            cos_vals.append(abs(float(torch.dot(u, eps / norm).item())))
        mean_cos = statistics.mean(cos_vals) if cos_vals else float("nan")
        key_prefix = geometry.lower()
        out[f"cos_{key_prefix}_u{k}"] = mean_cos
        out[f"cos_{key_prefix}_u{k}_status"] = "ok"
    return out


def expected_sharpness(
    network: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    sigma: float,
    geometry: str,
    max_batches: int,
    n_samples: int,
    fisher_basis: Optional[List[torch.Tensor]] = None,
    fisher_eigvals: Optional[torch.Tensor] = None,
    ogd_directions: Optional[Sequence[torch.Tensor]] = None,
    fisher_eps: float = 1e-5,
    project_ogd: bool = False,
    base_loss: Optional[float] = None,
) -> Dict[str, float]:
    """
    Estimate E[L(theta + eps)] - L(theta) for a concrete perturbation geometry.
    """
    geometry = str(geometry or "none").lower()
    if loader is None:
        return {
            f"es_{geometry}": float("nan"),
            f"es_{geometry}_status": "skipped_no_loader",
        }

    if base_loss is None:
        base_loss = _compute_loss(network, loader, device, max_batches)

    if sigma <= 0.0 or geometry in {"none", "off", "disabled", "zero"}:
        return {
            f"es_{geometry}": 0.0,
            f"es_{geometry}_std": 0.0,
            f"es_{geometry}_base": float(base_loss),
            f"es_{geometry}_status": "zero_noise",
            f"es_{geometry}_n_samples": int(n_samples),
        }

    params = [p for p in network.parameters() if p.requires_grad]
    if not params:
        return {
            f"es_{geometry}": float("nan"),
            f"es_{geometry}_status": "skipped_no_params",
        }

    total_dim = sum(p.numel() for p in params)
    if total_dim <= 0:
        return {
            f"es_{geometry}": float("nan"),
            f"es_{geometry}_status": "skipped_zero_dim",
        }

    if geometry in {"fisher", "structured", "fisher_inv"} and (
        not fisher_basis or fisher_eigvals is None or fisher_eigvals.numel() == 0
    ):
        return {
            f"es_{geometry}": float("nan"),
            f"es_{geometry}_status": "skipped_no_fisher",
        }

    backup = _clone_params(params)
    deltas: List[float] = []
    pointer = None
    try:
        for _ in range(int(n_samples)):
            eps = sample_noise_by_geometry(
                total_dim,
                geometry,
                sigma,
                fisher_basis=fisher_basis,
                fisher_eigvals=fisher_eigvals,
                ogd_directions=ogd_directions,
                fisher_eps=fisher_eps,
                device=device,
                dtype=params[0].dtype,
                project_ogd=project_ogd,
            )
            pointer = 0
            with torch.no_grad():
                for p in params:
                    numel = p.numel()
                    delta = eps[pointer : pointer + numel].reshape_as(p)
                    p.add_(delta.to(device=p.device, dtype=p.dtype))
                    pointer += numel
            noisy_loss = _compute_loss(network, loader, device, max_batches)
            deltas.append(float(noisy_loss - float(base_loss)))
            _restore_params(params, backup)
    finally:
        _restore_params(params, backup)

    mean_delta = statistics.mean(deltas) if deltas else float("nan")
    std_delta = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    return {
        f"es_{geometry}": mean_delta,
        f"es_{geometry}_std": std_delta,
        f"es_{geometry}_base": float(base_loss),
        f"es_{geometry}_status": "ok",
        f"es_{geometry}_n_samples": int(n_samples),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  D. λ_max(H_old) / λ_max(H_new)  (S_old / S_new)
# ══════════════════════════════════════════════════════════════════════════════

def compute_lambda_max(
    network: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
    power_iters: int = 15,
    backend: str = "emp_fisher",
) -> float:
    """
    λ_max of the empirical Fisher (or GGN/Hessian) on a specific loader.

    Wraps build_mvp_fns + _power_iteration_lambda_max without modification.
    backend : "hessian" | "ggn" | "emp_fisher"
    """
    mvp_h, mvp_g, mvp_f = build_mvp_fns(
        network, loader, device,
        loss_eval_max_batches=max_batches,
    )
    mvp_map = {"hessian": mvp_h, "ggn": mvp_g, "emp_fisher": mvp_f}
    mvp_fn  = mvp_map.get(backend, mvp_f)

    params    = [p for p in network.parameters() if p.requires_grad]
    total_dim = sum(p.numel() for p in params)

    lmax = _power_iteration_lambda_max(
        mvp_fn, total_dim, power_iters, device
    )
    return float(lmax)
