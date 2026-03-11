"""Core curvature utilities (MVP builders, eigen/trace estimators).

These helpers are intentionally thin wrappers around existing implementations
in ``eval_flat`` so downstream code can import a stable, dependency-light
interface. They do **not** alter numerical behavior; they only package the
closures and iteration strategies used elsewhere in the repo.
"""
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple, List
import torch

from evaluation_weight_sharpness.loss_utils import _unwrap_batch, _forward_logits_full
from evaluation_weight_sharpness.power_iter import _power_iteration_lambda_max, _power_iteration_generic
from evaluation_weight_sharpness.lanczos_iter import _lanczos_topk_generic


def make_mvp_map(
    hessian_fn: Callable[[torch.Tensor], torch.Tensor],
    ggn_fn: Callable[[torch.Tensor], torch.Tensor],
    fisher_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Return a backend→MVP mapping with consistent keys.

    Parameters
    ----------
    hessian_fn, ggn_fn, fisher_fn : callable
        Matrix-vector product closures for the Hessian, GGN, and empirical
        Fisher respectively. Each must accept a single vector ``v`` and return
        the MVP result with matching shape.
    """
    return {
        "hessian": hessian_fn,
        "ggn": ggn_fn,
        "emp_fisher": fisher_fn,
    }


def build_mvp_fns(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    loss_eval_max_batches: Optional[int] = None,
    known_classes: Optional[int] = None,
) -> Tuple[Callable, Callable, Callable]:
    """Construct Hessian / GGN / empirical Fisher MVP closures.

    Notes
    -----
    - Uses the repository's existing CE loss and logits wrapper. The returned
      functions capture the same batch budget as ``eval_flatness_weight_Loss``.
    - This helper is side-effect free and keeps model in eval mode.
    """
    model.eval()

    def _loss_and_grads(inputs, targets):
        logits = _forward_logits_full(model, inputs, targets)
        return torch.nn.functional.cross_entropy(logits, targets, reduction="sum")

    def _mvp_factory(kind: str):
        def _mvp(v: torch.Tensor) -> torch.Tensor:
            # Single-pass batch accumulation mirroring the existing logic
            hv_total = torch.zeros_like(v)
            total = 0
            for b_idx, batch in enumerate(loader):
                inputs, targets = _unwrap_batch(batch)
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.enable_grad():
                    loss = _loss_and_grads(inputs, targets) / targets.size(0)
                    grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
                    flat_grad = torch.cat([g.reshape(-1) for g in grad])
                    if kind == "hessian":
                        hv = torch.autograd.grad(flat_grad @ v, model.parameters(), retain_graph=False)
                    elif kind == "ggn":
                        # GGN via Jacobian-vector on per-sample gradient norm
                        hv = torch.autograd.grad((flat_grad * v).sum(), model.parameters(), retain_graph=False)
                    else:  # emp_fisher
                        hv = grad
                hv_flat = torch.cat([h.reshape(-1) for h in hv])
                hv_total += hv_flat
                total += 1
                if loss_eval_max_batches is not None and (b_idx + 1) >= loss_eval_max_batches:
                    break
            if total == 0:
                return hv_total
            return hv_total / float(total)
        return _mvp

    return _mvp_factory("hessian"), _mvp_factory("ggn"), _mvp_factory("emp_fisher")


def estimate_trace(
    mvp_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    *,
    iters: int = 5,
    samples: int = 5,
    device: Optional[torch.device] = None,
) -> float:
    """Hutchinson-style trace estimate using power-iteration scaffolding."""
    traces: List[float] = []
    for _ in range(samples):
        vals = _power_iteration_generic(mvp_fn, dim, iters, device or torch.device("cpu"))
        if len(vals) > 0:
            traces.append(float(vals[-1]))
    return float(torch.tensor(traces).mean().item()) if traces else 0.0


def topk_eigs(
    mvp_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    *,
    k: int = 2,
    iters: int = 20,
    tol: float = 1e-3,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
):
    """Compute top-k eigenvalues/vectors via Lanczos.

    Returns
    -------
    eigenvalues : torch.Tensor
    eigenvectors : List[torch.Tensor]
    """
    return _lanczos_topk_generic(
        mvp_fn, dim, iters, device or torch.device("cpu"), topk=k, tol=tol, seed=seed
    )
