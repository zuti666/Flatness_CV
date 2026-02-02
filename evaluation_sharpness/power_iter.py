from typing import Any, Callable, List, Optional, Tuple

import torch


def _power_iteration_generic(
    mv: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_iters: int,
    device: torch.device,
    *,
    return_vec: bool = False,
    topk: int = 1,
    tol: Optional[float] = None,
    patience: int = 2,
    seed: Optional[int] = None,
) -> Any:
    """Generic power iteration; optionally return eigenvector(s).

    If topk==2, compute v1 via power iteration and v2 on the orthogonal complement via deflated power.
    """
    if num_iters <= 0 or dim == 0:
        return (0.0, None) if return_vec else 0.0

    if return_vec:
        vals, vecs = _deflated_power_iteration(
            mv,
            dim,
            num_iters,
            device,
            topk=max(1, int(topk)),
            tol=tol,
            patience=patience,
            seed=seed,
            use_rayleigh=False,
        )
        if not vals:
            return (0.0, None)
        if int(topk) <= 1:
            return (float(vals[0]), vecs[0])
        return [float(v) for v in vals], vecs

    eig1, _ = _deflated_power_iteration(
        mv,
        dim,
        num_iters,
        device,
        topk=1,
        tol=tol,
        patience=patience,
        seed=seed,
        use_rayleigh=False,
    )
    return float(eig1[0]) if eig1 else 0.0


def _deflated_power_iteration(
    mv: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_iters: int,
    device: torch.device,
    *,
    topk: int = 1,
    tol: Optional[float] = None,
    patience: int = 2,
    seed: Optional[int] = None,
    use_rayleigh: bool = True,
    use_abs_eig: bool = False,
) -> Tuple[List[float], List[torch.Tensor]]:
    """Compute top-k eigenpairs using simple deflated power iteration.

    - ``use_rayleigh=True``: eigenvalue from Rayleigh quotient v^T A v
    - ``use_rayleigh=False``: eigenvalue from ||A v|| (spectral radius proxy)
    """
    if dim == 0 or num_iters <= 0 or topk <= 0:
        return [], []

    if seed is not None:
        torch.manual_seed(int(seed))

    eigvals: List[float] = []
    eigvecs: List[torch.Tensor] = []

    def _project_out(vec: torch.Tensor, basis: List[torch.Tensor]) -> torch.Tensor:
        if not basis:
            return vec
        for b in basis:
            vec = vec - torch.dot(vec, b) * b
        return vec

    for _ in range(max(1, int(topk))):
        v = torch.randn(dim, device=device)
        v = _project_out(v, eigvecs)
        v = v / (v.norm() + 1e-12)
        prev = None
        hit = 0
        eig = 0.0
        for _ in range(num_iters):
            w = mv(v)
            w = _project_out(w, eigvecs)
            nrm = w.norm()
            if not torch.isfinite(nrm) or nrm.item() == 0.0:
                eig = 0.0
                break
            if use_rayleigh:
                eig_cur = float(torch.dot(v, w).item())
                eig = abs(eig_cur) if use_abs_eig else eig_cur
            else:
                eig = float(nrm.item())
            v = w / (nrm + 1e-12)
            if tol is not None and prev is not None:
                rel = abs(eig - prev) / (abs(eig) + 1e-12)
                if rel < tol:
                    hit += 1
                    if hit >= patience:
                        break
                else:
                    hit = 0
            prev = eig
        eigvals.append(float(eig))
        eigvecs.append(v)
    return eigvals, eigvecs


def _power_iteration_lambda_max(
    mv: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_iters: int,
    device: torch.device,
    *,
    tol: float = 1e-3,
    patience: int = 2,
    return_vec: bool = False,
    topk: int = 1,
    seed: Optional[int] = None,
    use_abs_eig: bool = False,
) -> Any:
    """Estimate the dominant eigenvalue via power iteration."""
    if num_iters <= 0 or dim == 0:
        return (0.0, None) if return_vec else 0.0

    if return_vec:
        vals, vecs = _deflated_power_iteration(
            mv,
            dim,
            num_iters,
            device,
            topk=max(1, int(topk)),
            tol=tol,
            patience=patience,
            seed=seed,
            use_rayleigh=True,
            use_abs_eig=use_abs_eig,
        )
        if not vals:
            return (0.0, None)
        if int(topk) <= 1:
            return (float(vals[0]), vecs[0])
        return [float(v) for v in vals], vecs

    vals_only, _ = _deflated_power_iteration(
        mv,
        dim,
        num_iters,
        device,
        topk=1,
        tol=tol,
        patience=patience,
        seed=seed,
        use_rayleigh=True,
        use_abs_eig=use_abs_eig,
    )
    return float(vals_only[0]) if vals_only else 0.0


def _deflated_power_iteration_masked(
    mv: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    mask: torch.Tensor,
    device: torch.device,
    num_iters: int,
    *,
    topk: int = 1,
    tol: Optional[float] = None,
    patience: int = 2,
    seed: Optional[int] = None,
) -> Tuple[List[float], List[torch.Tensor]]:
    if dim == 0 or topk <= 0:
        return [], []
    if seed is not None:
        torch.manual_seed(int(seed))
    mask = mask.to(device=device, dtype=torch.float32)
    if mask.numel() != dim:
        return [], []

    eigvals: List[float] = []
    eigvecs: List[torch.Tensor] = []

    def _project_out(vec: torch.Tensor, basis: List[torch.Tensor]) -> torch.Tensor:
        if not basis:
            return vec
        for b in basis:
            vec = vec - torch.dot(vec, b) * b
        return vec

    for _ in range(max(1, int(topk))):
        v = torch.randn(dim, device=device) * mask
        v = _project_out(v, eigvecs)
        v = v / (v.norm() + 1e-12)
        prev = None
        hit = 0
        eig = 0.0
        for _ in range(int(max(1, num_iters))):
            w = mv(v * mask) * mask
            w = _project_out(w, eigvecs)
            nrm = w.norm()
            if not torch.isfinite(nrm) or nrm.item() == 0.0:
                eig = 0.0
                break
            eig_cur = float(torch.dot(v, w).item())
            eig = eig_cur
            v = w / (nrm + 1e-12)
            if tol is not None and prev is not None:
                rel = abs(eig - prev) / (abs(eig) + 1e-12)
                if rel < tol:
                    hit += 1
                    if hit >= patience:
                        break
                else:
                    hit = 0
            prev = eig
        eigvals.append(float(eig))
        eigvecs.append(v)
    return eigvals, eigvecs


def _power_iteration_masked(
    mv: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    mask: torch.Tensor,
    device: torch.device,
    num_iters: int,
    *,
    topk: int = 1,
    tol: Optional[float] = None,
    patience: int = 2,
    seed: Optional[int] = None,
) -> Tuple[List[float], List[torch.Tensor]]:
    return _deflated_power_iteration_masked(
        mv,
        dim,
        mask,
        device,
        num_iters,
        topk=topk,
        tol=tol,
        patience=patience,
        seed=seed,
    )
