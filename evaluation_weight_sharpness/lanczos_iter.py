from typing import Any, Callable, List, Optional, Tuple

import torch


def _lanczos_lambda_max_generic(
    mv: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_iters: int,
    device: torch.device,
    tol: float = 1e-3,
    reorth: bool = False,
    *,
    return_vec: bool = False,
    seed: Optional[int] = None,
) -> Any:
    m = int(max(0, num_iters))
    if m == 0 or dim == 0:
        return (0.0, None) if return_vec else 0.0
    if seed is not None:
        torch.manual_seed(int(seed))
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
            for i in range(min(len(betas), k - 1)):
                beta_val = betas[i]
                T[i, i + 1] = beta_val
                T[i + 1, i] = beta_val
            cur_ritz = float(torch.linalg.eigvalsh(T.cpu()).max().item())
            if prev_ritz is not None:
                rel = abs(cur_ritz - prev_ritz) / (abs(cur_ritz) + 1e-12)
                if rel < tol:
                    prev_ritz = cur_ritz
                    break
            prev_ritz = cur_ritz
    k = len(alphas)
    if k == 0:
        return (0.0, None) if return_vec else 0.0
    T = torch.zeros((k, k), dtype=torch.float64, device=device)
    for i in range(k):
        T[i, i] = alphas[i]
    for i in range(min(len(betas), k - 1)):
        beta_val = betas[i]
        T[i, i + 1] = beta_val
        T[i + 1, i] = beta_val
    lam = float(torch.linalg.eigvalsh(T.cpu()).max().item())
    if return_vec:
        return lam, v
    return lam


def _lanczos_lambda_max(
    mv: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_iters: int,
    device: torch.device,
    tol: float = 1e-3,
    reorth: bool = False,
    seed: Optional[int] = None,
    *,
    patience: int = 2,
    return_vec: bool = False,
) -> Any:
    """Estimate the dominant (algebraic) eigenvalue via Lanczos."""
    m = int(max(0, num_iters))
    if m == 0 or dim == 0:
        return (0.0, None) if return_vec else 0.0

    if seed is not None:
        torch.manual_seed(int(seed))

    v = torch.randn(dim, device=device)
    v = v / (v.norm() + 1e-12)
    v_prev = torch.zeros_like(v)
    beta_prev = 0.0

    alphas: List[float] = []
    betas: List[float] = []
    basis: List[torch.Tensor] = [v] if reorth else []

    prev_ritz: Optional[float] = None
    hit = 0

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
            for i in range(min(len(betas), k - 1)):
                beta_val = betas[i]
                T[i, i + 1] = beta_val
                T[i + 1, i] = beta_val
            cur_ritz = float(torch.linalg.eigvalsh(T.cpu()).max().item())
            if prev_ritz is not None:
                rel = abs(cur_ritz - prev_ritz) / (abs(cur_ritz) + 1e-12)
                if rel < tol:
                    hit += 1
                    if hit >= patience:
                        prev_ritz = cur_ritz
                        break
                else:
                    hit = 0
            prev_ritz = cur_ritz

    k = len(alphas)
    if k == 0:
        return (0.0, None) if return_vec else 0.0

    T = torch.zeros((k, k), dtype=torch.float64, device=device)
    for i in range(k):
        T[i, i] = alphas[i]
    for i in range(min(len(betas), k - 1)):
        beta_val = betas[i]
        T[i, i + 1] = beta_val
        T[i + 1, i] = beta_val

    lam_max = float(torch.linalg.eigvalsh(T.cpu()).max().item())

    if return_vec:
        return lam_max, v
    return lam_max


def _lanczos_topk_generic(
    mv: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_iters: int,
    device: torch.device,
    *,
    topk: int = 1,
    tol: Optional[float] = None,
    reorth: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[float], List[torch.Tensor]]:
    m = int(max(0, num_iters))
    if m == 0 or dim == 0 or topk <= 0:
        return [], []
    if seed is not None:
        torch.manual_seed(int(seed))
    v = torch.randn(dim, device=device)
    v = v / (v.norm() + 1e-12)
    v_prev = torch.zeros_like(v)
    beta_prev = 0.0
    alphas: List[float] = []
    betas: List[float] = []
    basis: List[torch.Tensor] = []
    prev_ritz: Optional[float] = None
    for it in range(m):
        hv = mv(v)
        alpha = torch.dot(v, hv).item()
        alphas.append(alpha)
        w = hv - alpha * v - beta_prev * v_prev
        if reorth and basis:
            for q in basis:
                w = w - torch.dot(w, q) * q
        beta = w.norm().item()
        basis.append(v)
        if it < m - 1:
            betas.append(beta)
        if beta <= 1e-12 or torch.isnan(torch.tensor(beta)):
            break
        v_prev = v
        v = w / (beta + 1e-12)
        beta_prev = beta
        if tol is not None and it >= 1:
            k = len(alphas)
            T = torch.zeros((k, k), dtype=torch.float64, device=device)
            for i in range(k):
                T[i, i] = alphas[i]
            for i in range(min(len(betas), k - 1)):
                beta_val = betas[i]
                T[i, i + 1] = beta_val
                T[i + 1, i] = beta_val
            cur_ritz = float(torch.linalg.eigvalsh(T.cpu()).max().item())
            if prev_ritz is not None:
                rel = abs(cur_ritz - prev_ritz) / (abs(cur_ritz) + 1e-12)
                if rel < tol:
                    prev_ritz = cur_ritz
                    break
            prev_ritz = cur_ritz

    k = len(alphas)
    if k == 0:
        return [], []
    T = torch.zeros((k, k), dtype=torch.float64, device=device)
    for i in range(k):
        T[i, i] = alphas[i]
    for i in range(min(len(betas), k - 1)):
        beta_val = betas[i]
        T[i, i + 1] = beta_val
        T[i + 1, i] = beta_val
    evals, evecs = torch.linalg.eigh(T.cpu())
    evals = evals.flip(0)
    evecs = evecs.flip(1)
    k_use = int(min(k, int(topk)))
    evals = evals[:k_use]
    evecs = evecs[:, :k_use]
    Q = torch.stack(basis, dim=1)
    eigvals: List[float] = []
    eigvecs: List[torch.Tensor] = []
    for i in range(k_use):
        y = evecs[:, i].to(Q.device, dtype=Q.dtype)
        vec = Q @ y
        vec = vec / (vec.norm() + 1e-12)
        eigvals.append(float(evals[i].item()))
        eigvecs.append(vec)
    return eigvals, eigvecs


def _lanczos_topk_masked(
    mv: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    mask: torch.Tensor,
    device: torch.device,
    num_iters: int,
    *,
    topk: int = 1,
    tol: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[List[float], List[torch.Tensor]]:
    if dim == 0 or topk <= 0:
        return [], []
    mask = mask.to(device=device, dtype=torch.float32)
    if mask.numel() != dim:
        return [], []

    def mv_masked(v: torch.Tensor) -> torch.Tensor:
        return mv(v * mask) * mask

    return _lanczos_topk_generic(
        mv_masked,
        dim,
        num_iters,
        device,
        topk=topk,
        tol=tol,
        reorth=True,
        seed=seed,
    )
