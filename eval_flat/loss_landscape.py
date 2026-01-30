import copy
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_flat.lanczos_iter import _lanczos_lambda_max_generic, _lanczos_topk_generic
from eval_flat.loss_utils import _compute_loss, _unwrap_batch
from eval_flat.param_utils import (
    _clone_params,
    _restore_params,
    _unflatten_to_param_like,
)
from eval_flat.power_iter import _power_iteration_lambda_max


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
    scale = (param.norm() / (d.norm() + eps)) if d.norm() > 0 else 1.0
    return d * scale


def _build_direction_list(
    params: List[torch.nn.Parameter],
    device: torch.device,
    filter_norm: bool = False,
    seed: Optional[int] = 42,
) -> List[torch.Tensor]:
    if seed is not None:
        torch.manual_seed(int(seed))
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
    proj = _dir_inner(dirs_v, dirs_u) / (_dir_inner(dirs_u, dirs_u) + eps)
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
            p.add_(alpha * d.to(device=p.device, dtype=p.dtype))


def _loss_landscape_1d(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    radius: float = 0.5,
    num_points: int = 21,
    max_batches: Optional[int] = None,
    filter_norm: bool = True,
    known_classes: Optional[int] = None,
    dirs_override: Optional[List[torch.Tensor]] = None,
) -> Dict[str, np.ndarray]:
    backup = _clone_params(params)
    if dirs_override is not None and len(dirs_override) > 0:
        dirs = dirs_override
    else:
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
    max_batches: Optional[int] = None,
    filter_norm: bool = True,
    known_classes: Optional[int] = None,
    dirs_override: Optional[List[List[torch.Tensor]]] = None,
) -> Dict[str, np.ndarray]:
    backup = _clone_params(params)
    if dirs_override is not None and len(dirs_override) == 2:
        d1, d2 = dirs_override
    else:
        d1 = _build_direction_list(params, device=device, filter_norm=filter_norm)
        d2 = _build_direction_list(params, device=device, filter_norm=filter_norm)
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


def compute_loss_landscape_v1(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    criterion: Callable,
    output_dir: str,
    save_file_name: str = "lossLandscape",
    eval_task_id: Optional[int] = 200,
    class_incremental: bool = False,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
    num_points: int = 20,
    max_batches: int = 5,
    sample_batches: bool = False,
    param_name_exclude_substr: Optional[str] = "shared",
    seed: int = 42,
) -> str:
    """Compute 2D loss-landscape surface using per-parameter random directions."""
    work_model = copy.deepcopy(model).to(device)
    work_model.eval()

    original_params_to_perturb: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, param in work_model.named_parameters():
            if param_name_exclude_substr is not None and param_name_exclude_substr in name:
                continue
            original_params_to_perturb[name] = param.data.detach().clone().cpu()

    torch.manual_seed(int(seed))
    perturb_x: Dict[str, torch.Tensor] = {}
    perturb_y: Dict[str, torch.Tensor] = {}
    eps = 1e-8
    for name, p0 in original_params_to_perturb.items():
        p_norm = p0.norm().item()
        d_x = torch.randn_like(p0)
        d_x_norm = d_x.norm().item()
        if d_x_norm > 0:
            d_x = d_x / d_x_norm * (p_norm + eps)
        else:
            d_x = torch.zeros_like(p0)

        d_y = torch.randn_like(p0)
        dx_norm_sq = max(float(d_x.norm().item()) ** 2, eps)
        proj = torch.sum(d_y * d_x) * d_x / dx_norm_sq
        d_y = d_y - proj
        d_y_norm = d_y.norm().item()
        if d_y_norm > 0:
            d_y = d_y / d_y_norm * (p_norm + eps)
        else:
            d_y = torch.zeros_like(p0)

        perturb_x[name] = d_x.to(device)
        perturb_y[name] = d_y.to(device)

    all_batches = []
    for i, batch in enumerate(test_loader):
        if (max_batches is not None) and (i >= max_batches):
            break
        all_batches.append(batch)

    x_coords = np.linspace(x_range[0], x_range[1], int(num_points)).astype(np.float64)
    y_coords = np.linspace(y_range[0], y_range[1], int(num_points)).astype(np.float64)
    loss_grid = np.zeros((int(num_points), int(num_points)), dtype=np.float64)

    with torch.no_grad():
        state = work_model.state_dict()
        for i, xv in enumerate(tqdm(x_coords, desc="Grid X")):
            xv = float(xv)
            for j, yv in enumerate(y_coords):
                yv = float(yv)

                for name, tensor in original_params_to_perturb.items():
                    state[name].copy_(tensor.to(device=device, dtype=state[name].dtype))

                for name in original_params_to_perturb.keys():
                    delta = xv * perturb_x[name] + yv * perturb_y[name]
                    state[name].add_(delta)

                total = 0.0
                count = 0
                for batch in all_batches:
                    inputs, labels = _unwrap_batch(batch)
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = work_model(inputs)
                    logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs
                    loss = criterion(logits, labels)
                    total += float(loss.item())
                    count += 1
                loss_grid[i, j] = total / max(1, count)

                if (j % 10 == 0) and torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()

    os.makedirs(output_dir, exist_ok=True)
    surf_file = os.path.join(output_dir, f"{save_file_name}_task{eval_task_id}.h5")
    with h5py.File(surf_file, "w") as f:
        f.create_dataset("xcoordinates", data=x_coords)
        f.create_dataset("ycoordinates", data=y_coords)
        f.create_dataset("train_loss", data=loss_grid)

    logging.info("[LossLandV1] Saved loss surface to %s", surf_file)
    return surf_file


def compute_curvature_direction(
    mvp_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    device: torch.device,
    *,
    num_iters: int = 5,
    method: str = "power",
    topk: int = 1,
    tol: Optional[float] = None,
    patience: int = 2,
    seed: Optional[int] = None,
    use_abs_eig: bool = False,
) -> Tuple[List[float], List[torch.Tensor]]:
    if dim <= 0:
        return [], []
    method = str(method).lower()
    topk = max(1, int(topk))
    num_iters = int(max(1, num_iters))

    if method == "lanczos":
        if topk == 1:
            lam, vec = _lanczos_lambda_max_generic(
                mvp_fn,
                dim,
                num_iters=num_iters,
                device=device,
                tol=tol or 1e-3,
                reorth=True,
                return_vec=True,
                seed=seed,
            )
            if vec is None:
                return [], []
            return [float(lam)], [vec]
        vals, vecs = _lanczos_topk_generic(
            mvp_fn,
            dim,
            num_iters=num_iters,
            device=device,
            topk=topk,
            tol=tol,
            reorth=True,
            seed=seed,
        )
        return [float(v) for v in vals], vecs

    res = _power_iteration_lambda_max(
        mvp_fn,
        dim,
        num_iters=num_iters,
        device=device,
        tol=tol or 1e-3,
        patience=int(patience),
        return_vec=True,
        topk=topk,
        seed=seed,
        use_abs_eig=bool(use_abs_eig),
    )
    if isinstance(res, tuple) and isinstance(res[0], float):
        if res[1] is None:
            return [], []
        return [float(res[0])], [res[1]]
    vals, vecs = res
    return [float(v) for v in vals], vecs


def _compute_loss_v1_batches(
    model: nn.Module,
    batches: List,
    device: torch.device,
    criterion: Callable,
) -> float:
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            inputs, labels = _unwrap_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs
            loss = criterion(logits, labels)
            total += float(loss.item())
            count += 1
    if count == 0:
        return 0.0
    return total / count


def compute_loss_curve_1d(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    direction_list: List[torch.Tensor],
    *,
    radius: float = 0.5,
    num_points: int = 21,
    max_batches: Optional[int] = None,
    known_classes: Optional[int] = None,
    normalize: bool = False,
    base_loss: Optional[float] = None,
    batches: Optional[List] = None,
    criterion: Optional[Callable] = None,
) -> Dict[str, np.ndarray]:
    if not params or not direction_list:
        return {}

    backup = _clone_params(params)
    dirs = [d.to(device=device) for d in direction_list]
    if normalize:
        concat_norm = _dir_norm(dirs)
        if concat_norm > 0:
            dirs = [d / concat_norm for d in dirs]

    def _eval_loss() -> float:
        if batches is not None and criterion is not None:
            return _compute_loss_v1_batches(model, batches, device, criterion)
        return _compute_loss(model, loader, device, max_batches=max_batches, known_classes=known_classes)

    if base_loss is None:
        _restore_params(params, backup)
        base_loss = float(_eval_loss())

    xs = np.linspace(-radius, radius, int(num_points)).astype(np.float64)
    losses = np.zeros_like(xs, dtype=np.float64)

    for i, x in enumerate(xs):
        _restore_params(params, backup)
        _apply_direction(params, dirs, float(x))
        losses[i] = _eval_loss()

    _restore_params(params, backup)
    return {
        "x": xs,
        "loss": losses,
        "delta_loss": losses - float(base_loss),
        "base_loss": float(base_loss),
    }


def compute_full_vs_lora_curvature_1d(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params_full: List[torch.nn.Parameter],
    params_lora: List[torch.nn.Parameter],
    mvp_full: Callable[[torch.Tensor], torch.Tensor],
    mvp_lora: Callable[[torch.Tensor], torch.Tensor],
    *,
    curvature_method: str = "power",
    num_iters: int = 5,
    topk: int = 1,
    radius_full: float = 0.5,
    radius_lora: float = 0.5,
    num_points: int = 21,
    max_batches: Optional[int] = None,
    known_classes: Optional[int] = None,
    seed: Optional[int] = None,
    tol: Optional[float] = None,
    patience: int = 2,
    use_abs_eig: bool = False,
    base_loss: Optional[float] = None,
    normalize_dir: bool = False,
    criterion: Optional[Callable] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    fixed_batches = None
    if criterion is not None:
        fixed_batches = []
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            fixed_batches.append(batch)
        if not fixed_batches:
            fixed_batches = None

    if fixed_batches is not None and criterion is not None:
        base_loss = _compute_loss_v1_batches(model, fixed_batches, device, criterion)
    elif base_loss is None:
        base_loss = _compute_loss(model, loader, device, max_batches=max_batches, known_classes=known_classes)

    if params_full:
        dim_full = int(sum(p.numel() for p in params_full))
        vals_f, vecs_f = compute_curvature_direction(
            mvp_full,
            dim_full,
            device,
            num_iters=num_iters,
            method=curvature_method,
            topk=topk,
            tol=tol,
            patience=patience,
            seed=seed,
            use_abs_eig=use_abs_eig,
        )
        if vecs_f:
            dirs_f = _unflatten_to_param_like(vecs_f[0].to(device), params_full)
            curve_f = compute_loss_curve_1d(
                model,
                loader,
                device,
                params_full,
                dirs_f,
                radius=radius_full,
                num_points=num_points,
                max_batches=max_batches,
                known_classes=known_classes,
                normalize=normalize_dir,
                base_loss=base_loss,
                batches=fixed_batches,
                criterion=criterion,
            )
            curve_f["eigval"] = np.array(vals_f, dtype=np.float64)
            out["full"] = curve_f

    if params_lora:
        dim_lora = int(sum(p.numel() for p in params_lora))
        vals_l, vecs_l = compute_curvature_direction(
            mvp_lora,
            dim_lora,
            device,
            num_iters=num_iters,
            method=curvature_method,
            topk=topk,
            tol=tol,
            patience=patience,
            seed=seed,
            use_abs_eig=use_abs_eig,
        )
        if vecs_l:
            dirs_l = _unflatten_to_param_like(vecs_l[0].to(device), params_lora)
            curve_l = compute_loss_curve_1d(
                model,
                loader,
                device,
                params_lora,
                dirs_l,
                radius=radius_lora,
                num_points=num_points,
                max_batches=max_batches,
                known_classes=known_classes,
                normalize=normalize_dir,
                base_loss=base_loss,
                batches=fixed_batches,
                criterion=criterion,
            )
            curve_l["eigval"] = np.array(vals_l, dtype=np.float64)
            out["lora"] = curve_l

    return out
