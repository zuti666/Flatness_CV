from typing import Any, Callable, Dict, List, Optional

import logging
import numpy as np
import torch
import torch.nn as nn

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from eval_flat.lanczos_iter import _lanczos_topk_generic, _lanczos_topk_masked
from eval_flat.power_iter import _power_iteration_generic, _power_iteration_masked
from eval_flat.loss_utils import _forward_logits_full, _unwrap_batch


def _projection_ratios(vecs: Optional[List[torch.Tensor]], mask: torch.Tensor) -> List[float]:
    """Compute ||P_mask v||^2 / ||v||^2 for each vector."""
    if vecs is None or len(vecs) == 0 or mask.numel() == 0:
        return []
    ratios: List[float] = []
    for v in vecs:
        device = v.device
        mask_dev = mask.to(device=device, dtype=v.dtype)
        v_use = v.to(device=device, dtype=v.dtype)
        num = torch.dot(v_use * mask_dev, v_use * mask_dev).item()
        den = torch.dot(v_use, v_use).item()
        ratios.append(float(num / (den + 1e-12)))
    return ratios


def _rayleigh_along_mask(
    mv: Callable[[torch.Tensor], torch.Tensor],
    mask: torch.Tensor,
    num_samples: int,
    device: torch.device,
) -> List[float]:
    """Sample Rayleigh quotients restricted to a masked subspace."""
    if num_samples <= 0 or mask.numel() == 0:
        return []
    mask_dev = mask.to(device=device, dtype=torch.float32)
    dim = mask_dev.numel()
    if mask_dev.sum().item() == 0:
        return []
    vals: List[float] = []
    for _ in range(num_samples):
        v = torch.randn(dim, device=device)
        v = v * mask_dev
        nrm = v.norm()
        if nrm.item() == 0.0:
            continue
        v = v / (nrm + 1e-12)
        hv = mv(v)
        vals.append(float(torch.dot(v, hv).item()))
    return vals


def _build_subspace_mask(
    names: List[str],
    splits: List[int],
    substrs: Optional[List[str]],
    *,
    total_dim: Optional[int] = None,
) -> torch.Tensor:
    """Return a boolean mask over the flattened parameter vector."""
    tot = int(total_dim) if total_dim is not None else int(sum(int(s) for s in splits))
    if tot == 0:
        return torch.zeros(0, dtype=torch.bool)
    mask = torch.zeros(tot, dtype=torch.bool)
    if substrs is None or len(substrs) == 0:
        return mask
    ptr = 0
    for name, k in zip(names, splits):
        k_int = int(k)
        hit = any(s in name for s in substrs)
        if hit and k_int > 0:
            mask[ptr : ptr + k_int] = True
        ptr += k_int
    return mask


def _curvature_localization_metrics(
    vecs: Optional[List[torch.Tensor]],
    mask: torch.Tensor,
    mv: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    *,
    num_samples: int = 32,
    basis_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Projection ratios + Rayleigh stats for a given subspace mask."""
    if mask.numel() == 0:
        return {}
    out: Dict[str, Any] = {}

    ratios = _projection_ratios(vecs, mask)
    if ratios:
        out["curv_proj_ratio_space"] = "param_mask_coords"
        out["curv_proj_ratio_note"] = "Parameter-coordinate mask subspace (LoRA param names)."
        out["curv_proj_ratio_top1"] = float(ratios[0])
        out["curv_proj_ratios"] = [float(r) for r in ratios]
        cumsum: List[float] = []
        total = 0.0
        for r in ratios:
            total += float(r)
            cumsum.append(total)
        out["curv_proj_ratio_cumsum"] = cumsum

    rng_state = None
    if basis_seed is not None:
        try:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(int(basis_seed))
        except Exception:
            rng_state = None

    try:
        ray_sub = _rayleigh_along_mask(mv, mask, num_samples, device)
        ray_orth = _rayleigh_along_mask(mv, ~mask.bool(), num_samples, device)
    finally:
        if rng_state is not None:
            try:
                torch.random.set_rng_state(rng_state)
            except Exception:
                pass

    if ray_sub:
        out["curv_rayleigh_sub_mean"] = float(np.mean(ray_sub))
        out["curv_rayleigh_sub_std"] = float(np.std(ray_sub))
    if ray_orth:
        out["curv_rayleigh_orth_mean"] = float(np.mean(ray_orth))
        out["curv_rayleigh_orth_std"] = float(np.std(ray_orth))
    if ray_sub and ray_orth:
        out["curv_rayleigh_gap"] = float(out["curv_rayleigh_sub_mean"] - out["curv_rayleigh_orth_mean"])
    return out





def _normalize_ranges(ranges: Optional[List[List[int]]], max_dim: int) -> List[Tuple[int, int]]:
    if not ranges:
        return [(0, max_dim)]
    out: List[Tuple[int, int]] = []
    for rng in ranges:
        if not rng or len(rng) < 2:
            continue
        start, end = int(rng[0]), int(rng[1])
        start = max(0, min(start, max_dim))
        end = max(0, min(end, max_dim))
        if end > start:
            out.append((start, end))
    if not out:
        out.append((0, max_dim))
    return out


def _infer_qkv_block_ranges(
    shape: Tuple[int, ...],
    tag: str,
    split_dim: Optional[str] = None,
) -> Tuple[Optional[List[List[int]]], Optional[List[List[int]]]]:
    """Infer row/col ranges for qkv blocks when rows or cols are packed as [Q, K, V]."""
    if len(shape) != 2:
        return None, None
    rows, cols = int(shape[0]), int(shape[1])
    tag_key = str(tag or "").strip().lower()
    tag_map = {"q": 0, "k": 1, "v": 2}
    if tag_key not in tag_map:
        return None, None
    idx = tag_map[tag_key]
    if split_dim is not None:
        split_dim = str(split_dim).strip().lower()
    if not split_dim:
        if rows % 3 == 0:
            split_dim = "row"
        elif cols % 3 == 0:
            split_dim = "col"
        else:
            return None, None
    if split_dim == "row":
        if rows % 3 != 0:
            return None, None
        span = rows // 3
        return [[idx * span, (idx + 1) * span]], None
    if split_dim == "col":
        if cols % 3 != 0:
            return None, None
        span = cols // 3
        return None, [[idx * span, (idx + 1) * span]]
    return None, None


def _build_qkv_block_mask(
    names: List[str],
    shapes: List[Tuple[int, ...]],
    splits: List[int],
    target_name: str,
    qkv_tag: Optional[str],
    row_ranges: Optional[List[List[int]]],
    col_ranges: Optional[List[List[int]]],
    *,
    total_dim: int,
    split_dim: Optional[str] = None,
    block_index: Optional[int] = None,
) -> torch.Tensor:
    """Build a mask for a qkv weight block, optionally inferring ranges from q/k/v tag."""
    if total_dim <= 0:
        return torch.zeros(0, dtype=torch.bool)
    if not target_name:
        return torch.zeros(total_dim, dtype=torch.bool)

    block_key = None
    if block_index is not None:
        block_key = f"blocks.{int(block_index)}."

    target_shape: Optional[Tuple[int, int]] = None
    for name, shape in zip(names, shapes):
        if target_name not in name:
            continue
        if block_key is not None and block_key not in name:
            continue
        if len(shape) != 2:
            continue
        target_shape = (int(shape[0]), int(shape[1]))
        break

    if target_shape is None:
        return torch.zeros(total_dim, dtype=torch.bool)

    if row_ranges is None and col_ranges is None:
        inferred_rows, inferred_cols = _infer_qkv_block_ranges(
            target_shape,
            qkv_tag or "",
            split_dim=split_dim,
        )
        row_ranges = inferred_rows
        col_ranges = inferred_cols
        if row_ranges is None and col_ranges is None:
            return torch.zeros(total_dim, dtype=torch.bool)

    mask = torch.zeros(total_dim, dtype=torch.bool)
    ptr = 0
    for name, shape, k in zip(names, shapes, splits):
        k_int = int(k)
        if k_int <= 0:
            ptr += k_int
            continue
        if target_name not in name:
            ptr += k_int
            continue
        if block_key is not None and block_key not in name:
            ptr += k_int
            continue
        if len(shape) != 2:
            ptr += k_int
            continue
        rows, cols = int(shape[0]), int(shape[1])
        row_spans = _normalize_ranges(row_ranges, rows)
        col_spans = _normalize_ranges(col_ranges, cols)
        block_mask = torch.zeros((rows, cols), dtype=torch.bool)
        for rs, re in row_spans:
            for cs, ce in col_spans:
                block_mask[rs:re, cs:ce] = True
        mask[ptr:ptr + k_int] = block_mask.reshape(-1)
        break
    return mask


def _build_param_block_mask(
    names: List[str],
    shapes: List[Tuple[int, ...]],
    splits: List[int],
    target_name: str,
    row_ranges: Optional[List[List[int]]],
    col_ranges: Optional[List[List[int]]],
    *,
    total_dim: int,
) -> torch.Tensor:
    if total_dim <= 0:
        return torch.zeros(0, dtype=torch.bool)
    mask = torch.zeros(total_dim, dtype=torch.bool)
    if not target_name:
        return mask
    ptr = 0
    for name, shape, k in zip(names, shapes, splits):
        k_int = int(k)
        if k_int <= 0:
            ptr += k_int
            continue
        if target_name not in name:
            ptr += k_int
            continue
        if len(shape) != 2:
            ptr += k_int
            continue
        rows, cols = int(shape[0]), int(shape[1])
        row_spans = _normalize_ranges(row_ranges, rows)
        col_spans = _normalize_ranges(col_ranges, cols)
        block_mask = torch.zeros((rows, cols), dtype=torch.bool)
        for rs, re in row_spans:
            for cs, ce in col_spans:
                block_mask[rs:re, cs:ce] = True
        flat = block_mask.reshape(-1)
        mask[ptr:ptr + k_int] = flat
        break
    return mask


def _extract_param_block(
    vec: torch.Tensor,
    names: List[str],
    shapes: List[Tuple[int, ...]],
    splits: List[int],
    target_name: str,
    row_ranges: Optional[List[List[int]]],
    col_ranges: Optional[List[List[int]]],
) -> Optional[torch.Tensor]:
    ptr = 0
    for name, shape, k in zip(names, shapes, splits):
        k_int = int(k)
        if target_name not in name:
            ptr += k_int
            continue
        if len(shape) != 2:
            return None
        rows, cols = int(shape[0]), int(shape[1])
        row_spans = _normalize_ranges(row_ranges, rows)
        col_spans = _normalize_ranges(col_ranges, cols)
        full = vec[ptr:ptr + k_int].reshape(rows, cols)
        if len(row_spans) == 1 and len(col_spans) == 1:
            rs, re = row_spans[0]
            cs, ce = col_spans[0]
            return full[rs:re, cs:ce]
        row_blocks = []
        for rs, re in row_spans:
            col_blocks = [full[rs:re, cs:ce] for cs, ce in col_spans]
            row_blocks.append(torch.cat(col_blocks, dim=1))
        return torch.cat(row_blocks, dim=0)
    return None


def _get_scale_value(scale_module: Any) -> float:
    if isinstance(scale_module, torch.nn.ModuleList) and len(scale_module) > 0:
        scale_module = scale_module[0]
    param = getattr(scale_module, "param", None)
    if param is not None:
        try:
            return float(param.detach().item())
        except Exception:
            return float(param)
    return 1.0


def _get_scale_value_at(scale_module: Any, idx: int) -> float:
    if isinstance(scale_module, torch.nn.ModuleList) and len(scale_module) > 0:
        if 0 <= idx < len(scale_module):
            scale_module = scale_module[idx]
        else:
            scale_module = scale_module[0]
    param = getattr(scale_module, "param", None)
    if param is not None:
        try:
            return float(param.detach().item())
        except Exception:
            return float(param)
    return 1.0


def _infer_task_id_from_prefix(save_prefix: Optional[str]) -> Optional[int]:
    if not save_prefix:
        return None
    match = re.search(r"_t(\d+)$", str(save_prefix))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _infer_task_id_from_args(config: Any) -> Optional[int]:
    args = getattr(config, "args", None)
    if not isinstance(args, dict):
        return None
    for key in ("cur_task", "task_id", "eval_task_id", "task", "current_task"):
        if key not in args:
            continue
        try:
            return int(args[key])
        except Exception:
            return None
    return None


def _infer_lora_save_dir(config: Any, model: torch.nn.Module) -> Optional[str]:
    args = getattr(config, "args", None)
    if isinstance(args, dict):
        path = args.get("filepath", None) or args.get("lora_save_dir", None) or args.get("save_dir", None)
        if path:
            return str(path)
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        save_file = getattr(backbone, "save_file", None)
        if save_file:
            return str(save_file)
    return None


def _get_qkv_module(model: torch.nn.Module, block_index: Optional[int]) -> Optional[torch.nn.Module]:
    if block_index is None:
        return None
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None
    vit = getattr(backbone, "lora_vit", None) or getattr(backbone, "base_vit", None) or backbone
    blocks = getattr(vit, "blocks", None)
    if blocks is None or block_index < 0 or block_index >= len(blocks):
        return None
    return getattr(getattr(blocks[block_index], "attn", None), "qkv", None)


def _load_lora_task_lists(save_dir: Optional[str], task_id: Optional[int]) -> Optional[Tuple[Any, Any]]:
    if not save_dir or task_id is None or task_id < 0:
        return None
    file_a = os.path.join(save_dir, f"lora_w_a_{int(task_id)}.pt")
    file_b = os.path.join(save_dir, f"lora_w_b_{int(task_id)}.pt")
    if not os.path.exists(file_a) or not os.path.exists(file_b):
        return None
    try:
        A_list = torch.load(file_a, map_location="cpu")
        B_list = torch.load(file_b, map_location="cpu")
    except Exception:
        return None
    return A_list, B_list


def _extract_qv_weights_from_lists(
    A_list: Any,
    B_list: Any,
    block_index: int,
    *,
    device: torch.device,
    scale: float,
    normalize: bool,
    eps: float,
) -> Optional[Dict[str, torch.Tensor]]:
    if A_list is None or B_list is None:
        return None
    idx = int(block_index) * 2
    if idx + 1 >= len(A_list) or idx + 1 >= len(B_list):
        return None
    try:
        A_q = A_list[idx].weight.detach().to(device)
        B_q = B_list[idx].weight.detach().to(device)
        A_v = A_list[idx + 1].weight.detach().to(device)
        B_v = B_list[idx + 1].weight.detach().to(device)
    except Exception:
        return None
    delta_q = B_q @ A_q
    delta_v = B_v @ A_v
    if normalize:
        denom_q = (B_q.norm() * A_q.norm()).clamp_min(eps)
        denom_v = (B_v.norm() * A_v.norm()).clamp_min(eps)
        delta_q = delta_q / denom_q
        delta_v = delta_v / denom_v
    if scale != 1.0:
        delta_q = delta_q * float(scale)
        delta_v = delta_v * float(scale)
    return {"q": delta_q, "v": delta_v}


def _sum_saved_lora_qv(
    saved_A: Dict[str, Any],
    saved_B: Dict[str, Any],
    block_index: int,
    *,
    device: torch.device,
    scale_module_prev: Any,
    normalize_prev: bool,
    eps: float,
) -> Optional[Dict[str, torch.Tensor]]:
    if not saved_A or not saved_B:
        return None
    task_ids: List[int] = []
    for key in saved_A.keys():
        if not str(key).startswith("saved_A_"):
            continue
        try:
            task_ids.append(int(str(key).split("_")[-1]))
        except Exception:
            continue
    if not task_ids:
        return None
    sum_q = None
    sum_v = None
    for task_id in sorted(set(task_ids)):
        key_a = f"saved_A_{task_id}"
        key_b = f"saved_B_{task_id}"
        if key_a not in saved_A or key_b not in saved_B:
            continue
        scale = _get_scale_value_at(scale_module_prev, int(task_id))
        delta = _extract_qv_weights_from_lists(
            saved_A[key_a],
            saved_B[key_b],
            block_index,
            device=device,
            scale=scale,
            normalize=normalize_prev,
            eps=eps,
        )
        if delta is None:
            continue
        if sum_q is None:
            sum_q = delta["q"].clone()
            sum_v = delta["v"].clone()
        else:
            sum_q = sum_q + delta["q"]
            sum_v = sum_v + delta["v"]
    if sum_q is None or sum_v is None:
        return None
    return {"q": sum_q, "v": sum_v}


def _current_lora_qv(
    qkv: torch.nn.Module,
    block_index: int,
    *,
    device: torch.device,
    scale: float,
    eps: float,
) -> Optional[Dict[str, torch.Tensor]]:
    if hasattr(qkv, "linear_a_q") and hasattr(qkv, "linear_b_q"):
        try:
            A_q = qkv.linear_a_q.weight.detach().to(device)
            B_q = qkv.linear_b_q.weight.detach().to(device)
            A_v = qkv.linear_a_v.weight.detach().to(device)
            B_v = qkv.linear_b_v.weight.detach().to(device)
        except Exception:
            return None
        delta_q = (B_q @ A_q) * float(scale)
        delta_v = (B_v @ A_v) * float(scale)
        return {"q": delta_q, "v": delta_v}

    task_id = getattr(qkv, "task_id", None)
    saved_A = getattr(qkv, "saved_A", {})
    saved_B = getattr(qkv, "saved_B", {})
    if task_id is None:
        return None
    key_a = f"saved_A_{task_id}"
    key_b = f"saved_B_{task_id}"
    if key_a not in saved_A or key_b not in saved_B:
        return None
    return _extract_qv_weights_from_lists(
        saved_A[key_a],
        saved_B[key_b],
        block_index,
        device=device,
        scale=scale,
        normalize=False,
        eps=eps,
    )


def _infer_w_delta_mode(config: Any, saved_A: Optional[Dict[str, Any]]) -> str:
    mode = str(getattr(config, "w_delta_effective_mode", "auto") or "auto").lower()
    if mode in {"auto", "default", ""}:
        name = str(getattr(config, "model_name", "")).lower()
        if not name:
            args = getattr(config, "args", None)
            if isinstance(args, dict):
                name = str(args.get("model_name", "")).lower()
        if "seq" in name:
            return "sequential"
        if any(tag in name for tag in ("inclora", "inflora", "olora", "sdlora", "inc")):
            return "cumulative"
        if saved_A:
            return "cumulative"
        return "sequential"
    if mode in {"cumulative", "cum", "sum", "merge", "merged", "inc"}:
        return "cumulative"
    return "sequential"


def _extract_lora_qv_weights(model: torch.nn.Module, block_index: int) -> Optional[Dict[str, torch.Tensor]]:
    if block_index is None:
        return None
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None
    vit = getattr(backbone, "lora_vit", None) or getattr(backbone, "base_vit", None) or backbone
    blocks = getattr(vit, "blocks", None)
    if blocks is None or block_index < 0 or block_index >= len(blocks):
        return None
    qkv = getattr(getattr(blocks[block_index], "attn", None), "qkv", None)
    if qkv is None:
        return None

    if hasattr(qkv, "linear_a_q") and hasattr(qkv, "linear_b_q"):
        A_q = qkv.linear_a_q.weight
        B_q = qkv.linear_b_q.weight
        A_v = qkv.linear_a_v.weight
        B_v = qkv.linear_b_v.weight
    else:
        task_id = getattr(qkv, "task_id", None)
        saved_A = getattr(qkv, "saved_A", {})
        saved_B = getattr(qkv, "saved_B", {})
        t_layer_i = getattr(qkv, "t_layer_i", None)
        if task_id is None or t_layer_i is None:
            return None
        key_a = f"saved_A_{task_id}"
        key_b = f"saved_B_{task_id}"
        if key_a not in saved_A or key_b not in saved_B:
            return None
        A_list = saved_A[key_a]
        B_list = saved_B[key_b]
        idx = int(t_layer_i) * 2
        if idx + 1 >= len(A_list) or idx + 1 >= len(B_list):
            return None
        A_q = A_list[idx].weight
        B_q = B_list[idx].weight
        A_v = A_list[idx + 1].weight
        B_v = B_list[idx + 1].weight

    scale = _get_scale_value(getattr(qkv, "scaling_factor", None))
    delta_q = (B_q @ A_q) * scale
    delta_v = (B_v @ A_v) * scale
    return {"q": delta_q, "v": delta_v}


def _extract_lora_qv_factors(
    model: torch.nn.Module,
    block_index: int,
) -> Optional[Dict[str, Any]]:
    if block_index is None:
        return None
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None
    vit = getattr(backbone, "lora_vit", None) or getattr(backbone, "base_vit", None) or backbone
    blocks = getattr(vit, "blocks", None)
    if blocks is None or block_index < 0 or block_index >= len(blocks):
        return None
    qkv = getattr(getattr(blocks[block_index], "attn", None), "qkv", None)
    if qkv is None:
        return None

    if hasattr(qkv, "linear_a_q") and hasattr(qkv, "linear_b_q"):
        A_q = qkv.linear_a_q.weight
        B_q = qkv.linear_b_q.weight
        A_v = qkv.linear_a_v.weight
        B_v = qkv.linear_b_v.weight
    else:
        task_id = getattr(qkv, "task_id", None)
        saved_A = getattr(qkv, "saved_A", {})
        saved_B = getattr(qkv, "saved_B", {})
        t_layer_i = getattr(qkv, "t_layer_i", None)
        if task_id is None or t_layer_i is None:
            return None
        key_a = f"saved_A_{task_id}"
        key_b = f"saved_B_{task_id}"
        if key_a not in saved_A or key_b not in saved_B:
            return None
        A_list = saved_A[key_a]
        B_list = saved_B[key_b]
        idx = int(t_layer_i) * 2
        if idx + 1 >= len(A_list) or idx + 1 >= len(B_list):
            return None
        A_q = A_list[idx].weight
        B_q = B_list[idx].weight
        A_v = A_list[idx + 1].weight
        B_v = B_list[idx + 1].weight

    scale = _get_scale_value(getattr(qkv, "scaling_factor", None))
    return {"q": {"A": A_q, "B": B_q}, "v": {"A": A_v, "B": B_v}, "scale": scale}


def _infer_lora_substr(target_name: str, tag: str, kind: str) -> str:
    if target_name and "qkv.weight" in target_name:
        return target_name.replace("qkv.weight", f"linear_{kind}_{tag}.weight")
    return f"linear_{kind}_{tag}"


def _find_param_by_substring(
    model: torch.nn.Module,
    target_name: str,
    block_index: Optional[int] = None,
) -> Tuple[Optional[str], Optional[torch.nn.Parameter]]:
    if not target_name:
        return None, None
    matches: List[Tuple[str, torch.nn.Parameter]] = []
    for name, p in model.named_parameters():
        if target_name in name:
            matches.append((name, p))
    if not matches:
        return None, None
    if block_index is not None:
        block_key = f"blocks.{int(block_index)}."
        for name, p in matches:
            if block_key in name:
                return name, p
    return matches[0]


def _slice_weight_block(
    weight: torch.Tensor,
    row_ranges: Optional[List[List[int]]],
    col_ranges: Optional[List[List[int]]],
) -> Optional[torch.Tensor]:
    if weight.ndim != 2:
        return None
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    row_spans = _normalize_ranges(row_ranges, rows)
    col_spans = _normalize_ranges(col_ranges, cols)
    row_blocks = []
    for rs, re in row_spans:
        row_blocks.append(weight[rs:re, :])
    if not row_blocks:
        return None
    block = torch.cat(row_blocks, dim=0)
    if len(col_spans) == 1 and col_spans[0] == (0, cols):
        return block
    col_blocks = []
    for cs, ce in col_spans:
        col_blocks.append(block[:, cs:ce])
    if not col_blocks:
        return None
    return torch.cat(col_blocks, dim=1)


def _slice_delta_block(
    delta: torch.Tensor,
    tag: str,
    row_ranges: Optional[List[List[int]]],
    col_ranges: Optional[List[List[int]]],
) -> Optional[torch.Tensor]:
    if delta.ndim != 2:
        return None
    rows, cols = int(delta.shape[0]), int(delta.shape[1])
    row_spans = _normalize_ranges(row_ranges, rows)
    col_spans = _normalize_ranges(col_ranges, cols)

    if row_ranges:
        max_end = None
        for rng in row_ranges:
            if not rng or len(rng) < 2:
                continue
            try:
                max_end = max(max_end or 0, int(rng[1]))
            except Exception:
                continue
        if max_end is not None and max_end > rows and tag in {"q", "k", "v"}:
            offset = {"q": 0, "k": rows, "v": 2 * rows}[tag]
            mapped: List[Tuple[int, int]] = []
            for rs, re in _normalize_ranges(row_ranges, rows * 3):
                if rs < offset or re > offset + rows:
                    continue
                mapped.append((rs - offset, re - offset))
            if mapped:
                row_spans = mapped

    row_blocks = [delta[rs:re, :] for rs, re in row_spans]
    if not row_blocks:
        return None
    block = torch.cat(row_blocks, dim=0)
    if len(col_spans) == 1 and col_spans[0] == (0, cols):
        return block
    col_blocks = [block[:, cs:ce] for cs, ce in col_spans]
    if not col_blocks:
        return None
    return torch.cat(col_blocks, dim=1)


def _extract_qv_factors_from_lists(
    A_list: Any,
    B_list: Any,
    block_index: int,
    *,
    device: torch.device,
) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
    if A_list is None or B_list is None:
        return None
    idx = int(block_index) * 2
    if idx + 1 >= len(A_list) or idx + 1 >= len(B_list):
        return None
    try:
        A_q = A_list[idx].weight.detach().to(device)
        B_q = B_list[idx].weight.detach().to(device)
        A_v = A_list[idx + 1].weight.detach().to(device)
        B_v = B_list[idx + 1].weight.detach().to(device)
    except Exception:
        return None
    return {
        "q": {"A": A_q, "B": B_q},
        "v": {"A": A_v, "B": B_v},
    }


def _get_qv_param_tensors(
    qkv: torch.nn.Module,
    block_index: int,
    tag: str,
) -> Optional[List[torch.nn.Parameter]]:
    if tag not in {"q", "v"}:
        return None
    if hasattr(qkv, "linear_a_q") and hasattr(qkv, "linear_b_q"):
        if tag == "q":
            return [qkv.linear_a_q.weight, qkv.linear_b_q.weight]
        return [qkv.linear_a_v.weight, qkv.linear_b_v.weight]

    task_id = getattr(qkv, "task_id", None)
    saved_A = getattr(qkv, "saved_A", {})
    saved_B = getattr(qkv, "saved_B", {})
    use_block = getattr(qkv, "t_layer_i", None)
    if use_block is None:
        use_block = block_index
    if task_id is None:
        return None
    key_a = f"saved_A_{task_id}"
    key_b = f"saved_B_{task_id}"
    if key_a not in saved_A or key_b not in saved_B:
        return None
    A_list = saved_A[key_a]
    B_list = saved_B[key_b]
    idx = int(use_block) * 2 + (0 if tag == "q" else 1)
    if idx >= len(A_list) or idx >= len(B_list):
        return None
    return [A_list[idx].weight, B_list[idx].weight]


def _clone_param_data(params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _restore_param_data(params: List[torch.nn.Parameter], saved: List[torch.Tensor]) -> None:
    for p, s in zip(params, saved):
        try:
            p.data.copy_(s)
        except Exception:
            pass


def _estimate_diag_fisher(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    *,
    max_batches: Optional[int],
    known_classes: Optional[int],
    micro_bs: int,
) -> Optional[torch.Tensor]:
    if loader is None or not params:
        return None
    micro_bs = int(micro_bs) if micro_bs and micro_bs > 0 else 1
    diag_accum = [torch.zeros_like(p, device=device) for p in params]
    total_samples = 0
    criterion = nn.CrossEntropyLoss(reduction="mean")
    model.eval()

    with torch.enable_grad():
        for batch_idx, batch in enumerate(loader):
            inputs, targets = _unwrap_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            B = int(targets.shape[0])
            for start in range(0, B, micro_bs):
                end = min(start + micro_bs, B)
                x_mb = inputs[start:end]
                y_mb = targets[start:end]

                model.zero_grad(set_to_none=True)
                logits = _forward_logits_full(model, x_mb, y_mb)
                num_classes = logits.size(-1)
                use_split = isinstance(known_classes, int) and 0 < known_classes < num_classes
                if use_split:
                    all_new = (y_mb >= known_classes).all() and (y_mb < num_classes).all()
                else:
                    all_new = False
                if use_split and all_new:
                    logits_use = logits[:, known_classes:]
                    targets_use = y_mb - known_classes
                else:
                    logits_use = logits
                    targets_use = y_mb

                loss = criterion(logits_use, targets_use)
                grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
                for idx, g in enumerate(grads):
                    if g is None:
                        continue
                    diag_accum[idx] += (g.detach() ** 2)
                total_samples += (end - start)

            if max_batches is not None and batch_idx + 1 >= int(max_batches):
                break

    if total_samples <= 0:
        return None
    flat = torch.cat([d.view(-1) for d in diag_accum])
    return flat / float(total_samples)


def _kl_diag_gaussians(
    mu_p: torch.Tensor,
    sigma2_p: torch.Tensor,
    mu_q: torch.Tensor,
    sigma2_q: torch.Tensor,
    *,
    eps: float,
) -> float:
    sigma2_p = sigma2_p.clamp_min(eps)
    sigma2_q = sigma2_q.clamp_min(eps)
    diff = mu_p - mu_q
    term = sigma2_p / sigma2_q + (diff ** 2) / sigma2_q - 1.0 + torch.log(sigma2_q / sigma2_p)
    return float(0.5 * term.sum().item())


def _flatten_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors]) if tensors else torch.zeros(0)


def _unflatten_like(vec: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    offset = 0
    for p in params:
        num = int(p.numel())
        out.append(vec[offset:offset + num].view_as(p))
        offset += num
    return out


def _add_flat_to_params(params: List[torch.nn.Parameter], vec: torch.Tensor, *, alpha: float = 1.0) -> None:
    offset = 0
    for p in params:
        num = int(p.numel())
        p.data.add_(vec[offset:offset + num].view_as(p), alpha=alpha)
        offset += num


def _grad_list_from_batch(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    params: List[torch.nn.Parameter],
    *,
    known_classes: Optional[int],
) -> List[torch.Tensor]:
    logits = _forward_logits_full(model, inputs, targets)
    num_classes = logits.size(-1)
    use_split = isinstance(known_classes, int) and 0 < known_classes < num_classes
    if use_split:
        all_new = (targets >= known_classes).all() and (targets < num_classes).all()
    else:
        all_new = False
    if use_split and all_new:
        logits_use = logits[:, known_classes:]
        targets_use = targets - known_classes
    else:
        logits_use = logits
        targets_use = targets
    loss = nn.CrossEntropyLoss(reduction="mean")(logits_use, targets_use)
    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    out: List[torch.Tensor] = []
    for p, g in zip(params, grads):
        out.append(torch.zeros_like(p) if g is None else g.detach())
    return out


def _grad_vector_from_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    *,
    known_classes: Optional[int],
    max_batches: Optional[int],
) -> Optional[torch.Tensor]:
    if loader is None or not params:
        return None
    accum = [torch.zeros_like(p, device=device) for p in params]
    batches = 0
    for batch_idx, batch in enumerate(loader):
        inputs, targets = _unwrap_batch(batch)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        grads = _grad_list_from_batch(
            model,
            inputs,
            targets,
            params,
            known_classes=known_classes,
        )
        for i, g in enumerate(grads):
            accum[i] += g
        batches += 1
        if max_batches is not None and batches >= int(max_batches):
            break
    if batches == 0:
        return None
    accum = [g / float(batches) for g in accum]
    return _flatten_tensors(accum)


def _adaptive_grad_norm(grad_vec: torch.Tensor, params: List[torch.nn.Parameter]) -> float:
    if grad_vec.numel() == 0:
        return 0.0
    p_flat = _flatten_tensors([p.detach() for p in params]).abs()
    return float((p_flat * grad_vec).pow(2).sum().sqrt().item())


def _compute_sam_direction(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    *,
    known_classes: Optional[int],
    max_batches: Optional[int],
    rho: float,
    adaptive: bool,
    eps: float,
) -> Optional[Dict[str, Any]]:
    g = _grad_vector_from_loader(
        model,
        loader,
        device,
        params,
        known_classes=known_classes,
        max_batches=max_batches,
    )
    if g is None or g.numel() == 0:
        return None
    if adaptive:
        p_flat = _flatten_tensors([p.detach() for p in params])
        grad_norm = _adaptive_grad_norm(g, params)
        scale = float(rho) / (grad_norm + eps)
        e_w = (p_flat * p_flat) * g * scale
    else:
        grad_norm = float(g.norm().item())
        scale = float(rho) / (grad_norm + eps)
        e_w = g * scale
    v_norm = float(e_w.norm().item())
    if v_norm <= eps:
        return None
    return {
        "direction": e_w / v_norm,
        "rho": float(rho),
        "grad_norm": float(grad_norm),
        "dir_norm": float(v_norm),
    }


def _compute_gam_direction(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    *,
    known_classes: Optional[int],
    max_batches: Optional[int],
    grad_rho: float,
    grad_norm_rho: float,
    beta1: float,
    beta2: float,
    beta3: float,
    gamma: float,
    adaptive: bool,
    eps: float,
) -> Optional[Dict[str, Any]]:
    if loader is None or not params:
        return None
    g_accum = torch.zeros(sum(int(p.numel()) for p in params), device=device)
    batches = 0
    saved_params = _clone_param_data(params)
    for batch_idx, batch in enumerate(loader):
        inputs, targets = _unwrap_batch(batch)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        g0_list = _grad_list_from_batch(
            model, inputs, targets, params, known_classes=known_classes
        )
        g0 = _flatten_tensors(g0_list)
        g0_norm = _adaptive_grad_norm(g0, params) if adaptive else float(g0.norm().item())
        if g0_norm <= eps:
            _restore_param_data(params, saved_params)
            continue
        scale0 = float(grad_rho) / (g0_norm + eps)
        if adaptive:
            p_flat = _flatten_tensors([p.detach() for p in params])
            e_w0 = (p_flat * p_flat) * g0 * scale0
        else:
            e_w0 = g0 * scale0
        _add_flat_to_params(params, e_w0)

        g1_list = _grad_list_from_batch(
            model, inputs, targets, params, known_classes=known_classes
        )
        g1 = _flatten_tensors(g1_list)
        _restore_param_data(params, saved_params)

        delta = g1 - g0
        delta_norm = _adaptive_grad_norm(delta, params) if adaptive else float(delta.norm().item())
        if delta_norm <= eps:
            continue
        scale1 = float(grad_norm_rho) / (delta_norm + eps)
        if adaptive:
            p_flat = _flatten_tensors([p.detach() for p in params])
            e_w1 = (p_flat * p_flat) * delta * scale1
        else:
            e_w1 = delta * scale1
        _add_flat_to_params(params, e_w1)

        g2_list = _grad_list_from_batch(
            model, inputs, targets, params, known_classes=known_classes
        )
        g2 = _flatten_tensors(g2_list)
        g2_norm = _adaptive_grad_norm(g2, params) if adaptive else float(g2.norm().item())
        if g2_norm <= eps:
            _restore_param_data(params, saved_params)
            continue
        scale2 = float(grad_rho) / (g2_norm + eps)
        if adaptive:
            p_flat = _flatten_tensors([p.detach() for p in params])
            e_w2 = (p_flat * p_flat) * g2 * scale2
        else:
            e_w2 = g2 * scale2
        _add_flat_to_params(params, e_w2)

        g3_list = _grad_list_from_batch(
            model, inputs, targets, params, known_classes=known_classes
        )
        g3 = _flatten_tensors(g3_list)
        _restore_param_data(params, saved_params)

        pro_m = g0 + abs(float(beta2)) * g2
        g_new = float(beta1) * g1 + float(beta3) * g3
        inner = float(torch.dot(pro_m, g_new).item())
        new_norm = float(g_new.norm().item())
        old_norm = float(pro_m.norm().item())
        if new_norm <= eps or old_norm <= eps:
            continue
        cosine = inner / (new_norm * old_norm + eps)
        vertical = pro_m - cosine * old_norm * g_new / (new_norm + eps)
        g_final = g_new - float(gamma) * vertical
        g_accum += g_final
        batches += 1

        if max_batches is not None and batches >= int(max_batches):
            break
    _restore_param_data(params, saved_params)
    if batches == 0:
        return None
    g_final = g_accum / float(batches)
    g_norm = float(g_final.norm().item())
    if g_norm <= eps:
        return None
    return {
        "direction": g_final / g_norm,
        "rho": float(grad_rho),
        "dir_norm": float(g_norm),
        "batches": int(batches),
    }


def _curv_ts_eval(
    wrapped_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    total_dim: int,
    mvp_map: Dict[str, callable],
    config: Any,
    *,
    save_prefix: str,
    known_classes: Optional[int] = None,
) -> Dict[str, Any]:
    if not bool(getattr(config, "curv_ts", False)):
        return {}
    if loader is None or not params:
        return {}

    args = getattr(config, "args", {}) or {}
    opt_type = str(args.get("optimizer_type", "")).lower()
    if opt_type not in {"sam", "gam"}:
        return {}

    backend = str(getattr(config, "curv_ts_backend", "ggn") or "ggn").lower()
    mvp_fn = mvp_map.get(backend)
    if mvp_fn is None:
        return {}

    eps = float(getattr(config, "curv_ts_eps", 1e-12))
    dir_batches = getattr(config, "curv_ts_dir_max_batches", None)
    if dir_batches is None:
        dir_batches = getattr(config, "loss_eval_max_batches", None)

    for p in params:
        if not p.requires_grad:
            p.requires_grad_(True)

    prev_mode = wrapped_model.training
    wrapped_model.train()
    try:
        if opt_type == "sam":
            rho = float(getattr(config, "curv_ts_rho", None) or args.get("sam_rho", 0.05))
            adaptive = bool(args.get("sam_adaptive", False))
            info = _compute_sam_direction(
                wrapped_model,
                loader,
                device,
                params,
                known_classes=known_classes,
                max_batches=dir_batches,
                rho=rho,
                adaptive=adaptive,
                eps=eps,
            )
            if info is None:
                return {}
            direction = info["direction"].to(device)
            rho = float(info.get("rho", rho))
        else:
            rho = float(getattr(config, "curv_ts_rho", None) or args.get("gam_grad_rho", 0.2))
            grad_norm_rho = float(args.get("gam_grad_norm_rho", 0.2))
            adaptive = bool(args.get("gam_adaptive", False))
            info = _compute_gam_direction(
                wrapped_model,
                loader,
                device,
                params,
                known_classes=known_classes,
                max_batches=dir_batches,
                grad_rho=rho,
                grad_norm_rho=grad_norm_rho,
                beta1=float(args.get("gam_grad_beta_1", 1.0)),
                beta2=float(args.get("gam_grad_beta_2", 1.0)),
                beta3=float(args.get("gam_grad_beta_3", 1.0)),
                gamma=float(args.get("gam_grad_gamma", 0.1)),
                adaptive=adaptive,
                eps=float(args.get("gam_perturb_eps", 1e-12)),
            )
            if info is None:
                return {}
            direction = info["direction"].to(device)
            rho = float(info.get("rho", rho))

        if direction.numel() != total_dim:
            return {}
        hv = mvp_fn(direction)
        rayleigh = float(torch.dot(direction, hv).item())
        ts_val = 0.5 * (rho ** 2) * rayleigh
        metrics = {
            "curv_ts_backend": backend,
            "curv_ts_optimizer": opt_type,
            "curv_ts_rho": float(rho),
            "curv_ts_rayleigh": float(rayleigh),
            "curv_ts_value": float(ts_val),
            "curv_ts_dim": int(total_dim),
        }
        for k in ("dir_norm", "grad_norm", "batches"):
            if k in info:
                metrics[f"curv_ts_{k}"] = float(info[k]) if k != "batches" else int(info[k])
        return metrics
    finally:
        wrapped_model.train(prev_mode)


def _curv_noise_cov_eval(
    wrapped_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    total_dim: int,
    mvp_map: Dict[str, callable],
    config: Any,
    *,
    save_prefix: str,
    known_classes: Optional[int] = None,
) -> Dict[str, Any]:
    if not bool(getattr(config, "curv_noise", False)):
        return {}
    if loader is None or not params:
        return {}

    backend = str(getattr(config, "curv_noise_backend", "ggn") or "ggn").lower()
    mvp_fn = mvp_map.get(backend)
    if mvp_fn is None:
        return {}
    eig_method = str(getattr(config, "curv_noise_eig_method", "lanczos") or "lanczos").lower()
    topk = int(getattr(config, "curv_noise_topk", None) or getattr(config, "curv_topk", 1))
    num_iters = int(getattr(config, "hessian_power_iters", 5))
    tol = getattr(config, "eig_tol", None)
    seed = getattr(config, "loss_land_seed", None)
    eps = float(getattr(config, "curv_noise_eps", 1e-12))

    if eig_method == "lanczos":
        eigvals, eigvecs = _lanczos_topk_generic(
            mvp_fn,
            total_dim,
            num_iters,
            device,
            topk=topk,
            tol=tol,
            seed=seed,
        )
    else:
        res = _power_iteration_generic(
            mvp_fn,
            total_dim,
            num_iters,
            device,
            return_vec=True,
            topk=topk,
            tol=tol,
            patience=int(getattr(config, "eig_patience", 2)),
            seed=seed,
        )
        if isinstance(res, tuple) and isinstance(res[0], float):
            eigvals = [float(res[0])]
            eigvecs = [res[1]] if res[1] is not None else []
        else:
            eigvals, eigvecs = res
    if not eigvals or not eigvecs:
        return {}

    max_batches = getattr(config, "curv_noise_max_batches", None)
    if max_batches is None:
        max_batches = getattr(config, "loss_eval_max_batches", None)

    grads: List[torch.Tensor] = []
    for batch_idx, batch in enumerate(loader):
        inputs, targets = _unwrap_batch(batch)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        g_list = _grad_list_from_batch(
            wrapped_model,
            inputs,
            targets,
            params,
            known_classes=known_classes,
        )
        g_vec = _flatten_tensors(g_list).to(device)
        grads.append(g_vec)
        if max_batches is not None and len(grads) >= int(max_batches):
            break

    B = len(grads)
    if B < 2:
        return {}
    g_mean = torch.stack(grads, dim=0).mean(dim=0)

    s_sums = torch.zeros(len(eigvals), device=device)
    sigma_trace = 0.0
    for g in grads:
        gc = g - g_mean
        sigma_trace += float((gc * gc).sum().item())
        for i, u in enumerate(eigvecs):
            a = torch.dot(u.to(device), gc)
            s_sums[i] += a * a

    denom = float(B - 1)
    proj_vars = (s_sums / denom).detach().cpu().tolist()
    tr_sigma = float(sigma_trace / denom)
    tr_h_topk = float(sum(float(l) for l in eigvals))
    tr_hsigma = float(sum(float(l) * float(s) for l, s in zip(eigvals, proj_vars)))
    tracebar = float(tr_h_topk * tr_sigma / max(1, int(total_dim)))
    align_gain = float(tr_hsigma - tracebar)

    return {
        "curv_noise_backend": backend,
        "curv_noise_eig_method": eig_method,
        "curv_noise_topk": int(topk),
        "curv_noise_batches": int(B),
        "curv_noise_tr_hsigma_topk": tr_hsigma,
        "curv_noise_tr_sigma": tr_sigma,
        "curv_noise_tr_h_topk": tr_h_topk,
        "curv_noise_tracebar": tracebar,
        "curv_noise_align_gain": align_gain,
        "curv_noise_eigvals": [float(x) for x in eigvals],
        "curv_noise_proj_vars": [float(x) for x in proj_vars],
    }


def _random_svd_subspaces(
    rows: int,
    cols: int,
    rank: int,
    *,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random baseline via top-r singular vectors of a random matrix (paper-consistent)."""
    rng_state = None
    if seed is not None:
        try:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(int(seed))
        except Exception:
            rng_state = None
    try:
        mat = torch.randn(rows, cols, device=device, dtype=dtype)
        U, _, Vh = torch.linalg.svd(mat, full_matrices=False)
        r = int(min(rank, U.shape[1], Vh.shape[0]))
        return U[:, :r], Vh[:r, :].T
    finally:
        if rng_state is not None:
            try:
                torch.random.set_rng_state(rng_state)
            except Exception:
                pass


def _random_orthonormal(
    rows: int,
    cols: int,
    *,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return a random orthonormal basis via SVD of a random matrix."""
    U, _ = _random_svd_subspaces(
        rows,
        cols,
        rank=min(rows, cols),
        seed=seed,
        device=device,
        dtype=dtype,
    )
    return U[:, :cols]


def _delta_w_projection_eval(
    wrapped_model: torch.nn.Module,
    names: List[str],
    shapes: List[Tuple[int, ...]],
    splits: List[int],
    total_dim: int,
    device: torch.device,
    mvp_map: Dict[str, callable],
    config: Any,
    *,
    save_dir: Optional[str],
    save_prefix: str,
) -> Dict[str, Any]:
    if not bool(getattr(config, "delta_w_projection", False)):
        return {}
    target_name = getattr(config, "delta_w_param_name", None)
    if not target_name:
        return {}
    blocks = getattr(config, "delta_w_blocks", None) or []
    if not blocks:
        return {}
    backend = str(getattr(config, "delta_w_backend", None) or getattr(config, "curv_mvp", "hessian")).lower()
    mvp_fn = mvp_map.get(backend)
    if mvp_fn is None:
        return {}

    block_index = getattr(config, "delta_w_block_index", None)
    qkv = _get_qkv_module(wrapped_model, block_index)
    if qkv is None:
        return {}
    _, weight_param = _find_param_by_substring(wrapped_model, target_name, block_index=block_index)
    w_full = weight_param.detach() if weight_param is not None else None

    lora_rank = getattr(config, "delta_w_rank", None)
    if lora_rank is None:
        try:
            lora_rank = int(getattr(config, "args", {}).get("lora_rank", 0))
        except Exception:
            lora_rank = 0
    lora_rank = int(lora_rank) if lora_rank else None

    topk = int(getattr(config, "delta_w_topk", 5))
    num_iters = int(getattr(config, "hessian_power_iters", 5))
    tol = getattr(config, "eig_tol", None)
    patience = int(getattr(config, "eig_patience", 2))
    seed = getattr(config, "loss_land_seed", None)
    eps = float(getattr(config, "delta_w_svd_eps", 1e-12))
    eig_method = str(getattr(config, "delta_w_eig_method", "power")).lower()

    saved_A = getattr(qkv, "saved_A", {})
    saved_B = getattr(qkv, "saved_B", {})
    mode = _infer_w_delta_mode(config, saved_A)
    scale_cur = _get_scale_value_at(getattr(qkv, "scaling_factor", None), 0)
    scale_prev_mod = getattr(qkv, "scaling_factor_prev", None)
    delta_cur = _current_lora_qv(qkv, block_index, device=device, scale=scale_cur, eps=eps)
    if mode == "cumulative":
        delta_prev = _sum_saved_lora_qv(
            saved_A,
            saved_B,
            block_index,
            device=device,
            scale_module_prev=scale_prev_mod,
            normalize_prev=True,
            eps=eps,
        )
        if delta_prev is None:
            delta_w = delta_cur
        elif delta_cur is None:
            delta_w = delta_prev
        else:
            delta_w = {"q": delta_prev["q"] + delta_cur["q"], "v": delta_prev["v"] + delta_cur["v"]}
    else:
        delta_w = delta_cur
    if delta_w is None:
        return {}

    metrics: Dict[str, Any] = {}
    for block in blocks:
        tag = str(block.get("tag", "")).strip()
        row_ranges = block.get("row_ranges", None)
        col_ranges = block.get("col_ranges", None)
        if not tag:
            continue
        dw = delta_w.get(tag)
        if dw is None:
            continue

        mask = _build_param_block_mask(
            names,
            shapes,
            splits,
            target_name,
            row_ranges,
            col_ranges,
            total_dim=total_dim,
        )
        if mask.numel() == 0 or mask.sum().item() == 0:
            continue

        if eig_method == "lanczos":
            eigvals, eigvecs = _lanczos_topk_masked(
                mvp_fn,
                total_dim,
                mask,
                device,
                num_iters,
                topk=topk,
                tol=tol,
                seed=seed,
            )
        else:
            eigvals, eigvecs = _power_iteration_masked(
                mvp_fn,
                total_dim,
                mask,
                device,
                num_iters,
                topk=topk,
                tol=tol,
                patience=patience,
                seed=seed,
            )

        try:
            U, _, Vh = torch.linalg.svd(dw, full_matrices=False)
        except Exception:
            continue
        r = int(min(U.shape[1], Vh.shape[0], lora_rank or U.shape[1]))
        U_r = U[:, :r]
        V_r = Vh[:r, :].T

        w_block = _slice_weight_block(w_full, row_ranges, col_ranges) if w_full is not None else None
        if w_block is not None and w_block.shape != dw.shape:
            w_block = None
        w_r = 0
        U_w_r = None
        V_w_r = None
        if w_block is not None:
            try:
                U_w, _, Vh_w = torch.linalg.svd(w_block, full_matrices=False)
                w_r = int(min(U_w.shape[1], Vh_w.shape[0], r))
                if w_r > 0:
                    U_w_r = U_w[:, :w_r]
                    V_w_r = Vh_w[:w_r, :].T
            except Exception:
                w_r = 0

        ratios: List[float] = []
        cores: List[torch.Tensor] = []
        w_ratios: List[float] = []
        w_cores: List[torch.Tensor] = []
        for v in eigvecs:
            V = _extract_param_block(v, names, shapes, splits, target_name, row_ranges, col_ranges)
            if V is None:
                ratios.append(0.0)
                cores.append(torch.zeros((U_r.shape[1], V_r.shape[1]), device=device))
                if w_r > 0:
                    w_ratios.append(0.0)
                    w_cores.append(torch.zeros((w_r, w_r), device=device))
                continue
            denom = float(V.norm().item() ** 2)
            if denom <= eps:
                ratios.append(0.0)
                cores.append(torch.zeros((U_r.shape[1], V_r.shape[1]), device=device))
                if w_r > 0:
                    w_ratios.append(0.0)
                    w_cores.append(torch.zeros((w_r, w_r), device=device))
                continue
            tmp = U_r.T @ V @ V_r
            proj = U_r @ tmp @ V_r.T
            num = float(proj.norm().item() ** 2)
            ratios.append(float(num / (denom + eps)))
            cores.append(tmp)
            if w_r > 0 and U_w_r is not None and V_w_r is not None and w_block is not None and V.shape == w_block.shape:
                tmp_w = U_w_r.T @ V @ V_w_r
                proj_w = U_w_r @ tmp_w @ V_w_r.T
                num_w = float(proj_w.norm().item() ** 2)
                w_ratios.append(float(num_w / (denom + eps)))
                w_cores.append(tmp_w)
            elif w_r > 0:
                w_ratios.append(0.0)
                w_cores.append(torch.zeros((w_r, w_r), device=device))

        metrics[f"delta_w_proj_{tag}_backend"] = backend
        metrics[f"delta_w_proj_{tag}_eig_method"] = eig_method
        metrics[f"delta_w_proj_{tag}_space"] = "delta_w_bilinear_subspace"
        metrics[f"delta_w_proj_{tag}_note"] = "Projection onto U_r (.) V_r^T from delta_w SVD."
        metrics[f"delta_w_proj_{tag}_eigvals"] = [float(x) for x in eigvals]
        metrics[f"delta_w_proj_{tag}_ratios"] = [float(x) for x in ratios]
        metrics[f"delta_w_proj_{tag}_proj_ratio_LoRA"] = [float(x) for x in ratios]
        metrics[f"delta_w_proj_{tag}_proj_ratio_LoRA_coord"] = [float(x) for x in ratios]
        if ratios:
            metrics[f"delta_w_proj_{tag}_ratio_top1"] = float(ratios[0])
            metrics[f"delta_w_proj_{tag}_proj_ratio_LoRA_top1"] = float(ratios[0])
            metrics[f"delta_w_proj_{tag}_proj_ratio_LoRA_coord_top1"] = float(ratios[0])
        if eigvals and ratios and len(eigvals) == len(ratios):
            num = sum(float(l) * float(r) for l, r in zip(eigvals, ratios))
            den = sum(float(l) for l in eigvals)
            metrics[f"delta_w_proj_{tag}_ratio_weighted"] = float(num / (den + eps))
            metrics[f"delta_w_proj_{tag}_proj_ratio_LoRA_weighted"] = float(num / (den + eps))
            metrics[f"delta_w_proj_{tag}_proj_ratio_LoRA_coord_weighted"] = float(num / (den + eps))
        if w_ratios:
            metrics[f"delta_w_proj_w_{tag}_ratios"] = [float(x) for x in w_ratios]
            metrics[f"delta_w_proj_{tag}_proj_ratio_W"] = [float(x) for x in w_ratios]
            metrics[f"delta_w_proj_w_{tag}_ratio_top1"] = float(w_ratios[0])
            metrics[f"delta_w_proj_{tag}_proj_ratio_W_top1"] = float(w_ratios[0])
            if eigvals and len(eigvals) == len(w_ratios):
                num = sum(float(l) * float(r) for l, r in zip(eigvals, w_ratios))
                den = sum(float(l) for l in eigvals)
                metrics[f"delta_w_proj_w_{tag}_ratio_weighted"] = float(num / (den + eps))
                metrics[f"delta_w_proj_{tag}_proj_ratio_W_weighted"] = float(num / (den + eps))

        if cores:
            try:
                G = torch.stack([c.reshape(-1) for c in cores], dim=1)
                svals = torch.linalg.svdvals(G)
                p = int(min(G.shape[0], G.shape[1]))
                cos2 = (svals[:p] ** 2).detach().cpu().tolist()
                k_p = float(sum(cos2))
                metrics[f"delta_w_proj_{tag}_grassmann_cos2"] = [float(x) for x in cos2]
                metrics[f"delta_w_proj_{tag}_grassmann_kp"] = k_p
                metrics[f"delta_w_proj_{tag}_grassmann_ap"] = float(k_p / max(1, p))
                metrics[f"delta_w_proj_{tag}_grassmann_dp"] = float(max(0.0, p - k_p) ** 0.5)
            except Exception:
                pass

        rand_ratios: List[float] = []
        rand_cores: List[torch.Tensor] = []
        if r > 0:
            seed_offset = sum(ord(c) for c in tag)
            seed_rand = None if seed is None else int(seed) + seed_offset
            U_rand, V_rand = _random_svd_subspaces(
                U_r.shape[0],
                V_r.shape[0],
                r,
                seed=seed_rand,
                device=dw.device,
                dtype=dw.dtype,
            )
            for v in eigvecs:
                V = _extract_param_block(v, names, shapes, splits, target_name, row_ranges, col_ranges)
                if V is None:
                    rand_ratios.append(0.0)
                    rand_cores.append(torch.zeros((r, r), device=device))
                    continue
                denom = float(V.norm().item() ** 2)
                if denom <= eps:
                    rand_ratios.append(0.0)
                    rand_cores.append(torch.zeros((r, r), device=device))
                    continue
                tmp = U_rand.T @ V @ V_rand
                proj = U_rand @ tmp @ V_rand.T
                num = float(proj.norm().item() ** 2)
                rand_ratios.append(float(num / (denom + eps)))
                rand_cores.append(tmp)
            metrics[f"delta_w_proj_{tag}_rand_ratios"] = [float(x) for x in rand_ratios]
            if rand_ratios:
                metrics[f"delta_w_proj_{tag}_rand_ratio_top1"] = float(rand_ratios[0])
            if eigvals and rand_ratios and len(eigvals) == len(rand_ratios):
                num = sum(float(l) * float(r) for l, r in zip(eigvals, rand_ratios))
                den = sum(float(l) for l in eigvals)
                metrics[f"delta_w_proj_{tag}_rand_ratio_weighted"] = float(num / (den + eps))
                if f"delta_w_proj_{tag}_proj_ratio_LoRA_weighted" in metrics:
                    metrics[f"delta_w_proj_{tag}_gain_LoRA"] = float(
                        metrics[f"delta_w_proj_{tag}_proj_ratio_LoRA_weighted"] / (metrics[f"delta_w_proj_{tag}_rand_ratio_weighted"] + eps)
                    )
                if f"delta_w_proj_{tag}_proj_ratio_W_weighted" in metrics:
                    metrics[f"delta_w_proj_{tag}_gain_W"] = float(
                        metrics[f"delta_w_proj_{tag}_proj_ratio_W_weighted"] / (metrics[f"delta_w_proj_{tag}_rand_ratio_weighted"] + eps)
                    )
                if f"delta_w_proj_{tag}_proj_ratio_W_weighted" in metrics and f"delta_w_proj_{tag}_proj_ratio_LoRA_weighted" in metrics:
                    metrics[f"delta_w_proj_{tag}_pref_LoRA_over_W"] = float(
                        metrics[f"delta_w_proj_{tag}_proj_ratio_LoRA_weighted"] / (metrics[f"delta_w_proj_{tag}_proj_ratio_W_weighted"] + eps)
                    )
            if rand_cores:
                try:
                    G_rand = torch.stack([c.reshape(-1) for c in rand_cores], dim=1)
                    svals = torch.linalg.svdvals(G_rand)
                    p = int(min(G_rand.shape[0], G_rand.shape[1]))
                    cos2 = (svals[:p] ** 2).detach().cpu().tolist()
                    k_p = float(sum(cos2))
                    metrics[f"delta_w_proj_{tag}_rand_grassmann_cos2"] = [float(x) for x in cos2]
                    metrics[f"delta_w_proj_{tag}_rand_grassmann_kp"] = k_p
                    metrics[f"delta_w_proj_{tag}_rand_grassmann_ap"] = float(k_p / max(1, p))
                    metrics[f"delta_w_proj_{tag}_rand_grassmann_dp"] = float(max(0.0, p - k_p) ** 0.5)
                except Exception:
                    pass

        if bool(getattr(config, "delta_w_save_tensors", True)) and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{save_prefix}_delta_w_proj_{tag}.pt")
            payload = {
                "param_name": target_name,
                "block_tag": tag,
                "row_ranges": row_ranges,
                "col_ranges": col_ranges,
                "backend": backend,
                "eigvals": [float(x) for x in eigvals],
                "ratios": [float(x) for x in ratios],
                "delta_w": dw.detach().cpu(),
                "U_r": U_r.detach().cpu(),
                "V_r": V_r.detach().cpu(),
                "eigvecs": [v.detach().cpu() for v in eigvecs],
                "core_mats": [c.detach().cpu() for c in cores],
            }
            if w_block is not None and w_r > 0 and U_w_r is not None and V_w_r is not None and w_ratios:
                payload["W_block"] = w_block.detach().cpu()
                payload["U_w_r"] = U_w_r.detach().cpu()
                payload["V_w_r"] = V_w_r.detach().cpu()
                payload["w_proj_ratios"] = [float(x) for x in w_ratios]
                payload["w_core_mats"] = [c.detach().cpu() for c in w_cores]
            if rand_cores:
                payload["U_rand"] = U_rand.detach().cpu()
                payload["V_rand"] = V_rand.detach().cpu()
                payload["rand_ratios"] = [float(x) for x in rand_ratios]
                payload["rand_core_mats"] = [c.detach().cpu() for c in rand_cores]
            torch.save(payload, out_path)
            metrics[f"delta_w_proj_{tag}_path"] = out_path

    return metrics


def _delta_w_full_projection_eval(
    wrapped_model: torch.nn.Module,
    names: List[str],
    shapes: List[Tuple[int, ...]],
    splits: List[int],
    total_dim: int,
    device: torch.device,
    mvp_map: Dict[str, callable],
    config: Any,
    *,
    save_dir: Optional[str],
    save_prefix: str,
) -> Dict[str, Any]:
    if not bool(getattr(config, "delta_w_full_projection", False)):
        return {}
    target_name = getattr(config, "delta_w_full_param_name", None) or getattr(config, "delta_w_param_name", None)
    if not target_name:
        return {}
    blocks = getattr(config, "delta_w_full_blocks", None) or getattr(config, "delta_w_blocks", None) or []
    if not blocks:
        return {}
    backend = str(
        getattr(config, "delta_w_full_backend", None)
        or getattr(config, "delta_w_backend", None)
        or getattr(config, "curv_mvp", "hessian")
    ).lower()
    mvp_fn = mvp_map.get(backend)
    if mvp_fn is None:
        return {}

    block_index = getattr(config, "delta_w_full_block_index", None)
    if block_index is None:
        block_index = getattr(config, "delta_w_block_index", None)
    factors = _extract_lora_qv_factors(wrapped_model, block_index)
    if factors is None:
        return {}

    lora_rank = getattr(config, "delta_w_full_rank", None)
    if lora_rank is None:
        lora_rank = getattr(config, "delta_w_rank", None)
    if lora_rank is None:
        try:
            lora_rank = int(getattr(config, "args", {}).get("lora_rank", 0))
        except Exception:
            lora_rank = 0
    lora_rank = int(lora_rank) if lora_rank else None

    topk = getattr(config, "delta_w_full_topk", None)
    if topk is None:
        topk = getattr(config, "delta_w_topk", 5)
    topk = int(topk)
    num_iters = int(getattr(config, "hessian_power_iters", 5))
    tol = getattr(config, "eig_tol", None)
    patience = int(getattr(config, "eig_patience", 2))
    seed = getattr(config, "loss_land_seed", None)
    eps = float(
        getattr(config, "delta_w_full_svd_eps", None)
        or getattr(config, "delta_w_svd_eps", 1e-12)
    )
    eig_method = str(
        getattr(config, "delta_w_full_eig_method", None)
        or getattr(config, "delta_w_eig_method", "power")
    ).lower()
    save_tensors = getattr(config, "delta_w_full_save_tensors", None)
    if save_tensors is None:
        save_tensors = bool(getattr(config, "delta_w_save_tensors", True))

    name_to_shape: Dict[str, Tuple[int, ...]] = {}
    for name, shape in zip(names, shapes):
        if name and name not in name_to_shape:
            name_to_shape[name] = tuple(int(x) for x in shape)

    w_full = None
    if target_name:
        _, weight_param = _find_param_by_substring(wrapped_model, target_name, block_index=block_index)
        if weight_param is not None:
            w_full = weight_param.detach()
            if w_full.ndim != 2:
                w_full = None

    metrics: Dict[str, Any] = {}
    scale = float(factors.get("scale", 1.0))
    for block in blocks:
        tag = str(block.get("tag", "")).strip()
        row_ranges = block.get("row_ranges", None)
        col_ranges = block.get("col_ranges", None)
        if not tag:
            continue
        factor = factors.get(tag)
        if factor is None:
            continue
        A = factor.get("A", None)
        B = factor.get("B", None)
        if A is None or B is None:
            continue

        A = A.detach()
        B = B.detach()
        try:
            delta_w = (B @ A) * scale
        except Exception:
            continue

        mask = _build_param_block_mask(
            names,
            shapes,
            splits,
            target_name,
            row_ranges,
            col_ranges,
            total_dim=total_dim,
        )
        if mask.numel() == 0:
            mask = torch.zeros(total_dim, dtype=torch.bool)
        a_substr = block.get("a_substr", None) or block.get("a_name", None)
        b_substr = block.get("b_substr", None) or block.get("b_name", None)
        if not a_substr:
            a_substr = _infer_lora_substr(target_name, tag, "a")
        if not b_substr:
            b_substr = _infer_lora_substr(target_name, tag, "b")
        logging.info(
            "[FlatEval] DeltaWFull block=%s tag=%s row_ranges=%s col_ranges=%s A=%s B=%s delta_w=%s scale=%.6g a_substr=%s b_substr=%s",
            str(block_index),
            tag,
            str(row_ranges),
            str(col_ranges),
            tuple(A.shape),
            tuple(B.shape),
            tuple(delta_w.shape),
            float(scale),
            a_substr,
            b_substr,
        )
        ptr = 0
        for name, k in zip(names, splits):
            k_int = int(k)
            if k_int <= 0:
                ptr += k_int
                continue
            if (a_substr and a_substr in name) or (b_substr and b_substr in name):
                mask[ptr:ptr + k_int] = True
            ptr += k_int
        mask_sum = int(mask.sum().item()) if mask.numel() else 0
        matched_names = [
            name for name in names
            if name == target_name or name == a_substr or name == b_substr
        ]
        target_hits = [name for name in names if target_name and target_name in name]
        a_hits = [name for name in names if a_substr and a_substr in name]
        b_hits = [name for name in names if b_substr and b_substr in name]
        param_hits = []
        ptr = 0
        for name, k in zip(names, splits):
            k_int = int(k)
            if k_int <= 0:
                ptr += k_int
                continue
            if mask[ptr:ptr + k_int].any().item():
                param_hits.append(name)
            ptr += k_int
        logging.info(
            "[FlatEval] DeltaWFull mask tag=%s mask_sum=%d param_hits=%d matched=%d target_hits=%d a_hits=%d b_hits=%d",
            tag,
            mask_sum,
            len(param_hits),
            len(matched_names),
            len(target_hits),
            len(a_hits),
            len(b_hits),
        )
        logging.info(
            "[FlatEval] DeltaWFull mask_names tag=%s param_hits=%s matched_names=%s",
            tag,
            param_hits,
            matched_names,
        )
        logging.info(
            "[FlatEval] DeltaWFull hit_shapes tag=%s target=%s a=%s b=%s",
            tag,
            [(n, name_to_shape.get(n)) for n in target_hits],
            [(n, name_to_shape.get(n)) for n in a_hits],
            [(n, name_to_shape.get(n)) for n in b_hits],
        )
        if mask.sum().item() == 0:
            continue

        if eig_method == "lanczos":
            eigvals, eigvecs = _lanczos_topk_masked(
                mvp_fn,
                total_dim,
                mask,
                device,
                num_iters,
                topk=topk,
                tol=tol,
                seed=seed,
            )
        else:
            eigvals, eigvecs = _power_iteration_masked(
                mvp_fn,
                total_dim,
                mask,
                device,
                num_iters,
                topk=topk,
                tol=tol,
                patience=patience,
                seed=seed,
            )

        try:
            U, _, Vh = torch.linalg.svd(delta_w, full_matrices=False)
        except Exception:
            continue
        r = int(min(U.shape[1], Vh.shape[0], lora_rank or U.shape[1]))
        if r <= 0:
            continue
        U_r = U[:, :r]
        V_r = Vh[:r, :].T

        ratios: List[float] = []
        cores: List[torch.Tensor] = []
        full_vecs: List[torch.Tensor] = []
        w_ratios: List[float] = []
        w_cores: List[torch.Tensor] = []
        U_w_r = None
        V_w_r = None
        if w_full is not None:
            w_block = _slice_weight_block(w_full, row_ranges, col_ranges)
            logging.info(
                "[FlatEval] DeltaWFull W-block tag=%s w_block=%s delta_w=%s",
                tag,
                None if w_block is None else tuple(w_block.shape),
                tuple(delta_w.shape),
            )
            if w_block is not None and w_block.shape == delta_w.shape:
                try:
                    U_w, _, Vh_w = torch.linalg.svd(w_block, full_matrices=False)
                    r_w = int(min(U_w.shape[1], Vh_w.shape[0], r))
                    if r_w > 0:
                        U_w_r = U_w[:, :r_w]
                        V_w_r = Vh_w[:r_w, :].T
                        logging.info(
                            "[FlatEval] DeltaWFull W-subspace tag=%s r_w=%d U_w_r=%s V_w_r=%s",
                            tag,
                            r_w,
                            tuple(U_w_r.shape),
                            tuple(V_w_r.shape),
                        )
                except Exception:
                    U_w_r = None
                    V_w_r = None
        logged_extract = False
        logged_merge = False
        for v in eigvecs:
            dw0 = _extract_param_block(v, names, shapes, splits, target_name, row_ranges, col_ranges)
            dA = _extract_param_block(v, names, shapes, splits, a_substr, None, None)
            dB = _extract_param_block(v, names, shapes, splits, b_substr, None, None)
            if not logged_extract:
                logging.info(
                    "[FlatEval] DeltaWFull extract tag=%s dw0=%s dA=%s dB=%s match_dw0=%s match_A=%s match_B=%s",
                    tag,
                    None if dw0 is None else tuple(dw0.shape),
                    None if dA is None else tuple(dA.shape),
                    None if dB is None else tuple(dB.shape),
                    bool(dw0 is not None and dw0.shape == delta_w.shape),
                    bool(dA is not None and dA.shape == A.shape),
                    bool(dB is not None and dB.shape == B.shape),
                )
                logged_extract = True
            if dw0 is None or dA is None or dB is None:
                ratios.append(0.0)
                cores.append(torch.zeros((U_r.shape[1], V_r.shape[1]), device=device))
                continue
            if dA.shape != A.shape or dB.shape != B.shape:
                ratios.append(0.0)
                cores.append(torch.zeros((U_r.shape[1], V_r.shape[1]), device=device))
                continue
            if dw0.shape != delta_w.shape:
                ratios.append(0.0)
                cores.append(torch.zeros((U_r.shape[1], V_r.shape[1]), device=device))
                continue

            try:
                dw_full = dw0 + (scale * (dB @ A + B @ dA))
            except Exception:
                ratios.append(0.0)
                cores.append(torch.zeros((U_r.shape[1], V_r.shape[1]), device=device))
                continue
            if not logged_merge:
                logging.info(
                    "[FlatEval] DeltaWFull merged tag=%s dw_full=%s",
                    tag,
                    tuple(dw_full.shape),
                )
                logged_merge = True

            denom = float(dw_full.norm().item() ** 2)
            if denom <= eps:
                ratios.append(0.0)
                cores.append(torch.zeros((U_r.shape[1], V_r.shape[1]), device=device))
                continue
            tmp = U_r.T @ dw_full @ V_r
            proj = U_r @ tmp @ V_r.T
            num = float(proj.norm().item() ** 2)
            ratios.append(float(num / (denom + eps)))
            cores.append(tmp)
            full_vecs.append(dw_full)
            if U_w_r is not None and V_w_r is not None:
                tmp_w = U_w_r.T @ dw_full @ V_w_r
                proj_w = U_w_r @ tmp_w @ V_w_r.T
                num_w = float(proj_w.norm().item() ** 2)
                w_ratios.append(float(num_w / (denom + eps)))
                w_cores.append(tmp_w)

        metrics[f"delta_w_full_proj_{tag}_backend"] = backend
        metrics[f"delta_w_full_proj_{tag}_eig_method"] = eig_method
        metrics[f"delta_w_full_proj_{tag}_space"] = "delta_w_bilinear_subspace"
        metrics[f"delta_w_full_proj_{tag}_note"] = "Projection of dW_full onto U_r (.) V_r^T from delta_w SVD."
        metrics[f"delta_w_full_proj_{tag}_eigvals"] = [float(x) for x in eigvals]
        metrics[f"delta_w_full_proj_{tag}_ratios"] = [float(x) for x in ratios]
        metrics[f"delta_w_full_proj_{tag}_proj_ratio_LoRA"] = [float(x) for x in ratios]
        metrics[f"delta_w_full_proj_{tag}_proj_ratio_LoRA_coord"] = [float(x) for x in ratios]
        if ratios:
            metrics[f"delta_w_full_proj_{tag}_ratio_top1"] = float(ratios[0])
            metrics[f"delta_w_full_proj_{tag}_proj_ratio_LoRA_top1"] = float(ratios[0])
            metrics[f"delta_w_full_proj_{tag}_proj_ratio_LoRA_coord_top1"] = float(ratios[0])
        if eigvals and ratios and len(eigvals) == len(ratios):
            num = sum(float(l) * float(r) for l, r in zip(eigvals, ratios))
            den = sum(float(l) for l in eigvals)
            metrics[f"delta_w_full_proj_{tag}_ratio_weighted"] = float(num / (den + eps))
            metrics[f"delta_w_full_proj_{tag}_proj_ratio_LoRA_weighted"] = float(num / (den + eps))
            metrics[f"delta_w_full_proj_{tag}_proj_ratio_LoRA_coord_weighted"] = float(num / (den + eps))
        if w_ratios:
            metrics[f"delta_w_full_proj_w_{tag}_space"] = "w_bilinear_subspace"
            metrics[f"delta_w_full_proj_w_{tag}_note"] = "Projection of dW_full onto W SVD subspace."
            metrics[f"delta_w_full_proj_w_{tag}_ratios"] = [float(x) for x in w_ratios]
            metrics[f"delta_w_full_proj_{tag}_proj_ratio_W"] = [float(x) for x in w_ratios]
            metrics[f"delta_w_full_proj_w_{tag}_ratio_top1"] = float(w_ratios[0])
            metrics[f"delta_w_full_proj_{tag}_proj_ratio_W_top1"] = float(w_ratios[0])
            if eigvals and len(eigvals) == len(w_ratios):
                num = sum(float(l) * float(r) for l, r in zip(eigvals, w_ratios))
                den = sum(float(l) for l in eigvals)
                metrics[f"delta_w_full_proj_w_{tag}_ratio_weighted"] = float(num / (den + eps))
                metrics[f"delta_w_full_proj_{tag}_proj_ratio_W_weighted"] = float(num / (den + eps))

        if cores:
            try:
                G = torch.stack([c.reshape(-1) for c in cores], dim=1)
                svals = torch.linalg.svdvals(G)
                p = int(min(G.shape[0], G.shape[1]))
                cos2 = (svals[:p] ** 2).detach().cpu().tolist()
                k_p = float(sum(cos2))
                metrics[f"delta_w_full_proj_{tag}_grassmann_cos2"] = [float(x) for x in cos2]
                metrics[f"delta_w_full_proj_{tag}_grassmann_kp"] = k_p
                metrics[f"delta_w_full_proj_{tag}_grassmann_ap"] = float(k_p / max(1, p))
                metrics[f"delta_w_full_proj_{tag}_grassmann_dp"] = float(max(0.0, p - k_p) ** 0.5)
            except Exception:
                pass

        rand_ratios: List[float] = []
        rand_cores: List[torch.Tensor] = []
        if r > 0:
            seed_offset = sum(ord(c) for c in tag)
            seed_rand = None if seed is None else int(seed) + seed_offset
            U_rand, V_rand = _random_svd_subspaces(
                U_r.shape[0],
                V_r.shape[0],
                r,
                seed=seed_rand,
                device=delta_w.device,
                dtype=delta_w.dtype,
            )
            for v in eigvecs:
                dw0 = _extract_param_block(v, names, shapes, splits, target_name, row_ranges, col_ranges)
                dA = _extract_param_block(v, names, shapes, splits, a_substr, None, None)
                dB = _extract_param_block(v, names, shapes, splits, b_substr, None, None)
                if dw0 is None or dA is None or dB is None:
                    rand_ratios.append(0.0)
                    rand_cores.append(torch.zeros((r, r), device=device))
                    continue
                if dA.shape != A.shape or dB.shape != B.shape or dw0.shape != delta_w.shape:
                    rand_ratios.append(0.0)
                    rand_cores.append(torch.zeros((r, r), device=device))
                    continue
                try:
                    dw_full = dw0 + (scale * (dB @ A + B @ dA))
                except Exception:
                    rand_ratios.append(0.0)
                    rand_cores.append(torch.zeros((r, r), device=device))
                    continue
                denom = float(dw_full.norm().item() ** 2)
                if denom <= eps:
                    rand_ratios.append(0.0)
                    rand_cores.append(torch.zeros((r, r), device=device))
                    continue
                tmp = U_rand.T @ dw_full @ V_rand
                proj = U_rand @ tmp @ V_rand.T
                num = float(proj.norm().item() ** 2)
                rand_ratios.append(float(num / (denom + eps)))
                rand_cores.append(tmp)
            metrics[f"delta_w_full_proj_{tag}_rand_ratios"] = [float(x) for x in rand_ratios]
            if rand_ratios:
                metrics[f"delta_w_full_proj_{tag}_rand_ratio_top1"] = float(rand_ratios[0])
            if eigvals and rand_ratios and len(eigvals) == len(rand_ratios):
                num = sum(float(l) * float(r) for l, r in zip(eigvals, rand_ratios))
                den = sum(float(l) for l in eigvals)
                metrics[f"delta_w_full_proj_{tag}_rand_ratio_weighted"] = float(num / (den + eps))
                if f"delta_w_full_proj_{tag}_proj_ratio_LoRA_weighted" in metrics:
                    metrics[f"delta_w_full_proj_{tag}_gain_LoRA"] = float(
                        metrics[f"delta_w_full_proj_{tag}_proj_ratio_LoRA_weighted"] / (metrics[f"delta_w_full_proj_{tag}_rand_ratio_weighted"] + eps)
                    )
                if f"delta_w_full_proj_{tag}_proj_ratio_W_weighted" in metrics:
                    metrics[f"delta_w_full_proj_{tag}_gain_W"] = float(
                        metrics[f"delta_w_full_proj_{tag}_proj_ratio_W_weighted"] / (metrics[f"delta_w_full_proj_{tag}_rand_ratio_weighted"] + eps)
                    )
                if f"delta_w_full_proj_{tag}_proj_ratio_W_weighted" in metrics and f"delta_w_full_proj_{tag}_proj_ratio_LoRA_weighted" in metrics:
                    metrics[f"delta_w_full_proj_{tag}_pref_LoRA_over_W"] = float(
                        metrics[f"delta_w_full_proj_{tag}_proj_ratio_LoRA_weighted"] / (metrics[f"delta_w_full_proj_{tag}_proj_ratio_W_weighted"] + eps)
                    )
            if rand_cores:
                try:
                    G_rand = torch.stack([c.reshape(-1) for c in rand_cores], dim=1)
                    svals = torch.linalg.svdvals(G_rand)
                    p = int(min(G_rand.shape[0], G_rand.shape[1]))
                    cos2 = (svals[:p] ** 2).detach().cpu().tolist()
                    k_p = float(sum(cos2))
                    metrics[f"delta_w_full_proj_{tag}_rand_grassmann_cos2"] = [float(x) for x in cos2]
                    metrics[f"delta_w_full_proj_{tag}_rand_grassmann_kp"] = k_p
                    metrics[f"delta_w_full_proj_{tag}_rand_grassmann_ap"] = float(k_p / max(1, p))
                    metrics[f"delta_w_full_proj_{tag}_rand_grassmann_dp"] = float(max(0.0, p - k_p) ** 0.5)
                except Exception:
                    pass

        if bool(save_tensors) and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{save_prefix}_delta_w_full_proj_{tag}.pt")
            payload = {
                "param_name": target_name,
                "block_tag": tag,
                "row_ranges": row_ranges,
                "col_ranges": col_ranges,
                "a_substr": a_substr,
                "b_substr": b_substr,
                "backend": backend,
                "eigvals": [float(x) for x in eigvals],
                "ratios": [float(x) for x in ratios],
                "delta_w": delta_w.detach().cpu(),
                "U_r": U_r.detach().cpu(),
                "V_r": V_r.detach().cpu(),
                "eigvecs": [v.detach().cpu() for v in eigvecs],
                "core_mats": [c.detach().cpu() for c in cores],
                "delta_w_full": [v.detach().cpu() for v in full_vecs],
            }
            if w_ratios and U_w_r is not None and V_w_r is not None:
                payload["w_proj_ratios"] = [float(x) for x in w_ratios]
                payload["U_w_r"] = U_w_r.detach().cpu()
                payload["V_w_r"] = V_w_r.detach().cpu()
                if w_cores:
                    payload["w_core_mats"] = [c.detach().cpu() for c in w_cores]
            torch.save(payload, out_path)
            metrics[f"delta_w_full_proj_{tag}_path"] = out_path

    return metrics


def _w_delta_alignment_eval(
    wrapped_model: torch.nn.Module,
    config: Any,
    *,
    save_dir: Optional[str],
    save_prefix: str,
) -> Dict[str, Any]:
    if not bool(getattr(config, "w_delta_alignment", False)):
        return {}
    target_name = getattr(config, "w_delta_param_name", None) or getattr(config, "delta_w_param_name", None)
    if not target_name:
        return {}
    blocks = getattr(config, "w_delta_blocks", None) or getattr(config, "delta_w_blocks", None) or []
    if not blocks:
        return {}

    block_index = getattr(config, "w_delta_block_index", None)
    if block_index is None:
        block_index = getattr(config, "delta_w_block_index", None)
    qkv = _get_qkv_module(wrapped_model, block_index)
    if qkv is None:
        return {}

    _, weight_param = _find_param_by_substring(wrapped_model, target_name, block_index=block_index)
    if weight_param is None:
        return {}
    w_full = weight_param.detach()
    if w_full.ndim != 2:
        return {}

    rank = getattr(config, "w_delta_rank", None)
    if rank is None:
        rank = getattr(config, "delta_w_rank", None)
    if rank is None:
        try:
            rank = int(getattr(config, "args", {}).get("lora_rank", 0))
        except Exception:
            rank = 0
    rank = int(rank) if rank else None

    seed = getattr(config, "w_delta_seed", 42)
    eps = float(getattr(config, "w_delta_eps", 1e-12))

    device = w_full.device
    saved_A = getattr(qkv, "saved_A", {})
    saved_B = getattr(qkv, "saved_B", {})
    mode = _infer_w_delta_mode(config, saved_A)
    scale_cur = _get_scale_value_at(getattr(qkv, "scaling_factor", None), 0)
    scale_prev_mod = getattr(qkv, "scaling_factor_prev", None)

    delta_pre: Optional[Dict[str, torch.Tensor]] = None
    delta_post: Optional[Dict[str, torch.Tensor]] = None
    delta_cur = _current_lora_qv(qkv, block_index, device=device, scale=scale_cur, eps=eps)

    if mode == "cumulative":
        delta_pre = _sum_saved_lora_qv(
            saved_A,
            saved_B,
            block_index,
            device=device,
            scale_module_prev=scale_prev_mod,
            normalize_prev=True,
            eps=eps,
        )
        if delta_pre is None:
            delta_post = delta_cur
        elif delta_cur is None:
            delta_post = delta_pre
        else:
            delta_post = {
                "q": delta_pre["q"] + delta_cur["q"],
                "v": delta_pre["v"] + delta_cur["v"],
            }
    else:
        delta_post = delta_cur
        task_id = _infer_task_id_from_args(config)
        if task_id is None:
            task_id = _infer_task_id_from_prefix(save_prefix)
        prev_task_id = None if task_id is None else int(task_id) - 1
        lora_save_dir = _infer_lora_save_dir(config, wrapped_model)
        prev_lists = _load_lora_task_lists(lora_save_dir, prev_task_id)
        if prev_lists is not None:
            delta_pre = _extract_qv_weights_from_lists(
                prev_lists[0],
                prev_lists[1],
                block_index,
                device=device,
                scale=scale_cur,
                normalize=False,
                eps=eps,
            )

    metrics: Dict[str, Any] = {}
    for block in blocks:
        tag = str(block.get("tag", "")).strip()
        row_ranges = block.get("row_ranges", None)
        col_ranges = block.get("col_ranges", None)
        if not tag:
            continue
        dw = delta_post.get(tag) if delta_post is not None else None
        if dw is None:
            continue
        dw_prev = delta_pre.get(tag) if delta_pre is not None else None

        w_block = _slice_weight_block(w_full, row_ranges, col_ranges)
        if w_block is None:
            continue
        if w_block.shape != dw.shape:
            continue

        try:
            U_d, S_d, Vh_d = torch.linalg.svd(dw, full_matrices=False)
        except Exception:
            continue
        try:
            U_w, S_w, Vh_w = torch.linalg.svd(w_block, full_matrices=False)
        except Exception:
            continue

        r = int(min(U_d.shape[1], Vh_d.shape[0], U_w.shape[1], Vh_w.shape[0]))
        if rank is not None:
            r = int(min(r, rank))
        if r <= 0:
            continue

        U_r = U_d[:, :r]
        V_r = Vh_d[:r, :].T
        U_w_r = U_w[:, :r]
        V_w_r = Vh_w[:r, :].T

        if dw_prev is not None and dw_prev.shape == dw.shape:
            try:
                U_prev, _, _ = torch.linalg.svd(dw_prev, full_matrices=False)
            except Exception:
                U_prev = None
            if U_prev is not None:
                m = int(min(r, U_prev.shape[1]))
                if m > 0:
                    U_prev_m = U_prev[:, :m]
                    U_cur_m = U_r[:, :m]
                    overlap = U_prev_m.T @ U_cur_m
                    svals = torch.linalg.svdvals(overlap)
                    svals = torch.clamp(svals, 0.0, 1.0)
                    dp = float(torch.sqrt(torch.tensor(float(m), device=svals.device) - (svals ** 2).sum()).item())
                    metrics[f"w_delta_{tag}_proj_metric"] = dp
                    metrics[f"w_delta_{tag}_proj_metric_m"] = m

        w_norm = float(w_block.norm().item())
        delta_norm = float(dw.norm().item())
        w_post = w_block + dw
        w_post_norm = float(w_post.norm().item())

        proj = U_r.T @ w_block @ V_r
        proj_norm = float(proj.norm().item())
        proj_ratio = float(proj_norm / (w_norm + eps))

        w_self_norm = float(S_w[:r].norm().item())
        w_self_ratio = float(w_self_norm / (w_norm + eps))

        seed_offset = sum(ord(c) for c in tag)
        seed_rand = (int(seed) + seed_offset) if seed is not None else None
        U_rand, V_rand = _random_svd_subspaces(
            w_block.shape[0],
            w_block.shape[1],
            r,
            seed=seed_rand,
            device=w_block.device,
            dtype=w_block.dtype,
        )
        rand_proj = U_rand.T @ w_block @ V_rand
        rand_norm = float(rand_proj.norm().item())
        rand_ratio = float(rand_norm / (w_norm + eps))

        amp = float(delta_norm / (proj_norm + eps))

        phi_u = float((U_w_r.T @ U_r).pow(2).sum().item() / max(1, r))
        phi_v = float((V_w_r.T @ V_r).pow(2).sum().item() / max(1, r))

        post_proj_w = U_w_r.T @ w_post @ V_w_r
        post_in_w_norm = float(post_proj_w.norm().item())
        post_in_w_ratio = float(post_in_w_norm / (w_post_norm + eps))

        metrics[f"w_delta_{tag}_rank"] = r
        metrics[f"w_delta_{tag}_w_norm"] = w_norm
        metrics[f"w_delta_{tag}_delta_norm"] = delta_norm
        metrics[f"w_delta_{tag}_proj_norm"] = proj_norm
        metrics[f"w_delta_{tag}_proj_ratio"] = proj_ratio
        metrics[f"w_delta_{tag}_self_norm"] = w_self_norm
        metrics[f"w_delta_{tag}_self_ratio"] = w_self_ratio
        metrics[f"w_delta_{tag}_rand_norm"] = rand_norm
        metrics[f"w_delta_{tag}_rand_ratio"] = rand_ratio
        metrics[f"w_delta_{tag}_amp"] = amp
        metrics[f"w_delta_{tag}_phi_u"] = phi_u
        metrics[f"w_delta_{tag}_phi_v"] = phi_v
        metrics[f"w_delta_{tag}_post_norm"] = w_post_norm
        metrics[f"w_delta_{tag}_post_in_w_norm"] = post_in_w_norm
        metrics[f"w_delta_{tag}_post_in_w_ratio"] = post_in_w_ratio

        if bool(getattr(config, "w_delta_save_tensors", True)) and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{save_prefix}_w_delta_{tag}.pt")
            payload = {
                "param_name": target_name,
                "block_tag": tag,
                "row_ranges": row_ranges,
                "col_ranges": col_ranges,
                "rank": r,
                "w_norm": w_norm,
                "delta_norm": delta_norm,
                "proj_norm": proj_norm,
                "proj_ratio": proj_ratio,
                "self_norm": w_self_norm,
                "self_ratio": w_self_ratio,
                "rand_norm": rand_norm,
                "rand_ratio": rand_ratio,
                "amp": amp,
                "phi_u": phi_u,
                "phi_v": phi_v,
                "post_norm": w_post_norm,
                "post_in_w_norm": post_in_w_norm,
                "post_in_w_ratio": post_in_w_ratio,
                "W": w_block.detach().cpu(),
                "W_post": w_post.detach().cpu(),
                "delta_w": dw.detach().cpu(),
                "U_r": U_r.detach().cpu(),
                "V_r": V_r.detach().cpu(),
                "U_w_r": U_w_r.detach().cpu(),
                "V_w_r": V_w_r.detach().cpu(),
                "U_rand": U_rand.detach().cpu(),
                "V_rand": V_rand.detach().cpu(),
            }
            torch.save(payload, out_path)
            metrics[f"w_delta_{tag}_path"] = out_path

    return metrics


def _mean_drift_eval(
    wrapped_model: torch.nn.Module,
    config: Any,
    *,
    save_dir: Optional[str],
    save_prefix: str,
) -> Dict[str, Any]:
    if not bool(getattr(config, "mean_drift", False)):
        return {}
    blocks = (
        getattr(config, "mean_drift_blocks", None)
        or getattr(config, "w_delta_blocks", None)
        or getattr(config, "delta_w_blocks", None)
        or []
    )
    if not blocks:
        return {}

    block_index = getattr(config, "mean_drift_block_index", None)
    if block_index is None:
        block_index = getattr(config, "w_delta_block_index", None)
    if block_index is None:
        block_index = getattr(config, "delta_w_block_index", None)
    qkv = _get_qkv_module(wrapped_model, block_index)
    if qkv is None:
        return {}

    device = None
    try:
        device = next(qkv.parameters()).device
    except Exception:
        device = None
    if device is None:
        try:
            device = next(wrapped_model.parameters()).device
        except Exception:
            device = torch.device("cpu")

    eps = float(getattr(config, "mean_drift_eps", 1e-12))
    sigma2 = float(getattr(config, "mean_drift_sigma2", 1.0))
    if sigma2 <= 0.0:
        sigma2 = 1.0

    saved_A = getattr(qkv, "saved_A", {})
    saved_B = getattr(qkv, "saved_B", {})
    mode_override = str(getattr(config, "mean_drift_effective_mode", "") or "").lower()
    if mode_override in {"cumulative", "cum", "sum", "merge", "merged", "inc"}:
        mode = "cumulative"
    elif mode_override in {"sequential", "seq"}:
        mode = "sequential"
    else:
        mode = _infer_w_delta_mode(config, saved_A)

    scale_cur = _get_scale_value_at(getattr(qkv, "scaling_factor", None), 0)
    scale_prev_mod = getattr(qkv, "scaling_factor_prev", None)
    delta_cur = _current_lora_qv(qkv, block_index, device=device, scale=scale_cur, eps=eps)
    if delta_cur is None:
        return {}

    delta_pre: Optional[Dict[str, torch.Tensor]] = None
    delta_post: Optional[Dict[str, torch.Tensor]] = None
    if mode == "cumulative":
        delta_pre = _sum_saved_lora_qv(
            saved_A,
            saved_B,
            block_index,
            device=device,
            scale_module_prev=scale_prev_mod,
            normalize_prev=True,
            eps=eps,
        )
        if delta_pre is None:
            delta_post = delta_cur
        elif delta_cur is None:
            delta_post = delta_pre
        else:
            delta_post = {
                "q": delta_pre["q"] + delta_cur["q"],
                "v": delta_pre["v"] + delta_cur["v"],
            }
    else:
        delta_post = delta_cur
        task_id = _infer_task_id_from_args(config)
        if task_id is None:
            task_id = _infer_task_id_from_prefix(save_prefix)
        prev_task_id = None if task_id is None else int(task_id) - 1
        lora_save_dir = _infer_lora_save_dir(config, wrapped_model)
        prev_lists = _load_lora_task_lists(lora_save_dir, prev_task_id)
        if prev_lists is not None:
            delta_pre = _extract_qv_weights_from_lists(
                prev_lists[0],
                prev_lists[1],
                block_index,
                device=device,
                scale=scale_cur,
                normalize=False,
                eps=eps,
            )

    metrics: Dict[str, Any] = {}
    for block in blocks:
        tag = str(block.get("tag", "")).strip()
        row_ranges = block.get("row_ranges", None)
        col_ranges = block.get("col_ranges", None)
        if not tag:
            continue
        dw_post = delta_post.get(tag) if delta_post is not None else None
        if dw_post is None:
            continue
        dw_post = _slice_delta_block(dw_post, tag, row_ranges, col_ranges)
        if dw_post is None:
            continue

        norm_sq = float((dw_post * dw_post).sum().item())
        metrics[f"mean_drift_{tag}_delta_norm_sq"] = norm_sq
        metrics[f"mean_drift_{tag}_delta_norm"] = float(norm_sq ** 0.5)
        metrics[f"mean_drift_{tag}_mean_abs"] = float(0.5 * norm_sq / sigma2)
        metrics[f"mean_drift_{tag}_sigma2"] = float(sigma2)
        metrics[f"mean_drift_{tag}_mode"] = str(mode)

        dw_pre = delta_pre.get(tag) if delta_pre is not None else None
        if dw_pre is None:
            continue
        dw_pre = _slice_delta_block(dw_pre, tag, row_ranges, col_ranges)
        if dw_pre is None or dw_pre.shape != dw_post.shape:
            continue
        diff = dw_post - dw_pre
        diff_sq = float((diff * diff).sum().item())
        metrics[f"mean_drift_{tag}_delta_diff_norm_sq"] = diff_sq
        metrics[f"mean_drift_{tag}_delta_diff_norm"] = float(diff_sq ** 0.5)
        metrics[f"mean_drift_{tag}_mean_inc"] = float(0.5 * diff_sq / sigma2)
        metrics[f"mean_drift_{tag}_mean_inc_ratio"] = float(diff_sq / (norm_sq + eps))

    return metrics


def _lora_kl_eval(
    wrapped_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Any,
    *,
    save_prefix: str,
    known_classes: Optional[int] = None,
) -> Dict[str, Any]:
    if not bool(getattr(config, "lora_kl", False)):
        return {}
    if loader is None:
        return {}

    blocks = (
        getattr(config, "lora_kl_blocks", None)
        or getattr(config, "w_delta_blocks", None)
        or getattr(config, "delta_w_blocks", None)
        or []
    )
    if not blocks:
        return {}

    block_index = getattr(config, "lora_kl_block_index", None)
    if block_index is None:
        block_index = getattr(config, "w_delta_block_index", None)
    if block_index is None:
        block_index = getattr(config, "delta_w_block_index", None)
    qkv = _get_qkv_module(wrapped_model, block_index)
    if qkv is None:
        return {}

    saved_A = getattr(qkv, "saved_A", {})
    mode = _infer_w_delta_mode(config, saved_A)
    if mode != "sequential":
        return {"lora_kl_mode": str(mode), "lora_kl_skipped": "unsupported_mode"}

    task_id = _infer_task_id_from_args(config)
    if task_id is None:
        task_id = _infer_task_id_from_prefix(save_prefix)
    prev_task_id = None if task_id is None else int(task_id) - 1
    lora_save_dir = _infer_lora_save_dir(config, wrapped_model)
    prev_lists = _load_lora_task_lists(lora_save_dir, prev_task_id)
    prev_factors = None
    if prev_lists is not None:
        prev_factors = _extract_qv_factors_from_lists(
            prev_lists[0],
            prev_lists[1],
            block_index,
            device=device,
        )

    max_batches = getattr(config, "lora_kl_max_batches", None)
    if max_batches is None:
        max_batches = getattr(config, "loss_eval_max_batches", None)
    micro_bs = int(getattr(config, "lora_kl_micro_bs", 1))
    damping = float(getattr(config, "lora_kl_damping", 1e-4))
    lam = float(getattr(config, "lora_kl_lambda", 1.0))
    eps = float(getattr(config, "lora_kl_eps", 1e-12))

    metrics: Dict[str, Any] = {}
    for block in blocks:
        tag = str(block.get("tag", "")).strip()
        if tag not in {"q", "v"}:
            continue
        if prev_factors is None or tag not in prev_factors:
            continue

        params = _get_qv_param_tensors(qkv, block_index, tag)
        if not params:
            continue
        saved_data = _clone_param_data(params)
        saved_requires = [bool(p.requires_grad) for p in params]
        for p in params:
            if not p.requires_grad:
                p.requires_grad_(True)

        try:
            A_prev = prev_factors[tag]["A"]
            B_prev = prev_factors[tag]["B"]
            if A_prev.shape != params[0].shape or B_prev.shape != params[1].shape:
                continue

            mu_post = torch.cat([p.detach().reshape(-1) for p in params])
            fisher_post = _estimate_diag_fisher(
                wrapped_model,
                loader,
                device,
                params,
                max_batches=max_batches,
                known_classes=known_classes,
                micro_bs=micro_bs,
            )
            if fisher_post is None:
                continue

            metrics[f"lora_kl_{tag}_dim"] = int(mu_post.numel())
            metrics[f"lora_kl_{tag}_lambda"] = float(lam)
            metrics[f"lora_kl_{tag}_damping"] = float(damping)

            params[0].data.copy_(A_prev)
            params[1].data.copy_(B_prev)

            mu_prev = torch.cat([p.detach().reshape(-1) for p in params])
            fisher_prev = _estimate_diag_fisher(
                wrapped_model,
                loader,
                device,
                params,
                max_batches=max_batches,
                known_classes=known_classes,
                micro_bs=micro_bs,
            )
            if fisher_prev is None:
                continue

            sigma2_post = lam / (fisher_post + damping)
            sigma2_prev = lam / (fisher_prev + damping)
            kl = _kl_diag_gaussians(mu_post, sigma2_post, mu_prev, sigma2_prev, eps=eps)
            metrics[f"lora_kl_{tag}_kl"] = float(kl)
            metrics[f"lora_kl_{tag}_kl_per_dim"] = float(kl / max(1, mu_post.numel()))
        finally:
            _restore_param_data(params, saved_data)
            for p, req in zip(params, saved_requires):
                if p.requires_grad != req:
                    p.requires_grad_(req)

    return metrics
