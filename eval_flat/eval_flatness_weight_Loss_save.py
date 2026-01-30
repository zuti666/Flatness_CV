import argparse
import json
import os
import logging
import math
from dataclasses import dataclass, field, fields as dc_fields, MISSING
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from contextlib import contextmanager, nullcontext
import copy
import h5py
from tqdm import tqdm

from evaluation.probe import TunaEvalWrapper


def _extract_logits(model: nn.Module, inputs: torch.Tensor):
    """Run model forward and return logits tensor.

    Accepts either dict outputs with key 'logits' or direct logits tensor.
    """
    outputs = model(inputs)
    if isinstance(outputs, dict):
        return outputs.get("logits", outputs)
    return outputs

def _forward_logits_full(model: nn.Module, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Best-effort to obtain logits over all seen classes.

    - First try regular forward.
    - If logits cols < max(targets)+1 (mismatch), and the model exposes an
      'interface' method (SiNet), call it to get concatenated logits over all
      heads.
    """
    logits = _extract_logits(model, inputs)
    try:
        need_classes = (int(targets.max().item()) + 1) if (targets is not None) else None
    except Exception:
        need_classes = None
    if need_classes is not None and logits.size(-1) < need_classes:
        if hasattr(model, "interface") and callable(getattr(model, "interface")):
            try:
                logits_full = model.interface(inputs)
                if isinstance(logits_full, torch.Tensor) and logits_full.size(-1) >= need_classes:
                    return logits_full
            except Exception:
                pass
    return logits

def _sdp_disable_context():
    """
    Disable Flash / MemEfficient SDPA kernels during second-order ops.
    Works across PyTorch 2.x variants:
      - New API (torch.nn.attention.sdpa_kernel): choose MATH backend
      - Legacy API (torch.backends.cuda.sdp_kernel): enable_* flags
      - CPU / missing API: no-op context
    """
    import torch

    # CPU 或未启用 CUDA：直接空上下文
    if not torch.cuda.is_available():
        return nullcontext()

    # --- 新 API（2.3+）：torch.nn.attention.sdpa_kernel(SDPBackend.MATH) ---
    try:
        import torch.nn.attention as _attn
        if hasattr(_attn, "sdpa_kernel") and hasattr(_attn, "SDPBackend"):
            # 强制使用 MATH 后端（禁用 Flash / MemEfficient）
            return _attn.sdpa_kernel(_attn.SDPBackend.MATH)
    except Exception:
        pass

    # --- 旧 API（2.0–2.2）：torch.backends.cuda.sdp_kernel(enable_*) ---
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )
    except Exception:
        pass

    # 兜底：什么都不做
    return nullcontext()

# ---------------------------------------------------------------------------
# Helpers for flattening perturbations
# ---------------------------------------------------------------------------



@dataclass
class FlatnessConfig:
    """Configuration knobs for the flatness/sharpness estimators."""

    # ---------------- core knobs ----------------
    model_name: Optional[str] = None
    sharpness_radius: float = 0.05
    esh_num_samples: int = 20
    esh_gaussian_std: Optional[float] = None
    loss_eval_max_batches: Optional[float] = None

    hessian_power_iters: int = 5
    hessian_trace_samples: int = 5

    first_order_grad_batches: Optional[int] = None
    flat_batch_size: Optional[float] = None

    max_examples_per_batch: Optional[int] = 128

    # Optional persistence
    save_metrics_path: Optional[str] = None
    save_prefix: str = "flatness"
    param_name_substrings: Optional[List[str]] = None  # e.g. ["loranew_"]
    include_frozen_params: bool = False

    # ---------------- weight loss landscape ----------------
    weight_loss_land_1d: bool = False
    weight_loss_land_2d: bool = False
    weight_loss_land_radius: float = 0.5
    weight_loss_land_num_points: int = 21
    weight_loss_land_max_batches: Optional[float] = None
    weight_loss_land_filter_norm: bool = True

    loss_land_modes: str = "lora"             # "lora" | "full" | "all"
    loss_land_include_frozen: bool = True
    loss_land_param_names: Optional[List[str]] = None

    # eig / basis / radius controls
    eig_save_vectors: bool = False
    eig_backend: str = "emp_fisher"              # "hessian" | "ggn" | "emp_fisher"
    eig_topk: int = 2
    eig_tol: float = 1e-3
    eig_patience: int = 2
    disable_power: bool = False
    eval_sharpness: bool = True   # compute grad-based sharpness (Sh0/Sh1/E-Sh)
    eval_hessian: bool = False    # compute Hessian spectral metrics
    eval_ggn: bool = False        # compute GGN spectral metrics
    eval_fisher: bool = True      # compute empirical Fisher metrics
    loss_land_basis: str = "random"           # "random" | "eig" | "sam" | "both"
    loss_land_radius_from_rho: bool = False
    loss_land_radius_scale: float = 1.5
    loss_land_seed: Optional[int] = None

    # ---------------- Fisher–Rao & Relative Flatness ----------------
    fisher_rao: bool = True
    relative_flatness: bool = True
    rf_scope: str = "custom"                    # "fc" | "lora" | "custom"
    rf_norm_mode: str = "fro"                 # "fro" | "spectral"
    rf_param_name_substrings: Optional[List[str]] = None
    rf_include_bias: bool = False
    rf_power_iters: Optional[int] = None
    rf_trace_samples: Optional[int] = None
    rf_max_batches: Optional[int] = None

    # ---------------- Curvature localization (LoRA vs. complement) ----------------
    curv_localization: bool = False
    curv_topk: int = 6
    curv_param_name_substrings: Optional[List[str]] = None
    curv_qkv_blocks: Optional[List[Dict[str, Any]]] = None
    curv_rayleigh_samples: int = 32
    curv_basis_seed: Optional[int] = None
    curv_mvp: str = "emp_fisher"  # "hessian" | "ggn" | "emp_fisher"
    curv_eig_method: str = "power"  # "power" | "lanczos"
    ratio_as_percent: bool = False

    # ---------------- Delta-W projection (weight-space LoRA subspace) ----------------
    delta_w_projection: bool = False
    delta_w_param_name: Optional[str] = None
    delta_w_block_index: Optional[int] = None
    delta_w_blocks: Optional[List[Dict[str, Any]]] = None
    delta_w_topk: int = 6
    delta_w_rank: Optional[int] = None
    delta_w_backend: Optional[str] = None
    delta_w_eig_method: str = "lanczos"  # "power" | "lanczos"
    delta_w_save_tensors: bool = True
    delta_w_svd_eps: float = 1e-12

    # ---------------- W vs Delta-W alignment (LoRA paper 7.3-style) ----------------
    w_delta_alignment: bool = False
    w_delta_param_name: Optional[str] = None
    w_delta_block_index: Optional[int] = None
    w_delta_blocks: Optional[List[Dict[str, Any]]] = None
    w_delta_rank: Optional[int] = None
    w_delta_seed: int = 42
    w_delta_save_tensors: bool = True
    w_delta_eps: float = 1e-12

    # ---------------- NEW: accept args mapping ----------------
    args: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __post_init__(self):
        """Populate fields from self.args using a key mapping, but do not
        override values explicitly provided via __init__.
        """
        if not isinstance(self.args, dict):
            return

        # Compute dataclass field defaults (to know what is "still default")
        defaults = {}
        for f in dc_fields(self):
            if f.name == "args":
                continue
            if f.default is not MISSING:
                defaults[f.name] = f.default
            else:
                defaults[f.name] = None

        def _get_seed(a):
            s = a.get("seed", None)
            if isinstance(s, list) and len(s) > 0:
                try:
                    return int(s[0])
                except Exception:
                    return s[0]
            return s

        # Mapping from args keys -> FlatnessConfig field names
        key_map = {
            # core
            "model_name": "model_name",
            "flat_eval_sharpness_radius": "sharpness_radius",
            "flat_eval_esh_samples": "esh_num_samples",
            "flat_eval_esh_gaussian_std": "esh_gaussian_std",
            "flat_eval_loss_max_batches": "loss_eval_max_batches",

            "flat_eval_hessian_power_iters": "hessian_power_iters",
            "flat_eval_hessian_trace_samples": "hessian_trace_samples",

            "flat_eval_first_order_grad_batches": "first_order_grad_batches",
            "flat_eval_batch_size": "flat_batch_size",
            "flat_eval_max_examples_per_batch": "max_examples_per_batch",
            "flat_eval_include_frozen": "include_frozen_params",

            # weight loss landscape
            "weight_loss_land_1d": "weight_loss_land_1d",
            "weight_loss_land_2d": "weight_loss_land_2d",
            "weight_loss_land_radius": "weight_loss_land_radius",
            "weight_loss_land_num_points": "weight_loss_land_num_points",
            "weight_loss_land_max_batches": "weight_loss_land_max_batches",
            "weight_loss_land_filter_norm": "weight_loss_land_filter_norm",
            "loss_land_modes": "loss_land_modes",
            "loss_land_include_frozen": "loss_land_include_frozen",
            "loss_land_param_names": "loss_land_param_names",

            "eig_save_vectors": "eig_save_vectors",
            "eig_backend": "eig_backend",
            "eig_topk": "eig_topk",
            "eig_tol": "eig_tol",
            "eig_patience": "eig_patience",
            "flat_eval_disable_power": "disable_power",
            "flat_eval_sharpness": "eval_sharpness",
            "flat_eval_hessian": "eval_hessian",
            "flat_eval_GGN": "eval_ggn",
            "flat_eval_ggn": "eval_ggn",
            "flat_eval_fisher": "eval_fisher",
            "loss_land_basis": "loss_land_basis",
            "loss_land_radius_from_rho": "loss_land_radius_from_rho",
            "loss_land_radius_scale": "loss_land_radius_scale",

            # Fisher–Rao / Relative Flatness
            "fisher_rao": "fisher_rao",
            "relative_flatness": "relative_flatness",
            "rf_scope": "rf_scope",
            "rf_norm_mode": "rf_norm_mode",
            "rf_param_name_substrings": "rf_param_name_substrings",
            "rf_include_bias": "rf_include_bias",
            "rf_power_iters": "rf_power_iters",
            "rf_trace_samples": "rf_trace_samples",
            "rf_max_batches": "rf_max_batches",

            # curvature localization (LoRA subspace)
            "flat_eval_curv_localization": "curv_localization",
            "flat_eval_curv_topk": "curv_topk",
            "flat_eval_curv_param_names": "curv_param_name_substrings",
            "flat_eval_curv_blocks": "curv_qkv_blocks",
            "flat_eval_curv_qkv_blocks": "curv_qkv_blocks",
            "flat_eval_curv_rayleigh_samples": "curv_rayleigh_samples",
            "flat_eval_curv_basis_seed": "curv_basis_seed",
            "flat_eval_curv_mvp": "curv_mvp",
            "flat_eval_curv_eig_method": "curv_eig_method",
            "flat_eval_ratio_as_percent": "ratio_as_percent",

            # delta-W projection
            "flat_eval_delta_w_projection": "delta_w_projection",
            "flat_eval_delta_w_param_name": "delta_w_param_name",
            "flat_eval_delta_w_block_index": "delta_w_block_index",
            "flat_eval_delta_w_blocks": "delta_w_blocks",
            "flat_eval_delta_w_topk": "delta_w_topk",
            "flat_eval_delta_w_rank": "delta_w_rank",
            "flat_eval_delta_w_backend": "delta_w_backend",
            "flat_eval_delta_w_eig_method": "delta_w_eig_method",
            "flat_eval_delta_w_save_tensors": "delta_w_save_tensors",
            "flat_eval_delta_w_svd_eps": "delta_w_svd_eps",

            # W vs Delta-W alignment (LoRA paper 7.3-style)
            "flat_eval_w_delta_alignment": "w_delta_alignment",
            "flat_eval_w_delta_param_name": "w_delta_param_name",
            "flat_eval_w_delta_block_index": "w_delta_block_index",
            "flat_eval_w_delta_blocks": "w_delta_blocks",
            "flat_eval_w_delta_rank": "w_delta_rank",
            "flat_eval_w_delta_seed": "w_delta_seed",
            "flat_eval_w_delta_save_tensors": "w_delta_save_tensors",
            "flat_eval_w_delta_eps": "w_delta_eps",

            # backward-compat (optional aliases)
            "flat_eval_param_names": "param_name_substrings",
        }

        # 1) apply key_map when the target field still equals its default
        for src_key, dst in key_map.items():
            if src_key in self.args:
                current = getattr(self, dst)
                if current == defaults.get(dst):
                    setattr(self, dst, self.args[src_key])

        # 2) derive loss_land_seed from args["seed"] when still default
        if self.loss_land_seed == defaults.get("loss_land_seed"):
            s = _get_seed(self.args)
            if s is not None:
                try:
                    self.loss_land_seed = int(s)
                except Exception:
                    self.loss_land_seed = s

        # 3) optional: radius sync with SAM rho if requested and no explicit radius
        if bool(self.loss_land_radius_from_rho) and \
           (self.weight_loss_land_radius == defaults.get("weight_loss_land_radius")):
            rho = self.args.get("sam_rho", None)
            try:
                if rho is not None:
                    self.weight_loss_land_radius = float(rho) * float(self.loss_land_radius_scale)
            except Exception:
                pass



def _get_fc_params(module: nn.Module, include_bias: bool = False) -> List[torch.nn.Parameter]:
    picked: List[torch.nn.Parameter] = []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("fc.weight") or name == "fc.weight":
            picked.append(p)
        elif include_bias and (name.endswith("fc.bias") or name == "fc.bias"):
            picked.append(p)
        elif name.endswith("classifier.weight") or name == "classifier.weight":
            picked.append(p)
        elif include_bias and (name.endswith("classifier.bias") or name == "classifier.bias"):
            picked.append(p)
    return picked


def _weight_norm_for_params(params: List[torch.nn.Parameter], mode: str = "fro") -> Tuple[float, str]:
    mode = (mode or "fro").lower()
    if not params:
        return 0.0, mode
    if mode == "fro":
        s = 0.0
        for p in params:
            t = p.detach().float()
            s += float(torch.sum(t * t).item())
        return float(math.sqrt(max(s, 0.0))), "fro"
    spectral_vals: List[float] = []
    fro2 = 0.0
    for p in params:
        t = p.detach().float()
        if t.ndim == 2 and t.numel() > 0:
            try:
                svals = torch.linalg.svdvals(t)
                spectral_vals.append(float(svals.max().item()))
            except Exception:
                pass
        fro2 += float(torch.sum(t * t).item())
    if spectral_vals:
        return float(max(spectral_vals)), "spectral"
    return float(math.sqrt(max(fro2, 0.0))), "fro"


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


def _param_names_and_shapes(module: nn.Module, params: List[torch.nn.Parameter]):
    """Return names, shapes, and splits for the given params in order.

    names: List[str]
    shapes: List[torch.Size]
    splits: List[int]
    """
    id2info = {}
    for name, p in module.named_parameters():
        id2info[id(p)] = (name, p.shape, p.numel())

    names: List[str] = []
    shapes: List[torch.Size] = []
    splits: List[int] = []
    for p in params:
        key = id(p)
        if key not in id2info:
            # unnamed or detached param; fabricate a placeholder
            names.append("")
            shapes.append(p.shape)
            splits.append(p.numel())
        else:
            n, s, k = id2info[key]
            names.append(n)
            shapes.append(s)
            splits.append(k)
    return names, shapes, splits


def _unflatten_to_param_like(vec: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    pointer = 0
    for p in params:
        k = p.numel()
        out.append(vec[pointer:pointer + k].view_as(p))
        pointer += k
    return out


def _save_eigvecs(save_path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(payload, save_path)


def _load_eigvecs_for_params(path: str, module: nn.Module, params: List[torch.nn.Parameter]):
    """Load flat eigenvectors and align to current params by names/shapes.

    Returns (vals, vecs_flat, dirs_list) on success, else None.
    dirs_list is a list of list[Tensor] shaped like params: [v1_list, v2_list, ...].
    """
    if not os.path.isfile(path):
        return None
    payload = torch.load(path, map_location="cpu")

    vals = payload.get("vals", None)
    vlist = []
    for key in ("v1", "v2", "v3"):
        if key in payload and isinstance(payload[key], torch.Tensor):
            vlist.append(payload[key].float().view(-1))
    names_saved = payload.get("names", None)
    shapes_saved = payload.get("shapes", None)
    splits_saved = payload.get("splits", None)
    if vals is None or not vlist or names_saved is None or shapes_saved is None or splits_saved is None:
        return None

    names_cur, shapes_cur, splits_cur = _param_names_and_shapes(module, params)
    if names_saved != names_cur or splits_saved != splits_cur:
        logging.info("[FlatEval] Saved eigvecs do not match current param ordering; fallback.")
        return None

    dirs = []
    for v in vlist:
        dirs.append(_unflatten_to_param_like(v, params))
    return vals, vlist, dirs


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

            logits = _forward_logits_full(model, inputs, targets)

            num_classes = logits.size(-1)


            if known_classes is not None and known_classes > 0:
                loss = criterion(logits[:, known_classes:], targets - known_classes)
            else:
                # 全量微调或混合标签的情形：直接用全部 logits 计算 CE
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

            logits = _forward_logits_full(model, inputs, targets)
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
    max_batches: Optional[int] = None,
    known_classes: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """Monte-Carlo estimate of ``H v`` using double-backprop."""

    # 向量放到目标设备；non_blocking 有助于流水
    vec = vec.to(device, non_blocking=True)

    # 累加器与计数
    hvp_accumulator = torch.zeros_like(vec, device=device)
    batches_processed = 0
    criterion = nn.CrossEntropyLoss(reduction="mean")

    # ---- 禁用易碎大核 + 禁用 AMP（建议在二阶时统一关闭 AMP）----
    from contextlib import nullcontext
    autocast_ctx = (
        torch.cuda.amp.autocast(enabled=False)
        if (device.type == "cuda" and torch.cuda.is_available())
        else nullcontext()
    )

    with _sdp_disable_context(), autocast_ctx:
        for batch_idx, batch in enumerate(loader):
            # 先默认不 break，等清理完显存再决定是否 break
            should_break = False
            try:
                inputs, targets = _unwrap_batch(batch)
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                model.zero_grad(set_to_none=True)
                logits = _forward_logits_full(model, inputs, targets)

                if known_classes is not None and known_classes > 0:
                    # 增量分类下的“新类”损失（与上游一致）
                    loss = criterion(logits[:, known_classes:], targets - known_classes)
                else:
                    loss = criterion(logits, targets)

                # 一阶梯度（建图，以便二次求导）
                grads = torch.autograd.grad(
                    loss,
                    params,
                    create_graph=True,
                    allow_unused=True,
                )

                # 拉平成向量：grad_vec
                grad_terms = []
                for p, g in zip(params, grads):
                    if g is None:
                        grad_terms.append(torch.zeros_like(p, device=p.device).view(-1))
                    else:
                        grad_terms.append(g.contiguous().view(-1))
                grad_vec = torch.cat(grad_terms)

                # 标量 grad_v = <grad_vec, vec>
                grad_v = torch.dot(grad_vec, vec)

                # 二阶：Hv = ∂(grad_v)/∂params
                hv = torch.autograd.grad(
                    grad_v,
                    params,
                    retain_graph=False,
                    allow_unused=True,
                )

                # 拉平成向量并累计（detach 避免继续挂图）
                hv_terms = []
                for p, h in zip(params, hv):
                    if h is None:
                        hv_terms.append(torch.zeros_like(p, device=p.device).reshape(-1))
                    else:
                        hv_terms.append(h.detach().reshape(-1))
                hv_flat = torch.cat(hv_terms)

                hvp_accumulator += hv_flat
                batches_processed += 1

                if max_batches is not None and batches_processed >= max_batches:
                    should_break = True

            finally:
                # ---- 显存卫生处理：释放大中间量，减少碎片 ----
                # 注：这些变量都在 try 块里定义；若某些分支未定义，忽略即可
                for _name in [
                    "outputs", "logits", "loss", "grads", "grad_terms",
                    "grad_vec", "grad_v", "hv", "hv_terms", "hv_flat",
                    "inputs", "targets"
                ]:
                    if _name in locals():
                        try:
                            del locals()[_name]
                        except Exception:
                            pass

                # 及时把已释放块归还 allocator，缓解碎片化
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()

                # 若达到批次数上限，清理完再跳出
                if should_break:
                    break

    if batches_processed == 0:
        return hvp_accumulator

    return hvp_accumulator / batches_processed



def _ggn_vector_product(
    model, loader, device, params, vec,
    loss_eval_max_batches: Optional[int] = None,
    known_classes: Optional[int] = None,
) -> torch.Tensor:
    vec = vec.to(device)
    dim = vec.numel()
    out_accum = torch.zeros(dim, device=device)
    batches_processed = 0

    # 预切 vec 片段（可留可去，不是瓶颈）
    pointer = 0
    vec_slices = []
    for p in params:
        n = p.numel()
        vec_slices.append(vec[pointer:pointer+n].view_as(p))
        pointer += n

    model.eval()
    with _sdp_disable_context():  # 你已有该上下文
        for batch_idx, batch in enumerate(loader):
            should_break = False
            try:
                inputs, targets = _unwrap_batch(batch)
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # base forward（需要图，用于后面的 VJP）
                outputs0 = model(inputs)
                logits0 = outputs0["logits"] if isinstance(outputs0, dict) else outputs0
                if known_classes is not None and known_classes > 0:
                    logits_use = logits0[:, known_classes:]
                    t_use = targets - known_classes
                else:
                    logits_use = logits0
                    t_use = targets

                # —— 中心差分 JVP：强制 no_grad，避免为 logits_p/m 构图 —— #
                eps = 1e-3
                saved = _clone_params(params)
                with torch.no_grad():
                    _add_vector_to_params(params, eps * vec)
                    logits_p = model(inputs)
                    logits_p = logits_p["logits"] if isinstance(logits_p, dict) else logits_p
                    _restore_params(params, saved)
                    _add_vector_to_params(params, -eps * vec)
                    logits_m = model(inputs)
                    logits_m = logits_m["logits"] if isinstance(logits_m, dict) else logits_m
                    _restore_params(params, saved)

                if known_classes is not None and known_classes > 0:
                    u = (logits_p[:, known_classes:] - logits_m[:, known_classes:]) / (2.0 * eps)
                else:
                    u = (logits_p - logits_m) / (2.0 * eps)
                u = u.detach()   # ★ 保证后续图只依赖 logits_use，而不是 logits_p/m

                # 输出空间 CE Hessian 作用：s = (I - p 1ᵀ) diag(p) u
                with torch.no_grad():
                    p = torch.softmax(logits_use, dim=-1)
                up = u * p
                pu = up.sum(dim=-1, keepdim=True)
                s = up - p * pu

                # VJP: J^T s，对 base logits_use 的图做一次 autograd.grad
                scalar = (logits_use * s).sum()
                grads = torch.autograd.grad(scalar, params, retain_graph=False, allow_unused=True)
                flat = []
                for p_, g in zip(params, grads):
                    flat.append((torch.zeros_like(p_) if g is None else g.detach()).view(-1))
                out_accum += torch.cat(flat)

                batches_processed += 1
                if loss_eval_max_batches is not None and batches_processed >= loss_eval_max_batches:
                    should_break = True

            finally:
                # —— 显存卫生：删除大中间量 + 归还缓存 —— #
                for _name in [
                    "outputs0","logits0","logits_use","t_use","logits_p","logits_m",
                    "u","p","up","pu","s","scalar","grads","flat","inputs","targets","saved"
                ]:
                    if _name in locals():
                        try: del locals()[_name]
                        except Exception: pass
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()
                if should_break:
                    break

    return out_accum if batches_processed == 0 else out_accum / batches_processed


def _empirical_fisher_vector_product(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    vec: torch.Tensor,
    max_batches: Optional[int] = None,
    known_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute EF v ≈ E[(g_i^T v) g_i] using per-sample gradients, with memory hardening:
    - micro-batch per-sample backward to avoid retaining the whole graph B times.
    - stream accumulation: never materialize [B, dim] gradient matrix.
    - try/finally + empty_cache() for better memory hygiene.
    - automatic fallback to batch-gradient approximation on OOM.
    """
    vec = vec.to(device, non_blocking=True)
    dim = vec.numel()
    out_accum = torch.zeros(dim, device=device)
    samples_accum = 0

    criterion_vec = nn.CrossEntropyLoss(reduction="none")
    # 逐样本或小微批：你可用 args 配个开关，例如 ef_microbatch；这里默认 1 更稳
    micro_bs = 1

    model.eval()

    def _batch_grad_fallback(inputs, targets):
        """biased fallback: (g_batch^T v) g_batch"""
        model.zero_grad(set_to_none=True)
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
        loss.backward()

        g_list = []
        for p in params:
            pg = p.grad
            g_list.append((torch.zeros_like(p) if pg is None else pg).view(-1))
        g = torch.cat(g_list)  # [dim]
        s = torch.dot(g, vec)  # scalar
        return s * g, targets.size(0)

    with _sdp_disable_context():
        batches_processed = 0
        for batch_idx, batch in enumerate(loader):
            try:
                inputs, targets = _unwrap_batch(batch)
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                B = targets.shape[0]
                # 逐 micro-batch 处理，避免整批图常驻
                for start in range(0, B, micro_bs):
                    end = min(start + micro_bs, B)
                    x_mb = inputs[start:end]
                    y_mb = targets[start:end]

                    # 先试 per-sample 精确 EF（micro_bs=1 最稳）
                    try:
                        # 单次前向建立图（仅对该 micro-batch）
                        logits = _forward_logits_full(model, x_mb, y_mb)
                        num_classes = logits.size(-1)

                        use_split = isinstance(known_classes, int) and 0 < known_classes < num_classes
                        if use_split:
                            # 只有当该 micro-batch 全是“新类”才切分
                            all_new = (y_mb >= known_classes).all() and (y_mb < num_classes).all()
                        else:
                            all_new = False

                        if use_split and all_new:
                            logits_use = logits[:, known_classes:]
                            targets_use = y_mb - known_classes
                        else:
                            logits_use = logits
                            targets_use = y_mb

                        losses = criterion_vec(logits_use, targets_use)  # [mb]
                        mb = losses.shape[0]

                        for i in range(mb):
                            model.zero_grad(set_to_none=True)
                            # 仅保留当前 micro-batch 内的图；每个 i 之间不保留
                            grads = torch.autograd.grad(
                                losses[i],
                                params,
                                retain_graph=(i < mb - 1),  # 同一 micro 内最后一次不留图
                                create_graph=False,
                                allow_unused=True,
                            )
                            gi = torch.cat([
                                (torch.zeros_like(p) if g is None else g.detach()).view(-1)
                                for p, g in zip(params, grads)
                            ])
                            si = torch.dot(gi, vec)   # scalar
                            out_accum += si * gi
                            samples_accum += 1
                            # 释放临时变量，降低峰值
                            del grads, gi, si

                    except RuntimeError as e:
                        # 显存告急：退化为 batch-grad 近似（对当前 micro-batch）
                        if "out of memory" in str(e).lower():
                            torch.cuda.empty_cache()
                            approx, n = _batch_grad_fallback(x_mb, y_mb)
                            out_accum += approx
                            samples_accum += n
                        else:
                            raise

                batches_processed += 1
                if max_batches is not None and batches_processed >= max_batches:
                    break

            finally:
                # 内存卫生：删除局部变量并归还缓存
                for _name in ["outputs", "logits", "logits_use", "targets_use",
                              "inputs", "targets", "x_mb", "y_mb", "losses"]:
                    if _name in locals():
                        try:
                            del locals()[_name]
                        except Exception:
                            pass
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()

    if samples_accum == 0:
        return out_accum  # 0 向量
    return out_accum / samples_accum

# ---------------------------------------------------------------------------
# Generic spectral/trace helpers for arbitrary matrix-vector products
# ---------------------------------------------------------------------------


def _power_iteration_generic(
    mv: callable,
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
    mv: callable,
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


def _lanczos_lambda_max_generic(
    mv: callable,
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
        # Return the last Lanczos basis vector as an approximation of the top eigenvector
        return lam, v
    return lam


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
    filter_norm: bool = False,
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
        d1 = _build_direction_list(params, device=device, filter_norm=filter_norm, )
        d2 = _build_direction_list(params, device=device, filter_norm=filter_norm, )
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
    """Compute 2D loss-landscape surface using per-parameter random directions.

    This mirrors the user-provided implementation:
      - Select parameters excluding names containing ``param_name_exclude_substr``
      - For each tensor, build two directions ``d_x, d_y``
        (normalize+scale by ``||p||``; ``d_y`` orthogonalized to ``d_x`` per-tensor)
      - Explore a grid over ``x_range × y_range`` without global concat normalization
      - Evaluate loss in eval mode and no-grad, averaging over up to ``max_batches``
      - Save an HDF5 file with datasets: xcoordinates, ycoordinates, train_loss

    Returns the file path of the saved surface.
    """
    # Deepcopy and eval-mode to avoid mutating the caller's model
    work_model = copy.deepcopy(model).to(device)
    was_training = work_model.training
    work_model.eval()

    # Choose parameters to perturb: exclude names containing the given substring
    original_params_to_perturb: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, param in work_model.named_parameters():
            if param_name_exclude_substr is not None and param_name_exclude_substr in name:
                continue
            original_params_to_perturb[name] = param.data.detach().clone().cpu()

    # Build per-parameter directions d_x, d_y (per-tensor normalization; no global concat norm)
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

    # Cache up to max_batches from the test loader
    all_batches = []
    for i, batch in enumerate(test_loader):
        if (max_batches is not None)  and (i >= max_batches):
            break
        all_batches.append(batch)

    # Grid coordinates
    x_coords = np.linspace(x_range[0], x_range[1], int(num_points)).astype(np.float64)
    y_coords = np.linspace(y_range[0], y_range[1], int(num_points)).astype(np.float64)
    loss_grid = np.zeros((int(num_points), int(num_points)), dtype=np.float64)

    # Task id handling: follow provided flags
    # current_task_id: Optional[int] = (0 if class_incremental else None)

    # Helper to forward and compute loss robustly
    # def _forward_loss(_inputs: torch.Tensor, _labels: torch.Tensor) -> torch.Tensor:
    #     # Call model with task_id if available, else fallback
    #     # try:
    #     #     if current_task_id is not None:
    #     #         outputs = work_model(_inputs, task_id=current_task_id)
    #     #     else:
    #     #         outputs = work_model(_inputs)
    #     # except TypeError:
    #     outputs = work_model(_inputs)

    #     logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs
    #     return criterion(logits, _labels)

    # Sweep grid; restore params per point; evaluate in no-grad
    with torch.no_grad():
        state = work_model.state_dict()
        for i, xv in enumerate(tqdm(x_coords, desc="Grid X")):
            xv = float(xv)
            for j, yv in enumerate(y_coords):
                yv = float(yv)

                # Restore original parameters
                for name, tensor in original_params_to_perturb.items():
                    state[name].copy_(tensor.to(device=device, dtype=state[name].dtype))

                # Apply perturbation
                for name in original_params_to_perturb.keys():
                    delta = xv * perturb_x[name] + yv * perturb_y[name]
                    state[name].add_(delta)

                # Average loss over cached batches
                total = 0.0
                count = 0
                for batch in all_batches:
                    inputs, labels = _unwrap_batch(batch)
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    # loss = _forward_loss(inputs, labels)
                    outputs = work_model(inputs)
                    logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs
                    loss = criterion(logits, labels)
                    total += float(loss.item())
                    count += 1
                loss_grid[i, j] = total / max(1, count)

                if (j % 10 == 0) and torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()

    # Save surface
    os.makedirs(output_dir, exist_ok=True)
    surf_file = os.path.join(output_dir, f"{save_file_name}_task{eval_task_id}.h5")
    with h5py.File(surf_file, "w") as f:
        f.create_dataset("xcoordinates", data=x_coords)
        f.create_dataset("ycoordinates", data=y_coords)
        f.create_dataset("train_loss", data=loss_grid)

    # No need to restore mode on deepcopy; return saved path
    logging.info("[LossLandV1] Saved loss surface to %s", surf_file)
    return surf_file

def _power_iteration_lambda_max(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    dim: int,
    num_iters: int,
    max_batches: Optional[int]= None,
    known_classes: Optional[int]= None,
    *,
    tol: float = 1e-3,          # 对齐 Lanczos 的默认收敛容差
    patience: int = 2,
    return_vec: bool = False,
    topk: int = 1,
    seed: Optional[int] = None,
    use_abs_eig: bool = False,  # False: 代数主特征值；True: 谱半径代理（取绝对值）
) -> Any:
    """Estimate the dominant Hessian eigenvalue ``lambda_max`` via power iteration.

    - Computes algebraic largest eigenvalue via Rayleigh quotient by default.
    - Switches the model to eval() during iteration and restores the previous mode.

    If ``return_vec`` is True, also return the eigenvector; if ``topk`` > 1, return top-k.
    """
    if num_iters <= 0 or dim == 0:
        return (0.0, None) if return_vec else 0.0

    # 统一 eval() 模式，避免 Dropout/BN 干扰；结束后恢复
    was_training = model.training
    model.eval()

    try:
        def _mvp(v: torch.Tensor) -> torch.Tensor:
            return _hessian_vector_product(
                model, loader, device, params, v, max_batches=max_batches, known_classes=known_classes
            )

        if return_vec:
            vals, vecs = _deflated_power_iteration(
                _mvp,
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
            _mvp,
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
    
    finally:
        if was_training:
            model.train()


def _lanczos_lambda_max(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    params: List[torch.nn.Parameter],
    dim: int,
    num_iters: int,
    max_batches: Optional[int] = None,
    known_classes: Optional[int] = None,
    tol: float = 1e-3,
    reorth: bool = False,
    seed: Optional[int] = None,
    *,
    patience: int = 2,           # 新增：需连续命中次数（默认 2 次更稳）
    return_vec: bool = False,
) -> Any:
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
        return (0.0, None) if return_vec else 0.0

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
    hit = 0  # 连续命中计数（修复未定义变量）

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

        # —— 新增：Ritz 最大特征值的相对变化 + 连续命中 —— #
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
        if was_training:
            model.train()
        return (0.0, None) if return_vec else 0.0

    T = torch.zeros((k, k), dtype=torch.float64, device=device)
    for i in range(k):
        T[i, i] = alphas[i]
    for i in range(min(len(betas), k - 1)):
        beta_val = betas[i]
        T[i, i + 1] = beta_val
        T[i + 1, i] = beta_val

    lam_max = float(torch.linalg.eigvalsh(T.cpu()).max().item())

    if was_training:
        model.train()
    if return_vec:
        return lam_max, v
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
    """Estimate tr(H) using the Hutchinson estimator with Rademacher noise.

    We sample v ~ Rademacher({-1,+1}^d) and use E[v^T H v] = tr(H).
    """
    if num_samples <= 0:
        return 0.0

    trace_estimate = 0.0
    for _ in range(num_samples):
        # 生成 Rademacher 噪声向量 v ∈ {-1, +1}^dim（float32, 在正确 device 上）
        v = torch.empty(dim, device=device, dtype=torch.float32).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        hv = _hessian_vector_product(
            model, loader, device, params, v, max_batches=max_batches, known_classes=known_classes
        )
        trace_estimate += torch.dot(v, hv).item()

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
    # wrapped_model = network.module if isinstance(network, nn.DataParallel) else network
    # params = [p for p in wrapped_model.parameters() if p.requires_grad]

    wrapped_model = network.module if isinstance(network, torch.nn.DataParallel) else network
    if getattr(config,"model_name", "").lower() == "tuna":
        module = wrapped_model._network
        module = module.module if hasattr(module, "module") else module
        bb = getattr(module, "backbone", None)
        fused_id = (len(bb.adapter_list) + 1) if (len(bb.adapter_list) > 0 and getattr(bb, "merged_adapter", None) is not None) else len(bb.adapter_list)
        wrapped_model = TunaEvalWrapper(module, fused_id)

    substrs = getattr(config, "param_name_substrings", None)
    include_frozen = bool(getattr(config, "include_frozen_params", False))
    if isinstance(substrs, str):
        if substrs.lower() in {"none", "all", ""}:
            substrs = None
        else:
            substrs = [substrs]
    _saved_requires_grad = []
    if include_frozen:
        for _name, _p in wrapped_model.named_parameters():
            _saved_requires_grad.append((_p, bool(_p.requires_grad)))
            if not _p.requires_grad:
                _p.requires_grad_(True)

    # 当 substrs 为 None 时匹配全部参数
    params = _select_params_by_name(wrapped_model, substrs, include_frozen=include_frozen)
    names_for_params, shapes_for_params, splits_for_params = _param_names_and_shapes(wrapped_model, params)

    if not params:
        # 没有匹配到参数，就至少返回一个 base loss，避免崩
        return {
            "base_loss": float(_compute_loss(wrapped_model, loader, device, max_batches=config.loss_eval_max_batches))
        }
    
    



    global _GLOBAL_MAX_EXAMPLES_PER_BATCH
    if not params:
        return {"base_loss": 0.0}

    prev_max_examples = _GLOBAL_MAX_EXAMPLES_PER_BATCH
    _GLOBAL_MAX_EXAMPLES_PER_BATCH = config.max_examples_per_batch
    try:
        flat_metrics: Dict[str, float] = {}
        vals_power = None
        vecs_power = None
        param_backup = _clone_params(params)
        want_vecs = bool(
            getattr(config, "eig_save_vectors", False)
            or (str(getattr(config, "loss_land_basis", "random")).lower() == "eig")
            or bool(getattr(config, "curv_localization", False))
        )
        eig_topk_req = max(
            1,
            int(getattr(config, "eig_topk", 1)),
            int(getattr(config, "curv_topk", 1)),
        )
        # Hessian MVP is needed for Hessian metrics and optional localization
        def _mvp_hessian(v: torch.Tensor) -> torch.Tensor:
            return _hessian_vector_product(
                wrapped_model,
                loader,
                device,
                params,
                v,
                max_batches=config.loss_eval_max_batches,
                known_classes=known_classes,
            )
        #  ---------------- GGN / Fisher / Empirical Fisher (optional) ----------------
        # Define MVP closures that reuse the same data/batch budget
        def _mvp_ggn(v: torch.Tensor) -> torch.Tensor:
            return _ggn_vector_product(
                wrapped_model, loader, device, params, v,
                loss_eval_max_batches=config.loss_eval_max_batches, known_classes=known_classes
            )
    
        def _mvp_emp_fisher(v: torch.Tensor) -> torch.Tensor:
            return _empirical_fisher_vector_product(
                wrapped_model, loader, device, params, v,
                max_batches=config.loss_eval_max_batches, known_classes=known_classes
            )
        
        # 1) Base loss (always)
        logging.info("[FlatEval] Start base_loss (max_batches=%s)", str(config.loss_eval_max_batches))
        base_loss = _compute_loss(
            wrapped_model, loader, device, max_batches=config.loss_eval_max_batches, known_classes=known_classes
        )
        flat_metrics["base_loss"] = float(base_loss)
        logging.info("[FlatEval] Done base_loss=%.6f", base_loss)

        total_dim = int(sum(p.numel() for p in params))

        disable_power = bool(getattr(config, "disable_power", False))
        ggn_vecs_power = None
        ef_vecs_power = None
        ef_vals_power = None
        if bool(getattr(config, "eval_sharpness", True)):
            # 2) First-order gradient and Sh^(1)
            logging.info(
                "[FlatEval] Start grad/first-order (grad_batches=%s, rho=%.4f)",
                str(config.first_order_grad_batches), float(config.sharpness_radius),
            )
            grad_vector = _compute_grad_vector(
                wrapped_model,
                loader,
                device,
                params,
                max_batches=config.first_order_grad_batches,
                known_classes=known_classes,
            )
            grad_norm = grad_vector.norm().item()
            flat_metrics["grad_norm"] = grad_norm
            flat_metrics["first_order_sharpness"] = config.sharpness_radius * grad_norm
            logging.info(
                "[FlatEval] Done grad_norm=%.6f, Sh1=%.6f",
                grad_norm,
                flat_metrics["first_order_sharpness"],
            )

            # max sharpness
            logging.info("[FlatEval] Start Sh0_max along grad (rho=%.4f)", float(config.sharpness_radius))
            if grad_norm > 0:
                direction = grad_vector / (grad_norm + 1e-12)
                perturb = direction * config.sharpness_radius
                _add_vector_to_params(params, perturb)
                perturbed_loss = _compute_loss(
                    wrapped_model, loader, device, max_batches=config.loss_eval_max_batches, known_classes=known_classes
                )
                sh0 = perturbed_loss - base_loss
                flat_metrics["sh0_perturbed_loss"] = float(perturbed_loss)
                flat_metrics["sh0_max"] = float(sh0)
                _restore_params(params, param_backup)
            else:
                flat_metrics["sh0_max"] = 0.0
            logging.info("[FlatEval] Done Sh0_max=%.6f", flat_metrics.get("sh0_max", 0.0))

            # Random expectation sharpness
            gaussian_std = config.esh_gaussian_std or (config.sharpness_radius / (total_dim ** 0.5))
            logging.info(
                "[FlatEval] Start E-Sh (samples=%d, sigma=%.6f)",
                int(config.esh_num_samples), float(gaussian_std),
            )
            rand_losses: List[float] = []
            for _ in range(config.esh_num_samples):
                noise = torch.randn(total_dim, device=device) * gaussian_std
                _add_vector_to_params(params, noise)
                loss = _compute_loss(
                    wrapped_model, loader, device, max_batches=config.loss_eval_max_batches, known_classes=known_classes
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
            logging.info(
                "[FlatEval] Done E-Sh mean=%.6f std=%.6f",
                flat_metrics["esh_mean"], flat_metrics["esh_std"],
            )

        
        if bool(getattr(config, "eval_hessian", False)):
            if want_vecs:
                if not disable_power:
                    res = _power_iteration_lambda_max(
                        wrapped_model,
                        loader,
                        device,
                        params,
                        dim=total_dim,
                        num_iters=config.hessian_power_iters,
                        max_batches=config.loss_eval_max_batches,
                        known_classes=known_classes,
                        tol=getattr(config, "eig_tol", 1e-2),
                        patience=getattr(config, "eig_patience", 2),
                        return_vec=True,
                        topk=eig_topk_req,
                        seed=getattr(config, "loss_land_seed", None),
                    )
                    if isinstance(res, tuple) and isinstance(res[0], float):
                        lambda_max_power, v1_power = res
                        vals_power = [lambda_max_power]
                        vecs_power = [v1_power.detach().cpu()]
                    else:
                        vals, vecs = res
                        lambda_max_power = float(vals[0])
                        vals_power = [float(x) for x in vals]
                        vecs_power = [v.detach().cpu() for v in vecs]
                else:
                    lambda_max_power = float("nan")
                    vals_power = None
                    vecs_power = None
            else:
                if not disable_power:
                    lambda_max_power = _power_iteration_lambda_max(
                        wrapped_model,
                        loader,
                        device,
                        params,
                        dim=total_dim,
                        num_iters=config.hessian_power_iters,
                        max_batches=config.loss_eval_max_batches,
                        known_classes=known_classes,
                        tol=1e-3,
                        patience=int(getattr(config, "eig_patience", 2))
                    )
                else:
                    lambda_max_power = float("nan")
            # Backward‑compat key + explicit method key
            flat_metrics["lambda_max"] = lambda_max_power
            flat_metrics["lambda_max_power"] = lambda_max_power
            if vals_power is not None:
                flat_metrics["hessian_topk_vals_power"] = [float(x) for x in vals_power]
            logging.info("[FlatEval] Hessian lambda_max (power)=%.6f", lambda_max_power)

            # Save eigenvectors (Hessian + power)
            if want_vecs and vecs_power is not None and (not disable_power) and getattr(config, "save_metrics_path", None):
                save_dir = config.save_metrics_path
                prefix = getattr(config, "save_prefix", "flatness")
                out_path = os.path.join(save_dir, f"{prefix}_eig_hessian_power_top{len(vecs_power)}_lora.pt")
                payload = {
                    "backend": "hessian",
                    "method": "power",
                    "impl": "hessian",
                    "vals": vals_power,
                    "v1": vecs_power[0].contiguous(),
                    "names": names_for_params,
                    "shapes": [tuple(s) for s in shapes_for_params],
                    "splits": splits_for_params,
                    "param_scope": "lora",
                    "iters": int(getattr(config, "hessian_power_iters", 0)),
                    "tol": float(getattr(config, "eig_tol", 1e-2)),
                    "patience": int(getattr(config, "eig_patience", 2)),
                    "seed": getattr(config, "loss_land_seed", None),
                    "known_classes": known_classes,
                    "max_batches": getattr(config, "loss_eval_max_batches", None),
                    "normalized": "unit_euclid",
                }
                if len(vecs_power) >= 2:
                    payload["v2"] = vecs_power[1].contiguous()
                _save_eigvecs(out_path, payload)
                flat_metrics["eigvecs_hessian_power_path"] = out_path
        
            # Hessian spectral proxies (Lanczos)
            if want_vecs:
                lam_l, v_l = _lanczos_lambda_max(
                    wrapped_model,
                    loader,
                    device,
                    params,
                    dim=total_dim,
                    num_iters=config.hessian_power_iters,
                    max_batches=config.loss_eval_max_batches,
                    known_classes=known_classes,
                    tol=1e-3,
                    reorth=False,
                    patience=getattr(config, "eig_patience", 2),
                    return_vec=True,
                )
                lambda_max_lanczos = float(lam_l)
                v_lanczos = v_l.detach().cpu()
            else:
                lambda_max_lanczos = _lanczos_lambda_max(
                    wrapped_model,
                    loader,
                    device,
                    params,
                    dim=total_dim,
                    num_iters=config.hessian_power_iters,
                    max_batches=config.loss_eval_max_batches,
                    known_classes=known_classes,
                    reorth=True,
                )
            flat_metrics["lambda_max_lanczos"] = lambda_max_lanczos
            logging.info("[FlatEval] Hessian lambda_max (lanczos)=%.6f", lambda_max_lanczos)
            # Save eigenvector (Hessian + Lanczos)
            if want_vecs and ("v_lanczos" in locals()) and getattr(config, "save_metrics_path", None):
                save_dir = config.save_metrics_path
                prefix = getattr(config, "save_prefix", "flatness")
                out_path = os.path.join(save_dir, f"{prefix}_eig_hessian_lanczos_top1_lora.pt")
                payload = {
                    "backend": "hessian",
                    "method": "lanczos",
                    "impl": "hessian",
                    "vals": [float(lambda_max_lanczos)],
                    "v1": v_lanczos.contiguous().cpu(),
                    "names": names_for_params,
                    "shapes": [tuple(s) for s in shapes_for_params],
                    "splits": splits_for_params,
                    "param_scope": "lora",
                    "iters": int(getattr(config, "hessian_power_iters", 0)),
                    "tol": 1e-3,
                    "patience": int(getattr(config, "eig_patience", 2)),
                    "seed": getattr(config, "loss_land_seed", None),
                    "known_classes": known_classes,
                    "max_batches": getattr(config, "loss_eval_max_batches", None),
                    "normalized": "unit_euclid",
                }
                _save_eigvecs(out_path, payload)
                flat_metrics["eigvecs_hessian_lanczos_path"] = out_path
        
            trace_est = _hutchinson_trace(
                wrapped_model,
                loader,
                device,
                params,
                dim=total_dim,
                num_samples=config.hessian_trace_samples,
                max_batches=config.loss_eval_max_batches,
                known_classes=known_classes,
            )
            flat_metrics["hessian_trace"] = trace_est
            logging.info("[FlatEval] Done Hessian trace=%.6f", trace_est)
    
        
        if bool(getattr(config, "eval_ggn", False)):
            logging.info(
                "[FlatEval] Start GGN/Fisher (iters=%d, trace_samples=%d)",
                int(config.hessian_power_iters), int(config.hessian_trace_samples)
            )
            if want_vecs:
                if not disable_power:
                    res = _power_iteration_generic(
                        _mvp_ggn, total_dim, config.hessian_power_iters, device,
                        return_vec=True, topk=eig_topk_req,
                        tol=getattr(config, "eig_tol", None), patience=getattr(config, "eig_patience", 2),
                        seed=getattr(config, "loss_land_seed", None),
                    )
                    if isinstance(res, tuple) and isinstance(res[0], float):
                        flat_metrics["ggn_lambda_max_power"] = float(res[0])
                        ggn_vals_power = [float(res[0])]
                        ggn_vecs_power = [res[1].detach().cpu()]
                    else:
                        ggn_vals_power, _vecs = res
                        flat_metrics["ggn_lambda_max_power"] = float(ggn_vals_power[0])
                        ggn_vecs_power = [v.detach().cpu() for v in _vecs]
                else:
                    flat_metrics["ggn_lambda_max_power"] = float("nan")
                    ggn_vals_power = None
                    ggn_vecs_power = None
                lam_l_g, v_l_g = _lanczos_lambda_max_generic(
                    _mvp_ggn, total_dim, config.hessian_power_iters, device,
                    tol=1e-3, reorth=True, return_vec=True, seed=getattr(config, "loss_land_seed", None)
                )
                flat_metrics["ggn_lambda_max_lanczos"] = float(lam_l_g)
            else:
                if not disable_power:
                    flat_metrics["ggn_lambda_max_power"] = _power_iteration_generic(
                        _mvp_ggn, total_dim, config.hessian_power_iters, device
                    )
                else:
                    flat_metrics["ggn_lambda_max_power"] = float("nan")
                flat_metrics["ggn_lambda_max_lanczos"] = _lanczos_lambda_max_generic(
                    _mvp_ggn, total_dim, config.hessian_power_iters, device
                )
                ggn_vecs_power = None
                ggn_vals_power = None
            flat_metrics["ggn_trace"] = _hutchinson_trace_generic(
                _mvp_ggn, total_dim, config.hessian_trace_samples, device
            )
            # For CE/NLL, Fisher == GGN
            flat_metrics["fisher_lambda_max_power"]   = flat_metrics["ggn_lambda_max_power"]
            flat_metrics["fisher_lambda_max_lanczos"] = flat_metrics["ggn_lambda_max_lanczos"]
            flat_metrics["fisher_trace"]              = flat_metrics["ggn_trace"]
            if getattr(config, "fisher_rao", True):
                try:
                    flat_metrics["fisher_rao_norm"] = float(math.sqrt(max(0.0, float(flat_metrics["fisher_trace"]))))
                except Exception:
                    pass
            # Save GGN eigvecs if requested
            if want_vecs and getattr(config, "save_metrics_path", None):
                save_dir = config.save_metrics_path
                prefix = getattr(config, "save_prefix", "flatness")
                if ggn_vecs_power is not None and (not disable_power):
                    out_path = os.path.join(save_dir, f"{prefix}_eig_ggn_power_top{len(ggn_vecs_power)}_lora.pt")
                    payload = {
                        "backend": "ggn",
                        "method": "power",
                        "impl": "generic",
                        "vals": ggn_vals_power,
                        "v1": ggn_vecs_power[0].contiguous(),
                        "names": names_for_params,
                        "shapes": [tuple(s) for s in shapes_for_params],
                        "splits": splits_for_params,
                        "param_scope": "lora",
                        "iters": int(getattr(config, "hessian_power_iters", 0)),
                        "tol": float(getattr(config, "eig_tol", 1e-2)),
                        "patience": int(getattr(config, "eig_patience", 2)),
                        "seed": getattr(config, "loss_land_seed", None),
                        "known_classes": known_classes,
                        "max_batches": getattr(config, "loss_eval_max_batches", None),
                        "normalized": "unit_euclid",
                    }
                    if len(ggn_vecs_power) >= 2:
                        payload["v2"] = ggn_vecs_power[1].contiguous()
                    _save_eigvecs(out_path, payload)
                    flat_metrics["eigvecs_ggn_power_path"] = out_path
                if 'v_l_g' in locals():
                    out_path2 = os.path.join(save_dir, f"{prefix}_eig_ggn_lanczos_top1_lora.pt")
                    payload2 = {
                        "backend": "ggn",
                        "method": "lanczos",
                        "impl": "generic",
                        "vals": [float(flat_metrics["ggn_lambda_max_lanczos"])] ,
                        "v1": v_l_g.detach().cpu().contiguous(),
                        "names": names_for_params,
                        "shapes": [tuple(s) for s in shapes_for_params],
                        "splits": splits_for_params,
                        "param_scope": "lora",
                        "iters": int(getattr(config, "hessian_power_iters", 0)),
                        "tol": 1e-3,
                        "patience": int(getattr(config, "eig_patience", 2)),
                        "seed": getattr(config, "loss_land_seed", None),
                        "known_classes": known_classes,
                        "max_batches": getattr(config, "loss_eval_max_batches", None),
                        "normalized": "unit_euclid",
                    }
                    _save_eigvecs(out_path2, payload2)
                    flat_metrics["eigvecs_ggn_lanczos_path"] = out_path2
            logging.info(
                "[FlatEval] Done GGN: power=%.6f, lanczos=%.6f, trace=%.6f (Fisher same)",
                flat_metrics["ggn_lambda_max_power"], flat_metrics["ggn_lambda_max_lanczos"], flat_metrics["ggn_trace"]
            )
    
        if bool(getattr(config, "eval_fisher", True)):
            logging.info("[FlatEval] Start Empirical Fisher (iters=%d)", int(config.hessian_power_iters))
            if want_vecs:
                if not disable_power:
                    res = _power_iteration_generic(
                        _mvp_emp_fisher, total_dim, config.hessian_power_iters, device,
                        return_vec=True, topk=eig_topk_req,
                        tol=getattr(config, "eig_tol", None), patience=getattr(config, "eig_patience", 2),
                        seed=getattr(config, "loss_land_seed", None),
                    )
                    if isinstance(res, tuple) and isinstance(res[0], float):
                        flat_metrics["emp_fisher_lambda_max_power"] = float(res[0])
                        ef_vals_power = [float(res[0])]
                        ef_vecs_power = [res[1].detach().cpu()]
                    else:
                        ef_vals_power, _vecs = res
                        flat_metrics["emp_fisher_lambda_max_power"] = float(ef_vals_power[0])
                        ef_vecs_power = [v.detach().cpu() for v in _vecs]
                else:
                    flat_metrics["emp_fisher_lambda_max_power"] = float("nan")
                    ef_vals_power = None
                    ef_vecs_power = None
                lam_l_e, v_l_e = _lanczos_lambda_max_generic(
                    _mvp_emp_fisher, total_dim, config.hessian_power_iters, device,
                    tol=1e-3, reorth=True, return_vec=True, seed=getattr(config, "loss_land_seed", None)
                )
                flat_metrics["emp_fisher_lambda_max_lanczos"] = float(lam_l_e)
            else:
                if not disable_power:
                    flat_metrics["emp_fisher_lambda_max_power"] = _power_iteration_generic(
                        _mvp_emp_fisher, total_dim, config.hessian_power_iters, device
                    )
                else:
                    flat_metrics["emp_fisher_lambda_max_power"] = float("nan")
                flat_metrics["emp_fisher_lambda_max_lanczos"] = _lanczos_lambda_max_generic(
                    _mvp_emp_fisher, total_dim, config.hessian_power_iters, device
                )
                ef_vecs_power = None
                ef_vals_power = None
            flat_metrics["emp_fisher_trace"] = _hutchinson_trace_generic(
                _mvp_emp_fisher, total_dim, max(1, config.hessian_trace_samples // 2), device
            )
            if want_vecs and getattr(config, "save_metrics_path", None):
                save_dir = config.save_metrics_path
                prefix = getattr(config, "save_prefix", "flatness")
                if ef_vecs_power is not None and not disable_power:
                    out_path = os.path.join(save_dir, f"{prefix}_eig_emp_fisher_power_top{len(ef_vecs_power)}_lora.pt")
                    payload = {
                        "backend": "emp_fisher",
                        "method": "power",
                        "impl": "generic",
                        "vals": ef_vals_power,
                        "v1": ef_vecs_power[0].contiguous(),
                        "names": names_for_params,
                        "shapes": [tuple(s) for s in shapes_for_params],
                        "splits": splits_for_params,
                        "param_scope": "lora",
                        "iters": int(getattr(config, "hessian_power_iters", 0)),
                        "tol": float(getattr(config, "eig_tol", 1e-2)),
                        "patience": int(getattr(config, "eig_patience", 2)),
                        "seed": getattr(config, "loss_land_seed", None),
                        "known_classes": known_classes,
                        "max_batches": getattr(config, "loss_eval_max_batches", None),
                        "normalized": "unit_euclid",
                    }
                    if len(ef_vecs_power) >= 2:
                        payload["v2"] = ef_vecs_power[1].contiguous()
                    _save_eigvecs(out_path, payload)
                    flat_metrics["eigvecs_emp_fisher_power_path"] = out_path
                if 'v_l_e' in locals():
                    out_path2 = os.path.join(save_dir, f"{prefix}_eig_emp_fisher_lanczos_top1_lora.pt")
                    payload2 = {
                        "backend": "emp_fisher",
                        "method": "lanczos",
                        "impl": "generic",
                        "vals": [float(flat_metrics["emp_fisher_lambda_max_lanczos"])],
                        "v1": v_l_e.detach().cpu().contiguous(),
                        "names": names_for_params,
                        "shapes": [tuple(s) for s in shapes_for_params],
                        "splits": splits_for_params,
                        "param_scope": "lora",
                        "iters": int(getattr(config, "hessian_power_iters", 0)),
                        "tol": 1e-3,
                        "patience": int(getattr(config, "eig_patience", 2)),
                        "seed": getattr(config, "loss_land_seed", None),
                        "known_classes": known_classes,
                        "max_batches": getattr(config, "loss_eval_max_batches", None),
                        "normalized": "unit_euclid",
                    }
                    _save_eigvecs(out_path2, payload2)
                    flat_metrics["eigvecs_emp_fisher_lanczos_path"] = out_path2
            logging.info(
                "[FlatEval] Done Empirical Fisher: power=%.6f, lanczos=%.6f, trace=%.6f",
                flat_metrics["emp_fisher_lambda_max_power"], flat_metrics["emp_fisher_lambda_max_lanczos"], flat_metrics["emp_fisher_trace"]
            )
    
        if bool(getattr(config, "curv_localization", False)):
            curv_substrs = getattr(config, "curv_param_name_substrings", None)
            if isinstance(curv_substrs, str):
                if curv_substrs.lower() in {"none", "all", ""}:
                    curv_substrs = None
                else:
                    curv_substrs = [curv_substrs]
            if curv_substrs is None:
                curv_substrs = substrs
            curv_substrs = curv_substrs or ["linear_a", "linear_b", "lora"]

            mask = _build_subspace_mask(
                names_for_params, splits_for_params, curv_substrs, total_dim=total_dim
            )
            curv_choice = str(getattr(config, "curv_mvp", "hessian")).lower()
            curv_context = {
                "mask": mask,
                "num_samples": int(getattr(config, "curv_rayleigh_samples", 32)),
                "basis_seed": getattr(config, "curv_basis_seed", None),
                "choice": curv_choice,
            }
            curv_pending = curv_context
        
            choice = str(curv_pending.get("choice", "emp_fisher")).lower()
            mask = curv_pending.get("mask", None)
            num_samples = int(curv_pending.get("num_samples", 32))
            basis_seed = curv_pending.get("basis_seed", None)
            mvp_map = {
                "hessian": _mvp_hessian,
                "ggn": _mvp_ggn,
                "emp_fisher": _mvp_emp_fisher,
            }
            mvp_fn = mvp_map.get(choice, _mvp_emp_fisher)
            curv_method = str(getattr(config, "curv_eig_method", "power")).lower()
            vecs_local = None
            if curv_method == "lanczos":
                eig_topk = int(getattr(config, "curv_topk", 1))
                num_iters = int(getattr(config, "hessian_power_iters", 5))
                tol = getattr(config, "eig_tol", None)
                seed = getattr(config, "loss_land_seed", None)
                _eigvals, vecs_local = _lanczos_topk_generic(
                    mvp_fn,
                    total_dim,
                    num_iters,
                    device,
                    topk=eig_topk,
                    tol=tol,
                    seed=seed,
                )
            else:
                if choice == "emp_fisher":
                    vecs_local = ef_vecs_power
                elif choice == "ggn":
                    vecs_local = ggn_vecs_power
                else:
                    vecs_local = vecs_power
                if vecs_local is None:
                    vecs_local = vecs_power
            if mask is not None:
                try:
                    curv_metrics = _curvature_localization_metrics(
                        vecs_local,
                        mask,
                        mvp_fn,
                        device,
                        num_samples=num_samples,
                        basis_seed=basis_seed,
                    )
                    flat_metrics.update(curv_metrics)
                except Exception:
                    logging.exception("[FlatEval] Curvature localization (%s) failed", str(choice))
            else:
                logging.warning("[FlatEval] Skip curvature localization (%s): mask is None", str(choice))
            curv_blocks = getattr(config, "curv_qkv_blocks", None)
            if isinstance(curv_blocks, dict):
                curv_blocks = [curv_blocks]
            if curv_blocks:
                for block in curv_blocks:
                    if not isinstance(block, dict):
                        continue
                    tag = str(block.get("tag") or block.get("qkv_tag") or "block").strip()
                    tag = tag.replace(" ", "_")
                    target_name = block.get("param_name") or block.get("target_name")
                    if not target_name:
                        continue
                    row_ranges = block.get("row_ranges", None)
                    col_ranges = block.get("col_ranges", None)
                    qkv_tag = block.get("qkv_tag", None)
                    if qkv_tag is None and tag.lower() in {"q", "k", "v"}:
                        qkv_tag = tag
                    split_dim = block.get("split_dim", None)
                    block_index = block.get("block_index", None)
                    block_mask = _build_qkv_block_mask(
                        names_for_params,
                        shapes_for_params,
                        splits_for_params,
                        target_name,
                        qkv_tag,
                        row_ranges,
                        col_ranges,
                        total_dim=total_dim,
                        split_dim=split_dim,
                        block_index=block_index,
                    )
                    if block_mask.numel() == 0 or block_mask.sum().item() == 0:
                        logging.warning(
                            "[FlatEval] Curvature qkv mask empty (tag=%s, param=%s)", tag, target_name
                        )
                        continue
                    try:
                        block_metrics = _curvature_localization_metrics(
                            vecs_local,
                            block_mask,
                            mvp_fn,
                            device,
                            num_samples=num_samples,
                            basis_seed=basis_seed,
                        )
                        prefixed = {f"curv_qkv_{tag}_{k}": v for k, v in block_metrics.items()}
                        prefixed[f"curv_qkv_{tag}_param_name"] = target_name
                        if qkv_tag is not None:
                            prefixed[f"curv_qkv_{tag}_qkv_tag"] = str(qkv_tag)
                        if row_ranges is not None:
                            prefixed[f"curv_qkv_{tag}_row_ranges"] = row_ranges
                        if col_ranges is not None:
                            prefixed[f"curv_qkv_{tag}_col_ranges"] = col_ranges
                        if split_dim is not None:
                            prefixed[f"curv_qkv_{tag}_split_dim"] = str(split_dim)
                        if block_index is not None:
                            prefixed[f"curv_qkv_{tag}_block_index"] = int(block_index)
                        flat_metrics.update(prefixed)
                    except Exception:
                        logging.exception("[FlatEval] Curvature qkv localization failed (tag=%s)", tag)
            curv_pending = None

        # -------- Delta-W projection (weight-space LoRA subspace) --------
        if bool(getattr(config, "delta_w_projection", False)):
            mvp_map = {
                "hessian": _mvp_hessian,
                "ggn": _mvp_ggn,
                "emp_fisher": _mvp_emp_fisher,
            }
            save_dir = getattr(config, "save_metrics_path", None)
            prefix = getattr(config, "save_prefix", "flatness")
            try:
                delta_metrics = _delta_w_projection_eval(
                    wrapped_model,
                    names_for_params,
                    shapes_for_params,
                    splits_for_params,
                    total_dim,
                    device,
                    mvp_map,
                    config,
                    save_dir=save_dir,
                    save_prefix=prefix,
                )
                if delta_metrics:
                    flat_metrics.update(delta_metrics)
            except Exception:
                logging.exception("[FlatEval] Delta-W projection failed")

        if bool(getattr(config, "w_delta_alignment", False)):
            save_dir = getattr(config, "save_metrics_path", None)
            prefix = getattr(config, "save_prefix", "flatness")
            try:
                w_metrics = _w_delta_alignment_eval(
                    wrapped_model,
                    config,
                    save_dir=save_dir,
                    save_prefix=prefix,
                )
                if w_metrics:
                    flat_metrics.update(w_metrics)
            except Exception:
                logging.exception("[FlatEval] W-DeltaW alignment failed")

        # Loss landscape slices (optional)
        # ---------- Loss landscape：由单一参数 loss_land_modes 控制 ----------
        if False:
            do_1d = bool(getattr(config, "weight_loss_land_1d", False))
            do_2d = bool(getattr(config, "weight_loss_land_2d", False))
            if do_1d or do_2d:
                # radius策略
                if bool(getattr(config, "loss_land_radius_from_rho", False)):
                    loss_radius = float(getattr(config, "loss_land_radius_scale", 1.5)) * float(getattr(config, "sharpness_radius", 0.05))
                else:
                    loss_radius  = float(getattr(config, "weight_loss_land_radius", 1))
                loss_points  = int(getattr(config, "weight_loss_land_num_points", 41))
                loss_batches = getattr(config, "weight_loss_land_max_batches", None)
                loss_filter  = bool(getattr(config, "weight_loss_land_filter_norm", True))
                basis       = str(getattr(config, "loss_land_basis", "random")).lower()
                save_dir = getattr(config, "save_metrics_path", None)
                prefix   = getattr(config, "save_prefix", "flatness")

                # NEW(单一开关)：'lora' | 'full' | 'all'
                mode = str(getattr(config, "loss_land_modes", "lora")).lower()
                if mode not in {"lora", "full", "all"}:
                    mode = "full"
                logging.info(
                    "[FlatEval] Start loss landscape (modes=%s, 1D=%s, 2D=%s, basis=%s)",
                    mode, str(do_1d), str(do_2d), basis
                )

                # 收集“全参”用于地形作图（可选含冻结；可选基于名字二次筛选）
                def _collect_full_params():
                    include_frozen = bool(getattr(config, "loss_land_include_frozen", True))
                    substrs_full = getattr(config, "loss_land_param_names", None)  # 例：["attn.", "mlp."]
                    selected = []
                    for n, p in wrapped_model.named_parameters():
                        if (not include_frozen) and (not p.requires_grad):
                            continue
                        if (substrs_full is None) or any(s in n for s in substrs_full):
                            selected.append(p)
                    return selected

                # 需要作图的模式集合 & 是否多份输出
                tags = ["lora", "full"] if mode == "all" else [mode]
                multi_output = (mode == "all")

                # prepare eigenvector directions for lora if basis=eig and hessian backend
                dirs_eig_1d = None
                dirs_eig_2d = None
                if basis == "eig" and str(getattr(config, "eig_backend", "hessian")).lower() == "hessian" and vals_power is not None and vecs_power is not None:
                    # unflatten for current LoRA param list
                    v1_list = _unflatten_to_param_like(vecs_power[0].to(device), params)
                    dirs_eig_1d = v1_list
                    if len(vecs_power) >= 2:
                        v2_list = _unflatten_to_param_like(vecs_power[1].to(device), params)
                        dirs_eig_2d = [v1_list, v2_list]

                rng_seed = getattr(config, "loss_land_seed", None)

                for tag in tags:
                    if tag == "lora":
                        land_params = params                       # 已筛好的 LoRA 子空间参数
                        suffix = "_lora" if multi_output else ""   # 'all' 时区分 key/文件名
                    else:
                        land_params = _collect_full_params()        # 全参（可含冻结）
                        suffix = "_full" if multi_output else ""

                    if not land_params:  # 兜底：若参数集合为空则跳过该模式
                        continue

                    # build override dirs if needed
                    override_1d = None
                    override_2d = None
                    if basis == "eig" and tag == "lora":
                        if dirs_eig_1d is not None:
                            override_1d = dirs_eig_1d
                        if dirs_eig_2d is not None:
                            override_2d = dirs_eig_2d
                    elif basis == "random":
                        pass


                    # New: use V1 implementation to plot/save 2D lossland
                    try:
                        do_plot_2d = bool(do_2d)
                        if do_plot_2d:
                            save_dir = getattr(config, "save_metrics_path", None)
                            prefix = getattr(config, "save_prefix", "flatness")
                            if save_dir:
                                os.makedirs(save_dir, exist_ok=True)
                                # Derive task id and CI flag from args if available
                                # eval_task_id_v = None
                                # class_incr = False
                                # if hasattr(config, "args") and isinstance(config.args, dict):
                                #     a = config.args
                                #     # Common keys candidates
                                #     for k in ("cur_task", "task_id", "eval_task_id"):
                                #         if k in a:
                                #             try:
                                #                 eval_task_id_v = int(a[k])
                                #             except Exception:
                                #                 eval_task_id_v = a[k]
                                #             break
                                #     class_incr = bool(a.get("class_incremental", False))

                                # Only compute for 'full' mode since V1 perturbs all (excl. 'shared')
                                if tag == "full":
                                    crit = nn.CrossEntropyLoss(reduction="mean")
                                    base_name = f"{prefix}_lossland_2d{suffix}_11"
                                    surf_path = compute_loss_landscape_v1(
                                        model=wrapped_model,
                                        test_loader=loader,
                                        device=device,
                                        criterion=crit,
                                        output_dir=save_dir,
                                        save_file_name=base_name,
                                        eval_task_id=200,
                                        class_incremental=False,
                                        x_range=(-1.0, 1.0),
                                        y_range=(-1.0, 1.0),
                                        num_points=loss_points,
                                        max_batches=loss_batches,
                                        sample_batches=False,
                                        param_name_exclude_substr=None,
                                        seed=int(rng_seed) if rng_seed is not None else 42,
                                    )
                                    flat_metrics[f"lossland_2d_file{suffix})_11"] = surf_path

                                    # Compute min/max for logging consistency
                                   
                                    with h5py.File(surf_path, "r") as f:
                                        z = np.array(f["train_loss"]) if "train_loss" in f else None
                                    if z is not None:
                                        flat_metrics[f"lossland_2d_min{suffix}_11"] = float(np.min(z))
                                        flat_metrics[f"lossland_2d_max{suffix}_11"] = float(np.max(z))
                                        logging.info(
                                            "[FlatEval] Done 2D (full%s): min=%.6f, max=%.6f",
                                            suffix,
                                            flat_metrics[f"lossland_2d_min{suffix}_11"],
                                            flat_metrics[f"lossland_2d_max{suffix}_11"]
                                        )

                                    # 再画一个小的
                                    base_name = f"{prefix}_lossland_2d{suffix}_0202"
                                    surf_path = compute_loss_landscape_v1(
                                        model=wrapped_model,
                                        test_loader=loader,
                                        device=device,
                                        criterion=crit,
                                        output_dir=save_dir,
                                        save_file_name=base_name,
                                        eval_task_id=200,
                                        class_incremental=False,
                                        x_range=(-0.2, 0.2),
                                        y_range=(-0.2, 0.2),
                                        num_points=loss_points,
                                        max_batches=loss_batches,
                                        sample_batches=False,
                                        param_name_exclude_substr=None,
                                        seed=int(rng_seed) if rng_seed is not None else 42,
                                    )
                                    flat_metrics[f"lossland_2d_file{suffix}_0202"] = surf_path

                                    # Compute min/max for logging consistency
                                   
                                    with h5py.File(surf_path, "r") as f:
                                        z = np.array(f["train_loss"]) if "train_loss" in f else None
                                    if z is not None:
                                        flat_metrics[f"lossland_2d_min{suffix}_0202"] = float(np.min(z))
                                        flat_metrics[f"lossland_2d_max{suffix}_0202"] = float(np.max(z))
                                        logging.info(
                                            "[FlatEval] Done 2D (full%s): min=%.6f, max=%.6f",
                                            suffix,
                                            flat_metrics[f"lossland_2d_min{suffix}_0202"],
                                            flat_metrics[f"lossland_2d_max{suffix}_0202"]
                                        )
                                    
                            else:
                                logging.info("[FlatEval] loss_landscape_v1 skipped (no save_metrics_path)")
                    except Exception:
                        logging.exception("[FlatEval] loss_landscape_v1 failed")

                    # # random override to infuse seed
                    # def _rand_dirs_for(land_params, k=1, base_seed=None):
                    #     out = []
                    #     if base_seed is None:
                    #         base_seed = 123
                    #     for i in range(k):
                    #         d = _build_direction_list(land_params, device=device, filter_norm=loss_filter, seed=int(base_seed) + i)
                    #         # normalize concat
                    #         nrm = _dir_norm(d)
                    #         if nrm > 0:
                    #             d = [t / nrm for t in d]
                    #         out.append(d)
                    #     return out

                    # if override_1d is None and basis == "random":
                    #     override_1d = _rand_dirs_for(land_params, k=1, base_seed=rng_seed or 123)[0]
                    # if override_2d is None and basis == "random":
                    #     pair = _rand_dirs_for(land_params, k=2, base_seed=rng_seed or 123)
                    #     d1 = pair[0]
                    #     d2 = _orthonormalize(pair[1], d1)
                    #     override_2d = [d1, d2]

                    # if do_1d:
                    #     # logging.info(
                    #     #     "[FlatEval] Loss landscape 1D (%s%s): radius=%.3f, points=%d"  
                    #     #     tag, suffix, float(loss_radius), int(loss_points)  
                    #     # )
                    #     res1d = _loss_landscape_1d(
                    #         wrapped_model, loader, device, land_params,
                    #         radius=loss_radius, num_points=loss_points,
                    #          filter_norm=loss_filter,
                    #         known_classes=known_classes,
                    #         dirs_override=override_1d,
                    #     )
                    #     flat_metrics[f"lossland_1d_min{suffix}"] = float(np.min(res1d["loss"]))
                    #     flat_metrics[f"lossland_1d_max{suffix}"] = float(np.max(res1d["loss"]))
                    #     logging.info(
                    #         "[FlatEval] Done 1D (%s%s): min=%.6f, max=%.6f",
                    #         tag, suffix, flat_metrics[f"lossland_1d_min{suffix}"], flat_metrics[f"lossland_1d_max{suffix}"]
                    #     )
                    #     if save_dir:
                    #         os.makedirs(save_dir, exist_ok=True)
                    #         path1d = os.path.join(save_dir, f"{prefix}_lossland_1d{suffix}.npz")
                    #         np.savez_compressed(path1d, x=res1d["x"], loss=res1d["loss"])
                    #         flat_metrics[f"lossland_1d_file{suffix}"] = path1d

                    # if do_2d:
                    #     logging.info(
                    #         "[FlatEval] Loss landscape 2D (%s%s): radius=%.3f,   basis=%s",
                    #         tag, suffix, float(loss_radius),  basis
                    #     )
                    #     res2d = _loss_landscape_2d(
                    #         wrapped_model, loader, device, land_params,
                    #         radius=loss_radius, num_points=loss_points,
                    #         filter_norm=loss_filter,
                    #         known_classes=known_classes,
                    #         dirs_override=override_2d,
                    #     )
                    #     flat_metrics[f"lossland_2d_min{suffix}"] = float(np.min(res2d["loss"]))
                    #     flat_metrics[f"lossland_2d_max{suffix}"] = float(np.max(res2d["loss"]))
                    #     logging.info(
                    #         "[FlatEval] Done 2D (%s%s): min=%.6f, max=%.6f",
                    #         tag, suffix, flat_metrics[f"lossland_2d_min{suffix}"], flat_metrics[f"lossland_2d_max{suffix}"]
                    #     )
                    #     if save_dir:
                    #         os.makedirs(save_dir, exist_ok=True)
                    #         path2d = os.path.join(save_dir, f"{prefix}_lossland_2d{suffix}.npz")
                    #         np.savez_compressed(path2d, x=res2d["x"], y=res2d["y"], loss=res2d["loss"])
                    #         flat_metrics[f"lossland_2d_file{suffix}"] = path2d

        
        #
        # ---------------- Relative Flatness (layerwise) ----------------
        if getattr(config, "relative_flatness", False):
            scope = str(getattr(config, "rf_scope", "custom")).lower()
            rf_power_iters = int(getattr(config, "rf_power_iters", 0) or getattr(config, "hessian_power_iters", 0) or 0)
            rf_trace_samples = int(getattr(config, "rf_trace_samples", 0) or getattr(config, "hessian_trace_samples", 0) or 0)
            rf_max_batches = getattr(config, "rf_max_batches", None)
            if rf_max_batches is None:
                rf_max_batches = getattr(config, "loss_eval_max_batches", None)

            if scope == "fc":
                layer_params = _get_fc_params(wrapped_model, include_bias=bool(getattr(config, "rf_include_bias", False)))
                scope_tag = "fc"
            elif scope == "custom":
                substrs = getattr(config, "rf_param_name_substrings", None) or getattr(config, "param_name_substrings", None)
                layer_params = _select_params_by_name(wrapped_model, substrs)
                scope_tag = "custom"
            else:  # lora
                substrs = getattr(config, "rf_param_name_substrings", None) or getattr(config, "param_name_substrings", None)
                if substrs is None:
                    substrs = ["lora", "lora_"]
                layer_params = _select_params_by_name(wrapped_model, substrs)
                scope_tag = "lora"

            if layer_params:
                dim_layer = int(sum(p.numel() for p in layer_params))
                wn, mode_used = _weight_norm_for_params(layer_params, getattr(config, "rf_norm_mode", "fro"))
                tr_layer = _hutchinson_trace(
                    wrapped_model,
                    loader,
                    device,
                    layer_params,
                    dim_layer,
                    int(max(0, rf_trace_samples)),
                    rf_max_batches,
                    known_classes,
                ) if rf_trace_samples and rf_trace_samples > 0 else 0.0
                lam_layer = _power_iteration_lambda_max(
                    wrapped_model,
                    loader,
                    device,
                    layer_params,
                    dim_layer,
                    int(max(0, rf_power_iters)),
                    rf_max_batches,
                    known_classes,
                ) if rf_power_iters and rf_power_iters > 0 else 0.0

                flat_metrics[f"rf_weight_norm_{scope_tag}_{mode_used}"] = float(wn)
                flat_metrics[f"rf_trace_{scope_tag}"] = float(tr_layer)
                flat_metrics[f"rf_lambda_{scope_tag}"] = float(lam_layer)
                flat_metrics[f"relative_flatness_trace_{scope_tag}"] = float(wn * tr_layer)
                flat_metrics[f"relative_flatness_lambda_{scope_tag}"] = float(wn * lam_layer)

        _restore_params(params, param_backup)
        wrapped_model.zero_grad(set_to_none=True)
        if bool(getattr(config, "ratio_as_percent", False)):
            def _scale_ratio_value(value: Any) -> Any:
                if isinstance(value, (int, float)):
                    return float(value) * 100.0
                if isinstance(value, list):
                    return [float(v) * 100.0 for v in value]
                return value

            scaled: Dict[str, Any] = {}
            for key, value in flat_metrics.items():
                if "ratio" in str(key):
                    scaled[key] = _scale_ratio_value(value)
                else:
                    scaled[key] = value
            scaled["ratio_unit"] = "%"
            scaled["ratio_scale"] = 100.0
            flat_metrics = scaled
        # Optional persistence
        if getattr(config, "save_metrics_path", None):
            os.makedirs(config.save_metrics_path, exist_ok=True)
            save_file = os.path.join(
                config.save_metrics_path,
                f"{getattr(config, 'save_prefix', 'flatness')}_metrics.json",
            )
            with open(save_file, "w", encoding="utf-8") as fh:
                json.dump(flat_metrics, fh, indent=2)
    
        return flat_metrics
    finally:
        # 恢复所有参数的 requires_grad 原状态
        try:
            for _p, _old in _saved_requires_grad:
                if bool(_p.requires_grad) != bool(_old):
                    _p.requires_grad_(bool(_old))
        except Exception:
            pass
        _GLOBAL_MAX_EXAMPLES_PER_BATCH = prev_max_examples


# ---------------------------------------------------------------------------
# Optional CLI to evaluate flatness from a stored config/checkpoint
# ---------------------------------------------------------------------------
def _load_args(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)

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
        # Fallback: concatenate spans into a dense block
        row_blocks = []
        for rs, re in row_spans:
            col_blocks = [full[rs:re, cs:ce] for cs, ce in col_spans]
            row_blocks.append(torch.cat(col_blocks, dim=1))
        return torch.cat(row_blocks, dim=0)
    return None


def _deflated_power_iteration_masked(
    mv: callable,
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
    mv: callable,
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


def _lanczos_topk_generic(
    mv: callable,
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
    Q = torch.stack(basis, dim=1)  # [dim, k]
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
    mv: callable,
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
    config: FlatnessConfig,
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
    delta_w = _extract_lora_qv_weights(wrapped_model, block_index)
    if delta_w is None:
        return {}

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
            U, S, Vh = torch.linalg.svd(dw, full_matrices=False)
        except Exception:
            continue
        r = int(min(U.shape[1], Vh.shape[0], lora_rank or U.shape[1]))
        U_r = U[:, :r]
        V_r = Vh[:r, :].T

        ratios: List[float] = []
        cores: List[torch.Tensor] = []
        for v in eigvecs:
            V = _extract_param_block(v, names, shapes, splits, target_name, row_ranges, col_ranges)
            if V is None:
                ratios.append(0.0)
                cores.append(torch.zeros((U_r.shape[1], V_r.shape[1]), device=device))
                continue
            denom = float(V.norm().item() ** 2)
            if denom <= eps:
                ratios.append(0.0)
                cores.append(torch.zeros((U_r.shape[1], V_r.shape[1]), device=device))
                continue
            tmp = U_r.T @ V @ V_r
            proj = U_r @ tmp @ V_r.T
            num = float(proj.norm().item() ** 2)
            ratios.append(float(num / (denom + eps)))
            cores.append(tmp)

        metrics[f"delta_w_proj_{tag}_backend"] = backend
        metrics[f"delta_w_proj_{tag}_eig_method"] = eig_method
        metrics[f"delta_w_proj_{tag}_space"] = "delta_w_bilinear_subspace"
        metrics[f"delta_w_proj_{tag}_note"] = "Projection onto U_r (·) V_r^T from ΔW SVD."
        metrics[f"delta_w_proj_{tag}_eigvals"] = [float(x) for x in eigvals]
        metrics[f"delta_w_proj_{tag}_ratios"] = [float(x) for x in ratios]
        if ratios:
            metrics[f"delta_w_proj_{tag}_ratio_top1"] = float(ratios[0])
        if eigvals and ratios and len(eigvals) == len(ratios):
            num = sum(float(l) * float(r) for l, r in zip(eigvals, ratios))
            den = sum(float(l) for l in eigvals)
            metrics[f"delta_w_proj_{tag}_ratio_weighted"] = float(num / (den + eps))

        # Grassmann-style alignment (principal angles)
        if cores:
            try:
                G = torch.stack([c.reshape(-1) for c in cores], dim=1)  # [r^2, k]
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

        # Random subspace baseline (same r)
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
            if rand_cores:
                payload["U_rand"] = U_rand.detach().cpu()
                payload["V_rand"] = V_rand.detach().cpu()
                payload["rand_ratios"] = [float(x) for x in rand_ratios]
                payload["rand_core_mats"] = [c.detach().cpu() for c in rand_cores]
            torch.save(payload, out_path)
            metrics[f"delta_w_proj_{tag}_path"] = out_path

    return metrics


def _w_delta_alignment_eval(
    wrapped_model: torch.nn.Module,
    config: FlatnessConfig,
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
    delta_w = _extract_lora_qv_weights(wrapped_model, block_index)
    if delta_w is None:
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


def _select_params_by_name(module: torch.nn.Module, substrs: Optional[List[str]], include_frozen: bool = False):
    """按名字包含某些子串筛选需要参与二阶的参数；substrs=None 则退化为取所有 requires_grad=True 的参数。"""
    picked = []
    for n, p in module.named_parameters():
        if (not include_frozen) and (not p.requires_grad):
            continue
        if substrs is None or any(s in n for s in substrs):
            picked.append(p)
    return picked


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
        shuffle=config.get("shuffle", False),
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
    torch.cuda.empty_cache()
    logging.info("Flatness metrics: %s", metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
