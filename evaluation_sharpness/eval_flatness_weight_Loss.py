import argparse
import json
import logging
import math
import os
from dataclasses import MISSING, dataclass, field, fields as dc_fields
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from contextlib import nullcontext
from torch.utils.data import DataLoader, Subset

from evaluation_performance.probe import TunaEvalWrapper
from evaluation_sharpness.power_iter import (
    _power_iteration_generic,
    _power_iteration_lambda_max,
)
from evaluation_sharpness.lanczos_iter import (
    _lanczos_lambda_max,
    _lanczos_lambda_max_generic,
    _lanczos_topk_generic,
)
from evaluation_sharpness.param_utils import (
    _add_vector_to_params,
    _clone_params,
    _get_fc_params,
    _param_names_and_shapes,
    _restore_params,
    _select_params_by_name,
    _unflatten_to_param_like,
    _weight_norm_for_params,
)
from evaluation_sharpness.loss_landscape import (
    compute_full_vs_lora_curvature_1d,
    compute_loss_landscape_v1,
)
from evaluation_sharpness.loss_utils import (
    _compute_loss,
    _forward_logits_full,
    _get_max_examples_per_batch,
    _set_max_examples_per_batch,
    _unwrap_batch,
)
from evaluation_sharpness.io_utils import (
    _load_args,
    _save_eigvecs,
)
from evaluation_sharpness.curv_localization import (
    _build_subspace_mask,
    _build_qkv_block_mask,
    _curvature_localization_metrics,
    _curv_noise_cov_eval,
    _curv_ts_eval,
    _delta_w_projection_eval,
    _delta_w_full_projection_eval,
    _w_delta_alignment_eval,
    _mean_drift_eval,
    _lora_kl_eval,
)
from sharpness_evaluation_core.curvature import make_mvp_map


# -----------------------------------------------------------------------------
# Small helpers to keep MVP mapping consistent and reusable across sections.
# These do not change numerical results; they only centralize backend→MVP wiring.
# -----------------------------------------------------------------------------
def _default_mvp_map(
    mvp_hessian: Callable[[torch.Tensor], torch.Tensor],
    mvp_ggn: Callable[[torch.Tensor], torch.Tensor],
    mvp_fisher: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Return the standard {hessian, ggn, emp_fisher} MVP lookup."""
    return make_mvp_map(mvp_hessian, mvp_ggn, mvp_fisher)
from sharpness_evaluation_core.curvature import make_mvp_map


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
    loss_land_enabled: bool = False
    weight_loss_land_1d: bool = False
    weight_loss_land_2d: bool = False
    weight_loss_land_radius: float = 0.5
    weight_loss_land_num_points: int = 21
    weight_loss_land_max_batches: Optional[float] = None
    weight_loss_land_filter_norm: bool = False

    loss_land_modes: str = "lora"             # "lora" | "full" | "all"
    loss_land_include_frozen: bool = True
    loss_land_param_names: Optional[List[str]] = None
    loss_land_curv_1d: bool = False
    loss_land_curv_backend: str = "emp_fisher"   # "hessian" | "ggn" | "emp_fisher"
    loss_land_curv_method: str = "power"         # "power" | "lanczos"
    loss_land_curv_iters: Optional[int] = None
    loss_land_curv_topk: int = 1
    loss_land_curv_num_points: Optional[int] = None
    loss_land_curv_radius_full: Optional[float] = None
    loss_land_curv_radius_lora: Optional[float] = None
    loss_land_curv_max_batches: Optional[int] = None
    loss_land_curv_use_abs_eig: bool = False
    loss_land_curv_normalize: bool = False
    loss_land_curv_lora_param_names: Optional[List[str]] = None

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
    relative_flatness: bool = False
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

    # ---------------- Perturbation curvature term (rank-one) ----------------
    curv_ts: bool = False
    curv_ts_backend: str = "ggn"  # "ggn" | "hessian" | "emp_fisher"
    curv_ts_rho: Optional[float] = None
    curv_ts_dir_max_batches: Optional[int] = None
    curv_ts_eps: float = 1e-12

    # ---------------- Noise covariance alignment (top-k) ----------------
    curv_noise: bool = False
    curv_noise_backend: str = "ggn"  # "ggn" | "hessian" | "emp_fisher"
    curv_noise_eig_method: str = "lanczos"  # "power" | "lanczos"
    curv_noise_topk: Optional[int] = None
    curv_noise_max_batches: Optional[int] = None
    curv_noise_eps: float = 1e-12

    # ---------------- Delta-W projection (weight-space LoRA subspace) ----------------
    delta_w_projection: bool = False
    delta_w_param_name: Optional[str] = None
    delta_w_block_index: Optional[int] = None
    delta_w_blocks: Optional[List[Dict[str, Any]]] = None
    delta_w_topk: int = 6
    delta_w_rank: Optional[int] = None
    delta_w_backend: Optional[str] = None
    delta_w_eig_method: str = "lanczos"  # "power" | "lanczos"
    delta_w_save_tensors: bool = False
    delta_w_svd_eps: float = 1e-12

    # ---------------- Delta-W full projection (param -> effective weight) ----------------
    delta_w_full_projection: bool = False
    delta_w_full_param_name: Optional[str] = None
    delta_w_full_block_index: Optional[int] = None
    delta_w_full_blocks: Optional[List[Dict[str, Any]]] = None
    delta_w_full_topk: Optional[int] = None
    delta_w_full_rank: Optional[int] = None
    delta_w_full_backend: Optional[str] = None
    delta_w_full_eig_method: Optional[str] = None
    delta_w_full_save_tensors: Optional[bool] = None
    delta_w_full_svd_eps: Optional[float] = None

    # ---------------- W vs Delta-W alignment (LoRA paper 7.3-style) ----------------
    w_delta_alignment: bool = False
    w_delta_param_name: Optional[str] = None
    w_delta_block_index: Optional[int] = None
    w_delta_blocks: Optional[List[Dict[str, Any]]] = None
    w_delta_rank: Optional[int] = None
    w_delta_seed: int = 42
    w_delta_save_tensors: bool = False
    w_delta_eps: float = 1e-12

    # ---------------- Mean drift (LoRA update displacement) ----------------
    mean_drift: bool = False
    mean_drift_param_name: Optional[str] = None
    mean_drift_block_index: Optional[int] = None
    mean_drift_blocks: Optional[List[Dict[str, Any]]] = None
    mean_drift_sigma2: float = 1.0
    mean_drift_eps: float = 1e-12
    mean_drift_effective_mode: Optional[str] = None

    # ---------------- LoRA KL (diag Fisher, taskwise) ----------------
    lora_kl: bool = False
    lora_kl_block_index: Optional[int] = None
    lora_kl_blocks: Optional[List[Dict[str, Any]]] = None
    lora_kl_lambda: float = 1.0
    lora_kl_damping: float = 1e-4
    lora_kl_eps: float = 1e-12
    lora_kl_max_batches: Optional[int] = None
    lora_kl_micro_bs: int = 1

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
            "loss_land_enabled": "loss_land_enabled",
            "weight_loss_land_1d": "weight_loss_land_1d",
            "weight_loss_land_2d": "weight_loss_land_2d",
            "weight_loss_land_radius": "weight_loss_land_radius",
            "weight_loss_land_num_points": "weight_loss_land_num_points",
            "weight_loss_land_max_batches": "weight_loss_land_max_batches",
            "weight_loss_land_filter_norm": "weight_loss_land_filter_norm",
            "loss_land_modes": "loss_land_modes",
            "loss_land_include_frozen": "loss_land_include_frozen",
            "loss_land_param_names": "loss_land_param_names",
            "loss_land_curv_1d": "loss_land_curv_1d",
            "loss_land_curv_backend": "loss_land_curv_backend",
            "loss_land_curv_method": "loss_land_curv_method",
            "loss_land_curv_iters": "loss_land_curv_iters",
            "loss_land_curv_topk": "loss_land_curv_topk",
            "loss_land_curv_num_points": "loss_land_curv_num_points",
            "loss_land_curv_radius_full": "loss_land_curv_radius_full",
            "loss_land_curv_radius_lora": "loss_land_curv_radius_lora",
            "loss_land_curv_max_batches": "loss_land_curv_max_batches",
            "loss_land_curv_use_abs_eig": "loss_land_curv_use_abs_eig",
            "loss_land_curv_normalize": "loss_land_curv_normalize",
            "loss_land_curv_lora_param_names": "loss_land_curv_lora_param_names",

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

            # perturbation curvature term (rank-one)
            "flat_eval_curv_ts": "curv_ts",
            "flat_eval_curv_ts_backend": "curv_ts_backend",
            "flat_eval_curv_ts_rho": "curv_ts_rho",
            "flat_eval_curv_ts_dir_max_batches": "curv_ts_dir_max_batches",
            "flat_eval_curv_ts_eps": "curv_ts_eps",

            # noise covariance alignment (top-k)
            "flat_eval_curv_noise": "curv_noise",
            "flat_eval_curv_noise_backend": "curv_noise_backend",
            "flat_eval_curv_noise_eig_method": "curv_noise_eig_method",
            "flat_eval_curv_noise_topk": "curv_noise_topk",
            "flat_eval_curv_noise_max_batches": "curv_noise_max_batches",
            "flat_eval_curv_noise_eps": "curv_noise_eps",

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

            "flat_eval_delta_w_full_projection": "delta_w_full_projection",
            "flat_eval_delta_w_full_param_name": "delta_w_full_param_name",
            "flat_eval_delta_w_full_block_index": "delta_w_full_block_index",
            "flat_eval_delta_w_full_blocks": "delta_w_full_blocks",
            "flat_eval_delta_w_full_topk": "delta_w_full_topk",
            "flat_eval_delta_w_full_rank": "delta_w_full_rank",
            "flat_eval_delta_w_full_backend": "delta_w_full_backend",
            "flat_eval_delta_w_full_eig_method": "delta_w_full_eig_method",
            "flat_eval_delta_w_full_save_tensors": "delta_w_full_save_tensors",
            "flat_eval_delta_w_full_svd_eps": "delta_w_full_svd_eps",

            # W vs Delta-W alignment (LoRA paper 7.3-style)
            "flat_eval_w_delta_alignment": "w_delta_alignment",
            "flat_eval_w_delta_param_name": "w_delta_param_name",
            "flat_eval_w_delta_block_index": "w_delta_block_index",
            "flat_eval_w_delta_blocks": "w_delta_blocks",
            "flat_eval_w_delta_rank": "w_delta_rank",
            "flat_eval_w_delta_seed": "w_delta_seed",
            "flat_eval_w_delta_save_tensors": "w_delta_save_tensors",
            "flat_eval_w_delta_eps": "w_delta_eps",

            # mean drift (LoRA update displacement)
            "flat_eval_mean_drift": "mean_drift",
            "flat_eval_mean_drift_param_name": "mean_drift_param_name",
            "flat_eval_mean_drift_block_index": "mean_drift_block_index",
            "flat_eval_mean_drift_blocks": "mean_drift_blocks",
            "flat_eval_mean_drift_sigma2": "mean_drift_sigma2",
            "flat_eval_mean_drift_eps": "mean_drift_eps",
            "flat_eval_mean_drift_effective_mode": "mean_drift_effective_mode",

            # LoRA KL (diag Fisher)
            "flat_eval_lora_kl": "lora_kl",
            "flat_eval_lora_kl_block_index": "lora_kl_block_index",
            "flat_eval_lora_kl_blocks": "lora_kl_blocks",
            "flat_eval_lora_kl_lambda": "lora_kl_lambda",
            "flat_eval_lora_kl_damping": "lora_kl_damping",
            "flat_eval_lora_kl_eps": "lora_kl_eps",
            "flat_eval_lora_kl_max_batches": "lora_kl_max_batches",
            "flat_eval_lora_kl_micro_bs": "lora_kl_micro_bs",

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
    params_override: Optional[List[torch.nn.Parameter]] = None,
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
    # Allow caller to inject a specific parameter list (e.g., LoRA-only) to avoid
    # re-selecting by substrings; falls back to name-based selection.
    params = params_override or _select_params_by_name(
        wrapped_model, substrs, include_frozen=include_frozen
    )
    names_for_params, shapes_for_params, splits_for_params = _param_names_and_shapes(wrapped_model, params)

    if not params:
        # 没有匹配到参数，就至少返回一个 base loss，避免崩
        return {
            "base_loss": float(_compute_loss(wrapped_model, loader, device, max_batches=config.loss_eval_max_batches))
        }
    
    if not params:
        return {"base_loss": 0.0}

    prev_max_examples = _get_max_examples_per_batch()
    _set_max_examples_per_batch(config.max_examples_per_batch)
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
        metrics_json_path = None
        if getattr(config, "save_metrics_path", None):
            metrics_json_path = os.path.join(
                config.save_metrics_path,
                f"{getattr(config, 'save_prefix', 'flatness')}_metrics.json",
            )

        disable_power = bool(getattr(config, "disable_power", False))
        ggn_vecs_power = None
        ef_vecs_power = None
        ef_vals_power = None
        if bool(getattr(config, "eval_sharpness", True)):
            logging.info("[FlatEval] Enable sharpness eval")
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
            if False:
                gaussian_std = config.esh_gaussian_std or (config.sharpness_radius)
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
            logging.info("[FlatEval] Done sharpness eval (metrics_json=%s)", metrics_json_path)
        else:
            logging.info("[FlatEval] Skip sharpness eval (disabled)")

        
        if bool(getattr(config, "eval_hessian", False)):
            logging.info("[FlatEval] Enable Hessian eval")
            if want_vecs:
                if not disable_power:
                    was_training = wrapped_model.training
                    wrapped_model.eval()
                    try:
                        res = _power_iteration_lambda_max(
                            _mvp_hessian,
                            total_dim,
                            config.hessian_power_iters,
                            device,
                            tol=getattr(config, "eig_tol", 1e-2),
                            patience=getattr(config, "eig_patience", 2),
                            return_vec=True,
                            topk=eig_topk_req,
                            seed=getattr(config, "loss_land_seed", None),
                        )
                    finally:
                        if was_training:
                            wrapped_model.train()
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
                    was_training = wrapped_model.training
                    wrapped_model.eval()
                    try:
                        lambda_max_power = _power_iteration_lambda_max(
                            _mvp_hessian,
                            total_dim,
                            config.hessian_power_iters,
                            device,
                            tol=1e-3,
                            patience=int(getattr(config, "eig_patience", 2)),
                        )
                    finally:
                        if was_training:
                            wrapped_model.train()
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
                was_training = wrapped_model.training
                wrapped_model.eval()
                try:
                    lam_l, v_l = _lanczos_lambda_max(
                        _mvp_hessian,
                        total_dim,
                        config.hessian_power_iters,
                        device,
                        tol=1e-3,
                        reorth=False,
                        patience=getattr(config, "eig_patience", 2),
                        return_vec=True,
                    )
                finally:
                    if was_training:
                        wrapped_model.train()
                lambda_max_lanczos = float(lam_l)
                v_lanczos = v_l.detach().cpu()
            else:
                was_training = wrapped_model.training
                wrapped_model.eval()
                try:
                    lambda_max_lanczos = _lanczos_lambda_max(
                        _mvp_hessian,
                        total_dim,
                        config.hessian_power_iters,
                        device,
                        reorth=True,
                    )
                finally:
                    if was_training:
                        wrapped_model.train()
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
            h_paths = []
            for key in ("eigvecs_hessian_power_path", "eigvecs_hessian_lanczos_path"):
                if key in flat_metrics:
                    h_paths.append(flat_metrics[key])
            logging.info("[FlatEval] Done Hessian eval (metrics_json=%s, paths=%s)", metrics_json_path, h_paths)
        else:
            logging.info("[FlatEval] Skip Hessian eval (disabled)")
    
        
        if bool(getattr(config, "eval_ggn", False)):
            logging.info("[FlatEval] Enable GGN eval")
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
            g_paths = []
            for key in ("eigvecs_ggn_power_path", "eigvecs_ggn_lanczos_path"):
                if key in flat_metrics:
                    g_paths.append(flat_metrics[key])
            logging.info("[FlatEval] Done GGN eval (metrics_json=%s, paths=%s)", metrics_json_path, g_paths)
        else:
            logging.info("[FlatEval] Skip GGN eval (disabled)")
    
        if bool(getattr(config, "eval_fisher", True)):
            logging.info("[FlatEval] Enable empirical Fisher eval")
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
            ef_paths = []
            for key in ("eigvecs_emp_fisher_power_path", "eigvecs_emp_fisher_lanczos_path"):
                if key in flat_metrics:
                    ef_paths.append(flat_metrics[key])
            logging.info("[FlatEval] Done empirical Fisher eval (metrics_json=%s, paths=%s)", metrics_json_path, ef_paths)
        else:
            logging.info("[FlatEval] Skip empirical Fisher eval (disabled)")
    
        if bool(getattr(config, "curv_localization", False)):
            logging.info("[FlatEval] Enable curvature localization eval")
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
            mvp_map = _default_mvp_map(_mvp_hessian, _mvp_ggn, _mvp_emp_fisher)
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
            logging.info("[FlatEval] Done curvature localization eval (metrics_json=%s)", metrics_json_path)
        else:
            logging.info("[FlatEval] Skip curvature localization eval (disabled)")

        if bool(getattr(config, "curv_ts", False)):
            logging.info("[FlatEval] Enable perturbation curvature term eval")
            mvp_map = _default_mvp_map(_mvp_hessian, _mvp_ggn, _mvp_emp_fisher)
            try:
                ts_metrics = _curv_ts_eval(
                    wrapped_model,
                    loader,
                    device,
                    params,
                    total_dim,
                    mvp_map,
                    config,
                    save_prefix=getattr(config, "save_prefix", "flatness"),
                    known_classes=known_classes,
                )
                if ts_metrics:
                    flat_metrics.update(ts_metrics)
                logging.info("[FlatEval] Done perturbation curvature term eval (metrics_json=%s)", metrics_json_path)
            except Exception:
                logging.exception("[FlatEval] Perturbation curvature term eval failed")
        else:
            logging.info("[FlatEval] Skip perturbation curvature term eval (disabled)")

        if bool(getattr(config, "curv_noise", False)):
            logging.info("[FlatEval] Enable noise covariance alignment eval")
            mvp_map = _default_mvp_map(_mvp_hessian, _mvp_ggn, _mvp_emp_fisher)
            try:
                noise_metrics = _curv_noise_cov_eval(
                    wrapped_model,
                    loader,
                    device,
                    params,
                    total_dim,
                    mvp_map,
                    config,
                    save_prefix=getattr(config, "save_prefix", "flatness"),
                    known_classes=known_classes,
                )
                if noise_metrics:
                    flat_metrics.update(noise_metrics)
                logging.info("[FlatEval] Done noise covariance alignment eval (metrics_json=%s)", metrics_json_path)
            except Exception:
                logging.exception("[FlatEval] Noise covariance alignment eval failed")
        else:
            logging.info("[FlatEval] Skip noise covariance alignment eval (disabled)")

        # -------- Delta-W projection (weight-space LoRA subspace) --------
        if bool(getattr(config, "delta_w_projection", False)):
            logging.info("[FlatEval] Enable Delta-W projection eval")
            mvp_map = _default_mvp_map(_mvp_hessian, _mvp_ggn, _mvp_emp_fisher)
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
                delta_paths = []
                if delta_metrics:
                    delta_paths = [v for k, v in delta_metrics.items() if str(k).endswith("_path")]
                logging.info("[FlatEval] Done Delta-W projection eval (metrics_json=%s, paths=%s)", metrics_json_path, delta_paths)
            except Exception:
                logging.exception("[FlatEval] Delta-W projection failed")
        else:
            logging.info("[FlatEval] Skip Delta-W projection eval (disabled)")

        if bool(getattr(config, "delta_w_full_projection", False)):
            logging.info("[FlatEval] Enable Delta-W full projection eval")
            mvp_map = _default_mvp_map(_mvp_hessian, _mvp_ggn, _mvp_emp_fisher)
            save_dir = getattr(config, "save_metrics_path", None)
            prefix = getattr(config, "save_prefix", "flatness")
            try:
                full_metrics = _delta_w_full_projection_eval(
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
                if full_metrics:
                    flat_metrics.update(full_metrics)
                full_paths = []
                if full_metrics:
                    full_paths = [v for k, v in full_metrics.items() if str(k).endswith("_path")]
                logging.info("[FlatEval] Done Delta-W full projection eval (metrics_json=%s, paths=%s)", metrics_json_path, full_paths)
            except Exception:
                logging.exception("[FlatEval] Delta-W full projection failed")
        else:
            logging.info("[FlatEval] Skip Delta-W full projection eval (disabled)")

        if bool(getattr(config, "w_delta_alignment", False)):
            logging.info("[FlatEval] Enable W-DeltaW alignment eval")
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
                w_paths = []
                if w_metrics:
                    w_paths = [v for k, v in w_metrics.items() if str(k).endswith("_path")]
                logging.info("[FlatEval] Done W-DeltaW alignment eval (metrics_json=%s, paths=%s)", metrics_json_path, w_paths)
            except Exception:
                logging.exception("[FlatEval] W-DeltaW alignment failed")
        else:
            logging.info("[FlatEval] Skip W-DeltaW alignment eval (disabled)")

        if bool(getattr(config, "mean_drift", False)):
            logging.info("[FlatEval] Enable mean drift eval")
            save_dir = getattr(config, "save_metrics_path", None)
            prefix = getattr(config, "save_prefix", "flatness")
            try:
                drift_metrics = _mean_drift_eval(
                    wrapped_model,
                    config,
                    save_dir=save_dir,
                    save_prefix=prefix,
                )
                if drift_metrics:
                    flat_metrics.update(drift_metrics)
                logging.info("[FlatEval] Done mean drift eval (metrics_json=%s)", metrics_json_path)
            except Exception:
                logging.exception("[FlatEval] Mean drift eval failed")
        else:
            logging.info("[FlatEval] Skip mean drift eval (disabled)")

        if bool(getattr(config, "lora_kl", False)):
            logging.info("[FlatEval] Enable LoRA KL eval")
            try:
                kl_metrics = _lora_kl_eval(
                    wrapped_model,
                    loader,
                    device,
                    config,
                    save_prefix=getattr(config, "save_prefix", "flatness"),
                    known_classes=known_classes,
                )
                if kl_metrics:
                    flat_metrics.update(kl_metrics)
                logging.info("[FlatEval] Done LoRA KL eval (metrics_json=%s)", metrics_json_path)
            except Exception:
                logging.exception("[FlatEval] LoRA KL eval failed")
        else:
            logging.info("[FlatEval] Skip LoRA KL eval (disabled)")

        # Loss landscape slices (optional)
        # ---------- Loss landscape：由单一参数 loss_land_modes 控制 ----------
        if bool(getattr(config, "loss_land_enabled", False)):
            do_1d = bool(getattr(config, "weight_loss_land_1d", False))
            do_2d = bool(getattr(config, "weight_loss_land_2d", False))
            do_curv_1d = bool(getattr(config, "loss_land_curv_1d", False))
            if do_1d or do_2d or do_curv_1d:
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
                rng_seed = getattr(config, "loss_land_seed", None)

                # Curvature 1D slices (full vs lora) along max-eigen directions
                if do_curv_1d:
                    curv_backend = str(getattr(config, "loss_land_curv_backend", "emp_fisher")).lower()
                    if curv_backend not in {"hessian", "ggn", "emp_fisher"}:
                        curv_backend = "emp_fisher"
                    curv_method = str(getattr(config, "loss_land_curv_method", "power")).lower()
                    if curv_method not in {"power", "lanczos"}:
                        curv_method = "power"
                    curv_iters = int(
                        getattr(config, "loss_land_curv_iters", None)
                        or getattr(config, "hessian_power_iters", 5)
                        or 5
                    )
                    curv_topk = int(getattr(config, "loss_land_curv_topk", 1))
                    curv_points = int(getattr(config, "loss_land_curv_num_points", None) or loss_points)
                    curv_max_batches = getattr(config, "loss_land_curv_max_batches", None)
                    if curv_max_batches is None:
                        curv_max_batches = loss_batches
                    if curv_max_batches is None:
                        curv_max_batches = getattr(config, "loss_eval_max_batches", None)
                    curv_radius_full = getattr(config, "loss_land_curv_radius_full", None)
                    if curv_radius_full is None:
                        curv_radius_full = loss_radius
                    curv_radius_lora = getattr(config, "loss_land_curv_radius_lora", None)
                    if curv_radius_lora is None:
                        curv_radius_lora = loss_radius
                    curv_use_abs = bool(getattr(config, "loss_land_curv_use_abs_eig", False))
                    curv_norm = bool(getattr(config, "loss_land_curv_normalize", False))

                    def _collect_full_params_with_grad():
                        include_frozen = bool(getattr(config, "loss_land_include_frozen", True))
                        substrs_full = getattr(config, "loss_land_param_names", None)
                        selected = []
                        saved = []
                        for n, p in wrapped_model.named_parameters():
                            if (not include_frozen) and (not p.requires_grad):
                                continue
                            if (substrs_full is None) or any(s in n for s in substrs_full):
                                if include_frozen and not p.requires_grad:
                                    saved.append((p, bool(p.requires_grad)))
                                    p.requires_grad_(True)
                                selected.append(p)
                        return selected, saved

                    full_params, full_saved = _collect_full_params_with_grad()
                    lora_params = params
                    lora_substrs = getattr(config, "loss_land_curv_lora_param_names", None)
                    if isinstance(lora_substrs, str):
                        if lora_substrs.lower() in {"none", "all", ""}:
                            lora_substrs = None
                        else:
                            lora_substrs = [lora_substrs]
                    if lora_substrs is not None:
                        lora_params = _select_params_by_name(
                            wrapped_model, lora_substrs, include_frozen=False
                        )

                    if not lora_params or not full_params:
                        logging.info("[FlatEval] Skip loss_land_curv_1d (empty params)")
                    else:
                        def _mvp_hessian_full(v: torch.Tensor) -> torch.Tensor:
                            return _hessian_vector_product(
                                wrapped_model,
                                loader,
                                device,
                                full_params,
                                v,
                                max_batches=curv_max_batches,
                                known_classes=known_classes,
                            )

                        def _mvp_hessian_lora(v: torch.Tensor) -> torch.Tensor:
                            return _hessian_vector_product(
                                wrapped_model,
                                loader,
                                device,
                                lora_params,
                                v,
                                max_batches=curv_max_batches,
                                known_classes=known_classes,
                            )

                        def _mvp_ggn_full(v: torch.Tensor) -> torch.Tensor:
                            return _ggn_vector_product(
                                wrapped_model,
                                loader,
                                device,
                                full_params,
                                v,
                                loss_eval_max_batches=curv_max_batches,
                                known_classes=known_classes,
                            )

                        def _mvp_ggn_lora(v: torch.Tensor) -> torch.Tensor:
                            return _ggn_vector_product(
                                wrapped_model,
                                loader,
                                device,
                                lora_params,
                                v,
                                loss_eval_max_batches=curv_max_batches,
                                known_classes=known_classes,
                            )

                        def _mvp_emp_fisher_full(v: torch.Tensor) -> torch.Tensor:
                            return _empirical_fisher_vector_product(
                                wrapped_model,
                                loader,
                                device,
                                full_params,
                                v,
                                max_batches=curv_max_batches,
                                known_classes=known_classes,
                            )

                        def _mvp_emp_fisher_lora(v: torch.Tensor) -> torch.Tensor:
                            return _empirical_fisher_vector_product(
                                wrapped_model,
                                loader,
                                device,
                                lora_params,
                                v,
                                max_batches=curv_max_batches,
                                known_classes=known_classes,
                            )

                        mvp_full_map = {
                            "hessian": _mvp_hessian_full,
                            "ggn": _mvp_ggn_full,
                            "emp_fisher": _mvp_emp_fisher_full,
                        }
                        mvp_lora_map = {
                            "hessian": _mvp_hessian_lora,
                            "ggn": _mvp_ggn_lora,
                            "emp_fisher": _mvp_emp_fisher_lora,
                        }

                        try:
                            crit = nn.CrossEntropyLoss(reduction="mean")
                            curves = compute_full_vs_lora_curvature_1d(
                                model=wrapped_model,
                                loader=loader,
                                device=device,
                                params_full=full_params,
                                params_lora=lora_params,
                                mvp_full=mvp_full_map[curv_backend],
                                mvp_lora=mvp_lora_map[curv_backend],
                                curvature_method=curv_method,
                                num_iters=curv_iters,
                                topk=curv_topk,
                                radius_full=float(curv_radius_full),
                                radius_lora=float(curv_radius_lora),
                                num_points=curv_points,
                                max_batches=curv_max_batches,
                                known_classes=known_classes,
                                seed=int(rng_seed) if rng_seed is not None else None,
                                tol=getattr(config, "eig_tol", None),
                                patience=int(getattr(config, "eig_patience", 2)),
                                use_abs_eig=curv_use_abs,
                                normalize_dir=curv_norm,
                                base_loss=None,
                                criterion=crit,
                            )
                            if curves:
                                if "full" in curves:
                                    full_curve = curves["full"]
                                    flat_metrics["lossland_curv1d_full_min"] = float(np.min(full_curve["delta_loss"]))
                                    flat_metrics["lossland_curv1d_full_max"] = float(np.max(full_curve["delta_loss"]))
                                    if "eigval" in full_curve and len(full_curve["eigval"]) > 0:
                                        flat_metrics["lossland_curv1d_full_eig"] = float(full_curve["eigval"][0])
                                if "lora" in curves:
                                    lora_curve = curves["lora"]
                                    flat_metrics["lossland_curv1d_lora_min"] = float(np.min(lora_curve["delta_loss"]))
                                    flat_metrics["lossland_curv1d_lora_max"] = float(np.max(lora_curve["delta_loss"]))
                                    if "eigval" in lora_curve and len(lora_curve["eigval"]) > 0:
                                        flat_metrics["lossland_curv1d_lora_eig"] = float(lora_curve["eigval"][0])

                                if save_dir:
                                    os.makedirs(save_dir, exist_ok=True)
                                    out_path = os.path.join(save_dir, f"{prefix}_lossland_curv1d.npz")
                                    curve_base_loss = float(base_loss)
                                    if "full" in curves and "base_loss" in curves["full"]:
                                        curve_base_loss = float(curves["full"]["base_loss"])
                                    elif "lora" in curves and "base_loss" in curves["lora"]:
                                        curve_base_loss = float(curves["lora"]["base_loss"])
                                    payload = {"base_loss": np.array([curve_base_loss], dtype=np.float64)}
                                    if "full" in curves:
                                        payload.update(
                                            {
                                                "x_full": curves["full"]["x"],
                                                "loss_full": curves["full"]["loss"],
                                                "delta_full": curves["full"]["delta_loss"],
                                                "eigval_full": curves["full"]["eigval"],
                                            }
                                        )
                                    if "lora" in curves:
                                        payload.update(
                                            {
                                                "x_lora": curves["lora"]["x"],
                                                "loss_lora": curves["lora"]["loss"],
                                                "delta_lora": curves["lora"]["delta_loss"],
                                                "eigval_lora": curves["lora"]["eigval"],
                                            }
                                        )
                                    np.savez_compressed(out_path, **payload)
                                    flat_metrics["lossland_curv1d_file"] = out_path
                                logging.info(
                                    "[FlatEval] Done loss_land_curv_1d (backend=%s, method=%s)",
                                    curv_backend,
                                    curv_method,
                                )
                        except Exception:
                            logging.exception("[FlatEval] loss_land_curv_1d failed")
                        finally:
                            for _p, _old in full_saved:
                                _p.requires_grad_(bool(_old))

                
                
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
            logging.info("[FlatEval] Enable relative flatness eval")
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
                if rf_power_iters and rf_power_iters > 0:
                    def _mvp_layer(v: torch.Tensor) -> torch.Tensor:
                        return _hessian_vector_product(
                            wrapped_model,
                            loader,
                            device,
                            layer_params,
                            v,
                            max_batches=rf_max_batches,
                            known_classes=known_classes,
                        )
                    was_training = wrapped_model.training
                    wrapped_model.eval()
                    try:
                        lam_layer = _power_iteration_lambda_max(
                            _mvp_layer,
                            dim_layer,
                            int(max(0, rf_power_iters)),
                            device,
                        )
                    finally:
                        if was_training:
                            wrapped_model.train()
                else:
                    lam_layer = 0.0

                flat_metrics[f"rf_weight_norm_{scope_tag}_{mode_used}"] = float(wn)
                flat_metrics[f"rf_trace_{scope_tag}"] = float(tr_layer)
                flat_metrics[f"rf_lambda_{scope_tag}"] = float(lam_layer)
                flat_metrics[f"relative_flatness_trace_{scope_tag}"] = float(wn * tr_layer)
                flat_metrics[f"relative_flatness_lambda_{scope_tag}"] = float(wn * lam_layer)
            logging.info("[FlatEval] Done relative flatness eval (metrics_json=%s)", metrics_json_path)
        else:
            logging.info("[FlatEval] Skip relative flatness eval (disabled)")

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
        _set_max_examples_per_batch(prev_max_examples)


# ---------------------------------------------------------------------------
# Optional CLI to evaluate flatness from a stored config/checkpoint
# ---------------------------------------------------------------------------


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
