"""
loss_tracker.py
---------------
Compute L_old / L_new / L_all (and optionally ΔL) at a task boundary.

Wraps the existing ``_compute_loss`` from
``evaluation_weight_sharpness.loss_utils`` — no new loss logic.

ΔL computation
--------------
If the caller supplies ``ref_params`` (a flat CPU tensor of θ_{t-1}), the
evaluator temporarily swaps model weights, computes L at the old params,
restores the current weights, and returns:

    ΔL_old  = L_old(θ_t)   - L_old(θ_{t-1})   should be small
    ΔL_new  = L_new(θ_{t-1}) - L_new(θ_t)     should be large (improvement)
"""
from __future__ import annotations

import logging
from typing import Optional, Dict

import torch
import torch.nn as nn

from evaluation_weight_sharpness.loss_utils import _compute_loss  # reuse, no change

logger = logging.getLogger(__name__)


def compute_tripled_loss(
    network: nn.Module,
    old_loader,
    new_loader,
    all_loader,
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    """
    Compute L_old, L_new, L_all using the existing _compute_loss helper.

    Parameters
    ----------
    network      : nn.Module  (the bare IncrementalNet, not the Learner wrapper)
    old_loader   : DataLoader | None
    new_loader   : DataLoader
    all_loader   : DataLoader
    device       : torch.device
    max_batches  : int  budget per loader

    Returns
    -------
    dict with keys L_old, L_new, L_all (float)
    """
    out: Dict[str, float] = {}

    if old_loader is not None:
        out["L_old"] = _compute_loss(network, old_loader, device, max_batches)
    else:
        out["L_old"] = float("nan")

    out["L_new"] = _compute_loss(network, new_loader, device, max_batches)
    out["L_all"] = _compute_loss(network, all_loader, device, max_batches)

    logger.debug(
        "[loss_tracker] L_old=%.4f  L_new=%.4f  L_all=%.4f",
        out["L_old"], out["L_new"], out["L_all"],
    )
    return out


def compute_delta_loss(
    network: nn.Module,
    old_loader,
    new_loader,
    device: torch.device,
    max_batches: int,
    ref_params_flat: Optional[torch.Tensor],
    exclude_prefix: Optional[str] = None,
) -> Dict[str, float]:
    """
    ΔL requires knowing L at θ_{t-1}.

    ref_params_flat : flat CPU tensor from _snapshot_params() below.
                      If None, returns empty dict (caller decides).
    """
    if ref_params_flat is None:
        return {}

    # Parameter selection must exactly match the snapshot convention.
    params = [
        p for name, p in network.named_parameters()
        if p.requires_grad and (exclude_prefix is None or not name.startswith(exclude_prefix))
    ]
    current_flat = _flatten_params(params).cpu().clone()

    # swap to θ_{t-1}
    _load_flat_params(params, ref_params_flat.to(device))

    out: Dict[str, float] = {}
    if old_loader is not None:
        l_old_prev = _compute_loss(network, old_loader, device, max_batches)
        out["delta_L_old_prev"] = float(l_old_prev)

    l_new_prev = _compute_loss(network, new_loader, device, max_batches)
    out["delta_L_new_prev"] = float(l_new_prev)

    # restore θ_t
    _load_flat_params(params, current_flat.to(device))

    return out


# ── internal helpers ──────────────────────────────────────────────────────────

def _flatten_params(params) -> torch.Tensor:
    return torch.cat([p.detach().cpu().reshape(-1) for p in params])


def _load_flat_params(params, flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(flat[offset: offset + numel].reshape_as(p))
        offset += numel


def snapshot_params(network: nn.Module, exclude_prefix: Optional[str] = "fc.") -> torch.Tensor:
    """
    Capture a lightweight CPU snapshot of trainable params.
    Call this BEFORE incremental_train() to get θ_{t-1}.
    Returns a flat float32 CPU tensor.

    exclude_prefix : skip params whose name starts with this prefix.
                     Pass None to snapshot all trainable parameters.
    """
    params = [
        p for name, p in network.named_parameters()
        if p.requires_grad and (exclude_prefix is None or not name.startswith(exclude_prefix))
    ]
    return _flatten_params(params)
