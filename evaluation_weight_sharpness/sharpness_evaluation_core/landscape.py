"""Thin wrappers around loss-landscape utilities."""
from __future__ import annotations
from typing import Dict, Optional
import torch
from evaluation_weight_sharpness.loss_landscape import (
    compute_loss_landscape_v1,
    compute_full_vs_lora_curvature_1d,
)


def compute_loss_landscape(
    model: torch.nn.Module,
    loader,
    *,
    radius: float = 0.5,
    num_points: int = 21,
    max_batches: Optional[int] = None,
    filter_norm: bool = False,
    known_classes: Optional[int] = None,
) -> Dict[str, object]:
    """Convenience wrapper for the 1D random-direction loss sweep."""
    return compute_loss_landscape_v1(
        model,
        loader,
        model._device if hasattr(model, "_device") else next(model.parameters()).device,
        radius=radius,
        num_points=num_points,
        max_batches=max_batches,
        filter_norm=filter_norm,
        known_classes=known_classes,
    )


def compute_full_vs_lora_curve(
    model: torch.nn.Module,
    loader,
    *,
    backend: str = "emp_fisher",
    method: str = "power",
    topk: int = 1,
    num_points: Optional[int] = None,
    radius_full: Optional[float] = None,
    radius_lora: Optional[float] = None,
    max_batches: Optional[int] = None,
    normalize: bool = False,
    use_abs_eig: bool = False,
):
    """Wrap curvature-aligned 1D loss slicing for full vs LoRA params."""
    return compute_full_vs_lora_curvature_1d(
        model,
        loader,
        backend=backend,
        eig_method=method,
        topk=topk,
        num_points=num_points,
        radius_full=radius_full,
        radius_lora=radius_lora,
        max_batches=max_batches,
        normalize=normalize,
        use_abs_eig=use_abs_eig,
    )
