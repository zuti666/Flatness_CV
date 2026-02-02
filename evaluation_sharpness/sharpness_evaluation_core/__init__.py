"""Lightweight, reusable evaluation primitives.

This package exposes small, dependency-minimal helpers for curvature MVPs,
loss landscapes, probes, and drift metrics. Existing eval_flat wrappers can
import from here while keeping their current public API intact.
"""

from .curvature import make_mvp_map, build_mvp_fns, topk_eigs, estimate_trace
from .landscape import compute_loss_landscape, compute_full_vs_lora_curve
from .probe import fit_linear_probe, evaluate_linear_probe
from .drift import mean_drift_metrics, lora_kl_metrics

__all__ = [
    "make_mvp_map",
    "build_mvp_fns",
    "topk_eigs",
    "estimate_trace",
    "compute_loss_landscape",
    "compute_full_vs_lora_curve",
    "fit_linear_probe",
    "evaluate_linear_probe",
    "mean_drift_metrics",
    "lora_kl_metrics",
]
