"""LoRA/mean-drift related metrics in a reusable form."""
from __future__ import annotations
from typing import Any, Dict
from evaluation_weight_sharpness.curv_localization import _mean_drift_eval, _lora_kl_eval


def mean_drift_metrics(*args, **kwargs) -> Dict[str, Any]:
    """Compute mean-drift statistics for parameter updates."""
    return _mean_drift_eval(*args, **kwargs) or {}


def lora_kl_metrics(*args, **kwargs) -> Dict[str, Any]:
    """Compute diagonal-Fisher KL metrics for LoRA blocks."""
    return _lora_kl_eval(*args, **kwargs) or {}
