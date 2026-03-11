"""Reusable linear probe helpers."""
from __future__ import annotations
from typing import Any, Dict
from evaluation_performance.probe import (
    fit_linear_probe_softmax_head,
    evaluate_linear_probe_softmax_with_head,
)


def fit_linear_probe(*args, **kwargs) -> Dict[str, Any]:
    """Train a linear probe head and return the fitted module/statistics."""
    return fit_linear_probe_softmax_head(*args, **kwargs)


def evaluate_linear_probe(*args, **kwargs) -> Dict[str, Any]:
    """Evaluate a pre-fit linear probe head."""
    return evaluate_linear_probe_softmax_with_head(*args, **kwargs)
