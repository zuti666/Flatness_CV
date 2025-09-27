"""Feature-space flatness metrics based on the empirical feature matrix (EFM).

This module implements the EFM diagnostics discussed in the user prompt.  For a
classifier with weights :math:`W` and softmax probabilities :math:`p(y|x)`, the
gradient of the log-probability w.r.t. the feature vector has the closed-form

.. math::

   g_y(x) = W_y - \sum_c p_c(x) W_c,

which removes the need for explicit Jacobian computation.  The per-sample EFM is
then ``E_f(x) = E_{y\sim p}[g_y(x) g_y(x)^\top]``.  Averaging this matrix over a
dataset yields spectral proxies (trace, spectral radius, Frobenius norm,
effective rank) that quantify feature-space flatness and can be correlated with
continual-learning metrics such as BWT/FG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class FeatureFlatnessConfig:
    """Configuration knobs for empirical feature matrix evaluation."""

    max_batches: Optional[int] = None
    topk_eigen: int = 5
    eps: float = 1e-12
    rank_tol: float = 1e-6
    save_matrix_path: Optional[str] = None
    save_prefix: str = "feature"
    device_override: Optional[torch.device] = None


def _unwrap_batch(batch):
    """Handle loaders that yield ``(index, inputs, targets)`` tuples."""

    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            _, inputs, targets = batch
            return inputs, targets
        if len(batch) == 2:
            return batch
    raise ValueError("Unexpected batch structure for feature-flatness evaluation")


def _ensure_module(network: nn.Module) -> nn.Module:
    """Unwrap DataParallel wrappers to access the underlying module."""

    return network.module if isinstance(network, nn.DataParallel) else network


def _classifier_weights(module: nn.Module) -> torch.Tensor:
    """Extract a dense classifier weight matrix.

    The helper covers the linear heads used in this repository (``SimpleLinear``,
    ``CosineLinear`` and ``SplitCosineLinear``).  Extend it if new heads are
    introduced.
    """

    head = getattr(module, "fc", None) or getattr(module, "classifier", None)
    if head is None:
        raise AttributeError("Classifier module not found (expected 'fc' or 'classifier').")

    if hasattr(head, "weight") and head.weight is not None:
        return head.weight

    if hasattr(head, "fc1") and hasattr(head, "fc2"):
        return torch.cat((head.fc1.weight, head.fc2.weight), dim=0)

    raise AttributeError("Unsupported classifier type for feature-flatness evaluation.")


def evaluate_feature_metrics(
    network: nn.Module,
    loader: DataLoader,
    *,
    config: Optional[FeatureFlatnessConfig] = None,
) -> Dict[str, object]:
    """Estimate feature-space flatness statistics via the empirical feature matrix.

    Parameters
    ----------
    network:
        Model exposing ``fc`` (or ``classifier``) with a linear head and a
        forward pass that returns a dict containing ``logits`` and ``features``.
    loader:
        DataLoader iterating over the current task's dataset :math:`\mathcal{X}_t`.
    config:
        Optional :class:`FeatureFlatnessConfig` controlling batching limits and
        reporting options.

    Returns
    -------
    dict
        A dictionary with the averaged EFM diagnostics (trace, spectral radius,
        Frobenius norm, rank, sample count and optionally top eigenvalues).
    """

    if config is None:
        config = FeatureFlatnessConfig()

    module = _ensure_module(network)
    head_weight = _classifier_weights(module).detach()
    if head_weight.ndim != 2:
        raise ValueError("Classifier weight tensor must be 2-D (num_classes x feat_dim).")

    device = config.device_override or head_weight.device
    was_training = module.training
    module.eval()

    head_weight = head_weight.to(device=device, dtype=torch.float32)
    probs_eps = max(config.eps, 0.0)
    max_batches = config.max_batches

    accumulated = torch.zeros(
        head_weight.shape[1], head_weight.shape[1], device=device, dtype=torch.float64
    )
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs, _ = _unwrap_batch(batch)
            inputs = inputs.to(device, non_blocking=True)

            outputs = module(inputs)
            if not isinstance(outputs, dict) or "logits" not in outputs or "features" not in outputs:
                raise ValueError(
                    "Network forward must return a dict containing 'logits' and 'features'."
                )

            logits = outputs["logits"].to(device=device, dtype=torch.float32)
            probs = torch.clamp(F.softmax(logits, dim=1), min=probs_eps)

            batch_size = logits.shape[0]
            total_samples += batch_size

            weight_expanded = head_weight.unsqueeze(0).expand(batch_size, -1, -1)
            weighted_mean = torch.matmul(probs, head_weight)
            centered = weight_expanded - weighted_mean.unsqueeze(1)

            sqrt_probs = torch.sqrt(probs).unsqueeze(2)
            weighted_centered = centered * sqrt_probs
            efm_batch = torch.matmul(weighted_centered.transpose(1, 2), weighted_centered)
            accumulated += efm_batch.sum(dim=0).double()

    if total_samples == 0:
        module.train(was_training)
        raise ValueError("Feature flatness evaluation received an empty loader.")

    efm = accumulated / float(total_samples)
    efm = 0.5 * (efm + efm.transpose(0, 1))

    evals = torch.linalg.eigvalsh(efm)
    evals = torch.clamp(evals, min=0.0)

    trace = torch.sum(evals)
    spectral_radius = torch.max(evals)
    frob = torch.linalg.matrix_norm(efm, ord="fro")
    rank = int(torch.count_nonzero(evals > config.rank_tol).item())

    metrics: Dict[str, object] = {
        "trace": float(trace.item()),
        "spectral_radius": float(spectral_radius.item()),
        "frobenius_norm": float(frob.item()),
        "rank": rank,
        "num_samples": int(total_samples),
    }

    if config.topk_eigen > 0:
        topk = min(config.topk_eigen, evals.numel())
        metrics["top_eigenvalues"] = evals[-topk:].flip(0).detach().cpu().tolist()

    if config.save_matrix_path is not None:
        directory = config.save_matrix_path.rstrip("/")
        filename = f"{config.save_prefix}_efm.pt"
        torch.save({"efm": efm.cpu(), "eigenvalues": evals.cpu()}, f"{directory}/{filename}")

    module.train(was_training)
    return metrics


__all__ = ["FeatureFlatnessConfig", "evaluate_feature_metrics"]
