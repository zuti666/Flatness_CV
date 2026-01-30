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
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataclasses import dataclass, field, fields as dc_fields, MISSING
from typing import Optional, Dict, Any
import torch

@dataclass
class FeatureFlatnessConfig:
    """Configuration knobs for empirical feature-matrix / EFM evaluation."""

    # ---- core knobs ----
    max_batches: Optional[int] = None
    topk_eigen: int = 5
    eps: float = 1e-12
    rank_tol: float = 1e-6

    # ---- persistence ----
    save_matrix_path: Optional[str] = None
    save_prefix: str = "feature"

    # ---- device override (optional) ----
    device_override: Optional[torch.device] = None

    # ---- data sub-sampling knobs (与 FlatnessConfig 对齐，默认禁用，防止二次抽样) ----
    max_examples_per_batch: Optional[int] = None

    # ---- accept args mapping ----
    args: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __post_init__(self):
        if not isinstance(self.args, dict):
            return

        # 收集默认值以判断“是否仍为默认、可被映射覆盖”
        defaults = {}
        for f in dc_fields(self):
            if f.name == "args":
                continue
            defaults[f.name] = (f.default if f.default is not MISSING else None)

        def _get_seed(a: Dict[str, Any]):
            s = a.get("seed", None)
            if isinstance(s, list) and len(s) > 0:
                try:
                    return int(s[0])
                except Exception:
                    return s[0]
            return s

        def _maybe_device(x):
            if x is None:
                return None
            if isinstance(x, torch.device):
                return x
            if isinstance(x, str):
                try:
                    return torch.device(x)
                except Exception:
                    return None
            return None

        # args 键 → dataclass 字段
        key_map = {
            "feature_flat_max_batches": "max_batches",
            "feature_flat_topk": "topk_eigen",
            "feature_flat_eps": "eps",
            "feature_flat_rank_tol": "rank_tol",

            # 持久化/输出
            "feature_flat_save_dir": "save_matrix_path",
            "feature_flat_save_prefix": "save_prefix",

        }

        # 1) 常规映射：仅当目标字段仍等于默认值时覆盖
        for src_key, dst in key_map.items():
            if src_key in self.args:
                current = getattr(self, dst)
                if current == defaults.get(dst):
                    val = self.args[src_key]
                    if dst == "device_override":
                        val = _maybe_device(val)
                    setattr(self, dst, val)

        
                



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

    # 
    module = _ensure_module(network)
    
    
    head_weight = _classifier_weights(module).detach()
    
    if head_weight.ndim != 2:
        raise ValueError("Classifier weight tensor must be 2-D (num_classes x feat_dim).")

    device = config.device_override or head_weight.device
    was_training = module.training
    module.eval()

    head_weight = head_weight.to(device=device, dtype=torch.float32)
    # Robustly coerce numeric configs that might be provided as strings in YAML
    try:
        probs_eps = float(config.eps)
    except Exception:
        probs_eps = 1e-12
    probs_eps = max(probs_eps, 0.0)
    max_batches = None if config.max_batches is None else int(config.max_batches)
    

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
    try:
        rank_tol = float(config.rank_tol)
    except Exception:
        rank_tol = 1e-6
    rank = int(torch.count_nonzero(evals > rank_tol).item())

    metrics: Dict[str, object] = {
        "trace": float(trace.item()),
        "spectral_radius": float(spectral_radius.item()),
        "frobenius_norm": float(frob.item()),
        "rank": rank,
        "num_samples": int(total_samples),
    }
    # Derived spectral measures
    # Effective rank reff = (sum λ)^2 / sum λ^2 (guard denom)
    s1 = trace
    s2 = torch.sum(evals * evals) + 1e-12
    effective_rank = float((s1 * s1 / s2).item())
    d = float(evals.numel()) if evals.ndim > 0 else 1.0
    # Anisotropy index AI = ρ(E) / (tr(E)/d)
    mean_var = float(trace.item()) / max(d, 1.0)
    anisotropy_index = float(spectral_radius.item()) / (mean_var + 1e-12)
    metrics["effective_rank"] = float(effective_rank)
    metrics["anisotropy_index"] = float(anisotropy_index)

    if config.topk_eigen > 0:
        topk = min(config.topk_eigen, evals.numel())
        metrics["top_eigenvalues"] = evals[-topk:].flip(0).detach().cpu().tolist()

    if config.save_matrix_path is not None:
        directory = config.save_matrix_path.rstrip("/")
        filename = f"{config.save_prefix}_efm.pt"
        torch.save({"efm": efm.cpu(), "eigenvalues": evals.cpu()}, f"{directory}/{filename}")

    module.train(was_training)
    return metrics


# ------------------------------
# Feature extraction & linear CKA
# ------------------------------

def _flatten_features(feat: torch.Tensor) -> torch.Tensor:
    if feat.dim() == 3:
        return feat[:, 0, ...]
    if feat.dim() > 2:
        return feat.view(feat.size(0), -1)
    return feat


@torch.no_grad()
def extract_features_and_labels(
    network: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract backbone features and labels for a loader.

    Expects network(inputs) -> {"features": ..., "logits": ...} or a tensor.
    Returns tensors on ``device`` with shapes [N, D] and [N].
    """
    module = _ensure_module(network)
    was_training = module.training
    module.eval()

    feats_list, lbls_list = [], []
    seen = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs, targets = _unwrap_batch(batch)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        out = module(inputs)
        feat = out["features"] if isinstance(out, dict) and "features" in out else out
        feat = _flatten_features(feat).to(device)
        feats_list.append(feat)
        lbls_list.append(targets)
        seen += feat.size(0)
        if max_samples is not None and seen >= max_samples:
            break
    if not feats_list:
        module.train(was_training)
        return torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    X = torch.cat(feats_list, dim=0)
    y = torch.cat(lbls_list, dim=0)
    if max_samples is not None and X.size(0) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]
    module.train(was_training)
    return X.float(), y.long()


def linear_cka(X: torch.Tensor, Y: torch.Tensor, center: bool = True, eps: float = 1e-12) -> float:
    """Compute linear CKA between two feature matrices with matching rows.

    X, Y: [N, D] tensors. Optionally mean-center each.
    Returns a scalar in [0, 1] (numerical noise may produce tiny negatives).
    """
    if X.numel() == 0 or Y.numel() == 0:
        return float("nan")
    n = min(X.size(0), Y.size(0))
    X = X[:n].float()
    Y = Y[:n].float()
    if center:
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
    K = X.T @ Y  # [D, D]
    num = torch.linalg.matrix_norm(K, ord="fro") ** 2
    XX = X.T @ X
    YY = Y.T @ Y
    den = (torch.linalg.matrix_norm(XX, ord="fro") * torch.linalg.matrix_norm(YY, ord="fro") + eps)
    val = (num / den).clamp(min=0.0, max=1.0)
    return float(val.item())


__all__ = [
    "FeatureFlatnessConfig",
    "evaluate_feature_metrics",
    "extract_features_and_labels",
    "linear_cka",
]
