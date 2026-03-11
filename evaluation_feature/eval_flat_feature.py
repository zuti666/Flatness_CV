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

from dataclasses import dataclass, field, fields as dc_fields, MISSING
from typing import Optional, Dict, Any, Tuple
import logging
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_manager import fractional_loader

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


def _build_fractional_seen_loader(data_manager, start_seen: int, end_seen: int, args: Dict[str, Any]):
    """Build a fractional loader over seen classes (train/test source, train mode)."""
    data_source = str(args.get("flat_eval_data_source", "train")).lower()
    if data_source not in {"train", "test"}:
        logging.warning(
            "[FeatureFlat] Unknown flat_eval_data_source=%s, fallback to train", data_source
        )
        data_source = "train"
    data_mode = "train"
    dataset_seen = data_manager.get_dataset(
        np.arange(start_seen, end_seen),
        source=data_source,
        mode=data_mode,
    )
    loader_seen = DataLoader(
        dataset_seen,
        batch_size=args.get("flat_eval_batch_size", 32),
        shuffle=True,
        num_workers=0,
    )
    return fractional_loader(
        loader=loader_seen,
        fraction=args.get("flat_eval_dataset_fraction", 0.1),
        seed=args.get("flat_eval_dataset_fraction_seed", args.get("seed", 42)),
        balanced=True,
        batch_size=args.get("flat_eval_batch_size", 32),
    )


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


# -----------------------------------------------------------------------------
# FIRST-vs-LAST feature comparison (CKA + prototype drift)
# -----------------------------------------------------------------------------
def compare_first_last_features(
    network: nn.Module,
    data_manager,
    class_ranges,
    task_idx: int,
    log_dir: str,
    base_stub: str,
    args: dict,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compare first-task features/prototypes with the last-task model.

    Saves CKA/prototype drift JSONs alongside cached anchor/prototype files.
    Returns a metrics dictionary (may be empty if disabled).
    """
    do_cka = bool(args.get("feature_cka_eval", False))
    do_proto = bool(args.get("feature_proto_eval", False))
    if not (do_cka or do_proto):
        return {}

    feature_dir = os.path.join(log_dir, "feature_flatness")
    os.makedirs(feature_dir, exist_ok=True)
    step_tag_last = f"t{task_idx:02d}"
    step_tag_first = "t00"

    anchor_first_path = os.path.join(feature_dir, f"{base_stub}_{step_tag_first}_anchors_seen.pt")
    anchor_last_path  = os.path.join(feature_dir, f"{base_stub}_{step_tag_last}_anchors_seen.pt")
    proto_first_path  = os.path.join(feature_dir, f"{base_stub}_{step_tag_first}_prototypes.pt")
    proto_last_path   = os.path.join(feature_dir, f"{base_stub}_{step_tag_last}_prototypes.pt")

    start0, end0 = class_ranges[0][0], class_ranges[0][1]
    device = torch.device(device)
    metrics: Dict[str, float] = {}

    # ---- CKA ----
    if do_cka and os.path.exists(anchor_first_path):
        first = torch.load(anchor_first_path, map_location="cpu")
        X0 = first.get("features", None)
        y0 = first.get("labels", None)
        classes0 = first.get("classes", (start0, end0))
        start0, end0 = int(classes0[0]), int(classes0[1])

        XT = yT = None
        if os.path.exists(anchor_last_path):
            last = torch.load(anchor_last_path, map_location="cpu")
            XT_all = last.get("features", None)
            yT_all = last.get("labels", None)
            if (XT_all is not None) and (yT_all is not None):
                mask = (yT_all >= start0) & (yT_all < end0)
                if mask.any():
                    XT = XT_all[mask]
                    yT = yT_all[mask]

        if XT is None or (XT is not None and XT.numel() == 0):
            seen_test_loader_first = _build_fractional_seen_loader(data_manager, start0, end0, args)
            anchor_max_batches = int(args.get("feature_cka_max_batches", 8))
            anchor_max_samples = int(args.get("feature_cka_max_samples", 2048))
            XT, yT = extract_features_and_labels(
                network,
                seen_test_loader_first,
                device,
                max_batches=anchor_max_batches,
                max_samples=anchor_max_samples,
            )

        if X0 is not None and XT is not None and X0.numel() > 0 and XT.numel() > 0:
            n_pairs = min(int(X0.shape[0]), int(XT.shape[0]))
            if n_pairs > 0:
                cka_val = linear_cka(X0[:n_pairs].to(device), XT[:n_pairs].to(device))
                cka_json = os.path.join(feature_dir, f"{base_stub}_{step_tag_last}_cka_seen_first.json")
                with open(cka_json, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "ref_step": 0,
                            "cur_step": int(task_idx),
                            "classes": [start0, end0],
                            "cka_stab_seen_first": float(cka_val),
                            "n_pairs": int(n_pairs),
                        },
                        fh,
                        indent=2,
                    )
                metrics["cka_first_last"] = float(cka_val)

    # ---- Prototype drift ----
    if do_proto and os.path.exists(proto_first_path):
        prot0 = counts0 = None
        first = torch.load(proto_first_path, map_location="cpu")
        prot0 = first.get("prototypes", None) or {}
        counts0 = first.get("counts", None) or {}
        classes0 = first.get("classes", (start0, end0))
        start0, end0 = int(classes0[0]), int(classes0[1])

        protT = countsT = None
        if os.path.exists(proto_last_path):
            last = torch.load(proto_last_path, map_location="cpu")
            prot_all = last.get("prototypes", None) or {}
            protT = {int(k): v for k, v in prot_all.items() if (int(k) >= start0 and int(k) < end0)}
        if protT is None or len(protT) == 0:
            seen_test_loader_first = _build_fractional_seen_loader(data_manager, start0, end0, args)
            proto_max_batches = int(args.get("feature_proto_max_batches",
                                            int(args.get("feature_cka_max_batches", 8))))
            proto_max_samples = int(args.get("feature_proto_max_samples",
                                            int(args.get("feature_cka_max_samples", 2048))))
            XTp, yTp = extract_features_and_labels(
                network,
                seen_test_loader_first,
                device,
                max_batches=proto_max_batches,
                max_samples=proto_max_samples,
            )
            protT, countsT = {}, {}
            if XTp is not None and XTp.numel() > 0:
                for cls in torch.unique(yTp).tolist():
                    if int(cls) >= start0 and int(cls) < end0:
                        mask = yTp == cls
                        n = int(mask.sum().item())
                        if n > 0:
                            protT[int(cls)] = XTp[mask].mean(dim=0).cpu()
                            countsT[int(cls)] = n
            overwrite_last = bool(args.get("feature_overwrite_last", False))
            if len(protT) > 0 and (overwrite_last or not os.path.exists(proto_last_path)):
                torch.save(
                    {"prototypes": protT, "counts": countsT, "classes": (start0, end0)},
                    proto_last_path,
                )

        if protT is not None and len(protT) > 0 and prot0 is not None and len(prot0) > 0:
            l2_list, cos_list = [], []
            for cls, mu0 in prot0.items():
                cls = int(cls)
                if cls in protT:
                    v0 = mu0.to(torch.float32)
                    vT = protT[cls].to(torch.float32)
                    l2_list.append(torch.norm(v0 - vT, p=2).item())
                    cos_list.append(1.0 - float(torch.nn.functional.cosine_similarity(
                        v0.unsqueeze(0), vT.unsqueeze(0)).item()))
            drift_json = os.path.join(feature_dir, f"{base_stub}_{step_tag_last}_prototype_drift_first.json")
            mean_l2 = float(np.mean(l2_list)) if len(l2_list) > 0 else None
            mean_cos = float(np.mean(cos_list)) if len(cos_list) > 0 else None
            with open(drift_json, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "ref_step": 0,
                        "cur_step": int(task_idx),
                        "classes": [start0, end0],
                        "drift_l2": l2_list,
                        "drift_cos": cos_list,
                        "drift_l2_mean": mean_l2,
                        "drift_cos_mean": mean_cos,
                        "n_classes": len(l2_list),
                    },
                    fh,
                    indent=2,
                )
            metrics["proto_drift_l2_mean"] = mean_l2 if mean_l2 is not None else float("nan")
            metrics["proto_drift_cos_mean"] = mean_cos if mean_cos is not None else float("nan")

    return metrics


def compare_task_last_features(
    network: nn.Module,
    data_manager,
    class_ranges,
    ref_task_idx: int,
    last_task_idx: int,
    log_dir: str,
    base_stub: str,
    args: dict,
    device: torch.device,
) -> Dict[str, float]:
    """Compare a reference task cache against the last-task model.

    This generalizes compare_first_last_features by letting the reference task
    index vary. It expects per-task caches saved via save_task_feature_cache.
    """
    do_cka = bool(args.get("feature_cka_eval", False))
    do_proto = bool(args.get("feature_proto_eval", False))
    if not (do_cka or do_proto):
        return {}

    feature_dir = os.path.join(log_dir, "feature_flatness")
    os.makedirs(feature_dir, exist_ok=True)
    step_tag_ref = f"t{ref_task_idx:02d}"
    step_tag_last = f"t{last_task_idx:02d}"

    anchor_ref_path = os.path.join(feature_dir, f"{base_stub}_{step_tag_ref}_anchors_seen.pt")
    anchor_last_path = os.path.join(feature_dir, f"{base_stub}_{step_tag_last}_anchors_seen.pt")
    proto_ref_path = os.path.join(feature_dir, f"{base_stub}_{step_tag_ref}_prototypes.pt")
    proto_last_path = os.path.join(feature_dir, f"{base_stub}_{step_tag_last}_prototypes.pt")

    start_ref, end_ref = class_ranges[ref_task_idx][0], class_ranges[ref_task_idx][1]
    device = torch.device(device)
    metrics: Dict[str, float] = {}

    # ---- CKA ----
    if do_cka and os.path.exists(anchor_ref_path):
        ref = torch.load(anchor_ref_path, map_location="cpu")
        X0 = ref.get("features", None)
        y0 = ref.get("labels", None)
        classes0 = ref.get("classes", (start_ref, end_ref))
        start_ref, end_ref = int(classes0[0]), int(classes0[1])

        XT = yT = None
        if os.path.exists(anchor_last_path):
            last = torch.load(anchor_last_path, map_location="cpu")
            XT_all = last.get("features", None)
            yT_all = last.get("labels", None)
            if (XT_all is not None) and (yT_all is not None):
                mask = (yT_all >= start_ref) & (yT_all < end_ref)
                if mask.any():
                    XT = XT_all[mask]
                    yT = yT_all[mask]

        if XT is None or (XT is not None and XT.numel() == 0):
            loader_ref = _build_fractional_seen_loader(data_manager, start_ref, end_ref, args)
            anchor_max_batches = int(args.get("feature_cka_max_batches", 8))
            anchor_max_samples = int(args.get("feature_cka_max_samples", 2048))
            XT, yT = extract_features_and_labels(
                network,
                loader_ref,
                device,
                max_batches=anchor_max_batches,
                max_samples=anchor_max_samples,
            )

        if X0 is not None and XT is not None and X0.numel() > 0 and XT.numel() > 0:
            n_pairs = min(int(X0.shape[0]), int(XT.shape[0]))
            if n_pairs > 0:
                cka_val = linear_cka(X0[:n_pairs].to(device), XT[:n_pairs].to(device))
                cka_json = os.path.join(
                    feature_dir, f"{base_stub}_{step_tag_last}_cka_seen_t{ref_task_idx:02d}.json"
                )
                with open(cka_json, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "ref_step": int(ref_task_idx),
                            "cur_step": int(last_task_idx),
                            "classes": [start_ref, end_ref],
                            "cka_stab_seen_ref": float(cka_val),
                            "n_pairs": int(n_pairs),
                        },
                        fh,
                        indent=2,
                    )
                metrics["cka_ref_last"] = float(cka_val)

    # ---- Prototype drift ----
    if do_proto and os.path.exists(proto_ref_path):
        prot0 = counts0 = None
        ref = torch.load(proto_ref_path, map_location="cpu")
        prot0 = ref.get("prototypes", None) or {}
        counts0 = ref.get("counts", None) or {}
        classes0 = ref.get("classes", (start_ref, end_ref))
        start_ref, end_ref = int(classes0[0]), int(classes0[1])

        protT = countsT = None
        if os.path.exists(proto_last_path):
            last = torch.load(proto_last_path, map_location="cpu")
            prot_all = last.get("prototypes", None) or {}
            protT = {int(k): v for k, v in prot_all.items() if (int(k) >= start_ref and int(k) < end_ref)}
        if protT is None or len(protT) == 0:
            loader_ref = _build_fractional_seen_loader(data_manager, start_ref, end_ref, args)
            proto_max_batches = int(args.get("feature_proto_max_batches",
                                            int(args.get("feature_cka_max_batches", 8))))
            proto_max_samples = int(args.get("feature_proto_max_samples",
                                            int(args.get("feature_cka_max_samples", 2048))))
            XTp, yTp = extract_features_and_labels(
                network,
                loader_ref,
                device,
                max_batches=proto_max_batches,
                max_samples=proto_max_samples,
            )
            protT, countsT = {}, {}
            if XTp is not None and XTp.numel() > 0:
                for cls in torch.unique(yTp).tolist():
                    if int(cls) >= start_ref and int(cls) < end_ref:
                        mask = yTp == cls
                        n = int(mask.sum().item())
                        if n > 0:
                            protT[int(cls)] = XTp[mask].mean(dim=0).cpu()
                            countsT[int(cls)] = n
            overwrite_last = bool(args.get("feature_overwrite_last", False))
            if len(protT) > 0 and (overwrite_last or not os.path.exists(proto_last_path)):
                torch.save(
                    {"prototypes": protT, "counts": countsT, "classes": (start_ref, end_ref)},
                    proto_last_path,
                )

        if protT is not None and len(protT) > 0 and prot0 is not None and len(prot0) > 0:
            l2_list, cos_list = [], []
            for cls, mu0 in prot0.items():
                cls = int(cls)
                if cls in protT:
                    v0 = mu0.to(torch.float32)
                    vT = protT[cls].to(torch.float32)
                    l2_list.append(torch.norm(v0 - vT, p=2).item())
                    cos_list.append(1.0 - float(torch.nn.functional.cosine_similarity(
                        v0.unsqueeze(0), vT.unsqueeze(0)).item()))
            drift_json = os.path.join(
                feature_dir, f"{base_stub}_{step_tag_last}_prototype_drift_t{ref_task_idx:02d}.json"
            )
            mean_l2 = float(np.mean(l2_list)) if len(l2_list) > 0 else None
            mean_cos = float(np.mean(cos_list)) if len(cos_list) > 0 else None
            with open(drift_json, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "ref_step": int(ref_task_idx),
                        "cur_step": int(last_task_idx),
                        "classes": [start_ref, end_ref],
                        "drift_l2": l2_list,
                        "drift_cos": cos_list,
                        "drift_l2_mean": mean_l2,
                        "drift_cos_mean": mean_cos,
                        "n_classes": len(l2_list),
                    },
                    fh,
                    indent=2,
                )
            metrics["proto_drift_l2_mean"] = mean_l2 if mean_l2 is not None else float("nan")
            metrics["proto_drift_cos_mean"] = mean_cos if mean_cos is not None else float("nan")

    return metrics


def save_task_feature_cache(
    network: nn.Module,
    data_manager,
    class_ranges,
    task_idx: int,
    log_dir: str,
    base_stub: str,
    args: dict,
    device: torch.device,
    *,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Save per-task anchors/prototypes for later comparisons.

    This caches feature anchors (for CKA) and class prototypes for the current
    task using the *seen* class range [0, end_seen). The caller can decide which
    tasks to cache (e.g., first and last only).
    """
    do_cka = bool(args.get("feature_cka_eval", False))
    do_proto = bool(args.get("feature_proto_eval", False))
    if not (do_cka or do_proto):
        return {}

    feature_dir = os.path.join(log_dir, "feature_flatness")
    os.makedirs(feature_dir, exist_ok=True)
    step_tag = f"t{task_idx:02d}"
    save_prefix = f"{base_stub}_{step_tag}"

    anchor_path = os.path.join(feature_dir, f"{save_prefix}_anchors_seen.pt")
    proto_path = os.path.join(feature_dir, f"{save_prefix}_prototypes.pt")

    start_seen = class_ranges[0][0]
    end_seen = class_ranges[task_idx][1]

    device = torch.device(device)
    dataset = None
    seen_test_loader = None
    X_seen = None
    y_seen = None
    prot = None
    counts = None
    saved = {}

    try:
        seen_test_loader = _build_fractional_seen_loader(data_manager, start_seen, end_seen, args)

        if do_cka and (overwrite or not os.path.exists(anchor_path)):
            anchor_max_batches = int(args.get("feature_cka_max_batches", 8))
            anchor_max_samples = int(args.get("feature_cka_max_samples", 2048))
            X_seen, y_seen = extract_features_and_labels(
                network,
                seen_test_loader,
                device,
                max_batches=anchor_max_batches,
                max_samples=anchor_max_samples,
            )
            torch.save(
                {"features": X_seen.cpu(), "labels": y_seen.cpu(), "classes": (start_seen, end_seen)},
                anchor_path,
            )
            saved["anchors"] = anchor_path

        if do_proto and (overwrite or not os.path.exists(proto_path)):
            proto_max_batches = int(
                args.get("feature_proto_max_batches", int(args.get("feature_cka_max_batches", 8)))
            )
            proto_max_samples = int(
                args.get("feature_proto_max_samples", int(args.get("feature_cka_max_samples", 2048)))
            )
            need_resample = (
                X_seen is None
                or int(args.get("feature_cka_max_batches", 8)) != proto_max_batches
                or int(args.get("feature_cka_max_samples", 2048)) != proto_max_samples
            )
            if need_resample:
                X_seen, y_seen = extract_features_and_labels(
                    network,
                    seen_test_loader,
                    device,
                    max_batches=proto_max_batches,
                    max_samples=proto_max_samples,
                )
            if X_seen is not None and X_seen.numel() > 0:
                prot, counts = {}, {}
                for cls in torch.unique(y_seen).tolist():
                    mask = y_seen == cls
                    n = int(mask.sum().item())
                    if n > 0:
                        prot[int(cls)] = X_seen[mask].mean(dim=0).cpu()
                        counts[int(cls)] = n
                torch.save(
                    {"prototypes": prot, "counts": counts, "classes": (start_seen, end_seen)},
                    proto_path,
                )
                saved["prototypes"] = proto_path
    finally:
        seen_test_loader = None
        dataset = None
        X_seen = None
        y_seen = None
        prot = None
        counts = None

    return saved


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
