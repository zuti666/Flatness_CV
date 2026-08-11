"""
feature_tracker.py
------------------
Feature-side mechanism metrics for online CL evaluation.

This module keeps only lightweight reference artifacts at each task boundary:
  - anchor features for the just-learned task
  - per-class prototypes for the just-learned task

At later task boundaries it compares the current model against those cached
references to quantify per-old-task representation drift.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from evaluation_CL_mechanism.loader_factory import make_ref_loader
from evaluation_feature.eval_flat_feature import (
    FeatureFlatnessConfig,
    evaluate_feature_metrics,
    extract_features_and_labels,
    linear_cka,
)

logger = logging.getLogger(__name__)


def _ref_cache_dir(save_root: str) -> str:
    return os.path.join(save_root, "feature_refs")


def _ref_cache_path(save_root: str, task_idx: int) -> str:
    return os.path.join(_ref_cache_dir(save_root), f"task_{int(task_idx):02d}.pt")


def _build_feature_cfg(args: dict) -> FeatureFlatnessConfig:
    return FeatureFlatnessConfig(
        max_batches=int(args.get("cl_eval_feature_max_batches", args.get("cl_eval_max_batches", 8))),
        topk_eigen=int(args.get("cl_eval_feature_topk", 5)),
        rank_tol=float(args.get("cl_eval_feature_rank_tol", 1e-6)),
        eps=float(args.get("cl_eval_feature_eps", 1e-12)),
        args=None,
    )


def _extract_reference_payload(
    network,
    loader,
    device: torch.device,
    args: dict,
    task_idx: int,
) -> Dict[str, Any]:
    max_batches = int(args.get("cl_eval_feature_anchor_max_batches", args.get("cl_eval_feature_max_batches", 8)))
    max_samples = int(args.get("cl_eval_feature_anchor_max_samples", 512))
    X, y = extract_features_and_labels(
        network,
        loader,
        device,
        max_batches=max_batches,
        max_samples=max_samples,
    )

    prototypes = {}
    counts = {}
    if X is not None and y is not None and X.numel() > 0:
        for cls in torch.unique(y).tolist():
            mask = y == cls
            n = int(mask.sum().item())
            if n > 0:
                prototypes[int(cls)] = X[mask].mean(dim=0).cpu()
                counts[int(cls)] = n

    return {
        "task": int(task_idx),
        "features": X.detach().cpu(),
        "labels": y.detach().cpu(),
        "prototypes": prototypes,
        "counts": counts,
        "n_samples": int(X.shape[0]) if X is not None and X.ndim >= 1 else 0,
    }


def save_reference_task_features(
    network,
    data_manager,
    task_idx: int,
    save_root: str,
    device: torch.device,
    args: dict,
    *,
    overwrite: bool = False,
) -> Optional[str]:
    """Cache anchor features and prototypes for the current task."""
    path = _ref_cache_path(save_root, task_idx)
    if os.path.exists(path) and not overwrite:
        return path

    loader = make_ref_loader(data_manager, task_idx, args)
    payload = _extract_reference_payload(network, loader, device, args, task_idx)
    os.makedirs(_ref_cache_dir(save_root), exist_ok=True)
    torch.save(payload, path)
    logger.debug("[feature_tracker] saved task reference cache -> %s", path)
    return path


def _namespace_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in metrics.items():
        out[f"{prefix}_{key}"] = value
    return out


def compute_triplet_feature_flatness(
    network,
    old_loader,
    new_loader,
    all_loader,
    args: dict,
) -> Dict[str, Any]:
    """Compute EFM metrics on old/new/all loaders."""
    cfg = _build_feature_cfg(args)
    results: Dict[str, Any] = {}

    if old_loader is not None:
        try:
            results.update(_namespace_metrics("EFM_old", evaluate_feature_metrics(network, old_loader, config=cfg)))
        except Exception as exc:
            logger.warning("[feature_tracker] EFM_old failed: %s", exc)
            results["EFM_old_status"] = f"failed:{exc}"
    else:
        results["EFM_old_status"] = "skipped_no_old_loader"

    for prefix, loader in (("EFM_new", new_loader), ("EFM_all", all_loader)):
        try:
            results.update(_namespace_metrics(prefix, evaluate_feature_metrics(network, loader, config=cfg)))
        except Exception as exc:
            logger.warning("[feature_tracker] %s failed: %s", prefix, exc)
            results[f"{prefix}_status"] = f"failed:{exc}"

    return results


def _compare_current_to_reference(
    network,
    loader,
    ref_payload: Dict[str, Any],
    device: torch.device,
    args: dict,
) -> Dict[str, float]:
    max_batches = int(args.get("cl_eval_feature_anchor_max_batches", args.get("cl_eval_feature_max_batches", 8)))
    max_samples = int(args.get("cl_eval_feature_anchor_max_samples", 512))
    X_cur, y_cur = extract_features_and_labels(
        network,
        loader,
        device,
        max_batches=max_batches,
        max_samples=max_samples,
    )
    X_ref = ref_payload.get("features")
    y_ref = ref_payload.get("labels")

    out: Dict[str, float] = {
        "n_ref": int(X_ref.shape[0]) if X_ref is not None and X_ref.ndim >= 1 else 0,
        "n_cur": int(X_cur.shape[0]) if X_cur is not None and X_cur.ndim >= 1 else 0,
    }

    if X_ref is not None and X_cur is not None and X_ref.numel() > 0 and X_cur.numel() > 0:
        n_pairs = min(int(X_ref.shape[0]), int(X_cur.shape[0]))
        if n_pairs > 0:
            out["cka"] = float(linear_cka(X_ref[:n_pairs].to(device), X_cur[:n_pairs].to(device)))
            out["n_pairs"] = int(n_pairs)

    prot_ref = ref_payload.get("prototypes", {}) or {}
    prot_cur = {}
    if X_cur is not None and y_cur is not None and X_cur.numel() > 0:
        for cls in torch.unique(y_cur).tolist():
            mask = y_cur == cls
            if mask.any():
                prot_cur[int(cls)] = X_cur[mask].mean(dim=0).cpu()

    l2_list = []
    cos_list = []
    for cls, mu_ref in prot_ref.items():
        cls = int(cls)
        if cls not in prot_cur:
            continue
        v0 = mu_ref.to(torch.float32)
        v1 = prot_cur[cls].to(torch.float32)
        l2_list.append(float(torch.norm(v0 - v1, p=2).item()))
        cos = 1.0 - float(torch.nn.functional.cosine_similarity(v0.unsqueeze(0), v1.unsqueeze(0)).item())
        cos_list.append(cos)
    if l2_list:
        out["proto_l2_mean"] = float(np.mean(l2_list))
        out["proto_l2_max"] = float(np.max(l2_list))
    if cos_list:
        out["proto_cos_mean"] = float(np.mean(cos_list))
        out["proto_cos_max"] = float(np.max(cos_list))
    out["n_proto_classes"] = int(len(l2_list))
    return out


def compute_old_task_feature_drift(
    network,
    data_manager,
    cur_task: int,
    save_root: str,
    device: torch.device,
    args: dict,
) -> Dict[str, Any]:
    """
    Compare the current model against cached per-task references for all old tasks.
    """
    if cur_task <= 0:
        return {
            "old_task_feature_drift": {},
            "old_task_cka_status": "skipped_no_old_tasks",
        }

    per_task: Dict[str, Any] = {}
    cka_vals = []
    proto_l2_vals = []
    proto_cos_vals = []

    for ref_task in range(cur_task):
        cache_path = _ref_cache_path(save_root, ref_task)
        if not os.path.exists(cache_path):
            per_task[f"t{ref_task:02d}"] = {"status": "missing_reference_cache"}
            continue
        ref_payload = torch.load(cache_path, map_location="cpu")
        loader = make_ref_loader(data_manager, ref_task, args)
        try:
            drift = _compare_current_to_reference(network, loader, ref_payload, device, args)
            drift["status"] = "ok"
            per_task[f"t{ref_task:02d}"] = drift
            if "cka" in drift:
                cka_vals.append(float(drift["cka"]))
            if "proto_l2_mean" in drift:
                proto_l2_vals.append(float(drift["proto_l2_mean"]))
            if "proto_cos_mean" in drift:
                proto_cos_vals.append(float(drift["proto_cos_mean"]))
        except Exception as exc:
            logger.warning("[feature_tracker] old-task feature drift failed for t%02d: %s", ref_task, exc)
            per_task[f"t{ref_task:02d}"] = {"status": f"failed:{exc}"}

    out: Dict[str, Any] = {"old_task_feature_drift": per_task}
    if cka_vals:
        out["old_task_cka_mean"] = float(np.mean(cka_vals))
        out["old_task_cka_min"] = float(np.min(cka_vals))
    if proto_l2_vals:
        out["old_task_proto_l2_mean"] = float(np.mean(proto_l2_vals))
        out["old_task_proto_l2_max"] = float(np.max(proto_l2_vals))
    if proto_cos_vals:
        out["old_task_proto_cos_mean"] = float(np.mean(proto_cos_vals))
        out["old_task_proto_cos_max"] = float(np.max(proto_cos_vals))
    return out
