"""
io_utils.py
-----------
Save / load per-task mechanism JSON files.
Aggregate across tasks and seeds for summary tables.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import statistics
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_task_json(path: str, data: Dict[str, Any]) -> None:
    """Atomic write (write to .tmp then rename)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    logger.debug("[io_utils] saved → %s", path)


def load_task_results(save_root: str) -> List[Dict[str, Any]]:
    """
    Load all ``cl_mechanism_t??.json`` files from save_root,
    sorted by task index.

    Returns list of dicts (one per task).
    """
    pattern = os.path.join(save_root, "cl_mechanism_t*.json")
    files   = sorted(glob.glob(pattern))
    results = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception as exc:
            logger.warning("[io_utils] failed to load %s: %s", fp, exc)
    return results


# ── Aggregation ───────────────────────────────────────────────────────────────

_SCALAR_KEYS = [
    "L_old", "L_new", "L_all",
    "delta_L_old", "delta_L_new",
    "S_old", "S_new", "S_all",
    "ES_old_iso", "ES_new_iso", "ES_all_iso",
    "ES_old_method", "ES_new_method", "ES_all_method",
    "analytic_tr_h_sigma_fisher",
    "analytic_tr_h_sigma_gaussian",
    "analytic_tr_ratio_gauss_fisher",
    "mc_tr_h_sigma_fisher",
    "mc_tr_h_sigma_gaussian",
    "mc_tr_h_sigma_none",
    "mc_tr_ratio",
    "mc_tr_ratio_gauss_method",
    "drift_dot_vq", "drift_cos_vq",
    "drift_dot_vg", "drift_cos_vg",
    "drift_v_norm", "drift_q_norm", "drift_g_norm",
    "drift_projection_ratio",
    "old_task_cka_mean", "old_task_cka_min",
    "old_task_proto_l2_mean", "old_task_proto_l2_max",
    "old_task_proto_cos_mean", "old_task_proto_cos_max",
    "EFM_old_trace", "EFM_new_trace", "EFM_all_trace",
    "EFM_old_spectral_radius", "EFM_new_spectral_radius", "EFM_all_spectral_radius",
    "cos_fisher_u0", "cos_gaussian_u0", "cos_ratio_u0",
    "cos_fisher_u1", "cos_gaussian_u1", "cos_ratio_u1",
    "cos_none_u0", "cos_none_u1",
    "eval_time_s",
]


def summarize_results(task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate a list of per-task dicts into a summary with per-key
    arrays and mean/std values.

    Returns
    -------
    dict:
      per_task   : List of original task dicts
      arrays     : {key: [val_t0, val_t1, ...]}
      mean       : {key: float}
      std        : {key: float}
      n_tasks    : int
    """
    if not task_results:
        return {}

    arrays: Dict[str, List[float]] = {}
    for key in _SCALAR_KEYS:
        vals = []
        for row in task_results:
            v = row.get(key)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
        if vals:
            arrays[key] = vals

    mean_dict = {k: statistics.mean(v) for k, v in arrays.items()}
    std_dict  = {k: statistics.stdev(v) if len(v) > 1 else 0.0
                 for k, v in arrays.items()}

    return {
        "per_task": task_results,
        "arrays":   arrays,
        "mean":     mean_dict,
        "std":      std_dict,
        "n_tasks":  len(task_results),
    }


def load_and_summarize(save_root: str) -> Dict[str, Any]:
    """Convenience: load all task JSONs and summarize."""
    task_results = load_task_results(save_root)
    return summarize_results(task_results)


# ── Multi-run aggregation (across seeds / methods) ────────────────────────────

def aggregate_across_seeds(
    per_seed_summaries: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate summaries from multiple seeds.

    Parameters
    ----------
    per_seed_summaries : {seed_label: summary_dict}

    Returns
    -------
    {key: {"mean": float, "std": float, "per_seed": {seed: float}}}
    """
    all_keys: set = set()
    for s in per_seed_summaries.values():
        all_keys.update(s.get("mean", {}).keys())

    out: Dict[str, Any] = {}
    for key in sorted(all_keys):
        seed_vals = {}
        for seed_label, s in per_seed_summaries.items():
            v = s.get("mean", {}).get(key)
            if v is not None:
                seed_vals[seed_label] = float(v)
        if not seed_vals:
            continue
        vals = list(seed_vals.values())
        out[key] = {
            "mean":     statistics.mean(vals),
            "std":      statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "per_seed": seed_vals,
        }
    return out
