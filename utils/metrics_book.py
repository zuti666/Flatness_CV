"""Unified metrics book helpers (JSON + matrices)."""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


def metrics_json_path(log_dir: str, logfilename: str) -> str:
    """Return the consolidated metrics JSON path."""
    return os.path.join(log_dir, f"{os.path.basename(logfilename)}_cl_metrics.json")


def _load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_json(path: str, obj: dict, *, json_safe=None):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_safe)
    os.replace(tmp, path)


def _update_metrics_json(
    json_path: str,
    section: str,
    *,
    step: Optional[int] = None,
    metrics: Optional[dict] = None,
    matrix=None,
    final: bool = False,
    json_safe=None,
):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    J = _load_json(json_path)
    S = J.setdefault(section, {})
    if final:
        S["final"] = metrics if metrics is not None else {}
        if matrix is not None:
            S.setdefault("matrices", {})["final"] = np.asarray(matrix, dtype=float).tolist()
    else:
        if metrics is not None and step is not None:
            steps = S.setdefault("steps", {})
            steps[str(step)] = metrics
        if matrix is not None and step is not None:
            mats = S.setdefault("matrices", {})
            mats[f"t{step:02d}"] = np.asarray(matrix, dtype=float).tolist()
    _save_json(json_path, J, json_safe=json_safe)


def write_step_metrics(
    json_path: str,
    section: str,
    step: int,
    *,
    metrics: Optional[dict] = None,
    matrix=None,
    json_safe=None,
):
    """Append per-step metrics and/or matrix into consolidated JSON."""
    _update_metrics_json(
        json_path,
        section,
        step=step,
        metrics=metrics,
        matrix=matrix,
        final=False,
        json_safe=json_safe,
    )


def write_final_metrics(
    json_path: str,
    section: str,
    final_metrics: dict,
    final_matrix=None,
    *,
    json_safe=None,
):
    """Write final metrics (and optional final matrix) into consolidated JSON."""
    _update_metrics_json(
        json_path,
        section,
        metrics=final_metrics,
        matrix=final_matrix,
        final=True,
        json_safe=json_safe,
    )


def assemble_eval_matrix(seq_rows, T, orientation: str = "time_by_task"):
    """Assemble a lower-triangular accuracy matrix from per-step rows."""
    M = np.full((T, T), np.nan, dtype=float)
    for i, row in enumerate(seq_rows):
        if len(row) > 0:
            n = min(len(row), T)
            M[i, :n] = np.array(row, dtype=float)[:n]
    if orientation == "task_by_time":
        return M.T
    return M


def log_eval_matrix(M, name: str, orientation: str = "time_by_task"):
    """Pretty-print and log an accuracy matrix."""
    import logging

    print("\nAccuracy Matrix ({} | {}):".format(name, orientation))
    print("=" * 72)
    if orientation == "time_by_task":
        print("- Row i: model after learning task i (0-based)")
        print("- Col j: evaluation on task j (0-based)")
        print("- Meaning: R[i, j] is valid only for j ≤ i (lower triangle)")
    print("=" * 72)
    print(np.array2string(M, precision=2, suppress_small=True))
    logging.info("\nAccuracy Matrix (%s | %s):\n%s", name, orientation, M)


def save_eval_matrix(M, run_dir: str, run_stub: str, tag: str):
    """Save matrix as .npy and .csv."""
    npy_path = os.path.join(run_dir, f"{run_stub}_{tag}.npy")
    csv_path = os.path.join(run_dir, f"{run_stub}_{tag}.csv")
    np.save(npy_path, M)
    np.savetxt(csv_path, M, delimiter=",", fmt="%.6f")
