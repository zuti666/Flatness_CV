#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _metric(flat: dict, *names: str):
    for name in names:
        if name and name in flat:
            return flat.get(name)
    return math.nan


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman(x: Iterable[float], y: Iterable[float]) -> float:
    def _as_float(value: object) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    xs = np.asarray([_as_float(v) for v in x], dtype=float)
    ys = np.asarray([_as_float(v) for v in y], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 3:
        return float("nan")
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _task_from_flatness_path(path: str) -> Optional[int]:
    match = re.search(r"_t(\d+)_metrics\.json$", os.path.basename(path))
    if not match:
        return None
    return int(match.group(1))


def _find_run_dirs(outputs_root: str, rq: str) -> List[str]:
    pattern = os.path.join(
        outputs_root,
        "logs_inc_lora",
        "seqlora",
        "*",
        "imagenetr",
        "*",
        f"exp_7_{rq}_seqlora_imagenetr_r16_*",
        "exp_run",
        "10",
    )
    return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def _load_cl_matrix(run_dir: str, task: int) -> Optional[np.ndarray]:
    files = glob.glob(os.path.join(run_dir, "*_cl_metrics.json"))
    if not files:
        return None
    data = _load_json(files[0])
    matrix = data.get("cnn", {}).get("matrices", {}).get(f"t{task:02d}")
    if matrix is None:
        return None
    return np.asarray(matrix, dtype=float)


def _run_meta(run_dir: str) -> Tuple[str, str, str]:
    parts = Path(run_dir).parts
    try:
        idx = parts.index("logs_inc_lora")
        opt = parts[idx + 2]
        seed = parts[idx + 4]
        prefix = parts[idx + 5]
        return opt, seed, prefix
    except Exception:
        return "", "", Path(run_dir).name


def summarize_rq1(outputs_root: str) -> Tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    for run_dir in _find_run_dirs(outputs_root, "RQ1"):
        opt, seed, prefix = _run_meta(run_dir)
        for flat_path in sorted(glob.glob(os.path.join(run_dir, "flatness", "*_t*_metrics.json"))):
            task = _task_from_flatness_path(flat_path)
            if task is None or task <= 0:
                continue
            matrix = _load_cl_matrix(run_dir, task)
            if matrix is None or matrix.shape[0] <= task or matrix.shape[1] <= task:
                continue
            diag = np.diag(matrix)
            bwt = float(np.nanmean(matrix[task, :task] - diag[:task]))
            forget = -bwt
            flat = _load_json(flat_path)
            rows.append(
                {
                    "optimizer": opt,
                    "seed": seed,
                    "prefix": prefix,
                    "task": task,
                    "bwt_t": bwt,
                    "forget_t": forget,
                    "Sh_param_full": _metric(flat, "Sh_param_full", "sh0_max"),
                    "Sh_AB": _metric(flat, "Sh_AB", "sh_ab_max"),
                    "Sh_Delta/W_tangent": _metric(flat, "Sh_Delta/W_tangent", "Sh_Delta_tangent", "sh_delta_max"),
                    "base_loss": flat.get("base_loss", math.nan),
                }
            )

    corr_rows: List[dict] = []
    for scope_name, metric in [
        ("Sh_param_full", "Sh_param_full"),
        ("Sh_AB", "Sh_AB"),
        ("Sh_Delta/W_tangent", "Sh_Delta/W_tangent"),
    ]:
        corr_rows.append(
            {
                "group": "all",
                "sharpness": scope_name,
                "spearman_vs_forget_t": _spearman((r[metric] for r in rows), (r["forget_t"] for r in rows)),
                "n": sum(_finite(r.get(metric, math.nan)) and _finite(r.get("forget_t", math.nan)) for r in rows),
            }
        )
        for opt in sorted({r["optimizer"] for r in rows}):
            sub = [r for r in rows if r["optimizer"] == opt]
            corr_rows.append(
                {
                    "group": opt,
                    "sharpness": scope_name,
                    "spearman_vs_forget_t": _spearman((r[metric] for r in sub), (r["forget_t"] for r in sub)),
                    "n": sum(_finite(r.get(metric, math.nan)) and _finite(r.get("forget_t", math.nan)) for r in sub),
                }
            )
    return rows, corr_rows


def summarize_rq2(outputs_root: str) -> List[dict]:
    rows: List[dict] = []
    for run_dir in _find_run_dirs(outputs_root, "RQ2"):
        opt, seed, prefix = _run_meta(run_dir)
        files = sorted(glob.glob(os.path.join(run_dir, "flatness", "*_t10_metrics.json")))
        if not files:
            continue
        flat = _load_json(files[0])
        rows.append(
            {
                "optimizer": opt,
                "seed": seed,
                "prefix": prefix,
                "task": 10,
                "c": flat.get("lora_rescale_factor", 1.0),
                "rescaled_pairs": flat.get("lora_rescale_num_pairs", 0),
                "base_loss": flat.get("base_loss", math.nan),
                "Sh_AB": _metric(flat, "Sh_AB", "sh_ab_max"),
                "Sh_Delta/W_tangent": _metric(flat, "Sh_Delta/W_tangent", "Sh_Delta_tangent", "sh_delta_max"),
            }
        )
    rows.sort(key=lambda r: float(r["c"]))
    return rows


def _write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs_logs")
    parser.add_argument("--out-dir", default="outputs_logs/exp_7_RQ1RQ2_summary")
    args = parser.parse_args()

    rq1_rows, rq1_corr = summarize_rq1(args.outputs_root)
    rq2_rows = summarize_rq2(args.outputs_root)

    _write_csv(os.path.join(args.out_dir, "rq1_per_task.csv"), rq1_rows)
    _write_csv(os.path.join(args.out_dir, "rq1_spearman.csv"), rq1_corr)
    _write_csv(os.path.join(args.out_dir, "rq2_rescale.csv"), rq2_rows)

    print(f"RQ1 rows: {len(rq1_rows)}")
    print(f"RQ2 rows: {len(rq2_rows)}")
    print(f"Wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
