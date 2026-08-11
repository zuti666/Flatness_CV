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
from typing import Dict, List, Optional, Tuple


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _mean(values) -> float:
    vals = [_as_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _metric(flat: dict, *names: str) -> float:
    for name in names:
        if name in flat:
            return _as_float(flat.get(name))
    return float("nan")


def _task_from_flatness_path(path: str) -> Optional[int]:
    match = re.search(r"_t(\d+)_metrics\.json$", os.path.basename(path))
    if not match:
        return None
    return int(match.group(1))


def _find_run_dirs(outputs_root: str) -> List[str]:
    pattern = os.path.join(
        outputs_root,
        "logs_inc_lora",
        "seqlora",
        "*",
        "imagenetr",
        "*",
        "exp_B_imager_r16_t20_seed1993_seqlora_*",
        "exp_run",
        "10",
    )
    return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def _run_meta(run_dir: str) -> Tuple[str, str, str]:
    parts = Path(run_dir).parts
    try:
        idx = parts.index("logs_inc_lora")
        opt_dir = parts[idx + 2]
        seed = parts[idx + 4]
        prefix = parts[idx + 5]
        variant = prefix.replace("exp_B_imager_r16_t20_seed1993_seqlora_", "")
        return variant, seed, opt_dir
    except Exception:
        prefix = Path(run_dir).parts[-3]
        return prefix, "", ""


def _load_final_matrix(run_dir: str):
    files = glob.glob(os.path.join(run_dir, "*_cl_metrics.json"))
    if not files:
        return None
    data = _load_json(files[0])
    matrices = data.get("cnn", {}).get("matrices", {})
    if not matrices:
        return None
    keys = sorted(matrices.keys())
    last_key = keys[-1]
    return matrices[last_key]


def _matrix_summary(matrix) -> Dict[str, float]:
    if matrix is None:
        return {"final_seen_acc": float("nan"), "final_bwt": float("nan"), "final_forget": float("nan")}
    rows = [[_as_float(x) for x in row] for row in matrix]
    t = len(rows) - 1
    if t <= 0:
        return {"final_seen_acc": float("nan"), "final_bwt": float("nan"), "final_forget": float("nan")}
    diag = [rows[i][i] for i in range(min(len(rows), len(rows[0])))]
    final_old = [rows[t][i] for i in range(t)]
    bwt_terms = [final_old[i] - diag[i] for i in range(min(len(final_old), len(diag) - 1))]
    final_seen = rows[t][: t + 1]
    bwt = _mean(bwt_terms)
    return {
        "final_seen_acc": _mean(final_seen),
        "final_bwt": bwt,
        "final_forget": -bwt if math.isfinite(bwt) else float("nan"),
    }


def _write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(outputs_root: str) -> Tuple[List[dict], List[dict]]:
    per_task: List[dict] = []
    summary: List[dict] = []

    for run_dir in _find_run_dirs(outputs_root):
        variant, seed, opt_dir = _run_meta(run_dir)
        matrix_summary = _matrix_summary(_load_final_matrix(run_dir))

        task_rows = []
        for flat_path in sorted(glob.glob(os.path.join(run_dir, "flatness", "*_t*_metrics.json"))):
            task = _task_from_flatness_path(flat_path)
            if task is None:
                continue
            flat = _load_json(flat_path)
            row = {
                "variant": variant,
                "seed": seed,
                "optimizer_dir": opt_dir,
                "task": task,
                "base_loss": _as_float(flat.get("base_loss", float("nan"))),
                "Sh_param_full": _metric(flat, "Sh_param_full", "sh0_max"),
                "Sh_AB": _metric(flat, "Sh_AB", "sh_ab_max"),
                "Sh_Delta/W_tangent": _metric(flat, "Sh_Delta/W_tangent", "Sh_Delta_tangent", "sh_delta_max"),
                "Sh_rand/W_tangent": _metric(flat, "Sh_rand/W_tangent", "Sh_rand_tangent"),
                "Sh_frozen_coords": _metric(flat, "Sh_frozen_coords"),
            }
            per_task.append(row)
            task_rows.append(row)

        summary.append(
            {
                "variant": variant,
                "seed": seed,
                "optimizer_dir": opt_dir,
                **matrix_summary,
                "num_flat_tasks": len(task_rows),
                "mean_Sh_param_full": _mean(r["Sh_param_full"] for r in task_rows),
                "mean_Sh_AB": _mean(r["Sh_AB"] for r in task_rows),
                "mean_Sh_Delta/W_tangent": _mean(r["Sh_Delta/W_tangent"] for r in task_rows),
                "mean_Sh_rand/W_tangent": _mean(r["Sh_rand/W_tangent"] for r in task_rows),
                "mean_Sh_frozen_coords": _mean(r["Sh_frozen_coords"] for r in task_rows),
                "neg_Sh_param_full": sum(_as_float(r["Sh_param_full"]) < 0 for r in task_rows),
                "neg_Sh_AB": sum(_as_float(r["Sh_AB"]) < 0 for r in task_rows),
                "neg_Sh_Delta/W_tangent": sum(_as_float(r["Sh_Delta/W_tangent"]) < 0 for r in task_rows),
                "neg_Sh_rand/W_tangent": sum(_as_float(r["Sh_rand/W_tangent"]) < 0 for r in task_rows),
                "neg_Sh_frozen_coords": sum(_as_float(r["Sh_frozen_coords"]) < 0 for r in task_rows),
            }
        )

    order = {
        "sgd": 0,
        "sam_factor": 1,
        "sam_full": 2,
        "sam_delta": 3,
        "sam_random": 4,
        "sam_frozen": 5,
        "sam_all": 6,
        "random_factor": 7,
        "random_full": 8,
        "random_delta": 9,
        "random_all": 10,
        "random_frozen": 11,
    }
    per_task.sort(key=lambda r: (order.get(r["variant"], 99), r["seed"], r["task"]))
    summary.sort(key=lambda r: (order.get(r["variant"], 99), r["seed"]))
    return per_task, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs_logs")
    parser.add_argument("--out-dir", default="outputs_logs/exp_B_imager_r16_t20_seed1993_summary")
    args = parser.parse_args()

    per_task, summary = summarize(args.outputs_root)
    _write_csv(os.path.join(args.out_dir, "per_task_flatness.csv"), per_task)
    _write_csv(os.path.join(args.out_dir, "summary_by_variant.csv"), summary)
    print(f"runs: {len(summary)}")
    print(f"flatness rows: {len(per_task)}")
    print(f"wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
