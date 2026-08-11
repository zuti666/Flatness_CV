#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
from typing import List, Tuple


ORDER = {
    "sgd": 0,
    "sam_factor": 1,
    "sam_full": 2,
    "sam_delta": 3,
    "sam_all": 4,
    "sam_frozen": 5,
    "random_factor": 6,
    "random_full": 7,
    "random_delta": 8,
    "random_all": 9,
    "random_frozen": 10,
}


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
    return sum(vals) / len(vals) if vals else float("nan")


def _metric(flat: dict, *names: str) -> float:
    for name in names:
        if name in flat:
            return _as_float(flat.get(name))
    return float("nan")


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


def _find_run_dirs(outputs_root: str) -> List[str]:
    pattern = os.path.join(
        outputs_root,
        "logs_inc_lora",
        "seqlora",
        "*",
        "imagenetr",
        "*",
        "exp_E_imagenetr_r16_t20_support_direction_seqlora_*",
        "exp_run",
        "10",
    )
    return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def _run_meta(run_dir: str) -> Tuple[str, str, str]:
    parts = Path(run_dir).parts
    idx = parts.index("logs_inc_lora")
    opt_dir = parts[idx + 2]
    seed = parts[idx + 4]
    prefix = parts[idx + 5]
    variant = prefix.replace("exp_E_imagenetr_r16_t20_support_direction_seqlora_", "")
    return variant, seed, opt_dir


def _load_cl_metrics(run_dir: str) -> dict:
    files = glob.glob(os.path.join(run_dir, "*_cl_metrics.json"))
    return _load_json(files[0]) if files else {}


def _task_from_flatness_path(path: str):
    stem = os.path.basename(path)
    marker = "_t"
    idx = stem.rfind(marker)
    if idx < 0:
        return None
    try:
        return int(stem[idx + 2 : idx + 4])
    except Exception:
        return None


def summarize(outputs_root: str):
    summary_rows = []
    flat_rows = []
    for run_dir in _find_run_dirs(outputs_root):
        variant, seed, opt_dir = _run_meta(run_dir)
        metrics = _load_cl_metrics(run_dir)
        cnn_final = metrics.get("cnn", {}).get("final", {})
        nme_final = metrics.get("nme", {}).get("final", {})

        task_rows = []
        for path in sorted(glob.glob(os.path.join(run_dir, "flatness", "*_t*_metrics.json"))):
            task = _task_from_flatness_path(path)
            if task is None:
                continue
            flat = _load_json(path)
            row = {
                "variant": variant,
                "seed": seed,
                "optimizer_dir": opt_dir,
                "task": task,
                "base_loss": _as_float(flat.get("base_loss")),
                "Sh_param_full": _metric(flat, "Sh_param_full", "sh0_max"),
                "Sh_AB": _metric(flat, "Sh_AB", "sh_ab_max"),
                "Sh_Delta/W_tangent": _metric(flat, "Sh_Delta/W_tangent", "Sh_Delta_tangent", "sh_delta_max"),
                "Sh_rand/W_tangent": _metric(flat, "Sh_rand/W_tangent", "Sh_rand_tangent"),
                "Sh_frozen_coords": _metric(flat, "Sh_frozen_coords"),
            }
            flat_rows.append(row)
            task_rows.append(row)

        summary_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "optimizer_dir": opt_dir,
                "CNN_FAA": _as_float(cnn_final.get("FAA")),
                "CNN_AAA": _as_float(cnn_final.get("AAA")),
                "CNN_BWT": _as_float(cnn_final.get("BWT_final_avg")),
                "CNN_Forget": _as_float(cnn_final.get("Forget_avg")),
                "NME_FAA": _as_float(nme_final.get("FAA")),
                "NME_AAA": _as_float(nme_final.get("AAA")),
                "NME_BWT": _as_float(nme_final.get("BWT_final_avg")),
                "NME_Forget": _as_float(nme_final.get("Forget_avg")),
                "num_flat_tasks": len(task_rows),
                "mean_Sh_param_full": _mean(r["Sh_param_full"] for r in task_rows),
                "mean_Sh_AB": _mean(r["Sh_AB"] for r in task_rows),
                "mean_Sh_Delta/W_tangent": _mean(r["Sh_Delta/W_tangent"] for r in task_rows),
                "mean_Sh_rand/W_tangent": _mean(r["Sh_rand/W_tangent"] for r in task_rows),
                "mean_Sh_frozen_coords": _mean(r["Sh_frozen_coords"] for r in task_rows),
            }
        )

    summary_rows.sort(key=lambda r: (ORDER.get(r["variant"], 99), r["seed"]))
    flat_rows.sort(key=lambda r: (ORDER.get(r["variant"], 99), r["seed"], r["task"]))
    return summary_rows, flat_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs_logs")
    parser.add_argument("--out-dir", default="outputs_logs/exp_E_imagenetr_r16_t20_support_direction_summary")
    args = parser.parse_args()

    summary_rows, flat_rows = summarize(args.outputs_root)
    _write_csv(os.path.join(args.out_dir, "summary_by_variant.csv"), summary_rows)
    _write_csv(os.path.join(args.out_dir, "per_task_flatness.csv"), flat_rows)
    print(f"runs: {len(summary_rows)}")
    print(f"flatness rows: {len(flat_rows)}")
    print(f"wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
