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


SUPPORT_METRICS = [
    "Sh_param_full",
    "Sh_AB",
    "Sh_Delta/W_tangent",
    "Sh_rand/W_tangent",
    "Sh_frozen_coords",
]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _metric(data: dict, name: str) -> float:
    if name == "Sh_Delta/W_tangent":
        return _as_float(data.get("Sh_Delta/W_tangent", data.get("Sh_Delta_tangent", data.get("sh_delta_max"))))
    if name == "Sh_rand/W_tangent":
        return _as_float(data.get("Sh_rand/W_tangent", data.get("Sh_rand_tangent")))
    return _as_float(data.get(name))


def _mean(values) -> float:
    vals = [_as_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _find_run_dirs(outputs_root: str) -> List[str]:
    pattern = os.path.join(
        outputs_root,
        "logs_inc_lora",
        "seqlora",
        "*",
        "cifar10_224",
        "*",
        "exp_C_cifar10_taskcond_seqlora_*",
        "exp_run",
        "5",
    )
    return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def _run_meta(run_dir: str) -> Tuple[str, str, str]:
    parts = Path(run_dir).parts
    try:
        idx = parts.index("logs_inc_lora")
        opt_dir = parts[idx + 2]
        seed = parts[idx + 4]
        prefix = parts[idx + 5]
        variant = prefix.replace("exp_C_cifar10_taskcond_seqlora_", "")
        return variant, seed, opt_dir
    except Exception:
        return Path(run_dir).name, "", ""


def _load_matrix(run_dir: str, task_key: str):
    files = glob.glob(os.path.join(run_dir, "*_cl_metrics.json"))
    if not files:
        return None
    data = _load_json(files[0])
    return data.get("cnn", {}).get("matrices", {}).get(task_key)


def _parse_conditioned_path(path: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"_theta_t(\d+)_loss_task(\d+)_metrics\.json$", os.path.basename(path))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


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
    long_rows: List[dict] = []
    summary_rows: List[dict] = []

    for run_dir in _find_run_dirs(outputs_root):
        variant, seed, opt_dir = _run_meta(run_dir)
        by_key: Dict[Tuple[int, int], dict] = {}
        for path in sorted(glob.glob(os.path.join(run_dir, "flatness_task_conditioned", "*_metrics.json"))):
            parsed = _parse_conditioned_path(path)
            if parsed is None:
                continue
            model_task, loss_task = parsed
            data = _load_json(path)
            by_key[(model_task, loss_task)] = data
            row = {
                "variant": variant,
                "seed": seed,
                "optimizer_dir": opt_dir,
                "model_task": model_task,
                "loss_task": loss_task,
                "base_loss": _as_float(data.get("base_loss")),
            }
            for metric in SUPPORT_METRICS:
                row[metric] = _metric(data, metric)
            long_rows.append(row)

        mat0 = _load_matrix(run_dir, "t00")
        mat1 = _load_matrix(run_dir, "t01")
        acc_t1_theta1 = _as_float(mat0[0][0]) if mat0 and len(mat0) > 0 and len(mat0[0]) > 0 else float("nan")
        acc_t1_theta2 = _as_float(mat1[1][0]) if mat1 and len(mat1) > 1 and len(mat1[1]) > 0 else float("nan")
        acc_t2_theta2 = _as_float(mat1[1][1]) if mat1 and len(mat1) > 1 and len(mat1[1]) > 1 else float("nan")
        acc_forget_1_2 = acc_t1_theta1 - acc_t1_theta2 if math.isfinite(acc_t1_theta1) and math.isfinite(acc_t1_theta2) else float("nan")

        loss_t1_theta1 = _as_float(by_key.get((0, 0), {}).get("base_loss"))
        loss_t1_theta2 = _as_float(by_key.get((1, 0), {}).get("base_loss"))
        loss_t2_theta2 = _as_float(by_key.get((1, 1), {}).get("base_loss"))
        loss_forget_1_2 = loss_t1_theta2 - loss_t1_theta1 if math.isfinite(loss_t1_theta1) and math.isfinite(loss_t1_theta2) else float("nan")

        sh_old_delta = _metric(by_key.get((1, 0), {}), "Sh_Delta/W_tangent")
        sh_cur_delta = _metric(by_key.get((1, 1), {}), "Sh_Delta/W_tangent")
        denom = 0.5 * (sh_old_delta + sh_cur_delta) + 1e-12
        task_cond_gap_delta = abs(sh_old_delta - sh_cur_delta) / denom if math.isfinite(denom) and abs(denom) > 0 else float("nan")

        summary = {
            "variant": variant,
            "seed": seed,
            "optimizer_dir": opt_dir,
            "acc_t1_theta1": acc_t1_theta1,
            "acc_t1_theta2": acc_t1_theta2,
            "acc_t2_theta2": acc_t2_theta2,
            "acc_forget_1_2": acc_forget_1_2,
            "loss_t1_theta1": loss_t1_theta1,
            "loss_t1_theta2": loss_t1_theta2,
            "loss_t2_theta2": loss_t2_theta2,
            "loss_forget_1_2": loss_forget_1_2,
            "task_cond_gap_delta_theta2": task_cond_gap_delta,
        }
        for metric in SUPPORT_METRICS:
            summary[f"{metric}_theta1_loss_t1"] = _metric(by_key.get((0, 0), {}), metric)
            summary[f"{metric}_theta2_loss_t1"] = _metric(by_key.get((1, 0), {}), metric)
            summary[f"{metric}_theta2_loss_t2"] = _metric(by_key.get((1, 1), {}), metric)
        summary_rows.append(summary)

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
    long_rows.sort(key=lambda r: (order.get(r["variant"], 99), r["seed"], r["model_task"], r["loss_task"]))
    summary_rows.sort(key=lambda r: (order.get(r["variant"], 99), r["seed"]))
    return long_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs_logs")
    parser.add_argument("--out-dir", default="outputs_logs/exp_C_cifar10_task_conditioned_summary")
    args = parser.parse_args()

    long_rows, summary_rows = summarize(args.outputs_root)
    _write_csv(os.path.join(args.out_dir, "task_conditioned_flatness_long.csv"), long_rows)
    _write_csv(os.path.join(args.out_dir, "summary_by_variant.csv"), summary_rows)
    print(f"runs: {len(summary_rows)}")
    print(f"task-conditioned rows: {len(long_rows)}")
    print(f"wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
