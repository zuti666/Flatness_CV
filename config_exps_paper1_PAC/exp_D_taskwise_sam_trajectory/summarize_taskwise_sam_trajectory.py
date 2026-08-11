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
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


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


def _find_run_dirs(outputs_root: str, prefix_stem: str) -> List[str]:
    pattern = os.path.join(
        outputs_root,
        "logs_inc_lora",
        "seqlora",
        "*",
        "cifar10_224",
        "*",
        f"{prefix_stem}*",
        "exp_run",
        "5",
    )
    return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def _run_meta(run_dir: str, prefix_stem: str) -> Tuple[str, str, str]:
    parts = Path(run_dir).parts
    try:
        idx = parts.index("logs_inc_lora")
        opt_dir = parts[idx + 2]
        seed = parts[idx + 4]
        prefix = parts[idx + 5]
        variant = prefix.replace(prefix_stem, "")
        return variant, seed, opt_dir
    except Exception:
        return Path(run_dir).name, "", ""


def _load_matrix(run_dir: str):
    files = glob.glob(os.path.join(run_dir, "*_cl_metrics.json"))
    if not files:
        return None
    data = _load_json(files[0])
    mats = data.get("cnn", {}).get("matrices", {})
    return mats.get("final", mats.get("t01"))


def _parse_conditioned_path(path: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"_theta_t(\d+)_loss_task(\d+)_metrics\.json$", os.path.basename(path))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _metric(data: dict, name: str) -> float:
    if name == "Sh_Delta/W_tangent":
        return _as_float(data.get("Sh_Delta/W_tangent", data.get("Sh_Delta_tangent", data.get("sh_delta_max"))))
    return _as_float(data.get(name))


def _finite(values):
    vals = [_as_float(v) for v in values]
    return [v for v in vals if math.isfinite(v)]


def _aggregate(rows: List[dict], keys: List[str]) -> List[dict]:
    by_variant: Dict[str, List[dict]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    out = []
    for variant, group in sorted(by_variant.items()):
        agg = {"variant": variant, "n": len(group)}
        for key in keys:
            vals = _finite([r.get(key) for r in group])
            agg[f"{key}_mean"] = mean(vals) if vals else float("nan")
            agg[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0 if vals else float("nan")
        out.append(agg)
    return out


def summarize(outputs_root: str, prefix_stem: str):
    rows: List[dict] = []
    for run_dir in _find_run_dirs(outputs_root, prefix_stem):
        variant, seed, opt_dir = _run_meta(run_dir, prefix_stem)
        matrix = _load_matrix(run_dir)
        if not matrix or len(matrix) < 2 or len(matrix[1]) < 2:
            continue
        a11 = _as_float(matrix[0][0])
        a12 = _as_float(matrix[1][0])
        a22 = _as_float(matrix[1][1])

        by_key: Dict[Tuple[int, int], dict] = {}
        for path in sorted(glob.glob(os.path.join(run_dir, "flatness_task_conditioned", "*_metrics.json"))):
            parsed = _parse_conditioned_path(path)
            if parsed is not None:
                by_key[parsed] = _load_json(path)

        loss11 = _as_float(by_key.get((0, 0), {}).get("base_loss"))
        loss12 = _as_float(by_key.get((1, 0), {}).get("base_loss"))
        loss22 = _as_float(by_key.get((1, 1), {}).get("base_loss"))
        row = {
            "variant": variant,
            "seed": seed,
            "optimizer_dir": opt_dir,
            "A_1_1": a11,
            "A_1_2": a12,
            "A_2_2": a22,
            "Current": 0.5 * (a11 + a22),
            "Retention": a12,
            "F1_acc": a11 - a12,
            "BWT_1_2": a12 - a11,
            "FinalAvg": 0.5 * (a12 + a22),
            "Loss_1_1": loss11,
            "Loss_1_2": loss12,
            "Loss_2_2": loss22,
            "F1_loss": loss12 - loss11 if math.isfinite(loss11) and math.isfinite(loss12) else float("nan"),
            "ShDelta_theta1_loss1": _metric(by_key.get((0, 0), {}), "Sh_Delta/W_tangent"),
            "ShDelta_theta2_loss1": _metric(by_key.get((1, 0), {}), "Sh_Delta/W_tangent"),
            "ShDelta_theta2_loss2": _metric(by_key.get((1, 1), {}), "Sh_Delta/W_tangent"),
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["seed"], r["variant"]))

    by_seed: Dict[str, Dict[str, dict]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["variant"]] = row

    contrast_rows: List[dict] = []
    for seed, variants in sorted(by_seed.items()):
        if "sgd_sgd" not in variants:
            continue
        base = variants["sgd_sgd"]
        for perturb_sgd in sorted(name for name in variants if name.endswith("_sgd") and name != "sgd_sgd"):
            perturb_tag = perturb_sgd[:-4]
            sgd_perturb = f"sgd_{perturb_tag}"
            perturb_perturb = f"{perturb_tag}_{perturb_tag}"
            required = ["sgd_sgd", perturb_sgd, sgd_perturb, perturb_perturb]
            if not all(name in variants for name in required):
                continue
            t1 = variants[perturb_sgd]
            t2 = variants[sgd_perturb]
            both = variants[perturb_perturb]
            contrast_rows.append({
                "seed": seed,
                "perturb_tag": perturb_tag,
                "Delta_T1_flat_A12": t1["A_1_2"] - base["A_1_2"],
                "Delta_T2_protect_A12": t2["A_1_2"] - base["A_1_2"],
                "Delta_T1_current_A11": t1["A_1_1"] - base["A_1_1"],
                "Delta_T2_current_A22": t2["A_2_2"] - base["A_2_2"],
                "Delta_both_FinalAvg": both["FinalAvg"] - base["FinalAvg"],
                "Interaction_A12": (both["A_1_2"] - t1["A_1_2"]) - (t2["A_1_2"] - base["A_1_2"]),
                "Delta_T1_flat_F1_acc": t1["F1_acc"] - base["F1_acc"],
                "Delta_T2_protect_F1_acc": t2["F1_acc"] - base["F1_acc"],
            })

    agg_keys = ["A_1_1", "A_1_2", "A_2_2", "Current", "Retention", "F1_acc", "BWT_1_2", "FinalAvg", "F1_loss"]
    aggregate_rows = _aggregate(rows, agg_keys)
    return rows, aggregate_rows, contrast_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs_logs")
    parser.add_argument("--prefix-stem", default="exp_D_taskwise_sam_trajectory_")
    parser.add_argument("--out-dir", default="outputs_logs/exp_D_taskwise_sam_trajectory_summary")
    args = parser.parse_args()

    rows, aggregate_rows, contrast_rows = summarize(args.outputs_root, args.prefix_stem)
    _write_csv(os.path.join(args.out_dir, "summary_by_run.csv"), rows)
    _write_csv(os.path.join(args.out_dir, "summary_by_variant.csv"), aggregate_rows)
    _write_csv(os.path.join(args.out_dir, "contrasts_by_seed.csv"), contrast_rows)
    print(f"runs: {len(rows)}")
    print(f"contrasts: {len(contrast_rows)}")
    print(f"wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
