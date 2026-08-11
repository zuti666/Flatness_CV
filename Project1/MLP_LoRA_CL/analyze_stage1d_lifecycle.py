#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analyze_stage1c_temporal import (
    SGD_SAM,
    SGD_SGD,
    bootstrap,
    load_runs,
    matched_reductions,
    scalar_metrics,
)


def trajectory_effects(roots: list[Path]) -> dict[tuple, dict[int, float]]:
    runs = [run for run in load_runs(roots) if run["parameterization"] == "factor_lora"]
    index = {(run["order"], run["schedule"], int(run["seed"])): run for run in runs}
    output: dict[tuple, dict[int, float]] = {}
    for order in sorted({run["order"] for run in runs}):
        for seed in sorted({int(run["seed"]) for run in runs}):
            control = index.get((order, SGD_SGD, seed))
            treatment = index.get((order, SGD_SAM, seed))
            if control is None or treatment is None:
                continue
            values = {
                metric: scalar_metrics(control)[metric] - scalar_metrics(treatment)[metric]
                for metric in scalar_metrics(control).keys() & scalar_metrics(treatment).keys()
            }
            values.update(matched_reductions(control, treatment))
            for metric, value in values.items():
                output.setdefault((order, metric), {})[seed] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare persistent and merge-reset LoRA")
    parser.add_argument("--persistent", type=Path, action="append", required=True)
    parser.add_argument("--merge-reset", type=Path, action="append", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/focus_stage1d_lifecycle.csv")
    )
    args = parser.parse_args()
    persistent = trajectory_effects(args.persistent)
    merge = trajectory_effects(args.merge_reset)
    rows = []
    for row_index, key in enumerate(sorted(persistent.keys() & merge.keys(), key=str)):
        order, metric = key
        for label, samples in (("persistent", persistent[key]), ("merge_reset", merge[key])):
            values = list(samples.values())
            mean, low, high = bootstrap(values, 60_000 + row_index)
            rows.append({
                "task_order": str(order), "metric": metric, "contrast": label,
                "n_paired_seeds": len(values), "mean": mean,
                "ci95_low": low, "ci95_high": high,
            })
        common = sorted(persistent[key].keys() & merge[key].keys())
        differences = [merge[key][seed] - persistent[key][seed] for seed in common]
        mean, low, high = bootstrap(differences, 70_000 + row_index)
        rows.append({
            "task_order": str(order), "metric": metric,
            "contrast": "merge_reset_minus_persistent", "n_paired_seeds": len(common),
            "mean": mean, "ci95_low": low, "ci95_high": high,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} lifecycle rows to {args.output}")


if __name__ == "__main__":
    main()
