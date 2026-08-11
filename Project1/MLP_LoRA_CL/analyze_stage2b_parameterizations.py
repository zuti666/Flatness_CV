#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from analyze_stage1c_temporal import (
    SGD_SAM,
    SGD_SGD,
    bootstrap,
    load_runs,
    matched_reductions,
    scalar_metrics,
    write_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze common-endpoint Stage-2B controls")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/focus_stage2b_effects.csv")
    )
    parser.add_argument(
        "--cells-output", type=Path, default=Path("analysis/focus_stage2b_cells.csv")
    )
    parser.add_argument("--exclude-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--preferred-balanced-rho", type=float, default=None)
    parser.add_argument("--preferred-factor-rho", type=float, default=None)
    args = parser.parse_args()

    runs = [run for run in load_runs(args.input) if int(run["seed"]) not in args.exclude_seeds]
    if args.preferred_balanced_rho is not None:
        filtered = []
        for run in runs:
            if run["parameterization"] == "balanced_lora" and "sam" in run["schedule"]:
                task_radii = run.get("task_sam_rhos")
                observed = float(task_radii[1]) if task_radii else float(run["sam_rho"])
                if abs(observed - args.preferred_balanced_rho) > 1e-12:
                    continue
            filtered.append(run)
        runs = filtered
    if args.preferred_factor_rho is not None:
        filtered = []
        for run in runs:
            if run["parameterization"] == "factor_lora" and "sam" in run["schedule"]:
                task_radii = run.get("task_sam_rhos")
                observed = float(task_radii[1]) if task_radii else float(run["sam_rho"])
                if abs(observed - args.preferred_factor_rho) > 1e-12:
                    continue
            filtered.append(run)
        runs = filtered
    for run in runs:
        if run["parameterization"] in {"random_subspace", "fixed_lora_tangent"}:
            dimension = int(run["transition"]["start_tangent_dimension"])
            run["parameterization"] = f"{run['parameterization']}_d{dimension}"
    # A calibration rerun may coexist with its superseded pilot in one output
    # tree.  Prefer the run that records task-resolved radii, which identifies
    # the predeclared parameterization-specific calibration.
    index = {}
    for run in runs:
        key = (run["order"], run["parameterization"], run["schedule"], int(run["seed"]))
        previous = index.get(key)
        if previous is None or (
            "task_sam_rhos" in run and "task_sam_rhos" not in previous
        ):
            index[key] = run
    cell_samples: dict[tuple, list[float]] = defaultdict(list)
    for run in index.values():
        for metric, value in scalar_metrics(run).items():
            cell_samples[(run["order"], run["parameterization"], run["schedule"], metric)].append(value)
    cell_rows = []
    for row_index, ((order, parameterization, schedule, metric), values) in enumerate(
        sorted(cell_samples.items(), key=str)
    ):
        mean, low, high = bootstrap(values, 50_000 + row_index)
        cell_rows.append({
            "task_order": str(order), "parameterization": parameterization,
            "old_task_optimizer": schedule[0], "new_task_optimizer": schedule[1],
            "metric": metric, "n_seeds": len(values), "mean": mean,
            "ci95_low": low, "ci95_high": high,
        })
    write_rows(cell_rows, args.cells_output)
    samples: dict[tuple, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    orders = sorted({run["order"] for run in runs})
    parameterizations = sorted({run["parameterization"] for run in runs})
    seeds = sorted({int(run["seed"]) for run in runs})
    for order in orders:
        for parameterization in parameterizations:
            for seed in seeds:
                control = index.get((order, parameterization, SGD_SGD, seed))
                treatment = index.get((order, parameterization, SGD_SAM, seed))
                if control is None or treatment is None:
                    continue
                left, right = scalar_metrics(control), scalar_metrics(treatment)
                for metric in sorted(left.keys() & right.keys()):
                    samples[(order, metric)][parameterization][seed] = left[metric] - right[metric]
                for metric, value in matched_reductions(control, treatment).items():
                    samples[(order, metric)][parameterization][seed] = value

    rows = []
    for row_index, ((order, metric), by_parameterization) in enumerate(
        sorted(samples.items(), key=str)
    ):
        for parameterization, by_seed in sorted(by_parameterization.items()):
            values = list(by_seed.values())
            mean, low, high = bootstrap(values, 60_000 + row_index)
            rows.append({
                "task_order": str(order), "metric": metric,
                "contrast": parameterization, "n_paired_seeds": len(values),
                "mean": mean, "ci95_low": low, "ci95_high": high,
            })
        factor = by_parameterization.get("factor_lora", {})
        for control_name, control in sorted(by_parameterization.items()):
            if control_name == "factor_lora":
                continue
            common = sorted(factor.keys() & control.keys())
            if not common:
                continue
            values = [factor[seed] - control[seed] for seed in common]
            mean, low, high = bootstrap(values, 70_000 + row_index)
            rows.append({
                "task_order": str(order), "metric": metric,
                "contrast": f"factor_lora_minus_{control_name}",
                "n_paired_seeds": len(values), "mean": mean,
                "ci95_low": low, "ci95_high": high,
            })
    write_rows(rows, args.output)
    print(
        f"wrote {len(rows)} Stage-2B effect rows to {args.output} "
        f"and {len(cell_rows)} cells to {args.cells_output}"
    )


if __name__ == "__main__":
    main()
