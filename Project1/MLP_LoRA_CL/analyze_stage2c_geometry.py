#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_stage1c_temporal import (
    SGD_SAM,
    SGD_SGD,
    bootstrap,
    load_runs,
    matched_reductions,
    scalar_metrics,
    write_rows,
)


GEOMETRY = (
    "prospective_reachable_gradient_fraction",
    "prospective_old_gradient_alignment",
    "prospective_old_gradient_cosine",
    "prospective_hessian_curvature",
    "prospective_ggn_curvature",
)

BENEFITS = (
    "actual_loss_change",
    "endpoint_old_loss_damage",
    "loss_matched_old_loss_damage_reduction",
    "drift_matched_old_loss_damage_reduction",
    "pathwise_interference_sum",
    "pathwise_directional_curvature_sum",
    "pathwise_ggn_directional_cost_sum",
)


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 3 or np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def _ranks(values: list[float]) -> list[float]:
    array = np.asarray(values)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    ranks[order] = np.arange(len(array), dtype=float)
    return ranks.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze task-geometry predictors of SAM benefit")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--seed-output", type=Path, default=Path("analysis/focus_stage2c_seed_level.csv")
    )
    parser.add_argument(
        "--cell-output", type=Path, default=Path("analysis/focus_stage2c_cells.csv")
    )
    parser.add_argument(
        "--correlation-output",
        type=Path,
        default=Path("analysis/focus_stage2c_geometry_correlations.csv"),
    )
    args = parser.parse_args()

    runs = load_runs(args.input)
    index = {
        (run["order"], run["parameterization"], run["schedule"], int(run["seed"])): run
        for run in runs
    }
    seed_rows = []
    orders = sorted({run["order"] for run in runs})
    parameters = sorted({run["parameterization"] for run in runs})
    seeds = sorted({int(run["seed"]) for run in runs})
    for order in orders:
        for parameterization in parameters:
            for seed in seeds:
                control = index.get((order, parameterization, SGD_SGD, seed))
                treatment = index.get((order, parameterization, SGD_SAM, seed))
                if control is None or treatment is None:
                    continue
                left, right = scalar_metrics(control), scalar_metrics(treatment)
                row = {
                    "task_order": str(order),
                    "angle_magnitude": abs(float(order[0])),
                    "direction": "forward" if order[0] < order[1] else "reverse",
                    "parameterization": parameterization,
                    "seed": seed,
                }
                for metric in GEOMETRY:
                    if metric not in left or metric not in right:
                        raise ValueError(f"missing {metric} for {order}/{parameterization}/seed={seed}")
                    if abs(left[metric] - right[metric]) > 1e-5:
                        raise ValueError(
                            f"prospective geometry changed across optimizer pair: {metric}"
                        )
                    row[metric] = 0.5 * (left[metric] + right[metric])
                for metric in BENEFITS:
                    if metric in left and metric in right:
                        row[f"sam_benefit__{metric}"] = left[metric] - right[metric]
                for metric, value in matched_reductions(control, treatment).items():
                    row[f"sam_benefit__{metric}"] = value
                seed_rows.append(row)
    write_rows(seed_rows, args.seed_output)

    numeric_fields = [
        key for key in seed_rows[0]
        if key in GEOMETRY or key.startswith("sam_benefit__")
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["task_order"], row["parameterization"])].append(row)
    cell_rows = []
    for group_index, ((order, parameterization), records) in enumerate(sorted(grouped.items())):
        base = {
            "task_order": order,
            "angle_magnitude": records[0]["angle_magnitude"],
            "direction": records[0]["direction"],
            "parameterization": parameterization,
            "n_seeds": len(records),
        }
        for field_index, field in enumerate(numeric_fields):
            values = [float(record[field]) for record in records if field in record]
            mean, low, high = bootstrap(values, 90_000 + 100 * group_index + field_index)
            base[f"{field}__mean"] = mean
            base[f"{field}__ci95_low"] = low
            base[f"{field}__ci95_high"] = high
        cell_rows.append(base)
    write_rows(cell_rows, args.cell_output)

    correlation_rows = []
    for parameterization in parameters:
        selected = [row for row in cell_rows if row["parameterization"] == parameterization]
        for geometry in GEOMETRY:
            x = [float(row[f"{geometry}__mean"]) for row in selected]
            for benefit in [field for field in numeric_fields if field.startswith("sam_benefit__")]:
                y = [float(row[f"{benefit}__mean"]) for row in selected]
                correlation_rows.append({
                    "scope": parameterization,
                    "predictor": geometry,
                    "outcome": benefit,
                    "n_task_orders": len(selected),
                    "pearson_r": _correlation(x, y),
                    "spearman_r": _correlation(_ranks(x), _ranks(y)),
                })

    by_cell = {(row["task_order"], row["parameterization"]): row for row in cell_rows}
    interaction_cells = []
    for order in sorted({row["task_order"] for row in cell_rows}):
        dense = by_cell.get((order, "dense"))
        factor = by_cell.get((order, "factor_lora"))
        if dense is None or factor is None:
            continue
        interaction = {"task_order": order}
        for geometry in GEOMETRY:
            interaction[f"factor__{geometry}"] = float(factor[f"{geometry}__mean"])
            interaction[f"difference__{geometry}"] = (
                float(factor[f"{geometry}__mean"]) - float(dense[f"{geometry}__mean"])
            )
        for benefit in [field for field in numeric_fields if field.startswith("sam_benefit__")]:
            interaction[f"interaction__{benefit}"] = (
                float(factor[f"{benefit}__mean"]) - float(dense[f"{benefit}__mean"])
            )
        interaction_cells.append(interaction)
    for predictor_prefix in ("factor__", "difference__"):
        for geometry in GEOMETRY:
            predictor = f"{predictor_prefix}{geometry}"
            x = [float(row[predictor]) for row in interaction_cells]
            outcomes = sorted(
                key for key in interaction_cells[0] if key.startswith("interaction__sam_benefit__")
            ) if interaction_cells else []
            for outcome in outcomes:
                y = [float(row[outcome]) for row in interaction_cells]
                correlation_rows.append({
                    "scope": "factor_minus_dense_interaction",
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_task_orders": len(interaction_cells),
                    "pearson_r": _correlation(x, y),
                    "spearman_r": _correlation(_ranks(x), _ranks(y)),
                })
    write_rows(correlation_rows, args.correlation_output)
    print(
        f"wrote {len(seed_rows)} seed rows, {len(cell_rows)} cells, "
        f"and {len(correlation_rows)} descriptive correlations"
    )


if __name__ == "__main__":
    main()
