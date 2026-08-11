#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SGD_SGD = ("sgd", "sgd")
SAM_SGD = ("sam", "sgd")
SGD_SAM = ("sgd", "sam")
SAM_SAM = ("sam", "sam")
EFFECTS = {
    "protection_given_new_sgd": (SGD_SGD, SAM_SGD),
    "protection_given_new_sam": (SGD_SAM, SAM_SAM),
    "trajectory_given_old_sgd": (SGD_SGD, SGD_SAM),
    "trajectory_given_old_sam": (SAM_SGD, SAM_SAM),
    "combined_sam_sam_vs_sgd_sgd": (SGD_SGD, SAM_SAM),
}


def bootstrap(values: list[float], seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(10_000, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def interpolate_first_crossing(points: list[dict], x_key: str, y_key: str, target: float) -> float:
    points = sorted(points, key=lambda item: item["epoch"])
    for point in points:
        if abs(float(point[x_key]) - target) <= 1e-12:
            return float(point[y_key])
    for left, right in zip(points, points[1:]):
        x0, x1 = float(left[x_key]), float(right[x_key])
        if (x0 - target) * (x1 - target) <= 0 and x0 != x1:
            weight = (target - x0) / (x1 - x0)
            return float(left[y_key]) + weight * (float(right[y_key]) - float(left[y_key]))
    raise ValueError(f"trajectory does not cross {x_key}={target}")


def matched_reductions(control: dict, treatment: dict) -> dict[str, float]:
    left = [item for item in control["trajectory"] if item["task"] == 1]
    right = [item for item in treatment["trajectory"] if item["task"] == 1]
    if not left or not right:
        raise ValueError("missing task-1 trajectory")
    output = {}
    loss_target = max(
        min(float(item["current_loss"]) for item in left),
        min(float(item["current_loss"]) for item in right),
    )
    loss_ceiling = min(
        max(float(item["current_loss"]) for item in left),
        max(float(item["current_loss"]) for item in right),
    )
    if loss_target <= loss_ceiling:
        for damage in ("old_loss_damage", "old_accuracy_damage"):
            output[f"loss_matched_{damage}_reduction"] = (
                interpolate_first_crossing(left, "current_loss", damage, loss_target)
                - interpolate_first_crossing(right, "current_loss", damage, loss_target)
            )
    drift_target = min(
        max(float(item["effective_drift"]) for item in left),
        max(float(item["effective_drift"]) for item in right),
    )
    for damage in ("old_loss_damage", "old_accuracy_damage"):
        output[f"drift_matched_{damage}_reduction"] = (
            interpolate_first_crossing(left, "effective_drift", damage, drift_target)
            - interpolate_first_crossing(right, "effective_drift", damage, drift_target)
        )
    return output


def load_runs(roots: list[Path]) -> list[dict]:
    runs = []
    for root in roots:
        for metrics_path in root.rglob("metrics.json"):
            run = json.loads(metrics_path.read_text())
            schedule = run.get("optimizer_schedule")
            if not schedule or len(schedule) != 2:
                continue
            run["schedule"] = tuple(schedule)
            run["order"] = tuple(float(value) for value in run["angles"])
            run["trajectory"] = json.loads(
                (metrics_path.parent / "trajectory_history.json").read_text()
            )
            transitions = json.loads(
                (metrics_path.parent / "immediate_transitions.json").read_text()
            )
            if len(transitions) != 1:
                raise ValueError(f"expected one immediate transition in {metrics_path.parent}")
            run["transition"] = transitions[0]
            runs.append(run)
    return runs


def scalar_metrics(run: dict) -> dict[str, float]:
    accuracy = run["accuracy_matrix"]
    loss = run["loss_matrix"]
    transition = run["transition"]
    output = {
        "endpoint_old_accuracy_damage": float(accuracy[0][0] - accuracy[1][0]),
        "endpoint_old_loss_damage": float(loss[1][0] - loss[0][0]),
        "endpoint_current_accuracy": float(accuracy[1][1]),
        "endpoint_current_loss": float(loss[1][1]),
        "transition_delta_norm": float(transition["delta_norm"]),
    }
    for key in (
        "actual_loss_change",
        "interference_I",
        "directional_curvature_C",
        "normalized_hessian_curvature",
        "prospective_reachable_gradient_fraction",
        "prospective_old_gradient_norm",
        "prospective_old_gradient_alignment",
        "prospective_old_gradient_cosine",
        "prospective_hessian_curvature",
        "prospective_ggn_curvature",
    ):
        if transition.get(key) is not None:
            output[key] = float(transition[key])
    pathwise = transition.get("pathwise") or {}
    for key in (
        "interference_sum",
        "directional_curvature_sum",
        "ggn_directional_cost_sum",
        "pathwise_taylor_prediction",
        "taylor_residual_sum",
        "path_length",
    ):
        if pathwise.get(key) is not None:
            metric = key if key.startswith("pathwise_") else f"pathwise_{key}"
            output[metric] = float(pathwise[key])
    return output


def write_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage-1C temporal SAM roles")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("analysis/focus_stage1c_effects.csv"))
    parser.add_argument(
        "--cells-output", type=Path, default=Path("analysis/focus_stage1c_schedule_cells.csv")
    )
    args = parser.parse_args()

    runs = load_runs(args.input)
    index = {
        (run["order"], run["parameterization"], run["schedule"], int(run["seed"])): run
        for run in runs
    }
    cell_values: dict[tuple, list[float]] = defaultdict(list)
    for run in runs:
        for metric, value in scalar_metrics(run).items():
            cell_values[(run["order"], run["parameterization"], run["schedule"], metric)].append(value)

    cell_rows = []
    for row_index, (key, values) in enumerate(sorted(cell_values.items(), key=str)):
        order, parameterization, schedule, metric = key
        mean, low, high = bootstrap(values, 30_000 + row_index)
        cell_rows.append({
            "task_order": str(order),
            "parameterization": parameterization,
            "old_task_optimizer": schedule[0],
            "new_task_optimizer": schedule[1],
            "metric": metric,
            "n_seeds": len(values),
            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,
        })
    write_rows(cell_rows, args.cells_output)

    samples: dict[tuple, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    orders = sorted({run["order"] for run in runs})
    parameterizations = sorted({run["parameterization"] for run in runs})
    seeds = sorted({int(run["seed"]) for run in runs})
    for order in orders:
        for parameterization in parameterizations:
            for seed in seeds:
                for effect, (control_schedule, treatment_schedule) in EFFECTS.items():
                    control = index.get((order, parameterization, control_schedule, seed))
                    treatment = index.get((order, parameterization, treatment_schedule, seed))
                    if control is None or treatment is None:
                        continue
                    control_metrics = scalar_metrics(control)
                    treatment_metrics = scalar_metrics(treatment)
                    for metric in sorted(control_metrics.keys() & treatment_metrics.keys()):
                        # Positive means the SAM schedule lowers damage, movement, or
                        # curvature relative to the corresponding SGD schedule.
                        value = control_metrics[metric] - treatment_metrics[metric]
                        samples[(order, effect, metric)][parameterization].append((seed, value))
                    for metric, value in matched_reductions(control, treatment).items():
                        samples[(order, effect, metric)][parameterization].append((seed, value))

    output_rows = []
    for row_index, (key, by_parameterization) in enumerate(sorted(samples.items(), key=str)):
        order, effect, metric = key
        for parameterization in sorted(by_parameterization):
            pairs = by_parameterization.get(parameterization, [])
            if not pairs:
                continue
            values = [value for _, value in pairs]
            mean, low, high = bootstrap(values, 40_000 + row_index)
            output_rows.append({
                "task_order": str(order), "effect": effect, "metric": metric,
                "contrast": parameterization, "n_paired_seeds": len(values),
                "mean": mean, "ci95_low": low, "ci95_high": high,
            })
        dense = dict(by_parameterization.get("dense", []))
        lora = dict(by_parameterization.get("factor_lora", []))
        common = sorted(dense.keys() & lora.keys())
        if common:
            values = [lora[seed] - dense[seed] for seed in common]
            mean, low, high = bootstrap(values, 50_000 + row_index)
            output_rows.append({
                "task_order": str(order), "effect": effect, "metric": metric,
                "contrast": "factor_lora_minus_dense", "n_paired_seeds": len(values),
                "mean": mean, "ci95_low": low, "ci95_high": high,
            })
    write_rows(output_rows, args.output)
    print(
        f"wrote {len(output_rows)} effect rows to {args.output} and "
        f"{len(cell_rows)} cell rows to {args.cells_output}"
    )


if __name__ == "__main__":
    main()
