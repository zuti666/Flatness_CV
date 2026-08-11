#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def bootstrap(values: list[float], seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(10_000, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def load_runs(roots: list[Path]) -> list[dict]:
    runs = []
    for root in roots:
        for path in root.rglob("metrics.json"):
            record = json.loads(path.read_text())
            trajectory_path = path.parent / "trajectory_history.json"
            if not trajectory_path.exists():
                continue
            record["trajectory"] = json.loads(trajectory_path.read_text())
            record["order"] = tuple(record["angles"])
            runs.append(record)
    return runs


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


def matched_reductions(sgd: dict, sam: dict) -> dict[str, float]:
    output = defaultdict(list)
    task_ids = sorted(
        set(item["task"] for item in sgd["trajectory"] if item["task"] > 0)
        & set(item["task"] for item in sam["trajectory"] if item["task"] > 0)
    )
    for task_id in task_ids:
        left = [item for item in sgd["trajectory"] if item["task"] == task_id]
        right = [item for item in sam["trajectory"] if item["task"] == task_id]
        loss_low = max(
            min(item["current_loss"] for item in left),
            min(item["current_loss"] for item in right),
        )
        loss_high = min(
            max(item["current_loss"] for item in left),
            max(item["current_loss"] for item in right),
        )
        if loss_low <= loss_high:
            for damage in ("old_loss_damage", "old_accuracy_damage"):
                sgd_value = interpolate_first_crossing(left, "current_loss", damage, loss_low)
                sam_value = interpolate_first_crossing(right, "current_loss", damage, loss_low)
                output[f"loss_matched_{damage}_reduction"].append(sgd_value - sam_value)

        common_drift = min(
            max(item["effective_drift"] for item in left),
            max(item["effective_drift"] for item in right),
        )
        for damage in ("old_loss_damage", "old_accuracy_damage"):
            sgd_value = interpolate_first_crossing(left, "effective_drift", damage, common_drift)
            sam_value = interpolate_first_crossing(right, "effective_drift", damage, common_drift)
            output[f"drift_matched_{damage}_reduction"].append(sgd_value - sam_value)
    if not task_ids:
        raise ValueError("no immediate-old-task trajectory transitions found")
    return {key: float(np.mean(values)) for key, values in output.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage-1B endpoint/loss/drift matched gains")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("analysis/focus_stage1b_matching.csv"))
    args = parser.parse_args()
    runs = load_runs(args.input)
    index = {}
    for run in runs:
        radius = float(run["sam_rho"]) if run["optimizer"] == "sam" else None
        index[(run["order"], run["parameterization"], run["optimizer"],
               int(run["seed"]), float(run["learning_rate"]), radius)] = run

    gains = defaultdict(lambda: defaultdict(list))
    for sam in runs:
        if sam["optimizer"] != "sam":
            continue
        key = (sam["order"], sam["parameterization"], "sgd", int(sam["seed"]),
               float(sam["learning_rate"]), None)
        sgd = index.get(key)
        if sgd is None:
            continue
        group = (sam["order"], float(sam["learning_rate"]), float(sam["sam_rho"]))
        parameterization = sam["parameterization"]
        gains[group][(parameterization, "endpoint_accuracy_gain")].append(
            sam["final_average_accuracy"] - sgd["final_average_accuracy"]
        )
        gains[group][(parameterization, "endpoint_forgetting_reduction")].append(
            sgd["average_forgetting"] - sam["average_forgetting"]
        )
        for metric, value in matched_reductions(sgd, sam).items():
            gains[group][(parameterization, metric)].append(value)

    output_rows = []
    for group_index, (group, values) in enumerate(sorted(gains.items())):
        order, learning_rate, radius = group
        metrics = sorted({metric for _, metric in values})
        for metric_index, metric in enumerate(metrics):
            for parameterization in ("dense", "factor_lora"):
                samples = values.get((parameterization, metric), [])
                if not samples:
                    continue
                mean, low, high = bootstrap(samples, 10_000 + 100 * group_index + metric_index)
                output_rows.append({
                    "task_order": str(order), "learning_rate": learning_rate,
                    "sam_rho": radius, "contrast": parameterization, "metric": metric,
                    "n_paired_seeds": len(samples), "mean": mean,
                    "ci95_low": low, "ci95_high": high,
                })
            dense = values.get(("dense", metric), [])
            lora = values.get(("factor_lora", metric), [])
            if dense and len(dense) == len(lora):
                interaction = (np.asarray(lora) - np.asarray(dense)).tolist()
                mean, low, high = bootstrap(interaction, 20_000 + 100 * group_index + metric_index)
                output_rows.append({
                    "task_order": str(order), "learning_rate": learning_rate,
                    "sam_rho": radius, "contrast": "factor_lora_minus_dense_interaction",
                    "metric": metric, "n_paired_seeds": len(interaction), "mean": mean,
                    "ci95_low": low, "ci95_high": high,
                })
    if not output_rows:
        raise SystemExit("No complete SGD/SAM trajectory pairs found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} Stage-1B rows to {args.output}")


if __name__ == "__main__":
    main()
