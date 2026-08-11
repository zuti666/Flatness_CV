#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlp_lora_cl_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(root: Path):
    summaries, transitions = [], []
    for metrics_path in root.rglob("metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["run_dir"] = str(metrics_path.parent)
        summaries.append(summary)
        transition_path = metrics_path.parent / "immediate_transitions.json"
        if not transition_path.exists():
            continue
        with transition_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        for record in records:
            for key in (
                "run_dir", "parameterization", "rank", "lifecycle", "optimizer", "seed",
                "rotation_span", "num_tasks",
            ):
                record[key] = summary.get(key)
            metadata = summary.get("data_metadata") or {}
            record["teacher_target_rank"] = metadata.get("target_rank")
            record["teacher_target_principal_angle"] = metadata.get(
                "target_principal_angle_degrees"
            )
            transitions.append(record)
    return summaries, transitions


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _fit(rows: list[dict], features: list[str], target: str) -> dict:
    usable = [
        row for row in rows
        if _finite(row.get(target)) and all(_finite(row.get(feature)) for feature in features)
    ]
    if len(usable) <= len(features) + 1:
        return {
            "n": len(usable), "r2": None, "adjusted_r2": None, "mae": None,
            "loso_seed_r2": None, "loso_seed_mae": None,
        }
    x = np.asarray([[row[feature] for feature in features] for row in usable], dtype=float)
    y = np.asarray([row[target] for row in usable], dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    prediction = x @ coefficients
    residual = np.square(y - prediction).sum()
    total = np.square(y - y.mean()).sum()
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    adjusted = 1.0 - (1.0 - r2) * (len(y) - 1) / (len(y) - len(features) - 1)
    seeds = sorted({row.get("seed") for row in usable})
    held_out_targets, held_out_predictions = [], []
    if len(seeds) >= 3:
        for seed in seeds:
            train = [row for row in usable if row.get("seed") != seed]
            test = [row for row in usable if row.get("seed") == seed]
            train_x = np.asarray([[row[feature] for feature in features] for row in train])
            train_x = np.column_stack([np.ones(len(train_x)), train_x])
            train_y = np.asarray([row[target] for row in train])
            fold_coefficients = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
            test_x = np.asarray([[row[feature] for feature in features] for row in test])
            test_x = np.column_stack([np.ones(len(test_x)), test_x])
            held_out_targets.extend(row[target] for row in test)
            held_out_predictions.extend((test_x @ fold_coefficients).tolist())
    if held_out_targets:
        held_y = np.asarray(held_out_targets)
        held_prediction = np.asarray(held_out_predictions)
        held_total = np.square(held_y - held_y.mean()).sum()
        loso_r2 = 1.0 - np.square(held_y - held_prediction).sum() / held_total if held_total > 0 else float("nan")
        loso_mae = float(np.abs(held_y - held_prediction).mean())
    else:
        loso_r2, loso_mae = None, None
    return {
        "n": len(usable),
        "r2": float(r2),
        "adjusted_r2": float(adjusted),
        "mae": float(np.abs(y - prediction).mean()),
        "loso_seed_r2": None if loso_r2 is None else float(loso_r2),
        "loso_seed_mae": loso_mae,
        "coefficients": coefficients.tolist(),
    }


def rq1_table(transitions: list[dict]) -> list[dict]:
    rows = []
    for record in transitions:
        enriched = dict(record)
        pathwise = record.get("pathwise") or {}
        enriched["pathwise_prediction"] = pathwise.get("pathwise_taylor_prediction")
        rows.append(enriched)
    models = [
        ("update_norm_only", ["delta_norm"]),
        ("global_hessian_lambda_max_only", ["hessian_lambda_max"]),
        ("interference_only", ["interference_I"]),
        ("norm_plus_interference", ["delta_norm", "interference_I"]),
        (
            "norm_interference_plus_directional_curvature",
            ["delta_norm", "interference_I", "directional_curvature_C"],
        ),
        ("endpoint_taylor", ["pathwise_taylor_prediction"]),
        ("pathwise_taylor", ["pathwise_prediction"]),
    ]
    output = []
    for name, features in models:
        result = _fit(rows, features, "actual_loss_change")
        output.append({"model": name, "features": "+".join(features), **result})
    by_name = {row["model"]: row for row in output}
    base = by_name["norm_plus_interference"].get("r2")
    full = by_name["norm_interference_plus_directional_curvature"].get("r2")
    increment = None if base is None or full is None else full - base
    output.append(
        {
            "model": "incremental_directional_curvature",
            "features": "delta R2 over norm+interference",
            "n": by_name["norm_interference_plus_directional_curvature"]["n"],
            "r2": increment,
            "adjusted_r2": None,
            "mae": None,
            "loso_seed_r2": (
                None
                if by_name["norm_plus_interference"].get("loso_seed_r2") is None
                or by_name["norm_interference_plus_directional_curvature"].get("loso_seed_r2") is None
                else by_name["norm_interference_plus_directional_curvature"]["loso_seed_r2"]
                - by_name["norm_plus_interference"]["loso_seed_r2"]
            ),
            "loso_seed_mae": None,
        }
    )
    return output


def rq2_rows(transitions: list[dict]) -> list[dict]:
    fields = [
        "run_dir", "parameterization", "rank", "optimizer", "seed", "rotation_span",
        "teacher_target_rank", "teacher_target_principal_angle", "angle_gap_from_previous",
        "from_task", "to_task", "P_r_lambda", "tau_r", "tau_r_quality", "kappa_G",
        "ggn_directional_cost", "reachable_dimension", "new_gradient_reachable_fraction",
        "old_new_gradient_cosine", "old_new_ggn_top_overlap", "reachable_coverage",
    ]
    return [{field: record.get(field) for field in fields} for record in transitions]


def rq3_rows(summaries: list[dict], transitions: list[dict]) -> list[dict]:
    geometry = defaultdict(list)
    for record in transitions:
        geometry[record["run_dir"]].append(record)

    def aggregate(summary: dict) -> dict:
        records = geometry.get(summary["run_dir"], [])
        result = {}
        for field in ("P_r_lambda", "tau_r", "kappa_G", "directional_curvature_C"):
            values = [record[field] for record in records if _finite(record.get(field))]
            result[f"mean_{field}"] = float(np.mean(values)) if values else None
        path_values = [
            record["pathwise"]["directional_curvature_sum"]
            for record in records
            if record.get("pathwise") and _finite(record["pathwise"].get("directional_curvature_sum"))
        ]
        result["mean_pathwise_curvature_cost"] = float(np.mean(path_values)) if path_values else None
        path_ggn_values = [
            record["pathwise"]["ggn_directional_cost_sum"]
            for record in records
            if record.get("pathwise") and _finite(record["pathwise"].get("ggn_directional_cost_sum"))
        ]
        result["mean_pathwise_ggn_cost"] = float(np.mean(path_ggn_values)) if path_ggn_values else None
        fd_values = []
        for record in records:
            fd_values.extend((record.get("hvp_fd_relative_error") or {}).values())
        result["mean_fd_relative_error"] = float(np.mean(fd_values)) if fd_values else None
        return result

    keys = (
        "parameterization", "rank", "lifecycle", "seed", "rotation_span", "num_tasks"
    )
    baselines = {
        tuple(summary.get(key) for key in keys): summary
        for summary in summaries
        if summary.get("optimizer") == "sgd"
    }
    output = []
    for summary in summaries:
        if summary.get("optimizer") == "sgd":
            continue
        key = tuple(summary.get(field) for field in keys)
        baseline = baselines.get(key)
        if baseline is None:
            continue
        row = {field: summary.get(field) for field in keys}
        row["optimizer"] = summary["optimizer"]
        row["faa_gain_over_sgd"] = (
            summary["final_average_accuracy"] - baseline["final_average_accuracy"]
        )
        row["forgetting_reduction_over_sgd"] = (
            baseline["average_forgetting"] - summary["average_forgetting"]
        )
        current_geometry = aggregate(summary)
        baseline_geometry = aggregate(baseline)
        row.update(current_geometry)
        for field, value in current_geometry.items():
            base_value = baseline_geometry.get(field)
            row[f"delta_{field}_vs_sgd"] = (
                value - base_value if _finite(value) and _finite(base_value) else None
            )
        output.append(row)
    return output


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _scatter(transitions: list[dict], output: Path) -> None:
    usable = [
        row for row in transitions
        if _finite(row.get("actual_loss_change")) and _finite(row.get("pathwise_taylor_prediction"))
    ]
    if not usable:
        return
    actual = np.asarray([row["actual_loss_change"] for row in usable])
    predicted = np.asarray([row["pathwise_taylor_prediction"] for row in usable])
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(predicted, actual, alpha=0.6, s=22)
    axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Endpoint directional Taylor prediction")
    axis.set_ylabel("Actual immediate old-task loss change")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_rq1_models(rows: list[dict], output: Path) -> None:
    usable = [row for row in rows if _finite(row.get("r2")) and not row["model"].startswith("incremental")]
    if not usable:
        return
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(np.arange(len(usable)), [row["r2"] for row in usable])
    axis.set_xticks(np.arange(len(usable)), [row["model"] for row in usable], rotation=45, ha="right")
    axis.set_ylabel(r"In-sample $R^2$ (descriptive)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_rq2(rows: list[dict], output: Path) -> None:
    usable = [row for row in rows if row.get("rank") is not None]
    if not usable:
        return
    fields = [("P_r_lambda", r"$\mathcal{P}_r(\lambda)$"), ("tau_r", r"$\tau_r$"), ("kappa_G", r"$\kappa$")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, (field, label) in zip(axes, fields):
        for parameterization in sorted({row["parameterization"] for row in usable}):
            points = []
            for rank in sorted({row["rank"] for row in usable if row["parameterization"] == parameterization}):
                values = [
                    row[field] for row in usable
                    if row["parameterization"] == parameterization and row["rank"] == rank and _finite(row.get(field))
                ]
                if values:
                    points.append((rank, float(np.mean(values))))
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=parameterization)
        axis.set_xlabel("Learner rank")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_rq3(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x_fields = ["delta_mean_pathwise_curvature_cost_vs_sgd", "mean_fd_relative_error"]
    x_labels = ["Change in pathwise curvature vs SGD", "Finite-difference relative error"]
    plotted = False
    for axis, field, label in zip(axes, x_fields, x_labels):
        for optimizer in sorted({row["optimizer"] for row in rows}):
            subset = [row for row in rows if row["optimizer"] == optimizer and _finite(row.get(field))]
            if subset:
                plotted = True
                axis.scatter(
                    [row[field] for row in subset],
                    [row["faa_gain_over_sgd"] for row in subset],
                    label=optimizer,
                    alpha=0.65,
                )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel(label)
        axis.set_ylabel("FAA gain over matched SGD")
        axis.grid(alpha=0.25)
    if plotted:
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce RQ1–RQ3 mechanism tables")
    parser.add_argument("--input", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    summaries, transitions = _load(args.input)
    if not summaries:
        raise SystemExit(f"No metrics.json files below {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    rq1 = rq1_table(transitions)
    rq2 = rq2_rows(transitions)
    rq3 = rq3_rows(summaries, transitions)
    _write_csv(rq1, args.output / "rq1_predictor_comparison.csv")
    _write_csv(rq2, args.output / "rq2_safe_routes.csv")
    _write_csv(rq3, args.output / "rq3_optimizer_gains.csv")
    _scatter(transitions, args.output / "rq1_taylor_vs_actual.png")
    _plot_rq1_models(rq1, args.output / "rq1_predictor_r2.png")
    _plot_rq2(rq2, args.output / "rq2_rank_safe_routes.png")
    _plot_rq3(rq3, args.output / "rq3_gain_mechanisms.png")
    print(
        f"wrote {len(rq1)} RQ1 rows, {len(rq2)} RQ2 rows, "
        f"and {len(rq3)} matched RQ3 contrasts to {args.output}"
    )


if __name__ == "__main__":
    main()
