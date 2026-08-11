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
    scalar_metrics,
    write_rows,
)


MODELS = {
    "path_only": ("delta_path_length",),
    "path_plus_I": ("delta_path_length", "delta_I"),
    "path_plus_I_plus_C": ("delta_path_length", "delta_I", "delta_C"),
}


def _fit_predict(train: list[dict], test: list[dict], features: tuple[str, ...]) -> np.ndarray:
    x_train = np.asarray([[row[key] for key in features] for row in train], dtype=float)
    x_test = np.asarray([[row[key] for key in features] for row in test], dtype=float)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    design = np.column_stack([np.ones(len(x_train)), x_train])
    coefficients, *_ = np.linalg.lstsq(
        design, np.asarray([row["benefit_test"] for row in train]), rcond=None
    )
    return np.column_stack([np.ones(len(x_test)), x_test]) @ coefficients


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    denominator = float(((actual - actual.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "r2": float(1.0 - (error**2).sum() / denominator) if denominator > 1e-20 else float("nan"),
        "sign_accuracy": float((np.sign(actual) == np.sign(predicted)).mean()),
    }


def _first_crossing_curve(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return damage as a function of decreasing current loss using first crossings."""
    selected: list[tuple[float, float]] = []
    best = float("inf")
    for point in sorted(points, key=lambda item: int(item["epoch"])):
        loss = float(point["current_loss"])
        damage = float(point["old_loss_damage"])
        if loss < best - 1e-12:
            selected.append((loss, damage))
            best = loss
    if len(selected) < 2:
        raise ValueError("trajectory has fewer than two decreasing-loss first crossings")
    selected.sort()
    return (
        np.asarray([item[0] for item in selected], dtype=float),
        np.asarray([item[1] for item in selected], dtype=float),
    )


def _integral(curve: tuple[np.ndarray, np.ndarray], low: float, high: float) -> float:
    x, y = curve
    interior = x[(x > low) & (x < high)]
    grid = np.unique(np.concatenate([[low], interior, [high]]))
    values = np.interp(grid, x, y)
    return float(np.trapezoid(values, grid) / (high - low))


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    fields.extend(
        key for row in rows for key in row
        if key not in fields
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P0 existing-data path-safety and common-support AUC analysis"
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("analysis/stage3_p0"))
    args = parser.parse_args()
    runs = load_runs(args.input)
    index = {
        (run["order"], run["parameterization"], run["schedule"], int(run["seed"])): run
        for run in runs
    }
    seed_rows = []
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
                required = {
                    "endpoint_old_loss_damage",
                    "pathwise_path_length",
                    "pathwise_interference_sum",
                    "pathwise_directional_curvature_sum",
                    "pathwise_taylor_prediction",
                    "pathwise_taylor_residual_sum",
                }
                if not required <= left.keys() or not required <= right.keys():
                    raise ValueError(f"missing P0 metrics for {order}/{parameterization}/seed={seed}")
                row = {
                    "task_order": str(order),
                    "direction": "forward" if order[0] < order[1] else "reverse",
                    "angle_magnitude": abs(float(order[0])),
                    "parameterization": parameterization,
                    "seed": seed,
                    "benefit_test": left["endpoint_old_loss_damage"] - right["endpoint_old_loss_damage"],
                    "delta_path_length": left["pathwise_path_length"] - right["pathwise_path_length"],
                    "delta_I": left["pathwise_interference_sum"] - right["pathwise_interference_sum"],
                    "delta_C": left["pathwise_directional_curvature_sum"] - right["pathwise_directional_curvature_sum"],
                    "delta_IC": left["pathwise_taylor_prediction"] - right["pathwise_taylor_prediction"],
                    "delta_R": left["pathwise_taylor_residual_sum"] - right["pathwise_taylor_residual_sum"],
                    "delta_net_drift": left["transition_delta_norm"] - right["transition_delta_norm"],
                }
                seed_rows.append(row)
    if not seed_rows:
        raise ValueError("no complete P0 SGD/SAM pairs")
    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / "path_seed_level.csv", seed_rows)

    prediction_rows = []
    double_holdout_rows = []
    fold_rows = []
    summary_rows = []
    increment_rows = []
    for parameterization in sorted({row["parameterization"] for row in seed_rows}):
        selected = [row for row in seed_rows if row["parameterization"] == parameterization]
        selected_orders = sorted({row["task_order"] for row in selected})
        for held_out in selected_orders:
            train = [row for row in selected if row["task_order"] != held_out]
            test = [row for row in selected if row["task_order"] == held_out]
            fold_predictions = {}
            for model_name, features in MODELS.items():
                predicted = _fit_predict(train, test, features)
                actual = np.asarray([row["benefit_test"] for row in test])
                fold_predictions[model_name] = predicted
                values = _metrics(actual, predicted)
                fold_rows.append({
                    "parameterization": parameterization,
                    "held_out_task_order": held_out,
                    "model": model_name,
                    "n_train": len(train),
                    "n_test": len(test),
                    **values,
                })
                for row, prediction in zip(test, predicted):
                    prediction_rows.append({
                        **row,
                        "held_out_task_order": held_out,
                        "model": model_name,
                        "prediction": float(prediction),
                    })
            actual = np.asarray([row["benefit_test"] for row in test])
            mae_i = _metrics(actual, fold_predictions["path_plus_I"])["mae"]
            mae_ic = _metrics(actual, fold_predictions["path_plus_I_plus_C"])["mae"]
            increment_rows.append({
                "parameterization": parameterization,
                "held_out_task_order": held_out,
                "mae_reduction_from_C": mae_i - mae_ic,
            })
        for model_name in MODELS:
            records = [row for row in prediction_rows if row["parameterization"] == parameterization and row["model"] == model_name]
            values = _metrics(
                np.asarray([row["benefit_test"] for row in records]),
                np.asarray([row["prediction"] for row in records]),
            )
            summary_rows.append({
                "parameterization": parameterization,
                "analysis": "leave_one_direction_out",
                "model": model_name,
                "n_predictions": len(records),
                **values,
            })
        # Stronger generalization check: each prediction is made by a model
        # trained on neither the held-out task direction nor the held-out seed.
        for test in selected:
            train = [
                row for row in selected
                if row["task_order"] != test["task_order"] and row["seed"] != test["seed"]
            ]
            for model_name, features in MODELS.items():
                prediction = float(_fit_predict(train, [test], features)[0])
                double_holdout_rows.append({
                    **test,
                    "held_out_task_order": test["task_order"],
                    "held_out_seed": test["seed"],
                    "model": model_name,
                    "prediction": prediction,
                })
        for model_name in MODELS:
            records = [
                row for row in double_holdout_rows
                if row["parameterization"] == parameterization and row["model"] == model_name
            ]
            values = _metrics(
                np.asarray([row["benefit_test"] for row in records]),
                np.asarray([row["prediction"] for row in records]),
            )
            summary_rows.append({
                "parameterization": parameterization,
                "analysis": "leave_direction_and_seed_out",
                "model": model_name,
                "n_predictions": len(records),
                **values,
            })
        direct_actual = np.asarray([row["benefit_test"] for row in selected])
        for label, key in (("direct_I", "delta_I"), ("direct_I_plus_C", "delta_IC")):
            direct_prediction = np.asarray([row[key] for row in selected])
            summary_rows.append({
                "parameterization": parameterization,
                "analysis": "unfitted_taylor",
                "model": label,
                "n_predictions": len(selected),
                **_metrics(direct_actual, direct_prediction),
            })
        by_order = defaultdict(list)
        for row in selected:
            by_order[row["task_order"]].append(row)
        sign_by_order = [
            float(np.mean([np.sign(row["benefit_test"]) == np.sign(row["delta_IC"]) for row in records]))
            for records in by_order.values()
        ]
        mean, low, high = bootstrap(sign_by_order, 310_000 + len(summary_rows))
        summary_rows.append({
            "parameterization": parameterization,
            "analysis": "direction_block_sign",
            "model": "sign_delta_I_plus_C",
            "n_predictions": len(selected),
            "mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
            "sign_accuracy": mean,
            "block_ci95_low": low,
            "block_ci95_high": high,
        })
        increments = [row["mae_reduction_from_C"] for row in increment_rows if row["parameterization"] == parameterization]
        mean, low, high = bootstrap(increments, 320_000 + len(summary_rows))
        summary_rows.append({
            "parameterization": parameterization,
            "analysis": "direction_block_increment",
            "model": "path_plus_I_plus_C_minus_path_plus_I",
            "n_predictions": len(increments),
            "mae": -mean,
            "rmse": float("nan"), "r2": float("nan"), "sign_accuracy": float("nan"),
            "mae_reduction_from_C": mean,
            "block_ci95_low": low,
            "block_ci95_high": high,
        })
    _write(args.output / "lodo_predictions.csv", prediction_rows)
    _write(args.output / "double_holdout_predictions.csv", double_holdout_rows)
    _write(args.output / "lodo_folds.csv", fold_rows)
    _write(args.output / "lodo_summary.csv", summary_rows)
    _write(args.output / "lodo_increment_by_direction.csv", increment_rows)
    residual_rows = []
    for parameterization in sorted({row["parameterization"] for row in seed_rows}):
        selected = [row for row in seed_rows if row["parameterization"] == parameterization]
        direction_means = []
        for order in sorted({row["task_order"] for row in selected}):
            values = [row["delta_R"] for row in selected if row["task_order"] == order]
            direction_means.append(float(np.mean(values)))
            residual_rows.append({
                "parameterization": parameterization,
                "scope": "direction",
                "task_order": order,
                "n": len(values),
                "mean_residual": float(np.mean(values)),
                "median_absolute_residual": float(np.median(np.abs(values))),
                "p90_absolute_residual": float(np.quantile(np.abs(values), 0.9)),
                "rmse_residual": float(np.sqrt(np.mean(np.asarray(values) ** 2))),
            })
        values = [row["delta_R"] for row in selected]
        mean, low, high = bootstrap(direction_means, 325_000 + len(residual_rows))
        residual_rows.append({
            "parameterization": parameterization,
            "scope": "all_direction_blocks",
            "task_order": "all",
            "n": len(values),
            "mean_residual": float(np.mean(values)),
            "mean_residual_block_ci95_low": low,
            "mean_residual_block_ci95_high": high,
            "median_absolute_residual": float(np.median(np.abs(values))),
            "p90_absolute_residual": float(np.quantile(np.abs(values), 0.9)),
            "rmse_residual": float(np.sqrt(np.mean(np.asarray(values) ** 2))),
        })
    _write(args.output / "taylor_residual_summary.csv", residual_rows)

    # Common-support current-loss AUC for the Dense-versus-Factor interaction.
    required_parameters = {"dense", "factor_lora"}
    trajectory_index = {
        (run["order"], run["parameterization"], run["schedule"], int(run["seed"])):
        _first_crossing_curve([item for item in run["trajectory"] if int(item["task"]) == 1])
        for run in runs
        if run["parameterization"] in required_parameters
    }
    auc_seed_rows = []
    target_seed_rows = []
    support_rows = []
    for order in orders:
        complete_seeds = []
        for seed in seeds:
            keys = [
                (order, parameterization, schedule, seed)
                for parameterization in sorted(required_parameters)
                for schedule in (SGD_SGD, SGD_SAM)
            ]
            if all(key in trajectory_index for key in keys):
                complete_seeds.append(seed)
        if not complete_seeds:
            continue
        curves = [
            trajectory_index[(order, parameterization, schedule, seed)]
            for seed in complete_seeds
            for parameterization in sorted(required_parameters)
            for schedule in (SGD_SGD, SGD_SAM)
        ]
        low = max(float(curve[0].min()) for curve in curves)
        high = min(float(curve[0].max()) for curve in curves)
        if high - low <= 1e-8:
            raise ValueError(f"empty global current-loss support for {order}: [{low}, {high}]")
        targets = np.linspace(low, high, 5)[1:-1]
        support_rows.append({
            "task_order": str(order), "n_seeds": len(complete_seeds),
            "loss_support_low": low, "loss_support_high": high,
            "support_width": high - low,
        })
        for seed in complete_seeds:
            benefits = {}
            target_benefits = defaultdict(dict)
            for parameterization in sorted(required_parameters):
                control = trajectory_index[(order, parameterization, SGD_SGD, seed)]
                treatment = trajectory_index[(order, parameterization, SGD_SAM, seed)]
                benefit = _integral(control, low, high) - _integral(treatment, low, high)
                benefits[parameterization] = benefit
                auc_seed_rows.append({
                    "task_order": str(order), "seed": seed,
                    "contrast": parameterization, "normalized_damage_auc_reduction": benefit,
                    "loss_support_low": low, "loss_support_high": high,
                })
                for target in targets:
                    target_benefits[float(target)][parameterization] = (
                        float(np.interp(target, control[0], control[1]))
                        - float(np.interp(target, treatment[0], treatment[1]))
                    )
            interaction = benefits["factor_lora"] - benefits["dense"]
            auc_seed_rows.append({
                "task_order": str(order), "seed": seed,
                "contrast": "factor_lora_minus_dense",
                "normalized_damage_auc_reduction": interaction,
                "loss_support_low": low, "loss_support_high": high,
            })
            for target, values in target_benefits.items():
                for contrast, value in (
                    ("dense", values["dense"]),
                    ("factor_lora", values["factor_lora"]),
                    ("factor_lora_minus_dense", values["factor_lora"] - values["dense"]),
                ):
                    target_seed_rows.append({
                        "task_order": str(order), "seed": seed,
                        "loss_target": target, "contrast": contrast,
                        "old_loss_damage_reduction": value,
                    })
    auc_summary_rows = []
    grouped_auc = defaultdict(list)
    for row in auc_seed_rows:
        grouped_auc[(row["task_order"], row["contrast"])].append(
            float(row["normalized_damage_auc_reduction"])
        )
    for row_index, ((order, contrast), values) in enumerate(sorted(grouped_auc.items())):
        mean, low, high = bootstrap(values, 330_000 + row_index)
        auc_summary_rows.append({
            "task_order": order, "contrast": contrast, "n_seeds": len(values),
            "mean": mean, "ci95_low": low, "ci95_high": high,
        })
    target_summary_rows = []
    grouped_targets = defaultdict(list)
    for row in target_seed_rows:
        grouped_targets[(row["task_order"], row["loss_target"], row["contrast"])].append(
            float(row["old_loss_damage_reduction"])
        )
    for row_index, ((order, target, contrast), values) in enumerate(sorted(grouped_targets.items())):
        mean, low, high = bootstrap(values, 340_000 + row_index)
        target_summary_rows.append({
            "task_order": order, "loss_target": target, "contrast": contrast,
            "n_seeds": len(values), "mean": mean,
            "ci95_low": low, "ci95_high": high,
        })
    _write(args.output / "auc_support.csv", support_rows)
    _write(args.output / "auc_seed_level.csv", auc_seed_rows)
    _write(args.output / "auc_summary.csv", auc_summary_rows)
    _write(args.output / "fixed_loss_seed_level.csv", target_seed_rows)
    _write(args.output / "fixed_loss_summary.csv", target_summary_rows)
    print(
        f"P0 wrote {len(seed_rows)} paired path rows, {len(prediction_rows)} LODO "
        f"predictions and {len(auc_seed_rows)} common-support AUC rows to {args.output}"
    )


if __name__ == "__main__":
    main()
