from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _write(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _order_label(angles: list[float]) -> str:
    def item(value: float) -> str:
        return ("m" if value < 0 else "p") + f"{abs(float(value)):g}"

    return "-".join(item(value) for value in angles)


def _rho_key(value: float | None) -> float | None:
    return None if value is None else round(float(value), 10)


def _bootstrap(values: list[float], seed: int = 7319) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return mean, float(lower), float(upper)


def _summarize(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    value_keys: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, items in sorted(groups.items(), key=lambda pair: str(pair[0])):
        record = dict(zip(group_keys, group))
        record["n"] = len(items)
        for index, key in enumerate(value_keys):
            values = [float(item[key]) for item in items]
            mean, lower, upper = _bootstrap(values, 7319 + index)
            record[f"{key}_mean"] = mean
            record[f"{key}_ci_low"] = lower
            record[f"{key}_ci_high"] = upper
        output.append(record)
    return output


def _first_interpolation(
    trajectory: list[dict[str, Any]], x_key: str, y_key: str, target: float
) -> float:
    for first, second in zip(trajectory[:-1], trajectory[1:]):
        x0, x1 = float(first[x_key]), float(second[x_key])
        if min(x0, x1) - 1e-12 <= target <= max(x0, x1) + 1e-12:
            if abs(x1 - x0) < 1e-20:
                return 0.5 * (float(first[y_key]) + float(second[y_key]))
            fraction = (target - x0) / (x1 - x0)
            return float(first[y_key]) + fraction * (
                float(second[y_key]) - float(first[y_key])
            )
    raise ValueError(f"Target {target} is outside trajectory support for {x_key}")


def _monotone_curve(
    trajectory: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    decreasing: bool,
) -> tuple[np.ndarray, np.ndarray]:
    selected = []
    best = float("inf") if decreasing else -float("inf")
    for point in trajectory:
        x = float(point[x_key])
        improved = x < best - 1e-12 if decreasing else x > best + 1e-12
        if improved:
            selected.append((x, float(point[y_key])))
            best = x
    if len(selected) < 2:
        raise ValueError(f"Too few monotone points for {x_key}")
    selected.sort()
    return np.asarray([x for x, _ in selected]), np.asarray([y for _, y in selected])


def _curve_integral(curve: tuple[np.ndarray, np.ndarray], low: float, high: float) -> float:
    x, y = curve
    interior = x[(x > low) & (x < high)]
    grid = np.unique(np.concatenate([[low], interior, [high]]))
    return float(np.trapezoid(np.interp(grid, x, y), grid) / (high - low))


def _leave_one_seed_out(
    rows: list[dict[str, Any]],
    feature_sets: dict[str, list[str]],
    targets: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = []
    summary = []
    for order in sorted({row["order"] for row in rows}):
        order_rows = [row for row in rows if row["order"] == order]
        seeds = sorted({int(row["seed"]) for row in order_rows})
        for target in targets:
            for model_name, features in feature_sets.items():
                model_predictions = []
                for held_seed in seeds:
                    train = [row for row in order_rows if int(row["seed"]) != held_seed]
                    test = [row for row in order_rows if int(row["seed"]) == held_seed]
                    train_x = np.asarray(
                        [[float(row[key]) for key in features] for row in train], dtype=float
                    ).reshape(len(train), len(features))
                    test_x = np.asarray(
                        [[float(row[key]) for key in features] for row in test], dtype=float
                    ).reshape(len(test), len(features))
                    mean = train_x.mean(axis=0) if features else np.empty(0)
                    scale = train_x.std(axis=0) if features else np.empty(0)
                    scale[scale < 1e-12] = 1
                    train_design = np.column_stack(
                        [np.ones(len(train)), (train_x - mean) / scale]
                    )
                    test_design = np.column_stack(
                        [np.ones(len(test)), (test_x - mean) / scale]
                    )
                    coefficient = np.linalg.lstsq(
                        train_design,
                        np.asarray([float(row[target]) for row in train]),
                        rcond=None,
                    )[0]
                    predicted = test_design @ coefficient
                    for row, prediction in zip(test, predicted):
                        item = {
                            "order": order,
                            "seed": held_seed,
                            "transformation": row["transformation"],
                            "target": target,
                            "model": model_name,
                            "actual": float(row[target]),
                            "predicted": float(prediction),
                        }
                        predictions.append(item)
                        model_predictions.append(item)
                actual = np.asarray([row["actual"] for row in model_predictions])
                predicted = np.asarray([row["predicted"] for row in model_predictions])
                residual = actual - predicted
                denominator = ((actual - actual.mean()) ** 2).sum()
                summary.append(
                    {
                        "order": order,
                        "target": target,
                        "model": model_name,
                        "n_predictions": len(actual),
                        "mae": float(np.abs(residual).mean()),
                        "rmse": float(np.sqrt((residual**2).mean())),
                        "r2": float(1 - (residual**2).sum() / max(denominator, 1e-20)),
                        "sign_accuracy": float(
                            (np.sign(actual) == np.sign(predicted)).mean()
                        ),
                    }
                )
    return predictions, summary


def _load(input_root: Path) -> tuple[list[dict[str, Any]], dict[tuple, list[dict[str, Any]]]]:
    rows = []
    trajectories: dict[tuple, list[dict[str, Any]]] = {}
    for path in sorted(input_root.glob("angles-*/seed_*/transform_*/**/metrics.json")):
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        rho = _rho_key(value["sam_rho"])
        key = (
            _order_label(value["angles"]),
            int(value["seed"]),
            value["transformation"],
            value["optimizer_method"],
            rho,
        )
        pathwise = value["pathwise"]
        pullback = value["start_pullback"]
        factors = value["start_factor_diagnostics"]
        step = value["step_geometry"]
        audit = value["transform_audit"]
        rows.append(
            {
                "order": key[0],
                "seed": key[1],
                "transformation": key[2],
                "method": key[3],
                "sam_rho": rho,
                "old_damage": value["old_loss_damage_from_task_a"],
                "continuation_old_damage": value["continuation_old_loss_damage"],
                "current_loss": value["final_current_loss"],
                "current_accuracy": value["final_current_accuracy"],
                "effective_drift": value["effective_drift"],
                "step_path_length": value["cumulative_step_path_length"],
                "epoch_path_length": pathwise["path_length"],
                "interference_I": pathwise["interference_sum"],
                "directional_curvature_C": pathwise["directional_curvature_sum"],
                "ggn_cost": pathwise["ggn_directional_cost_sum"],
                "taylor_prediction": pathwise["pathwise_taylor_prediction"],
                "taylor_residual": pathwise["taylor_residual_sum"],
                "pullback_old_new_inner": pullback["pullback_old_new_inner"],
                "pullback_old_new_cosine": pullback["pullback_old_new_cosine"],
                "pullback_new_energy": pullback["pullback_new_energy"],
                "predicted_sgd_old_interference": pullback[
                    "predicted_sgd_old_interference"
                ],
                "factor_jacobian_condition": factors["factor_jacobian_condition"],
                "factor_jacobian_operator_norm": factors[
                    "factor_jacobian_operator_norm"
                ],
                "factor_bilinear_ratio_at_start": factors["factor_bilinear_ratio"],
                "mean_actual_bilinear_ratio": step[
                    "mean_bilinear_to_linear_ratio"
                ],
                "max_step_closure_error": step[
                    "max_factor_step_closure_relative_error"
                ],
                "weight_audit_error": audit["effective_weight_error"],
                "logit_audit_error": audit["max_logit_error"],
                "projector_audit_error": audit["tangent_projector_distance"],
                "path": str(path),
            }
        )
        trajectory_path = path.parent / "trajectory_history.json"
        with trajectory_path.open(encoding="utf-8") as handle:
            trajectories[key] = json.load(handle)
        history_path = path.parent / "training_history.json"
        with history_path.open(encoding="utf-8") as handle:
            history = json.load(handle)
        achieved = [
            float(item["effective_perturbation_norm"])
            for item in history
            if item["effective_perturbation_norm"] is not None
        ]
        rows[-1]["max_effective_radius_error"] = (
            max(abs(value - float(rho)) for value in achieved)
            if rho is not None and achieved
            else 0.0
        )
        rows[-1]["epoch1_continuation_old_damage"] = trajectories[key][1][
            "continuation_old_loss_damage"
        ]
        rows[-1]["epoch1_current_loss"] = trajectories[key][1]["current_loss"]
    return rows, trajectories


def analyze(input_root: Path, output_root: Path) -> None:
    rows, trajectories = _load(input_root)
    if not rows:
        raise RuntimeError(f"No P1 metrics found under {input_root}")
    _write(rows, output_root / "runs.csv")
    by_key = {
        (row["order"], row["seed"], row["transformation"], row["method"], row["sam_rho"]): row
        for row in rows
    }
    radii = sorted({row["sam_rho"] for row in rows if row["method"] == "sam"})
    effects = []
    matched = []
    for row in rows:
        if row["method"] != "sam":
            continue
        base_key = (row["order"], row["seed"], row["transformation"], "sgd", None)
        sgd = by_key[base_key]
        effects.append(
            {
                "order": row["order"],
                "seed": row["seed"],
                "transformation": row["transformation"],
                "sam_rho": row["sam_rho"],
                "old_damage_reduction": sgd["old_damage"] - row["old_damage"],
                "continuation_damage_reduction": (
                    sgd["continuation_old_damage"] - row["continuation_old_damage"]
                ),
                "current_loss_penalty": row["current_loss"] - sgd["current_loss"],
                "current_accuracy_change": row["current_accuracy"] - sgd["current_accuracy"],
                "drift_change": row["effective_drift"] - sgd["effective_drift"],
                "relative_drift_change": (
                    row["effective_drift"] / max(sgd["effective_drift"], 1e-20) - 1
                ),
                "interference_reduction": sgd["interference_I"] - row["interference_I"],
                "curvature_reduction": (
                    sgd["directional_curvature_C"] - row["directional_curvature_C"]
                ),
                "ggn_cost_reduction": sgd["ggn_cost"] - row["ggn_cost"],
                "path_length_change": row["step_path_length"] - sgd["step_path_length"],
            }
        )
        sgd_key = (row["order"], row["seed"], row["transformation"], "sgd", None)
        sam_key = (
            row["order"], row["seed"], row["transformation"], "sam", row["sam_rho"]
        )
        sgd_trajectory = trajectories[sgd_key]
        sam_trajectory = trajectories[sam_key]
        sgd_current_curve = _monotone_curve(
            sgd_trajectory, "current_loss", "old_loss_damage_from_task_a", decreasing=True
        )
        sam_current_curve = _monotone_curve(
            sam_trajectory, "current_loss", "old_loss_damage_from_task_a", decreasing=True
        )
        current_target = max(
            float(sgd_current_curve[0].min()), float(sam_current_curve[0].min())
        )
        sgd_drift_curve = _monotone_curve(
            sgd_trajectory,
            "effective_drift",
            "old_loss_damage_from_task_a",
            decreasing=False,
        )
        sam_drift_curve = _monotone_curve(
            sam_trajectory,
            "effective_drift",
            "old_loss_damage_from_task_a",
            decreasing=False,
        )
        drift_target = min(
            float(sgd_drift_curve[0].max()), float(sam_drift_curve[0].max())
        )
        matched.append(
            {
                "order": row["order"],
                "seed": row["seed"],
                "transformation": row["transformation"],
                "sam_rho": row["sam_rho"],
                "current_loss_target": current_target,
                "current_loss_matched_damage_reduction": (
                    float(np.interp(current_target, *sgd_current_curve))
                    - float(np.interp(current_target, *sam_current_curve))
                ),
                "effective_drift_target": drift_target,
                "drift_matched_damage_reduction": (
                    float(np.interp(drift_target, *sgd_drift_curve))
                    - float(np.interp(drift_target, *sam_drift_curve))
                ),
            }
        )
    _write(effects, output_root / "paired_effects_seed.csv")
    effect_values = [
        "old_damage_reduction",
        "continuation_damage_reduction",
        "current_loss_penalty",
        "current_accuracy_change",
        "drift_change",
        "relative_drift_change",
        "interference_reduction",
        "curvature_reduction",
        "ggn_cost_reduction",
        "path_length_change",
    ]
    effect_summary = _summarize(
        effects, ["order", "transformation", "sam_rho"], effect_values
    )
    _write(effect_summary, output_root / "paired_effects_summary.csv")
    _write(matched, output_root / "matched_effects_seed.csv")
    matched_summary = _summarize(
        matched,
        ["order", "transformation", "sam_rho"],
        ["current_loss_matched_damage_reduction", "drift_matched_damage_reduction"],
    )
    _write(matched_summary, output_root / "matched_effects_summary.csv")

    auc_support = []
    auc_seed = []
    measures = [
        ("current_loss", True),
        ("effective_drift", False),
        ("cumulative_step_path_length", False),
    ]
    orders = sorted({row["order"] for row in rows})
    transforms = sorted({row["transformation"] for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    for measure, decreasing in measures:
        curves = {
            key: _monotone_curve(
                value,
                measure,
                "old_loss_damage_from_task_a",
                decreasing=decreasing,
            )
            for key, value in trajectories.items()
        }
        for order in orders:
            for rho in radii:
                comparison_keys = [
                    (order, seed, transform, method, None if method == "sgd" else rho)
                    for seed in seeds
                    for transform in transforms
                    for method in ("sgd", "sam")
                ]
                comparison_curves = [curves[key] for key in comparison_keys]
                low = max(float(curve[0].min()) for curve in comparison_curves)
                high = min(float(curve[0].max()) for curve in comparison_curves)
                if high - low <= 1e-10:
                    raise ValueError(
                        f"Empty global P1 support for {order}/{rho}/{measure}: [{low}, {high}]"
                    )
                auc_support.append(
                    {
                        "order": order,
                        "sam_rho": rho,
                        "measure": measure,
                        "support_low": low,
                        "support_high": high,
                        "support_width": high - low,
                    }
                )
                for seed in seeds:
                    benefits = {}
                    for transform in transforms:
                        sgd_curve = curves[(order, seed, transform, "sgd", None)]
                        sam_curve = curves[(order, seed, transform, "sam", rho)]
                        benefit = _curve_integral(sgd_curve, low, high) - _curve_integral(
                            sam_curve, low, high
                        )
                        benefits[transform] = benefit
                        auc_seed.append(
                            {
                                "order": order,
                                "seed": seed,
                                "transformation": transform,
                                "sam_rho": rho,
                                "measure": measure,
                                "contrast": "sam_reduction",
                                "damage_auc_reduction": benefit,
                            }
                        )
                    for transform in transforms:
                        if transform == "identity":
                            continue
                        auc_seed.append(
                            {
                                "order": order,
                                "seed": seed,
                                "transformation": transform,
                                "sam_rho": rho,
                                "measure": measure,
                                "contrast": "interaction_vs_identity",
                                "damage_auc_reduction": (
                                    benefits[transform] - benefits["identity"]
                                ),
                            }
                        )
    _write(auc_support, output_root / "global_support_auc_support.csv")
    _write(auc_seed, output_root / "global_support_auc_seed.csv")
    _write(
        _summarize(
            auc_seed,
            ["order", "transformation", "sam_rho", "measure", "contrast"],
            ["damage_auc_reduction"],
        ),
        output_root / "global_support_auc_summary.csv",
    )

    gauge_contrasts = []
    for row in rows:
        if row["transformation"] == "identity":
            continue
        identity = by_key[
            (row["order"], row["seed"], "identity", row["method"], row["sam_rho"])
        ]
        gauge_contrasts.append(
            {
                "order": row["order"],
                "seed": row["seed"],
                "transformation": row["transformation"],
                "method": row["method"],
                "sam_rho": row["sam_rho"],
                "old_damage_change_from_identity": row["old_damage"] - identity["old_damage"],
                "current_loss_change_from_identity": row["current_loss"] - identity["current_loss"],
                "drift_change_from_identity": (
                    row["effective_drift"] - identity["effective_drift"]
                ),
                "interference_change_from_identity": (
                    row["interference_I"] - identity["interference_I"]
                ),
                "curvature_change_from_identity": (
                    row["directional_curvature_C"]
                    - identity["directional_curvature_C"]
                ),
                "epoch1_old_damage_change_from_identity": (
                    row["epoch1_continuation_old_damage"]
                    - identity["epoch1_continuation_old_damage"]
                ),
                "predicted_interference_change_from_identity": (
                    row["predicted_sgd_old_interference"]
                    - identity["predicted_sgd_old_interference"]
                ),
                "log_jacobian_condition_change_from_identity": (
                    np.log(max(float(row["factor_jacobian_condition"]), 1e-20))
                    - np.log(max(float(identity["factor_jacobian_condition"]), 1e-20))
                ),
                "bilinear_ratio_change_from_identity": (
                    row["mean_actual_bilinear_ratio"]
                    - identity["mean_actual_bilinear_ratio"]
                ),
            }
        )
    _write(gauge_contrasts, output_root / "gauge_endpoint_contrasts_seed.csv")
    gauge_contrast_values = [
        "old_damage_change_from_identity",
        "current_loss_change_from_identity",
        "drift_change_from_identity",
        "interference_change_from_identity",
        "curvature_change_from_identity",
        "epoch1_old_damage_change_from_identity",
        "predicted_interference_change_from_identity",
        "log_jacobian_condition_change_from_identity",
        "bilinear_ratio_change_from_identity",
    ]
    _write(
        _summarize(
            gauge_contrasts,
            ["order", "transformation", "method", "sam_rho"],
            gauge_contrast_values,
        ),
        output_root / "gauge_endpoint_contrasts_summary.csv",
    )

    orthogonal_equivalence = [
        {
            **row,
            **{
                f"abs_{key}": abs(float(row[key]))
                for key in gauge_contrast_values
            },
        }
        for row in gauge_contrasts
        if row["transformation"] == "orthogonal"
    ]
    _write(orthogonal_equivalence, output_root / "orthogonal_equivalence_seed.csv")
    _write(
        _summarize(
            orthogonal_equivalence,
            ["order", "method", "sam_rho"],
            [f"abs_{key}" for key in gauge_contrast_values],
        ),
        output_root / "orthogonal_equivalence_summary.csv",
    )
    predictor_rows = [
        row
        for row in gauge_contrasts
        if row["method"] == "sgd"
        and row["transformation"]
        in {"scalar_0.5", "scalar_2", "anisotropic_2", "anisotropic_4"}
    ]
    predictions, predictor_summary = _leave_one_seed_out(
        predictor_rows,
        {
            "constant": [],
            "start_pullback": ["predicted_interference_change_from_identity"],
            "jacobian_condition": ["log_jacobian_condition_change_from_identity"],
            "actual_bilinear": ["bilinear_ratio_change_from_identity"],
            "pullback_plus_bilinear": [
                "predicted_interference_change_from_identity",
                "bilinear_ratio_change_from_identity",
            ],
        },
        [
            "epoch1_old_damage_change_from_identity",
            "old_damage_change_from_identity",
        ],
    )
    _write(predictions, output_root / "lodo_gauge_predictor_predictions.csv")
    _write(predictor_summary, output_root / "lodo_gauge_predictor_summary.csv")

    effect_index = {
        (row["order"], row["seed"], row["transformation"], row["sam_rho"]): row
        for row in effects
    }
    interactions = []
    for row in effects:
        if row["transformation"] in {"identity", "orthogonal"}:
            continue
        identity = effect_index[(row["order"], row["seed"], "identity", row["sam_rho"])]
        interactions.append(
            {
                "order": row["order"],
                "seed": row["seed"],
                "transformation": row["transformation"],
                "sam_rho": row["sam_rho"],
                "sam_benefit_interaction": (
                    row["old_damage_reduction"] - identity["old_damage_reduction"]
                ),
                "interference_interaction": (
                    row["interference_reduction"] - identity["interference_reduction"]
                ),
                "curvature_interaction": (
                    row["curvature_reduction"] - identity["curvature_reduction"]
                ),
            }
        )
    _write(interactions, output_root / "gauge_interactions_seed.csv")
    _write(
        _summarize(
            interactions,
            ["order", "transformation", "sam_rho"],
            ["sam_benefit_interaction", "interference_interaction", "curvature_interaction"],
        ),
        output_root / "gauge_interactions_summary.csv",
    )

    pooled_radius = _summarize(
        effects,
        ["sam_rho"],
        [
            "old_damage_reduction",
            "current_loss_penalty",
            "relative_drift_change",
            "interference_reduction",
            "curvature_reduction",
        ],
    )
    _write(pooled_radius, output_root / "radius_summary.csv")
    audit = {
        "num_runs": len(rows),
        "num_seeds": len({row["seed"] for row in rows}),
        "orders": sorted({row["order"] for row in rows}),
        "radii": radii,
        "max_effective_weight_audit_error": max(row["weight_audit_error"] for row in rows),
        "max_logit_audit_error": max(row["logit_audit_error"] for row in rows),
        "max_tangent_projector_distance": max(row["projector_audit_error"] for row in rows),
        "max_factor_step_closure_relative_error": max(
            row["max_step_closure_error"] for row in rows
        ),
        "max_effective_sam_radius_error": max(
            row["max_effective_radius_error"] for row in rows
        ),
    }
    with (output_root / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(
        f"P1 analyzed {len(rows)} runs, {len(effects)} SAM-SGD pairs; "
        f"outputs: {output_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze mature-factor gauge experiments")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
