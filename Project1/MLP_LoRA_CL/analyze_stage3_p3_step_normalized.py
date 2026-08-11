from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from small_cl.config import load_config


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


def _order(angles: list[float]) -> str:
    return "-".join(
        ("m" if value < 0 else "p") + f"{abs(float(value)):g}"
        for value in angles
    )


def _mode(value: dict[str, Any]) -> str:
    if value["update_mode"] == "raw":
        return "raw"
    return f"normalized_q{float(value['target_scale']):g}"


def _bootstrap_interval(
    values: list[float], ci_level: float = 0.95, seed: int = 5519
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    tail = (1.0 - float(ci_level)) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return mean, float(low), float(high)


def _bootstrap(values: list[float], seed: int = 5519) -> tuple[float, float, float]:
    return _bootstrap_interval(values, ci_level=0.95, seed=seed)


def _summarize(
    rows: list[dict[str, Any]], group_keys: list[str], value_keys: list[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, items in sorted(groups.items(), key=lambda pair: str(pair[0])):
        record = {**dict(zip(group_keys, group)), "n_seeds": len(items)}
        for index, key in enumerate(value_keys):
            mean, low, high = _bootstrap(
                [float(item[key]) for item in items], 5519 + index
            )
            record[f"{key}_mean"] = mean
            record[f"{key}_ci_low"] = low
            record[f"{key}_ci_high"] = high
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
    raise ValueError(f"Target {target} outside support for {x_key}")


def _common_support_auc(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    x_key: str,
    y_key: str = "old_loss_damage",
    points: int = 101,
) -> tuple[float, float]:
    low = max(
        min(float(row[x_key]) for row in first),
        min(float(row[x_key]) for row in second),
    )
    high = min(
        max(float(row[x_key]) for row in first),
        max(float(row[x_key]) for row in second),
    )
    width = high - low
    if width <= 1e-10:
        raise ValueError(f"Insufficient common support for {x_key}: {low}, {high}")
    grid = np.linspace(low, high, int(points))
    differences = np.asarray(
        [
            _first_interpolation(first, x_key, y_key, float(target))
            - _first_interpolation(second, x_key, y_key, float(target))
            for target in grid
        ],
        dtype=float,
    )
    return float(np.trapezoid(differences, grid) / width), float(width)


def analyze(
    input_root: Path,
    output_root: Path,
    config: dict[str, Any] | None = None,
) -> None:
    runs: list[dict[str, Any]] = []
    trajectories: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for path in sorted(input_root.glob("angles-*/seed_*/transform_*/mode_*/*/metrics.json")):
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        with (path.parent / "step_geometry.json").open(encoding="utf-8") as handle:
            step_geometry = json.load(handle)
        row = {
            "order": _order(value["angles"]),
            "seed": int(value["seed"]),
            "transformation": value["transformation"],
            "method": value["optimizer_method"],
            "mode": _mode(value),
            "update_mode": value["update_mode"],
            "target_scale": value["target_scale"],
            "old_damage": value["continuation_old_loss_damage"],
            "current_loss": value["final_current_loss"],
            "current_accuracy": value["final_current_accuracy"],
            "effective_drift": value["effective_drift"],
            **value["step_summary"],
            "maximum_candidate_closure_error": max(
                float(item["candidate_closure_relative_error"])
                for item in step_geometry
            ),
            "path": str(path),
        }
        runs.append(row)
        with (path.parent / "progress_history.json").open(encoding="utf-8") as handle:
            trajectories[
                (
                    row["order"],
                    row["seed"],
                    row["transformation"],
                    row["method"],
                    row["mode"],
                )
            ] = json.load(handle)
    if not runs:
        raise RuntimeError(f"No P3 metrics found under {input_root}")

    by_cell = {
        (
            row["order"],
            row["seed"],
            row["transformation"],
            row["mode"],
            row["method"],
        ): row
        for row in runs
    }
    benefit_rows = []
    cell_prefixes = {
        (row["order"], row["seed"], row["transformation"], row["mode"])
        for row in runs
    }
    for order, seed, transformation, mode in sorted(cell_prefixes):
        sgd = by_cell[(order, seed, transformation, mode, "sgd")]
        sam = by_cell[(order, seed, transformation, mode, "sam")]
        sgd_path = trajectories[(order, seed, transformation, "sgd", mode)]
        sam_path = trajectories[(order, seed, transformation, "sam", mode)]
        progress_auc, progress_width = _common_support_auc(
            sgd_path, sam_path, "current_loss"
        )
        path_auc, path_width = _common_support_auc(
            sgd_path, sam_path, "cumulative_step_path_length"
        )
        benefit_rows.append(
            {
                "order": order,
                "seed": seed,
                "transformation": transformation,
                "mode": mode,
                "endpoint_sam_benefit": float(sgd["old_damage"])
                - float(sam["old_damage"]),
                "progress_auc_sam_benefit": progress_auc,
                "progress_support_width": progress_width,
                "path_auc_sam_benefit": path_auc,
                "path_support_width": path_width,
                "sgd_final_current_loss": sgd["current_loss"],
                "sam_final_current_loss": sam["current_loss"],
            }
        )

    gauge_rows = []
    for row in runs:
        if row["transformation"] == "identity":
            continue
        identity = by_cell[
            (row["order"], row["seed"], "identity", row["mode"], row["method"])
        ]
        gauge_path = trajectories[
            (
                row["order"], row["seed"], row["transformation"],
                row["method"], row["mode"],
            )
        ]
        identity_path = trajectories[
            (row["order"], row["seed"], "identity", row["method"], row["mode"])
        ]
        progress_auc, progress_width = _common_support_auc(
            gauge_path, identity_path, "current_loss"
        )
        path_auc, path_width = _common_support_auc(
            gauge_path, identity_path, "cumulative_step_path_length"
        )
        gauge_rows.append(
            {
                "order": row["order"],
                "seed": row["seed"],
                "transformation": row["transformation"],
                "method": row["method"],
                "mode": row["mode"],
                "endpoint_gauge_damage": float(row["old_damage"])
                - float(identity["old_damage"]),
                "progress_auc_gauge_damage": progress_auc,
                "progress_support_width": progress_width,
                "path_auc_gauge_damage": path_auc,
                "path_support_width": path_width,
                "path_ratio_to_identity": float(row["cumulative_step_path_length"])
                / max(float(identity["cumulative_step_path_length"]), 1e-20),
                "quadratic_budget_ratio_to_identity": float(
                    row["quadratic_step_budget"]
                )
                / max(float(identity["quadratic_step_budget"]), 1e-20),
            }
        )

    benefit_by_key = {
        (row["order"], row["seed"], row["transformation"], row["mode"]): row
        for row in benefit_rows
    }
    interactions = []
    for row in benefit_rows:
        if row["transformation"] == "identity":
            continue
        identity = benefit_by_key[
            (row["order"], row["seed"], "identity", row["mode"])
        ]
        interactions.append(
            {
                "order": row["order"],
                "seed": row["seed"],
                "transformation": row["transformation"],
                "mode": row["mode"],
                "endpoint_interaction": float(row["endpoint_sam_benefit"])
                - float(identity["endpoint_sam_benefit"]),
                "progress_auc_interaction": float(row["progress_auc_sam_benefit"])
                - float(identity["progress_auc_sam_benefit"]),
                "path_auc_interaction": float(row["path_auc_sam_benefit"])
                - float(identity["path_auc_sam_benefit"]),
            }
        )

    attenuation = []
    normalized_modes = sorted(
        {row["mode"] for row in interactions if row["mode"].startswith("normalized")}
    )
    interaction_by_key = {
        (row["order"], row["seed"], row["transformation"], row["mode"]): row
        for row in interactions
    }
    for order, seed, transformation, _ in sorted(
        {
            (row["order"], row["seed"], row["transformation"], row["mode"])
            for row in interactions if row["mode"] == "raw"
        }
    ):
        raw = interaction_by_key[(order, seed, transformation, "raw")]
        for mode in normalized_modes:
            normalized = interaction_by_key[(order, seed, transformation, mode)]
            attenuation.append(
                {
                    "order": order,
                    "seed": seed,
                    "transformation": transformation,
                    "normalized_mode": mode,
                    "endpoint_interaction_attenuation": float(raw["endpoint_interaction"])
                    - float(normalized["endpoint_interaction"]),
                    "progress_auc_interaction_attenuation": float(
                        raw["progress_auc_interaction"]
                    )
                    - float(normalized["progress_auc_interaction"]),
                    "path_auc_interaction_attenuation": float(
                        raw["path_auc_interaction"]
                    )
                    - float(normalized["path_auc_interaction"]),
                }
            )

    counterfactual_raw = []
    for path in sorted(input_root.glob("angles-*/seed_*/counterfactual_q*.json")):
        with path.open(encoding="utf-8") as handle:
            counterfactual_raw.extend(json.load(handle))
    counterfactual_by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in counterfactual_raw:
        key = (
            _order(row["angles"]), int(row["seed"]), float(row["target_scale"]),
            int(row["global_step"]), int(row["diagnostic_batch"]),
            row["transformation"],
        )
        counterfactual_by_key[key][row["method"]] = row
    counterfactual = []
    for key, methods in sorted(counterfactual_by_key.items(), key=lambda pair: str(pair[0])):
        if set(methods) != {"sgd", "sam"}:
            continue
        sgd, sam = methods["sgd"], methods["sam"]
        counterfactual.append(
            {
                "order": key[0],
                "seed": key[1],
                "target_scale": key[2],
                "global_step": key[3],
                "diagnostic_batch": key[4],
                "transformation": key[5],
                "old_direction_safety": float(sgd["old_loss_change"])
                - float(sam["old_loss_change"]),
                "new_task_cost": float(sam["new_loss_change"])
                - float(sgd["new_loss_change"]),
                "unit_interference_reduction": float(sgd["unit_interference"])
                - float(sam["unit_interference"]),
                "unit_curvature_reduction": float(sgd["unit_hessian_curvature"])
                - float(sam["unit_hessian_curvature"]),
                "sam_raw_correction_ratio": sam["sam_raw_correction_ratio"],
                "sam_resolved_correction_ratio": sam[
                    "sam_resolved_correction_ratio"
                ],
                "sam_raw_update_angle": 1.0 - float(sam["sam_raw_update_cosine"]),
                "sam_resolved_update_angle": 1.0
                - float(sam["sam_resolved_update_cosine"]),
                "sam_abs_taylor_residual": abs(float(sam["taylor_residual"])),
                "sgd_abs_taylor_residual": abs(float(sgd["taylor_residual"])),
            }
        )

    # Average checkpoints and diagnostic batches inside a seed before bootstrap.
    cf_seed_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in counterfactual:
        cf_seed_groups[
            (
                row["order"], row["seed"], row["target_scale"],
                row["transformation"],
            )
        ].append(row)
    cf_seed = []
    cf_values = [
        "old_direction_safety", "new_task_cost", "unit_interference_reduction",
        "unit_curvature_reduction", "sam_raw_correction_ratio",
        "sam_resolved_correction_ratio", "sam_raw_update_angle",
        "sam_resolved_update_angle", "sam_abs_taylor_residual",
        "sgd_abs_taylor_residual",
    ]
    for group, items in sorted(cf_seed_groups.items(), key=lambda pair: str(pair[0])):
        cf_seed.append(
            {
                "order": group[0],
                "seed": group[1],
                "target_scale": group[2],
                "transformation": group[3],
                **{
                    key: float(np.mean([float(item[key]) for item in items]))
                    for key in cf_values
                },
            }
        )

    audit_values = [
        "maximum_target_relative_error", "maximum_alpha",
        "minimum_raw_to_resolved_cosine", "cumulative_step_path_length",
        "quadratic_step_budget", "maximum_step_norm", "effective_step_count",
        "mean_sam_raw_correction_ratio", "mean_sam_resolved_correction_ratio",
        "mean_sam_raw_update_angle", "mean_sam_resolved_update_angle",
    ]
    run_summary = _summarize(
        runs, ["order", "transformation", "method", "mode"], audit_values
    )
    benefit_summary = _summarize(
        benefit_rows, ["order", "transformation", "mode"],
        [
            "endpoint_sam_benefit", "progress_auc_sam_benefit",
            "progress_support_width", "path_auc_sam_benefit", "path_support_width",
        ],
    )
    gauge_summary = _summarize(
        gauge_rows, ["order", "transformation", "method", "mode"],
        [
            "endpoint_gauge_damage", "progress_auc_gauge_damage",
            "path_auc_gauge_damage", "path_ratio_to_identity",
            "quadratic_budget_ratio_to_identity",
        ],
    )
    interaction_summary = _summarize(
        interactions, ["order", "transformation", "mode"],
        ["endpoint_interaction", "progress_auc_interaction", "path_auc_interaction"],
    )
    attenuation_summary = _summarize(
        attenuation, ["order", "transformation", "normalized_mode"],
        [
            "endpoint_interaction_attenuation",
            "progress_auc_interaction_attenuation",
            "path_auc_interaction_attenuation",
        ],
    )
    counterfactual_summary = _summarize(
        cf_seed, ["order", "target_scale", "transformation"], cf_values
    )

    formal_inference_rows: list[dict[str, Any]] = []
    formal = config.get("formal_inference", {}) if config is not None else {}
    if formal:
        ci_level = float(formal.get("equivalence_ci_level", 0.90))
        margin = float(formal["progress_auc_interaction_equivalence_margin"])
        selected_scale_for_inference = float(formal.get("selected_target_scale", 1.0))
        selected_mode = f"normalized_q{selected_scale_for_inference:g}"
        primary_groups: dict[str, list[float]] = defaultdict(list)
        for row in benefit_rows:
            if row["transformation"] == "identity" and row["mode"] == selected_mode:
                primary_groups[row["order"]].append(
                    float(row["progress_auc_sam_benefit"])
                )
        for order, values in sorted(primary_groups.items()):
            mean, low, high = _bootstrap_interval(values, ci_level=0.95, seed=7519)
            formal_inference_rows.append(
                {
                    "analysis": "primary_identity_sam_benefit",
                    "order": order,
                    "transformation": "identity",
                    "mode": selected_mode,
                    "metric": "progress_auc_sam_benefit",
                    "n_seeds": len(values),
                    "mean": mean,
                    "ci_level": 0.95,
                    "ci_low": low,
                    "ci_high": high,
                    "equivalence_margin": "",
                    "equivalent_to_zero": "",
                }
            )
        interaction_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in interactions:
            if row["mode"] == selected_mode and row["transformation"] != "orthogonal":
                interaction_groups[(row["order"], row["transformation"])].append(
                    float(row["progress_auc_interaction"])
                )
        for (order, transformation), values in sorted(interaction_groups.items()):
            mean, low, high = _bootstrap_interval(
                values, ci_level=ci_level, seed=8519
            )
            formal_inference_rows.append(
                {
                    "analysis": "gauge_x_sam_equivalence",
                    "order": order,
                    "transformation": transformation,
                    "mode": selected_mode,
                    "metric": "progress_auc_interaction",
                    "n_seeds": len(values),
                    "mean": mean,
                    "ci_level": ci_level,
                    "ci_low": low,
                    "ci_high": high,
                    "equivalence_margin": margin,
                    "equivalent_to_zero": bool(low > -margin and high < margin),
                }
            )

    rules = {
        "maximum_target_relative_error": 1e-5,
        "maximum_candidate_closure_error": 5e-5,
        "maximum_alpha": 16.0,
        "minimum_raw_to_resolved_cosine": 0.995,
        "maximum_counterfactual_median_abs_taylor_residual": 0.01,
        "maximum_counterfactual_p90_abs_taylor_residual": 0.03,
        "minimum_identity_progress_support_width": 0.05,
    }
    if config is not None:
        rules.update(
            config.get("step_normalization", {}).get("selection_rules", {})
        )
    selection_rows = []
    scales = sorted(
        {float(row["target_scale"]) for row in runs if row["target_scale"] is not None}
    )
    orders = sorted({row["order"] for row in runs})
    for order in orders:
        for scale in scales:
            mode = f"normalized_q{scale:g}"
            cells = [row for row in runs if row["order"] == order and row["mode"] == mode]
            residuals = [
                abs(float(row["taylor_residual"]))
                for row in counterfactual_raw
                if _order(row["angles"]) == order
                and float(row["target_scale"]) == scale
            ]
            supports = [
                float(row["progress_support_width"])
                for row in benefit_rows
                if row["order"] == order
                and row["mode"] == mode
                and row["transformation"] == "identity"
            ]
            record = {
                "order": order,
                "target_scale": scale,
                "maximum_target_relative_error": max(
                    float(row["maximum_target_relative_error"]) for row in cells
                ),
                "maximum_candidate_closure_error": max(
                    float(row["maximum_candidate_closure_error"]) for row in cells
                ),
                "maximum_alpha": max(float(row["maximum_alpha"]) for row in cells),
                "minimum_raw_to_resolved_cosine": min(
                    float(row["minimum_raw_to_resolved_cosine"]) for row in cells
                ),
                "counterfactual_median_abs_taylor_residual": float(
                    np.median(residuals)
                ),
                "counterfactual_p90_abs_taylor_residual": float(
                    np.quantile(residuals, 0.9)
                ),
                "minimum_identity_progress_support_width": min(supports),
            }
            record["passes_frozen_rules"] = bool(
                record["maximum_target_relative_error"]
                <= float(rules["maximum_target_relative_error"])
                and record["maximum_candidate_closure_error"]
                <= float(rules["maximum_candidate_closure_error"])
                and record["maximum_alpha"] <= float(rules["maximum_alpha"])
                and record["minimum_raw_to_resolved_cosine"]
                >= float(rules["minimum_raw_to_resolved_cosine"])
                and record["counterfactual_median_abs_taylor_residual"]
                <= float(rules["maximum_counterfactual_median_abs_taylor_residual"])
                and record["counterfactual_p90_abs_taylor_residual"]
                <= float(rules["maximum_counterfactual_p90_abs_taylor_residual"])
                and record["minimum_identity_progress_support_width"]
                >= float(rules["minimum_identity_progress_support_width"])
            )
            selection_rows.append(record)
    passing_scales = [
        scale
        for scale in scales
        if all(
            bool(row["passes_frozen_rules"])
            for row in selection_rows
            if float(row["target_scale"]) == scale
        )
    ]
    selected_scale = max(passing_scales) if passing_scales else None

    integrity = {
        "num_runs": len(runs),
        "num_prefixes": len({(row["order"], row["seed"]) for row in runs}),
        "num_seeds": len({row["seed"] for row in runs}),
        "num_counterfactual_rows": len(counterfactual_raw),
        "max_normalized_target_relative_error": max(
            float(row["maximum_target_relative_error"])
            for row in runs if row["update_mode"] == "normalized"
        ),
        "max_candidate_closure_error": max(
            float(row["maximum_candidate_closure_error"]) for row in runs
        ),
        "max_orthogonal_endpoint_damage_difference": max(
            abs(float(row["endpoint_gauge_damage"]))
            for row in gauge_rows if row["transformation"] == "orthogonal"
        ),
        "selected_scale_by_frozen_nonbenefit_rules": selected_scale,
    }
    _write(runs, output_root / "runs.csv")
    _write(run_summary, output_root / "run_summary.csv")
    _write(benefit_rows, output_root / "sam_benefits.csv")
    _write(benefit_summary, output_root / "sam_benefit_summary.csv")
    _write(gauge_rows, output_root / "gauge_effects.csv")
    _write(gauge_summary, output_root / "gauge_effect_summary.csv")
    _write(interactions, output_root / "sam_interactions.csv")
    _write(interaction_summary, output_root / "sam_interaction_summary.csv")
    _write(attenuation, output_root / "normalization_attenuation.csv")
    _write(attenuation_summary, output_root / "normalization_attenuation_summary.csv")
    _write(counterfactual, output_root / "counterfactual_rows.csv")
    _write(cf_seed, output_root / "counterfactual_seed_means.csv")
    _write(counterfactual_summary, output_root / "counterfactual_summary.csv")
    if formal_inference_rows:
        _write(formal_inference_rows, output_root / "formal_inference_summary.csv")
    _write(selection_rows, output_root / "scale_selection_audit.csv")
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "scale_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "rules": rules,
                "selection_policy": "largest scale passing every rule in every direction",
                "selected_scale": selected_scale,
            },
            handle,
            indent=2,
        )
    with (output_root / "integrity.json").open("w", encoding="utf-8") as handle:
        json.dump(integrity, handle, indent=2)
    print(json.dumps(integrity, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze P3 exact-step normalization")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config")
    args = parser.parse_args()
    analyze(
        args.input,
        args.output,
        load_config(args.config) if args.config else None,
    )


if __name__ == "__main__":
    main()
