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
    return "-".join(("m" if value < 0 else "p") + f"{abs(float(value)):g}" for value in angles)


def _bootstrap(values: list[float], seed: int = 9021) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return mean, float(low), float(high)


def _mean_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _load(input_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    dynamic_rows: list[dict[str, Any]] = []
    for path in sorted(input_root.glob("angles-*/seed_*/transform_*/**/metrics.json")):
        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        dynamic_path = path.parent / "dynamic_geometry.json"
        replicate_path = path.parent / "pathwise_replicates.json"
        if not dynamic_path.exists() or not replicate_path.exists():
            raise RuntimeError(f"P2 diagnostics missing beside {path}")
        with dynamic_path.open(encoding="utf-8") as handle:
            dynamic = json.load(handle)
        with replicate_path.open(encoding="utf-8") as handle:
            replicates = json.load(handle)

        order = _order_label(metrics["angles"])
        common = {
            "order": order,
            "seed": int(metrics["seed"]),
            "transformation": metrics["transformation"],
            "method": metrics["optimizer_method"],
            "sam_rho": metrics["sam_rho"],
        }
        by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in dynamic:
            batch_id = int(item["diagnostic_batch"])
            record = {**common, "diagnostic_batch": batch_id, **item}
            dynamic_rows.append(record)
            by_batch[batch_id].append(item)

        for replicate in replicates:
            batch_id = int(replicate["diagnostic_batch"])
            pathwise = replicate["pathwise"]
            points = sorted(by_batch[batch_id], key=lambda item: int(item["epoch"]))
            pre_segment = [item for item in points if int(item["epoch"]) < int(metrics["continuation_epochs"])]
            if len(points) != int(metrics["continuation_epochs"]) + 1:
                raise RuntimeError(f"Unexpected dynamic trajectory length beside {path}")
            rows.append(
                {
                    **common,
                    "diagnostic_batch": batch_id,
                    "actual_loss_change": pathwise["actual_loss_change_sum"],
                    "interference_I": pathwise["interference_sum"],
                    "directional_curvature_C": pathwise["directional_curvature_sum"],
                    "taylor_residual_R": pathwise["taylor_residual_sum"],
                    "ggn_cost": pathwise["ggn_directional_cost_sum"],
                    "epoch_path_length": pathwise["path_length"],
                    "static_predicted_interference": points[0]["predicted_sgd_old_interference"],
                    "dynamic_predicted_interference_mean": float(
                        np.mean([item["predicted_sgd_old_interference"] for item in pre_segment])
                    ),
                    "dynamic_pullback_cosine_mean": float(
                        np.mean([item["pullback_old_new_cosine"] for item in pre_segment])
                    ),
                    "dynamic_pullback_energy_mean": float(
                        np.mean([item["pullback_new_energy"] for item in pre_segment])
                    ),
                    "start_factor_condition": points[0]["factor_jacobian_condition"],
                    "start_log_condition": float(
                        np.log(max(float(points[0]["factor_jacobian_condition"]), 1e-20))
                    ),
                    "dynamic_log_condition_mean": float(
                        np.mean([np.log(max(float(item["factor_jacobian_condition"]), 1e-20)) for item in pre_segment])
                    ),
                    "dynamic_operator_norm_mean": float(
                        np.mean([item["factor_jacobian_operator_norm"] for item in pre_segment])
                    ),
                    "factor_a_norm_change": float(points[-1]["factor_a_norm"] - points[0]["factor_a_norm"]),
                    "factor_b_norm_change": float(points[-1]["factor_b_norm"] - points[0]["factor_b_norm"]),
                    "mean_actual_bilinear_ratio": metrics["step_geometry"]["mean_bilinear_to_linear_ratio"],
                    "effective_drift": metrics["effective_drift"],
                    "current_loss": metrics["final_current_loss"],
                    "full_test_continuation_old_damage": metrics["continuation_old_loss_damage"],
                    "path": str(path),
                }
            )
    return rows, dynamic_rows


FEATURE_KEYS = [
    "actual_loss_change",
    "interference_I",
    "directional_curvature_C",
    "taylor_residual_R",
    "ggn_cost",
    "epoch_path_length",
    "static_predicted_interference",
    "dynamic_predicted_interference_mean",
    "dynamic_pullback_cosine_mean",
    "dynamic_pullback_energy_mean",
    "start_log_condition",
    "dynamic_log_condition_mean",
    "dynamic_operator_norm_mean",
    "factor_a_norm_change",
    "factor_b_norm_change",
    "mean_actual_bilinear_ratio",
    "effective_drift",
    "current_loss",
    "full_test_continuation_old_damage",
]


def _contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["order"], row["seed"], row["method"], row["diagnostic_batch"])].append(row)
    output = []
    for group, items in groups.items():
        identity = next(item for item in items if item["transformation"] == "identity")
        for item in items:
            if item["transformation"] == "identity":
                continue
            record = {
                "order": group[0],
                "seed": group[1],
                "method": group[2],
                "diagnostic_batch": group[3],
                "transformation": item["transformation"],
            }
            for key in FEATURE_KEYS:
                record[f"{key}_change"] = float(item[key]) - float(identity[key])
            output.append(record)
    return output


def _average_batches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    id_keys = ["order", "seed", "method", "transformation"]
    for row in rows:
        groups[tuple(row[key] for key in id_keys)].append(row)
    value_keys = [key for key in rows[0] if key.endswith("_change")]
    return [
        {**dict(zip(id_keys, group)), "num_diagnostic_batches": len(items), **_mean_rows(items, value_keys)}
        for group, items in groups.items()
    ]


def _sam_interactions(contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["order"], row["seed"], row["diagnostic_batch"], row["transformation"], row["method"]): row
        for row in contrasts
    }
    output = []
    for key, sgd in by_key.items():
        if key[-1] != "sgd":
            continue
        sam = by_key.get((*key[:-1], "sam"))
        if sam is None:
            continue
        output.append(
            {
                "order": key[0],
                "seed": key[1],
                "diagnostic_batch": key[2],
                "transformation": key[3],
                "sam_benefit_interaction": float(sgd["actual_loss_change_change"])
                - float(sam["actual_loss_change_change"]),
                "sam_I_reduction_interaction": float(sgd["interference_I_change"])
                - float(sam["interference_I_change"]),
                "sam_C_reduction_interaction": float(sgd["directional_curvature_C_change"])
                - float(sam["directional_curvature_C_change"]),
            }
        )
    return output


def _effect_summaries(
    contrasts: list[dict[str, Any]], sam: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows = [
        {
            "order": row["order"],
            "seed": row["seed"],
            "diagnostic_batch": row["diagnostic_batch"],
            "transformation": row["transformation"],
            "effect": f"{row['method']}_gauge_damage",
            "value": row["actual_loss_change_change"],
        }
        for row in contrasts
    ]
    long_rows.extend(
        {
            "order": row["order"],
            "seed": row["seed"],
            "diagnostic_batch": row["diagnostic_batch"],
            "transformation": row["transformation"],
            "effect": "sam_benefit_interaction",
            "value": row["sam_benefit_interaction"],
        }
        for row in sam
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        groups[(row["order"], row["transformation"], row["effect"], row["diagnostic_batch"])].append(row)
    batch_summary = []
    for group, items in sorted(groups.items(), key=lambda pair: str(pair[0])):
        mean, low, high = _bootstrap([float(item["value"]) for item in items])
        batch_summary.append(
            {
                "order": group[0],
                "transformation": group[1],
                "effect": group[2],
                "diagnostic_batch": group[3],
                "n_seeds": len(items),
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )

    pooled_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        pooled_groups[(row["order"], row["transformation"], row["effect"], row["seed"])].append(row)
    seed_means = [
        {
            "order": group[0],
            "transformation": group[1],
            "effect": group[2],
            "seed": group[3],
            "value": float(np.mean([float(item["value"]) for item in items])),
        }
        for group, items in pooled_groups.items()
    ]
    pooled: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_means:
        pooled[(row["order"], row["transformation"], row["effect"])].append(row)
    robustness = []
    for group, items in sorted(pooled.items(), key=lambda pair: str(pair[0])):
        mean, low, high = _bootstrap([float(item["value"]) for item in items])
        batches = [
            item for item in batch_summary
            if (item["order"], item["transformation"], item["effect"]) == group
        ]
        sign = np.sign(mean)
        robustness.append(
            {
                "order": group[0],
                "transformation": group[1],
                "effect": group[2],
                "n_seeds": len(items),
                "seed_mean": mean,
                "ci_low": low,
                "ci_high": high,
                "batches_same_sign": int(sum(np.sign(float(item["mean"])) == sign for item in batches)),
                "batches_ci_excludes_zero": int(sum(float(item["ci_low"]) * float(item["ci_high"]) > 0 for item in batches)),
                "num_batches": len(batches),
            }
        )
    return batch_summary, robustness


def _leave_one_seed_out(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_sets = {
        "constant": [],
        "static_start_pullback": ["static_predicted_interference_change"],
        "static_pullback_condition": [
            "static_predicted_interference_change",
            "start_log_condition_change",
        ],
        "dynamic_pullback": ["dynamic_predicted_interference_mean_change"],
        "dynamic_pullback_condition": [
            "dynamic_predicted_interference_mean_change",
            "dynamic_log_condition_mean_change",
        ],
        "dynamic_plus_bilinear_posthoc": [
            "dynamic_predicted_interference_mean_change",
            "dynamic_log_condition_mean_change",
            "mean_actual_bilinear_ratio_change",
        ],
        "oracle_I_C": ["interference_I_change", "directional_curvature_C_change"],
    }
    predictions = []
    summaries = []
    for order in sorted({row["order"] for row in rows}):
        for method in sorted({row["method"] for row in rows}):
            subset = [row for row in rows if row["order"] == order and row["method"] == method]
            seeds = sorted({int(row["seed"]) for row in subset})
            if len(seeds) < 2:
                continue
            for model_name, features in feature_sets.items():
                model_rows = []
                if model_name == "oracle_I_C":
                    for row in subset:
                        value = float(row["interference_I_change"]) + float(
                            row["directional_curvature_C_change"]
                        )
                        record = {
                            "order": order,
                            "method": method,
                            "seed": int(row["seed"]),
                            "transformation": row["transformation"],
                            "model": model_name,
                            "actual": row["actual_loss_change_change"],
                            "predicted": float(value),
                        }
                        predictions.append(record)
                        model_rows.append(record)
                else:
                    for held_seed in seeds:
                        train = [row for row in subset if int(row["seed"]) != held_seed]
                        test = [row for row in subset if int(row["seed"]) == held_seed]
                        train_x = np.asarray([[float(row[key]) for key in features] for row in train], dtype=float).reshape(len(train), len(features))
                        test_x = np.asarray([[float(row[key]) for key in features] for row in test], dtype=float).reshape(len(test), len(features))
                        mean = train_x.mean(axis=0) if features else np.empty(0)
                        scale = train_x.std(axis=0) if features else np.empty(0)
                        scale[scale < 1e-12] = 1
                        train_design = np.column_stack([np.ones(len(train)), (train_x - mean) / scale])
                        test_design = np.column_stack([np.ones(len(test)), (test_x - mean) / scale])
                        target = np.asarray([float(row["actual_loss_change_change"]) for row in train])
                        coefficients = np.linalg.lstsq(train_design, target, rcond=None)[0]
                        predicted = test_design @ coefficients
                        for row, value in zip(test, predicted):
                            record = {
                                "order": order,
                                "method": method,
                                "seed": held_seed,
                                "transformation": row["transformation"],
                                "model": model_name,
                                "actual": row["actual_loss_change_change"],
                                "predicted": float(value),
                            }
                            predictions.append(record)
                            model_rows.append(record)
                actual = np.asarray([float(row["actual"]) for row in model_rows])
                predicted = np.asarray([float(row["predicted"]) for row in model_rows])
                residual = actual - predicted
                denominator = ((actual - actual.mean()) ** 2).sum()
                summaries.append(
                    {
                        "order": order,
                        "method": method,
                        "model": model_name,
                        "n_predictions": len(actual),
                        "mae": float(np.abs(residual).mean()),
                        "rmse": float(np.sqrt(np.mean(residual**2))),
                        "r2": float(1 - (residual**2).sum() / max(float(denominator), 1e-20)),
                        "sign_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
                    }
                )
    return predictions, summaries


def analyze(input_root: Path, output_root: Path) -> None:
    rows, dynamic = _load(input_root)
    if not rows:
        raise RuntimeError(f"No P2 results found under {input_root}")
    contrasts = _contrasts(rows)
    contrast_means = _average_batches(contrasts)
    sam = _sam_interactions(contrasts)
    batch_summary, robustness = _effect_summaries(contrasts, sam)
    predictions, predictor_summary = _leave_one_seed_out(contrast_means)

    _write(rows, output_root / "runs_by_batch.csv")
    _write(dynamic, output_root / "dynamic_epochs.csv")
    _write(contrasts, output_root / "gauge_contrasts_by_batch.csv")
    _write(contrast_means, output_root / "gauge_contrasts_seed_mean.csv")
    _write(sam, output_root / "sam_interactions_by_batch.csv")
    _write(batch_summary, output_root / "batch_effect_summary.csv")
    _write(robustness, output_root / "batch_robustness.csv")
    _write(predictions, output_root / "predictor_loso.csv")
    _write(predictor_summary, output_root / "predictor_summary.csv")

    identity_orth = [row for row in contrasts if row["transformation"] == "orthogonal"]
    integrity = {
        "num_run_batch_rows": len(rows),
        "num_dynamic_rows": len(dynamic),
        "num_unique_runs": len({(row["order"], row["seed"], row["transformation"], row["method"]) for row in rows}),
        "num_seeds": len({row["seed"] for row in rows}),
        "num_diagnostic_batches": len({row["diagnostic_batch"] for row in rows}),
        "max_taylor_closure_error": max(abs(float(row["actual_loss_change"]) - float(row["interference_I"]) - float(row["directional_curvature_C"]) - float(row["taylor_residual_R"])) for row in rows),
        "max_orthogonal_actual_change": max(abs(float(row["actual_loss_change_change"])) for row in identity_orth),
        "max_orthogonal_dynamic_pullback_change": max(abs(float(row["dynamic_predicted_interference_mean_change"])) for row in identity_orth),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "integrity.json").open("w", encoding="utf-8") as handle:
        json.dump(integrity, handle, indent=2)
    print(json.dumps(integrity, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze P2 dynamic factor-chart diagnostics")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.input, args.output)


if __name__ == "__main__":
    main()
