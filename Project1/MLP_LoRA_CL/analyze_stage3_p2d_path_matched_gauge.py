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


def _order(angles: list[float]) -> str:
    return "-".join(("m" if value < 0 else "p") + f"{abs(float(value)):g}" for value in angles)


def _bootstrap(values: list[float], seed: int = 4409) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return mean, float(low), float(high)


def _summarize(rows: list[dict[str, Any]], group_keys: list[str], value_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, items in sorted(groups.items(), key=lambda pair: str(pair[0])):
        record = {**dict(zip(group_keys, group)), "n_seeds": len(items)}
        for index, key in enumerate(value_keys):
            mean, low, high = _bootstrap([float(item[key]) for item in items], 4409 + index)
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


def analyze(input_root: Path, output_root: Path) -> None:
    runs = []
    trajectories: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for path in sorted(input_root.glob("angles-*/seed_*/transform_*/**/metrics.json")):
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        runs.append(
            {
                "order": _order(value["angles"]),
                "seed": int(value["seed"]),
                "transformation": value["transformation"],
                "method": value["optimizer_method"],
                "continuation_lr": value["continuation_lr"],
                "old_damage": value["continuation_old_loss_damage"],
                "current_loss": value["final_current_loss"],
                "effective_drift": value["effective_drift"],
                "step_path_length": value["cumulative_step_path_length"],
                "interference_I": value["pathwise"]["interference_sum"],
                "directional_curvature_C": value["pathwise"]["directional_curvature_sum"],
                "path": str(path),
            }
        )
        with (path.parent / "trajectory_history.json").open(encoding="utf-8") as handle:
            trajectories[
                (
                    runs[-1]["order"],
                    runs[-1]["seed"],
                    runs[-1]["method"],
                    runs[-1]["transformation"],
                )
            ] = json.load(handle)
    if not runs:
        raise RuntimeError(f"No P2d metrics under {input_root}")
    by_key = {
        (row["order"], row["seed"], row["method"], row["transformation"]): row
        for row in runs
    }
    contrasts = []
    for row in runs:
        if row["transformation"] == "identity":
            continue
        identity = by_key[(row["order"], row["seed"], row["method"], "identity")]
        row_trajectory = trajectories[
            (row["order"], row["seed"], row["method"], row["transformation"])
        ]
        identity_trajectory = trajectories[
            (row["order"], row["seed"], row["method"], "identity")
        ]
        current_target = max(float(row["current_loss"]), float(identity["current_loss"]))
        path_target = min(float(row["step_path_length"]), float(identity["step_path_length"]))
        drift_target = min(float(row["effective_drift"]), float(identity["effective_drift"]))
        contrasts.append(
            {
                "order": row["order"],
                "seed": row["seed"],
                "method": row["method"],
                "transformation": row["transformation"],
                "path_ratio_to_identity": float(row["step_path_length"]) / max(float(identity["step_path_length"]), 1e-20),
                "path_difference": float(row["step_path_length"]) - float(identity["step_path_length"]),
                "gauge_old_damage": float(row["old_damage"]) - float(identity["old_damage"]),
                "gauge_current_loss": float(row["current_loss"]) - float(identity["current_loss"]),
                "gauge_effective_drift": float(row["effective_drift"]) - float(identity["effective_drift"]),
                "gauge_I": float(row["interference_I"]) - float(identity["interference_I"]),
                "gauge_C": float(row["directional_curvature_C"]) - float(identity["directional_curvature_C"]),
                "current_loss_match_target": current_target,
                "current_loss_matched_gauge_damage": _first_interpolation(
                    row_trajectory,
                    "current_loss",
                    "continuation_old_loss_damage",
                    current_target,
                )
                - _first_interpolation(
                    identity_trajectory,
                    "current_loss",
                    "continuation_old_loss_damage",
                    current_target,
                ),
                "step_path_match_target": path_target,
                "step_path_matched_gauge_damage": _first_interpolation(
                    row_trajectory,
                    "cumulative_step_path_length",
                    "continuation_old_loss_damage",
                    path_target,
                )
                - _first_interpolation(
                    identity_trajectory,
                    "cumulative_step_path_length",
                    "continuation_old_loss_damage",
                    path_target,
                ),
                "drift_match_target": drift_target,
                "drift_matched_gauge_damage": _first_interpolation(
                    row_trajectory,
                    "effective_drift",
                    "continuation_old_loss_damage",
                    drift_target,
                )
                - _first_interpolation(
                    identity_trajectory,
                    "effective_drift",
                    "continuation_old_loss_damage",
                    drift_target,
                ),
            }
        )

    benefits = {}
    for row in runs:
        key = (row["order"], row["seed"], row["transformation"])
        benefits.setdefault(key, {})[row["method"]] = row
    interactions = []

    def matched_sam_benefit(
        order: str, seed: int, transformation: str, x_key: str
    ) -> float:
        sgd = trajectories[(order, seed, "sgd", transformation)]
        sam = trajectories[(order, seed, "sam", transformation)]
        if x_key == "current_loss":
            target = max(float(sgd[-1][x_key]), float(sam[-1][x_key]))
        else:
            target = min(float(sgd[-1][x_key]), float(sam[-1][x_key]))
        return _first_interpolation(
            sgd, x_key, "continuation_old_loss_damage", target
        ) - _first_interpolation(
            sam, x_key, "continuation_old_loss_damage", target
        )

    for key, methods in benefits.items():
        if "sgd" not in methods or "sam" not in methods:
            continue
        benefit = float(methods["sgd"]["old_damage"]) - float(methods["sam"]["old_damage"])
        if key[2] == "identity":
            continue
        identity = benefits[(key[0], key[1], "identity")]
        identity_benefit = float(identity["sgd"]["old_damage"]) - float(identity["sam"]["old_damage"])
        current_matched = matched_sam_benefit(
            key[0], key[1], key[2], "current_loss"
        )
        identity_current_matched = matched_sam_benefit(
            key[0], key[1], "identity", "current_loss"
        )
        path_matched = matched_sam_benefit(
            key[0], key[1], key[2], "cumulative_step_path_length"
        )
        identity_path_matched = matched_sam_benefit(
            key[0], key[1], "identity", "cumulative_step_path_length"
        )
        drift_matched = matched_sam_benefit(
            key[0], key[1], key[2], "effective_drift"
        )
        identity_drift_matched = matched_sam_benefit(
            key[0], key[1], "identity", "effective_drift"
        )
        interactions.append(
            {
                "order": key[0],
                "seed": key[1],
                "transformation": key[2],
                "sam_benefit": benefit,
                "identity_sam_benefit": identity_benefit,
                "sam_benefit_interaction": benefit - identity_benefit,
                "current_loss_matched_sam_benefit": current_matched,
                "current_loss_matched_sam_benefit_interaction": current_matched
                - identity_current_matched,
                "step_path_matched_sam_benefit": path_matched,
                "step_path_matched_sam_benefit_interaction": path_matched
                - identity_path_matched,
                "drift_matched_sam_benefit": drift_matched,
                "drift_matched_sam_benefit_interaction": drift_matched
                - identity_drift_matched,
            }
        )

    contrast_summary = _summarize(
        contrasts,
        ["order", "method", "transformation"],
        [
            "path_ratio_to_identity",
            "gauge_old_damage",
            "gauge_current_loss",
            "gauge_effective_drift",
            "gauge_I",
            "gauge_C",
            "current_loss_matched_gauge_damage",
            "step_path_matched_gauge_damage",
            "drift_matched_gauge_damage",
        ],
    )
    interaction_summary = _summarize(
        interactions,
        ["order", "transformation"],
        [
            "sam_benefit",
            "identity_sam_benefit",
            "sam_benefit_interaction",
            "current_loss_matched_sam_benefit",
            "current_loss_matched_sam_benefit_interaction",
            "step_path_matched_sam_benefit",
            "step_path_matched_sam_benefit_interaction",
            "drift_matched_sam_benefit",
            "drift_matched_sam_benefit_interaction",
        ],
    )
    integrity = {
        "num_runs": len(runs),
        "num_seeds": len({row["seed"] for row in runs}),
        "num_prefixes": len({(row["order"], row["seed"]) for row in runs}),
        "max_orthogonal_path_ratio_error": max(
            abs(float(row["path_ratio_to_identity"]) - 1)
            for row in contrasts if row["transformation"] == "orthogonal"
        ),
        "max_orthogonal_old_damage": max(
            abs(float(row["gauge_old_damage"]))
            for row in contrasts if row["transformation"] == "orthogonal"
        ),
    }
    _write(runs, output_root / "runs.csv")
    _write(contrasts, output_root / "gauge_contrasts.csv")
    _write(contrast_summary, output_root / "gauge_contrast_summary.csv")
    _write(interactions, output_root / "sam_interactions.csv")
    _write(interaction_summary, output_root / "sam_interaction_summary.csv")
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "integrity.json").open("w", encoding="utf-8") as handle:
        json.dump(integrity, handle, indent=2)
    print(json.dumps(integrity, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze path-matched mature gauges")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.input, args.output)


if __name__ == "__main__":
    main()
