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


def _bootstrap(values: list[float], seed: int = 2603) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, float(low), float(high)


def analyze(input_root: Path, output_root: Path) -> None:
    runs = []
    alignment_rows = []
    for path in sorted(input_root.glob("angles-*/seed_*/transform_*/**/metrics.json")):
        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        with (path.parent / "fine_pathwise.json").open(encoding="utf-8") as handle:
            fine = json.load(handle)
        with (path.parent / "sampled_step_alignment.json").open(encoding="utf-8") as handle:
            alignment = json.load(handle)
        coarse = metrics["pathwise"]
        common = {
            "order": _order(metrics["angles"]),
            "seed": int(metrics["seed"]),
            "transformation": metrics["transformation"],
            "method": metrics["optimizer_method"],
        }
        for item in alignment:
            alignment_rows.append({**common, **item})
        actual = np.asarray([float(item["actual_step_interference"]) for item in alignment])
        prediction = np.asarray([float(item["clean_plus_bilinear_prediction"]) for item in alignment])
        residual = actual - prediction
        denominator = ((actual - actual.mean()) ** 2).sum()
        runs.append(
            {
                **common,
                "coarse_num_segments": coarse["num_segments"],
                "fine_num_segments": fine["num_segments"],
                "coarse_actual": coarse["actual_loss_change_sum"],
                "fine_actual": fine["actual_loss_change_sum"],
                "coarse_I": coarse["interference_sum"],
                "fine_I": fine["interference_sum"],
                "coarse_C": coarse["directional_curvature_sum"],
                "fine_C": fine["directional_curvature_sum"],
                "coarse_R": coarse["taylor_residual_sum"],
                "fine_R": fine["taylor_residual_sum"],
                "coarse_abs_R": abs(float(coarse["taylor_residual_sum"])),
                "fine_abs_R": abs(float(fine["taylor_residual_sum"])),
                "absolute_residual_reduction": abs(float(coarse["taylor_residual_sum"])) - abs(float(fine["taylor_residual_sum"])),
                "fine_to_coarse_abs_residual_ratio": abs(float(fine["taylor_residual_sum"])) / max(abs(float(coarse["taylor_residual_sum"])), 1e-20),
                "sampled_steps": len(alignment),
                "clean_prediction_mae": float(np.abs(residual).mean()),
                "clean_prediction_relative_mae": float(np.abs(residual).mean() / max(np.abs(actual).mean(), 1e-20)),
                "clean_prediction_r2": float(1 - (residual**2).sum() / max(float(denominator), 1e-20)),
                "mean_signed_clean_prediction_gap": float(residual.mean()),
            }
        )
    if not runs:
        raise RuntimeError(f"No P2b results found under {input_root}")

    value_keys = [
        "coarse_R",
        "fine_R",
        "coarse_abs_R",
        "fine_abs_R",
        "absolute_residual_reduction",
        "fine_to_coarse_abs_residual_ratio",
        "clean_prediction_mae",
        "clean_prediction_relative_mae",
        "clean_prediction_r2",
        "mean_signed_clean_prediction_gap",
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        groups[(row["order"], row["transformation"], row["method"])].append(row)
    summary = []
    for group, items in sorted(groups.items(), key=lambda pair: str(pair[0])):
        record = {
            "order": group[0],
            "transformation": group[1],
            "method": group[2],
            "n_seeds": len(items),
        }
        for index, key in enumerate(value_keys):
            mean, low, high = _bootstrap([float(item[key]) for item in items], 2603 + index)
            record[f"{key}_mean"] = mean
            record[f"{key}_ci_low"] = low
            record[f"{key}_ci_high"] = high
        summary.append(record)

    integrity = {
        "num_runs": len(runs),
        "num_alignment_rows": len(alignment_rows),
        "num_seeds": len({row["seed"] for row in runs}),
        "max_coarse_fine_actual_error": max(abs(float(row["coarse_actual"]) - float(row["fine_actual"])) for row in runs),
        "max_sgd_clean_prediction_absolute_error": max(
            abs(float(row["clean_prediction_error"]))
            for row in alignment_rows
            if row["method"] == "sgd"
        ),
    }
    _write(runs, output_root / "runs.csv")
    _write(alignment_rows, output_root / "sampled_step_alignment.csv")
    _write(summary, output_root / "summary.csv")
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "integrity.json").open("w", encoding="utf-8") as handle:
        json.dump(integrity, handle, indent=2)
    print(json.dumps(integrity, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze P2b fine Taylor paths")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.input, args.output)


if __name__ == "__main__":
    main()
