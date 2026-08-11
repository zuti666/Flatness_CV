#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlp_lora_stage1e_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage-1E parameterization controls")
    parser.add_argument(
        "--input", type=Path, default=Path("analysis/focus_stage1e_parameterizations.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("mean", "ci95_low", "ci95_high"):
            row[key] = float(row[key])
    orders = [("(-30.0, 30.0)", "Forward"), ("(30.0, -30.0)", "Reverse")]
    parameters = [
        ("dense", "Dense"),
        ("random_subspace", "Random subspace"),
        ("fixed_lora_tangent", "Fixed LoRA tangent"),
        ("factor_lora", "Factor LoRA"),
    ]
    keys = [(order, parameter) for order, _ in orders for parameter, _ in parameters]
    labels = [f"{order_label} {parameter_label}" for _, order_label in orders for _, parameter_label in parameters]
    panels = [
        ("loss_matched_old_loss_damage_reduction", "Current-loss matched old-loss reduction"),
        ("drift_matched_old_loss_damage_reduction", "Effective-drift matched old-loss reduction"),
        ("pathwise_interference_sum", "Pathwise first-order reduction"),
        ("pathwise_directional_curvature_sum", "Pathwise directional-curvature reduction"),
    ]
    index = {
        (row["task_order"], row["contrast"], row["metric"]): row
        for row in rows if row["effect"] == "trajectory_given_old_sgd"
    }
    colors = ["#4C78A8", "#72B7B2", "#B279A2", "#F58518"] * 2
    y = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, panels):
        selected = [index[(order, parameter, metric)] for order, parameter in keys]
        means = np.asarray([row["mean"] for row in selected])
        lows = np.asarray([row["ci95_low"] for row in selected])
        highs = np.asarray([row["ci95_high"] for row in selected])
        for position, mean, low, high, color in zip(y, means, lows, highs, colors):
            axis.errorbar(
                mean, position,
                xerr=np.asarray([[mean - low], [high - mean]]),
                fmt="o", color=color, capsize=4,
            )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.axhline(3.5, color="0.75", linewidth=1)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("Positive = new-task SAM is safer")
        axis.grid(axis="x", alpha=0.25)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "stage1e_parameterization_controls.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
