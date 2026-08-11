#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlp_lora_stage1c_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FORWARD = "(-30.0, 30.0)"
REVERSE = "(30.0, -30.0)"
ROW_KEYS = [
    (FORWARD, "dense", "Forward Dense"),
    (FORWARD, "factor_lora", "Forward LoRA"),
    (FORWARD, "factor_lora_minus_dense", "Forward LoRA−Dense"),
    (REVERSE, "dense", "Reverse Dense"),
    (REVERSE, "factor_lora", "Reverse LoRA"),
    (REVERSE, "factor_lora_minus_dense", "Reverse LoRA−Dense"),
]


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("mean", "ci95_low", "ci95_high"):
            row[key] = float(row[key])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot held-out Stage-1C temporal effects")
    parser.add_argument(
        "--input", type=Path, default=Path("analysis/focus_stage1c_confirmation_effects.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    rows = load_rows(args.input)
    index = {
        (row["task_order"], row["effect"], row["metric"], row["contrast"]): row
        for row in rows
    }
    panels = [
        (
            "protection_given_new_sgd",
            "prospective_ggn_curvature",
            "Protection: prospective GGN curvature reduction",
        ),
        (
            "protection_given_new_sgd",
            "loss_matched_old_loss_damage_reduction",
            "Protection: loss-matched old-loss reduction",
        ),
        (
            "trajectory_given_old_sgd",
            "loss_matched_old_loss_damage_reduction",
            "Trajectory: loss-matched old-loss reduction",
        ),
        (
            "trajectory_given_old_sgd",
            "drift_matched_old_loss_damage_reduction",
            "Trajectory: drift-matched old-loss reduction",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    y = np.arange(len(ROW_KEYS))
    for axis, (effect, metric, title) in zip(axes.flat, panels):
        selected = [index[(order, effect, metric, contrast)] for order, contrast, _ in ROW_KEYS]
        means = np.asarray([row["mean"] for row in selected])
        lows = np.asarray([row["ci95_low"] for row in selected])
        highs = np.asarray([row["ci95_high"] for row in selected])
        colors = ["#4C78A8", "#F58518", "#B279A2"] * 2
        for position, mean, low, high, color in zip(y, means, lows, highs, colors):
            axis.errorbar(
                mean,
                position,
                xerr=np.asarray([[mean - low], [high - mean]]),
                fmt="o",
                color=color,
                capsize=4,
            )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.axhline(2.5, color="0.75", linewidth=1)
        axis.set_yticks(y, [label for _, _, label in ROW_KEYS])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlabel("Positive = SAM reduces sensitivity/damage")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "stage1c_confirmation_mechanisms.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
