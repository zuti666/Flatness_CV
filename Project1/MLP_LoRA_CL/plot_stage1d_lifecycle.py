#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlp_lora_stage1d_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot persistent versus merge-reset effects")
    parser.add_argument(
        "--input", type=Path, default=Path("analysis/focus_stage1d_lifecycle.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("mean", "ci95_low", "ci95_high"):
            row[key] = float(row[key])
    orders = ["(-30.0, 30.0)", "(30.0, -30.0)"]
    contrasts = ["persistent", "merge_reset", "merge_reset_minus_persistent"]
    labels = [
        "Forward persistent", "Forward merge-reset", "Forward merge−persistent",
        "Reverse persistent", "Reverse merge-reset", "Reverse merge−persistent",
    ]
    keys = [(order, contrast) for order in orders for contrast in contrasts]
    metrics = [
        ("loss_matched_old_loss_damage_reduction", "Current-loss matched"),
        ("drift_matched_old_loss_damage_reduction", "Effective-drift matched"),
    ]
    index = {(row["task_order"], row["contrast"], row["metric"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        selected = [index[(order, contrast, metric)] for order, contrast in keys]
        means = np.asarray([row["mean"] for row in selected])
        lows = np.asarray([row["ci95_low"] for row in selected])
        highs = np.asarray([row["ci95_high"] for row in selected])
        colors = ["#F58518", "#54A24B", "#B279A2"] * 2
        y = np.arange(len(labels))
        for position, mean, low, high, color in zip(y, means, lows, highs, colors):
            axis.errorbar(
                mean, position,
                xerr=np.asarray([[mean - low], [high - mean]]),
                fmt="o", color=color, capsize=4,
            )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.axhline(2.5, color="0.75", linewidth=1)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("Positive = new-task SAM lowers old-task loss")
        axis.grid(axis="x", alpha=0.25)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "stage1d_lifecycle_matching.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
