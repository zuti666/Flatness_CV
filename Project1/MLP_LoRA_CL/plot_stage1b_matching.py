#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlp_lora_stage1b_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FORWARD = "(0.0, 15.0, 30.0, 45.0, 60.0)"
REVERSE = "(60.0, 45.0, 30.0, 15.0, 0.0)"


def interaction_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        if row["contrast"] != "factor_lora_minus_dense_interaction":
            continue
        for key in ("learning_rate", "sam_rho", "mean", "ci95_low", "ci95_high"):
            row[key] = float(row[key])
        output.append(row)
    return output


def plot_pilot_heatmap(rows: list[dict], output: Path) -> None:
    learning_rates = [0.01, 0.03, 0.1]
    radii = [0.005, 0.02, 0.05]
    metric = "loss_matched_old_loss_damage_reduction"
    index = {
        (row["task_order"], row["learning_rate"], row["sam_rho"]): row["mean"]
        for row in rows if row["metric"] == metric
    }
    matrices = []
    for order in (FORWARD, REVERSE):
        matrices.append(np.asarray([[index[(order, lr, rho)] for rho in radii] for lr in learning_rates]))
    limit = max(abs(value) for matrix in matrices for value in matrix.flat)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    for axis, matrix, title in zip(axes, matrices, ("Forward", "Reverse")):
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_xticks(range(len(radii)), [str(value) for value in radii])
        axis.set_yticks(range(len(learning_rates)), [str(value) for value in learning_rates])
        axis.set_xlabel(r"Effective SAM radius $\rho_W$")
        axis.set_ylabel("Learning rate")
        axis.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:+.3f}",
                          ha="center", va="center", fontsize=9)
    fig.colorbar(
        image,
        ax=list(axes),
        label="LoRA-minus-Dense loss-matched interaction",
        shrink=0.86,
        pad=0.035,
    )
    fig.savefig(output / "stage1b_pilot_loss_matched_heatmap.png", dpi=220)
    plt.close(fig)


def plot_confirmation(rows: list[dict], output: Path) -> None:
    metrics = (
        ("endpoint_accuracy_gain", "Endpoint FAA interaction"),
        ("endpoint_forgetting_reduction", "Endpoint forgetting-reduction interaction"),
        ("loss_matched_old_loss_damage_reduction", "Loss-matched old-loss interaction"),
        ("drift_matched_old_loss_damage_reduction", "Drift-matched old-loss interaction"),
    )
    keys = [(FORWARD, 0.005), (REVERSE, 0.005), (FORWARD, 0.02), (REVERSE, 0.02)]
    labels = ["Forward / .005", "Reverse / .005", "Forward / .02", "Reverse / .02"]
    index = {(row["task_order"], row["sam_rho"], row["metric"]): row for row in rows}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, (metric, title) in zip(axes.flat, metrics):
        selected = [index[(order, radius, metric)] for order, radius in keys]
        means = np.asarray([row["mean"] for row in selected])
        lower = means - np.asarray([row["ci95_low"] for row in selected])
        upper = np.asarray([row["ci95_high"] for row in selected]) - means
        y = np.arange(len(labels))
        axis.errorbar(means, y, xerr=np.asarray([lower, upper]), fmt="o", capsize=4)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "stage1b_confirmation_interactions.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage-1B robustness and confirmation")
    parser.add_argument("--pilot", type=Path, default=Path("analysis/focus_stage1b_matching.csv"))
    parser.add_argument("--confirmation", type=Path, default=Path("analysis/focus_stage1b_confirmation.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_pilot_heatmap(interaction_rows(args.pilot), args.output)
    plot_confirmation(interaction_rows(args.confirmation), args.output)
    print(f"wrote Stage-1B figures to {args.output}")


if __name__ == "__main__":
    main()
