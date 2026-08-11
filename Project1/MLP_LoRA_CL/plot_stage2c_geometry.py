#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlp_lora_stage2c_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def order_parts(label: str) -> tuple[float, str]:
    left, right = ast.literal_eval(label)
    return abs(float(left)), "forward" if left < right else "reverse"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot task-geometry dependence of SAM benefit")
    parser.add_argument(
        "--effects", type=Path, default=Path("analysis/focus_stage2c_effects.csv")
    )
    parser.add_argument(
        "--cells", type=Path, default=Path("analysis/focus_stage2c_geometry_cells.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    effects = read_rows(args.effects)
    cells = read_rows(args.cells)
    for row in effects:
        for key in ("mean", "ci95_low", "ci95_high"):
            row[key] = float(row[key])

    metrics = [
        ("endpoint_old_loss_damage", "Endpoint", "#4C78A8"),
        ("loss_matched_old_loss_damage_reduction", "Loss matched", "#F58518"),
        ("drift_matched_old_loss_damage_reduction", "Drift matched", "#54A24B"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, contrast, title in (
        (axes[0, 0], "factor_lora", "Factor LoRA: SAM old-loss reduction"),
        (
            axes[0, 1],
            "factor_lora_minus_dense",
            "Factor-minus-Dense SAM interaction",
        ),
    ):
        for metric, metric_label, color in metrics:
            for direction, marker, linestyle in (
                ("forward", "o", "-"), ("reverse", "s", "--")
            ):
                selected = []
                for row in effects:
                    magnitude, row_direction = order_parts(row["task_order"])
                    if (
                        row["contrast"] == contrast
                        and row["metric"] == metric
                        and row_direction == direction
                    ):
                        selected.append((magnitude, row))
                selected.sort()
                x = np.asarray([item[0] for item in selected])
                y = np.asarray([item[1]["mean"] for item in selected])
                low = np.asarray([item[1]["ci95_low"] for item in selected])
                high = np.asarray([item[1]["ci95_high"] for item in selected])
                axis.errorbar(
                    x, y, yerr=np.vstack([y - low, high - y]), marker=marker,
                    linestyle=linestyle, color=color, capsize=3,
                    label=f"{metric_label} · {direction}",
                )
        axis.axhline(0.0, color="black", linestyle=":", linewidth=1)
        axis.set_xticks([15, 30, 45])
        axis.set_xlabel("Rotation magnitude (degrees)")
        axis.set_ylabel("Positive = Factor/SAM is safer")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)

    cell_index = {(row["task_order"], row["parameterization"]): row for row in cells}
    effect_index = {
        (row["task_order"], row["contrast"], row["metric"]): row for row in effects
    }
    factor_cells = [
        (order, row) for (order, parameterization), row in cell_index.items()
        if parameterization == "factor_lora"
    ]
    scatters = [
        (
            axes[1, 0], "prospective_old_gradient_cosine__mean",
            "Old-gradient / reachable-new-gradient cosine",
        ),
        (
            axes[1, 1], "prospective_ggn_curvature__mean",
            "Prospective old-task GGN curvature",
        ),
    ]
    for axis, predictor, x_label in scatters:
        for order, cell in sorted(factor_cells):
            magnitude, direction = order_parts(order)
            outcome = effect_index[(order, "factor_lora", "endpoint_old_loss_damage")]["mean"]
            color = "#4C78A8" if direction == "forward" else "#F58518"
            marker = "o" if direction == "forward" else "s"
            axis.scatter(float(cell[predictor]), outcome, color=color, marker=marker, s=55)
            axis.annotate(f"{direction[0].upper()}{magnitude:g}",
                          (float(cell[predictor]), outcome), xytext=(4, 4),
                          textcoords="offset points", fontsize=8)
        axis.axhline(0.0, color="black", linestyle=":", linewidth=1)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Factor LoRA endpoint SAM reduction")
        axis.grid(alpha=0.25)
    axes[1, 0].set_title("Pre-task alignment versus realized benefit")
    axes[1, 1].set_title("Pre-task curvature versus realized benefit")

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "stage2c_task_geometry.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
