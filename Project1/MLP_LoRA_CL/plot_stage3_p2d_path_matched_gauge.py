from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRANSFORMS = ["orthogonal", "scalar_0.5", "scalar_2", "anisotropic_4"]
LABELS = ["Orthogonal", "Scalar .5", "Scalar 2", "Aniso. 4"]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _errorbar(axis, x, cells, key, label, color, offset=0.0) -> None:
    means = np.asarray([float(row[f"{key}_mean"]) for row in cells])
    low = np.asarray([float(row[f"{key}_ci_low"]) for row in cells])
    high = np.asarray([float(row[f"{key}_ci_high"]) for row in cells])
    axis.errorbar(
        x + offset,
        means,
        yerr=[means - low, high - means],
        fmt="o",
        capsize=3,
        color=color,
        label=label,
    )


def plot(analysis: Path, output: Path) -> None:
    contrasts = _read(analysis / "gauge_contrast_summary.csv")
    interactions = _read(analysis / "sam_interaction_summary.csv")
    orders = sorted({row["order"] for row in contrasts})
    colors = ["#3b6fb6", "#d55e00"]
    x = np.arange(len(TRANSFORMS))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for order_index, order in enumerate(orders):
        path_cells = [
            next(row for row in contrasts if row["order"] == order and row["method"] == "sgd" and row["transformation"] == transform)
            for transform in TRANSFORMS
        ]
        interaction_cells = [
            next(row for row in interactions if row["order"] == order and row["transformation"] == transform)
            for transform in TRANSFORMS
        ]
        offset = (order_index - (len(orders) - 1) / 2) * 0.14
        _errorbar(axes[0], x, path_cells, "path_ratio_to_identity", order, colors[order_index], offset)
        _errorbar(
            axes[1],
            x,
            path_cells,
            "step_path_matched_gauge_damage",
            order,
            colors[order_index],
            offset,
        )
        _errorbar(
            axes[2],
            x,
            interaction_cells,
            "current_loss_matched_sam_benefit_interaction",
            order,
            colors[order_index],
            offset,
        )
    axes[0].axhline(1, color="0.4", linewidth=1)
    axes[0].set_ylabel("Cumulative step-path / Identity")
    axes[0].set_title("Independent-cohort path match")
    for axis, title, ylabel in (
        (axes[1], "SGD damage at matched step path", "Matched old-loss damage change"),
        (
            axes[2],
            "Current-loss-matched SAM interaction",
            "Gauge change in matched SAM benefit",
        ),
    ):
        axis.axhline(0, color="0.4", linewidth=1)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    for axis in axes:
        axis.set_xticks(x, LABELS, rotation=24, ha="right")
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "stage3_p2d_path_matched_gauge.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "stage3_p2d_path_matched_gauge.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot P2d path-matched gauges")
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", default=Path("figures"), type=Path)
    args = parser.parse_args()
    plot(args.analysis, args.output)


if __name__ == "__main__":
    main()
