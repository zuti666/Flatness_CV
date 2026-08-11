from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODE_ORDER = ["raw", "normalized_q0.25", "normalized_q0.5", "normalized_q1"]
MODE_LABELS = ["Raw", "q=.25", "q=.5", "q=1"]
COLORS = {"m30-p30": "#3b6fb6", "p45-m45": "#d55e00"}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cell(
    rows: list[dict[str, str]], order: str, transformation: str, mode: str
) -> dict[str, str] | None:
    return next(
        (
            row
            for row in rows
            if row["order"] == order
            and row["transformation"] == transformation
            and row["mode"] == mode
        ),
        None,
    )


def _mode_errorbars(
    axis,
    rows: list[dict[str, str]],
    transformation: str,
    key: str,
    title: str,
) -> None:
    orders = sorted({row["order"] for row in rows})
    for order in orders:
        cells = [
            _cell(rows, order, transformation, mode)
            for mode in MODE_ORDER
        ]
        valid = [(index, cell) for index, cell in enumerate(cells) if cell is not None]
        x = np.asarray([index for index, _ in valid], dtype=float)
        means = np.asarray([float(cell[f"{key}_mean"]) for _, cell in valid])
        lows = np.asarray([float(cell[f"{key}_ci_low"]) for _, cell in valid])
        highs = np.asarray([float(cell[f"{key}_ci_high"]) for _, cell in valid])
        axis.errorbar(
            x,
            means,
            yerr=[means - lows, highs - means],
            marker="o",
            capsize=3,
            linewidth=1.4,
            color=COLORS.get(order),
            label=order,
        )
    axis.axhline(0, color="0.45", linewidth=1)
    axis.set_xticks(range(len(MODE_ORDER)), MODE_LABELS)
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=8)


def plot(analysis: Path, output: Path) -> None:
    benefits = _read(analysis / "sam_benefit_summary.csv")
    interactions = _read(analysis / "sam_interaction_summary.csv")
    counterfactual = _read(analysis / "counterfactual_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.3))

    _mode_errorbars(
        axes[0, 0],
        benefits,
        "identity",
        "progress_auc_sam_benefit",
        "Identity Factor: progress-AUC SAM benefit",
    )
    axes[0, 0].set_ylabel("SGD damage − SAM damage")
    _mode_errorbars(
        axes[0, 1],
        interactions,
        "scalar_0.5",
        "progress_auc_interaction",
        "Scalar .5 × SAM interaction",
    )
    axes[0, 1].set_ylabel("Gauge change in SAM benefit")
    _mode_errorbars(
        axes[1, 0],
        interactions,
        "scalar_2",
        "progress_auc_interaction",
        "Scalar 2 × SAM interaction",
    )
    axes[1, 0].set_ylabel("Gauge change in SAM benefit")

    axis = axes[1, 1]
    markers = {0.25: "o", 0.5: "s", 1.0: "^"}
    identity = [row for row in counterfactual if row["transformation"] == "identity"]
    for row in identity:
        order = row["order"]
        scale = float(row["target_scale"])
        x = float(row["new_task_cost_mean"])
        y = float(row["old_direction_safety_mean"])
        xlow = float(row["new_task_cost_ci_low"])
        xhigh = float(row["new_task_cost_ci_high"])
        ylow = float(row["old_direction_safety_ci_low"])
        yhigh = float(row["old_direction_safety_ci_high"])
        axis.errorbar(
            x,
            y,
            xerr=[[x - xlow], [xhigh - x]],
            yerr=[[y - ylow], [yhigh - y]],
            marker=markers.get(scale, "o"),
            color=COLORS.get(order),
            capsize=2,
            linestyle="none",
            label=f"{order}, q={scale:g}",
        )
    axis.axhline(0, color="0.45", linewidth=1)
    axis.axvline(0, color="0.45", linewidth=1)
    axis.set_xlabel("New-task cost of SAM direction")
    axis.set_ylabel("Old-task safety of SAM direction")
    axis.set_title("Shared-state, equal-step counterfactual")
    axis.legend(frameon=False, fontsize=7, ncol=2)

    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "stage3_p3_step_normalized.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "stage3_p3_step_normalized.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot P3 exact-step normalization")
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", default=Path("figures"), type=Path)
    args = parser.parse_args()
    plot(args.analysis, args.output)


if __name__ == "__main__":
    main()
