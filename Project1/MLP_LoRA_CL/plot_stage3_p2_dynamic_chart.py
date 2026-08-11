from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRANSFORMS = ["orthogonal", "scalar_0.5", "scalar_2", "anisotropic_4"]
TRANSFORM_LABELS = ["Orthogonal", "Scalar .5", "Scalar 2", "Aniso. 4"]
MODELS = [
    "static_start_pullback",
    "static_pullback_condition",
    "dynamic_pullback",
    "dynamic_pullback_condition",
    "dynamic_plus_bilinear_posthoc",
    "oracle_I_C",
]
MODEL_LABELS = [
    "Static",
    "Static + cond.",
    "Dynamic",
    "Dynamic + cond.",
    "+ bilinear\n(post-hoc)",
    "I + C\noracle",
]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot(analysis: Path, output: Path) -> None:
    robustness = _read(analysis / "batch_robustness.csv")
    batch = _read(analysis / "batch_effect_summary.csv")
    predictors = _read(analysis / "predictor_summary.csv")
    orders = sorted({row["order"] for row in robustness})
    colors = {"p45-m45": "#d55e00", "m30-p30": "#3b6fb6"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    x = np.arange(len(TRANSFORMS))
    for index, order in enumerate(orders):
        cells = [
            next(
                row for row in robustness
                if row["order"] == order
                and row["transformation"] == transform
                and row["effect"] == "sam_benefit_interaction"
            )
            for transform in TRANSFORMS
        ]
        means = np.asarray([float(row["seed_mean"]) for row in cells])
        low = np.asarray([float(row["ci_low"]) for row in cells])
        high = np.asarray([float(row["ci_high"]) for row in cells])
        offset = (index - (len(orders) - 1) / 2) * 0.15
        axes[0].errorbar(
            x + offset,
            means,
            yerr=[means - low, high - means],
            fmt="o",
            capsize=3,
            label=order,
            color=colors.get(order),
        )
    axes[0].axhline(0, color="0.4", linewidth=1)
    axes[0].set_xticks(x, TRANSFORM_LABELS, rotation=24, ha="right")
    axes[0].set_ylabel("Gauge change in SAM benefit")
    axes[0].set_title("Seed-level effect (4 batches averaged)")
    axes[0].legend(frameon=False)

    batch_x = np.arange(4)
    for order in orders:
        cells = sorted(
            [
                row for row in batch
                if row["order"] == order
                and row["transformation"] == "scalar_2"
                and row["effect"] == "sam_benefit_interaction"
            ],
            key=lambda row: int(row["diagnostic_batch"]),
        )
        means = np.asarray([float(row["mean"]) for row in cells])
        low = np.asarray([float(row["ci_low"]) for row in cells])
        high = np.asarray([float(row["ci_high"]) for row in cells])
        axes[1].errorbar(
            batch_x,
            means,
            yerr=[means - low, high - means],
            marker="o",
            capsize=3,
            label=order,
            color=colors.get(order),
        )
    axes[1].axhline(0, color="0.4", linewidth=1)
    axes[1].set_xticks(batch_x, [f"Batch {value}" for value in batch_x])
    axes[1].set_ylabel("Scalar-2 SAM-benefit interaction")
    axes[1].set_title("Non-overlapping diagnostic batches")
    axes[1].legend(frameon=False)

    styles = {("m30-p30", "sgd"): "o-", ("m30-p30", "sam"): "o--", ("p45-m45", "sgd"): "s-", ("p45-m45", "sam"): "s--"}
    model_x = np.arange(len(MODELS))
    for order in orders:
        for method in ("sgd", "sam"):
            cells = [
                next(
                    row for row in predictors
                    if row["order"] == order and row["method"] == method and row["model"] == model
                )
                for model in MODELS
            ]
            axes[2].plot(
                model_x,
                [float(row["r2"]) for row in cells],
                styles[(order, method)],
                label=f"{order} / {method.upper()}",
                color=colors.get(order),
            )
    axes[2].axhline(0, color="0.4", linewidth=1)
    axes[2].set_xticks(model_x, MODEL_LABELS, rotation=20, ha="right")
    axes[2].set_ylabel("Leave-one-seed-out $R^2$")
    axes[2].set_title("Static vs time-resolved predictors")
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "stage3_p2_dynamic_chart.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "stage3_p2_dynamic_chart.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot P2 dynamic-chart results")
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", default=Path("figures"), type=Path)
    args = parser.parse_args()
    plot(args.analysis, args.output)


if __name__ == "__main__":
    main()
