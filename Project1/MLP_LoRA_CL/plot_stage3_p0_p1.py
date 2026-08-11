from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRANSFORMS = [
    "identity",
    "scalar_0.5",
    "scalar_2",
    "orthogonal",
    "anisotropic_2",
    "anisotropic_4",
]
LABELS = ["Identity", "Scalar 0.5", "Scalar 2", "Orthogonal", "Aniso. 2", "Aniso. 4"]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_p0(root: Path, output: Path) -> None:
    summary = _read(root / "lodo_summary.csv")
    auc = _read(root / "auc_summary.csv")
    models = ["path_only", "path_plus_I", "path_plus_I_plus_C"]
    model_labels = ["Path only", "Path + I", "Path + I + C"]
    parameterizations = [("dense", "Dense"), ("factor_lora", "Factor LoRA")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    positions = np.arange(len(models))
    width = 0.34
    for index, (parameterization, label) in enumerate(parameterizations):
        values = []
        for model in models:
            row = next(
                item
                for item in summary
                if item["parameterization"] == parameterization
                and item["analysis"] == "leave_direction_and_seed_out"
                and item["model"] == model
            )
            values.append(float(row["mae"]))
        axes[0].bar(positions + (index - 0.5) * width, values, width, label=label)
    axes[0].set_xticks(positions, model_labels)
    axes[0].set_ylabel("LODO MAE (old-loss damage)")
    axes[0].set_title("P0: pathwise directional terms improve prediction")
    axes[0].legend(frameon=False)

    orders = []
    for row in auc:
        if row["task_order"] not in orders:
            orders.append(row["task_order"])
    factor_rows = [row for row in auc if row["contrast"] == "factor_lora"]
    x = np.arange(len(orders))
    values = [float(next(row["mean"] for row in factor_rows if row["task_order"] == order)) for order in orders]
    lower = [float(next(row["ci95_low"] for row in factor_rows if row["task_order"] == order)) for order in orders]
    upper = [float(next(row["ci95_high"] for row in factor_rows if row["task_order"] == order)) for order in orders]
    axes[1].axhline(0, color="0.35", linewidth=1)
    axes[1].errorbar(
        x,
        values,
        yerr=[np.asarray(values) - np.asarray(lower), np.asarray(upper) - np.asarray(values)],
        fmt="o",
        capsize=3,
        color="#3b6fb6",
    )
    axes[1].set_xticks(x, [order.replace(".0", "") for order in orders], rotation=25, ha="right")
    axes[1].set_ylabel("SAM reduction in normalized damage AUC")
    axes[1].set_title("Factor LoRA: six direction-level dose responses")
    fig.tight_layout()
    _save(fig, output, "stage3_p0_path_safety")


def _cell(
    rows: list[dict[str, str]], order: str, transform: str, rho: float
) -> dict[str, str]:
    return next(
        row
        for row in rows
        if row["order"] == order
        and row["transformation"] == transform
        and abs(float(row["sam_rho"]) - rho) < 1e-12
    )


def plot_p1(root: Path, output: Path) -> None:
    effects = _read(root / "paired_effects_summary.csv")
    matched = _read(root / "matched_effects_summary.csv")
    contrasts = _read(root / "gauge_endpoint_contrasts_seed.csv")
    auc = _read(root / "global_support_auc_summary.csv")
    orders = sorted({row["order"] for row in effects})
    radii = sorted({float(row["sam_rho"]) for row in effects})
    colors = ["#3b6fb6", "#d55e00", "#009e73"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    x = np.arange(len(TRANSFORMS))
    for order_index, order in enumerate(orders):
        axis = axes[0, order_index]
        for radius_index, rho in enumerate(radii):
            cells = [_cell(effects, order, transform, rho) for transform in TRANSFORMS]
            means = np.asarray([float(row["old_damage_reduction_mean"]) for row in cells])
            lower = np.asarray([float(row["old_damage_reduction_ci_low"]) for row in cells])
            upper = np.asarray([float(row["old_damage_reduction_ci_high"]) for row in cells])
            offset = (radius_index - (len(radii) - 1) / 2) * 0.16
            axis.errorbar(
                x + offset,
                means,
                yerr=[means - lower, upper - means],
                fmt="o",
                capsize=2.5,
                color=colors[radius_index],
                label=fr"$\rho={rho:g}$",
            )
        axis.axhline(0, color="0.4", linewidth=1)
        axis.set_xticks(x, LABELS, rotation=28, ha="right")
        axis.set_ylabel("SAM old-damage reduction")
        axis.set_title(f"P1 endpoint effect: {order}")
        axis.legend(frameon=False)

    selected_rho = min(radii)
    for order_index, order in enumerate(orders):
        axis = axes[1, order_index]
        cells = [_cell(matched, order, transform, selected_rho) for transform in TRANSFORMS]
        current = np.asarray(
            [float(row["current_loss_matched_damage_reduction_mean"]) for row in cells]
        )
        drift = np.asarray([float(row["drift_matched_damage_reduction_mean"]) for row in cells])
        current_low = np.asarray(
            [float(row["current_loss_matched_damage_reduction_ci_low"]) for row in cells]
        )
        current_high = np.asarray(
            [float(row["current_loss_matched_damage_reduction_ci_high"]) for row in cells]
        )
        drift_low = np.asarray(
            [float(row["drift_matched_damage_reduction_ci_low"]) for row in cells]
        )
        drift_high = np.asarray(
            [float(row["drift_matched_damage_reduction_ci_high"]) for row in cells]
        )
        width = 0.36
        axis.bar(x - width / 2, current, width, label="Current-loss matched", color="#56b4e9")
        axis.bar(x + width / 2, drift, width, label="Drift matched", color="#e69f00")
        axis.errorbar(
            x - width / 2,
            current,
            yerr=[current - current_low, current_high - current],
            fmt="none",
            ecolor="0.25",
            capsize=2,
            linewidth=0.8,
        )
        axis.errorbar(
            x + width / 2,
            drift,
            yerr=[drift - drift_low, drift_high - drift],
            fmt="none",
            ecolor="0.25",
            capsize=2,
            linewidth=0.8,
        )
        axis.axhline(0, color="0.4", linewidth=1)
        axis.set_xticks(x, LABELS, rotation=28, ha="right")
        axis.set_ylabel("Matched old-damage reduction")
        axis.set_title(fr"Matched controls ($\rho={selected_rho:g}$): {order}")
        axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, output, "stage3_p1_gauge_effects")

    fig, axes = plt.subplots(1, len(orders), figsize=(11.5, 4.1), squeeze=False)
    for order_index, order in enumerate(orders):
        axis = axes[0, order_index]
        cells = [
            next(
                row for row in auc
                if row["order"] == order
                and row["transformation"] == transform
                and abs(float(row["sam_rho"]) - selected_rho) < 1e-12
                and row["measure"] == "current_loss"
                and row["contrast"] == "sam_reduction"
            )
            for transform in TRANSFORMS
        ]
        means = np.asarray([float(row["damage_auc_reduction_mean"]) for row in cells])
        lower = np.asarray([float(row["damage_auc_reduction_ci_low"]) for row in cells])
        upper = np.asarray([float(row["damage_auc_reduction_ci_high"]) for row in cells])
        axis.axhline(0, color="0.4", linewidth=1)
        axis.errorbar(
            x,
            means,
            yerr=[means - lower, upper - means],
            fmt="o",
            capsize=3,
            color="#7b3294",
        )
        axis.set_xticks(x, LABELS, rotation=28, ha="right")
        axis.set_ylabel("Common-support damage AUC reduction")
        axis.set_title(f"Current-loss AUC: {order}")
    fig.tight_layout()
    _save(fig, output, "stage3_p1_global_support_auc")

    fig, axes = plt.subplots(1, len(orders), figsize=(11.5, 4.1), squeeze=False)
    for order_index, order in enumerate(orders):
        axis = axes[0, order_index]
        subset = [
            row for row in contrasts
            if row["order"] == order and row["method"] == "sgd"
        ]
        for transform, label in zip(TRANSFORMS[1:], LABELS[1:]):
            group = [row for row in subset if row["transformation"] == transform]
            axis.scatter(
                [float(row["predicted_interference_change_from_identity"]) for row in group],
                [float(row["old_damage_change_from_identity"]) for row in group],
                label=label,
                alpha=0.8,
            )
        axis.axhline(0, color="0.75", linewidth=0.8)
        axis.axvline(0, color="0.75", linewidth=0.8)
        axis.set_xlabel(r"Gauge change in $-g_{old}^{\top}M g_{new}$")
        axis.set_ylabel("Gauge change in SGD old-loss damage")
        axis.set_title(order)
    axes[0, -1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, output, "stage3_p1_pullback_prediction")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage-3 P0 and P1 results")
    parser.add_argument("--p0", required=True)
    parser.add_argument("--p1", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    plot_p0(Path(args.p0), output)
    plot_p1(Path(args.p1), output)


if __name__ == "__main__":
    main()
