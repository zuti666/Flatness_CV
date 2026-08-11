#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlp_lora_stage2_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


ORDERS = [("(-30.0, 30.0)", "Forward"), ("(30.0, -30.0)", "Reverse")]
PARAMETERS = [
    ("dense", "Dense"),
    ("random_subspace_d128", "Random 128D"),
    ("random_subspace_d240", "Random 240D"),
    ("fixed_lora_tangent_d128", "Fixed init tangent 128D"),
    ("fixed_mature_tangent", "Fixed mature tangent 240D"),
    ("projected_rank", "Direct rank manifold"),
    ("balanced_lora", "Balanced Factor LoRA"),
    ("factor_lora", "Factor LoRA"),
]


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("mean", "ci95_low", "ci95_high"):
            row[key] = float(row[key])
    return rows


def errorbar(axis, position: float, row: dict, color: str) -> None:
    mean, low, high = row["mean"], row["ci95_low"], row["ci95_high"]
    axis.errorbar(
        mean,
        position,
        xerr=np.asarray([[mean - low], [high - mean]]),
        fmt="o",
        color=color,
        capsize=3,
        markersize=5,
    )


def plot_stage2a(rows: list[dict], output: Path) -> None:
    metrics = [
        ("endpoint_old_loss_damage", "Endpoint old-loss reduction"),
        ("loss_matched_old_loss_damage_reduction", "Current-loss matched reduction"),
        ("drift_matched_old_loss_damage_reduction", "Effective-drift matched reduction"),
    ]
    index = {
        (row["task_order"], row["metric"]): row
        for row in rows
        if row.get("effect") == "trajectory_given_old_sgd"
        and row["contrast"] == "factor_lora_minus_dense"
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    colors = ["#4C78A8", "#F58518"]
    labels = [label for _, label in ORDERS]
    for axis, (metric, title) in zip(axes, metrics):
        for position, ((order, _), color) in enumerate(zip(ORDERS, colors)):
            errorbar(axis, position, index[(order, metric)], color)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("Factor-minus-Dense SAM effect")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.grid(axis="x", alpha=0.25)
    target = output / "stage2a_common_endpoint_interaction.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    print(f"wrote {target}")


def plot_stage2b(rows: list[dict], output: Path) -> None:
    index = {(row["task_order"], row["contrast"], row["metric"]): row for row in rows}
    present = [
        item for item in PARAMETERS
        if any((order, item[0], "endpoint_old_loss_damage") in index for order, _ in ORDERS)
    ]
    keys = [(order, parameter) for order, _ in ORDERS for parameter, _ in present]
    labels = [
        f"{order_label} · {parameter_label}"
        for _, order_label in ORDERS for _, parameter_label in present
    ]
    palette = plt.get_cmap("tab10")
    colors = [palette(index % 10) for _ in ORDERS for index in range(len(present))]
    panels = [
        ("endpoint_old_loss_damage", "Endpoint old-loss reduction"),
        ("loss_matched_old_loss_damage_reduction", "Current-loss matched reduction"),
        ("drift_matched_old_loss_damage_reduction", "Effective-drift matched reduction"),
        ("pathwise_directional_curvature_sum", "Pathwise curvature-cost reduction"),
    ]
    y = np.arange(len(keys))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, panels):
        for position, ((order, parameter), color) in enumerate(zip(keys, colors)):
            errorbar(axis, position, index[(order, parameter, metric)], color)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.axhline(len(present) - 0.5, color="0.7", linewidth=1)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("Positive = Task-B SAM is safer")
        axis.grid(axis="x", alpha=0.25)
    target = output / "stage2b_parameterization_controls.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    print(f"wrote {target}")

    controls = [parameter for parameter, _ in present if parameter != "factor_lora"]
    interaction_panels = [
        ("endpoint_old_loss_damage", "Endpoint interaction"),
        ("loss_matched_old_loss_damage_reduction", "Current-loss matched interaction"),
        ("pathwise_interference_sum", "First-order path interaction"),
        ("pathwise_directional_curvature_sum", "Curvature path interaction"),
    ]
    interaction_keys = [
        (order, f"factor_lora_minus_{control}")
        for order, _ in ORDERS for control in controls
    ]
    interaction_labels = [
        f"{order_label} · vs {dict(PARAMETERS)[control]}"
        for _, order_label in ORDERS for control in controls
    ]
    y = np.arange(len(interaction_keys))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, interaction_panels):
        for position, (order, contrast) in enumerate(interaction_keys):
            errorbar(axis, position, index[(order, contrast, metric)], "#E45756")
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.axhline(len(controls) - 0.5, color="0.7", linewidth=1)
        axis.set_yticks(y, interaction_labels)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("Positive = Factor LoRA has larger SAM reduction")
        axis.grid(axis="x", alpha=0.25)
    target = output / "stage2b_factor_minus_controls.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    print(f"wrote {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot common-endpoint Stage 2 results")
    parser.add_argument(
        "--stage2a", type=Path, default=Path("analysis/focus_stage2a_formal_effects.csv")
    )
    parser.add_argument(
        "--stage2b", type=Path, default=Path("analysis/focus_stage2b_formal_effects.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_stage2a(read_rows(args.stage2a), args.output)
    plot_stage2b(read_rows(args.stage2b), args.output)


if __name__ == "__main__":
    main()
