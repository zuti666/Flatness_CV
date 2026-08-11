#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlp_lora_stage2d_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Factor-LoRA gauge sensitivity")
    parser.add_argument(
        "--effects", type=Path, default=Path("analysis/focus_stage2d_gauge_effects.csv")
    )
    parser.add_argument(
        "--cells", type=Path, default=Path("analysis/focus_stage2d_gauge_cells.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("figures/stage2d_gauge_sensitivity.png")
    )
    args = parser.parse_args()
    effects, cells = read_rows(args.effects), read_rows(args.cells)
    for row in effects + cells:
        for key in ("mean", "ci95_low", "ci95_high"):
            row[key] = float(row[key])

    orders = sorted({row["task_order"] for row in effects})
    gauges = sorted(
        {row["gauge"] for row in cells},
        key=lambda label: float(label.removeprefix("gauge_")),
    )
    gauge_tick_labels = [label.removeprefix("gauge_") for label in gauges]
    x = np.arange(len(gauges))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = (
        ("endpoint_old_loss_damage", "Endpoint old-loss reduction"),
        ("loss_matched_old_loss_damage_reduction", "Current-loss matched reduction"),
        ("drift_matched_old_loss_damage_reduction", "Drift-matched reduction"),
    )
    for column, order in enumerate(orders):
        left_angle, right_angle = ast.literal_eval(order)
        direction = "Forward" if left_angle < right_angle else "Reverse"
        axis = axes[0, column]
        for metric, label in metrics:
            selected = {(r["contrast"], r["metric"]): r for r in effects if r["task_order"] == order}
            rows = [selected.get((gauge, metric)) for gauge in gauges]
            y = np.array([r["mean"] if r else np.nan for r in rows])
            low = np.array([r["ci95_low"] if r else np.nan for r in rows])
            high = np.array([r["ci95_high"] if r else np.nan for r in rows])
            axis.errorbar(x, y, yerr=np.vstack([y-low, high-y]), marker="o", capsize=3, label=label)
        axis.axhline(0, color="black", linestyle=":", linewidth=1)
        axis.set_xticks(x, gauge_tick_labels)
        axis.set_xlabel("Gauge scale c")
        axis.set_ylabel("SGD damage − SAM damage")
        axis.set_title(f"{direction}: {left_angle:g}° → {right_angle:g}°")
        axis.grid(alpha=.25)
        if column == 0:
            axis.legend(fontsize=8)

        axis = axes[1, column]
        for schedule, label, marker in (("sgd", "SGD", "s"), ("sam", "SAM", "o")):
            chosen = {
                (r["gauge"], r["new_task_optimizer"], r["metric"]): r
                for r in cells if r["task_order"] == order
            }
            rows = [chosen.get((gauge, schedule, "endpoint_current_loss")) for gauge in gauges]
            y = np.array([r["mean"] if r else np.nan for r in rows])
            low = np.array([r["ci95_low"] if r else np.nan for r in rows])
            high = np.array([r["ci95_high"] if r else np.nan for r in rows])
            axis.errorbar(x, y, yerr=np.vstack([y-low, high-y]), marker=marker, capsize=3, label=label)
        axis.set_xticks(x, gauge_tick_labels)
        axis.set_xlabel("Gauge scale c")
        axis.set_ylabel("Final current-task loss")
        axis.set_title(f"{direction} plasticity check")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
