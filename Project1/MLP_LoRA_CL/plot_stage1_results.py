#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlp_lora_stage1_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARAMETERIZATIONS = ("dense", "random_subspace", "factor_lora")
OPTIMIZERS = ("sgd", "sam", "gam_fd")


def load_metrics(root: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in root.rglob("metrics.json")]


def mean_ci(values: list[float], seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(10_000, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def plot_group_metrics(records: dict[str, list[dict]], output: Path) -> None:
    metrics = (
        ("final_average_accuracy", "Final average accuracy"),
        ("average_forgetting", "Average forgetting (lower is better)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    width = 0.23
    offsets = (-width, 0.0, width)
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for column, (order, rows) in enumerate(records.items()):
        grouped = defaultdict(list)
        for row in rows:
            for metric, _ in metrics:
                grouped[(row["parameterization"], row["optimizer"], metric)].append(row[metric])
        for row_index, (metric, ylabel) in enumerate(metrics):
            axis = axes[row_index, column]
            for opt_index, optimizer in enumerate(OPTIMIZERS):
                means, lower, upper = [], [], []
                for param_index, parameterization in enumerate(PARAMETERIZATIONS):
                    mean, low, high = mean_ci(
                        grouped[(parameterization, optimizer, metric)],
                        1000 + 100 * column + 10 * row_index + opt_index + param_index,
                    )
                    means.append(mean)
                    lower.append(mean - low)
                    upper.append(high - mean)
                x = np.arange(len(PARAMETERIZATIONS)) + offsets[opt_index]
                axis.bar(
                    x,
                    means,
                    width,
                    yerr=np.asarray([lower, upper]),
                    capsize=3,
                    color=colors[opt_index],
                    label=optimizer.upper().replace("_", "-"),
                )
            axis.set_title(order.capitalize())
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
            axis.set_xticks(np.arange(len(PARAMETERIZATIONS)))
            axis.set_xticklabels(("Dense", "Random subspace", "Factor LoRA"))
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output / "stage1_group_metrics.png", dpi=220)
    plt.close(fig)


def load_interactions(paths: dict[str, Path]) -> list[dict]:
    rows = []
    for order, path in paths.items():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["contrast"] != "factor_lora_minus_dense_interaction":
                    continue
                row["order"] = order
                for key in ("mean", "ci95_low", "ci95_high"):
                    row[key] = 100.0 * float(row[key])
                rows.append(row)
    return rows


def plot_interactions(rows: list[dict], output: Path) -> None:
    metrics = (("accuracy_gain", "FAA interaction (percentage points)"),
               ("forgetting_reduction", "Forgetting-reduction interaction (percentage points)"))
    labels = ["Forward / SAM", "Forward / GAM-FD", "Reverse / SAM", "Reverse / GAM-FD"]
    keys = [("forward", "sam"), ("forward", "gam_fd"),
            ("reverse", "sam"), ("reverse", "gam_fd")]
    index = {(row["order"], row["optimizer"], row["metric"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, (metric, title) in zip(axes, metrics):
        selected = [index[(order, optimizer, metric)] for order, optimizer in keys]
        means = np.asarray([row["mean"] for row in selected])
        lower = means - np.asarray([row["ci95_low"] for row in selected])
        upper = np.asarray([row["ci95_high"] for row in selected]) - means
        y = np.arange(len(labels))
        axis.errorbar(means, y, xerr=np.asarray([lower, upper]), fmt="o", capsize=4)
        axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlabel(title)
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "stage1_interaction_ci.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot focused Stage-1 results")
    parser.add_argument("--forward", type=Path, default=Path("outputs/focus_stage1_interaction_forward"))
    parser.add_argument("--reverse", type=Path, default=Path("outputs/focus_stage1_interaction_reverse"))
    parser.add_argument("--forward-analysis", type=Path, default=Path("analysis/focus_stage1_forward.csv"))
    parser.add_argument("--reverse-analysis", type=Path, default=Path("analysis/focus_stage1_reverse.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = {"forward": load_metrics(args.forward), "reverse": load_metrics(args.reverse)}
    plot_group_metrics(records, args.output)
    interactions = load_interactions(
        {"forward": args.forward_analysis, "reverse": args.reverse_analysis}
    )
    plot_interactions(interactions, args.output)
    print(f"wrote Stage-1 figures to {args.output}")


if __name__ == "__main__":
    main()
