#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlp_lora_cl_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_records(root: Path):
    summaries = []
    transitions = []
    for metrics_path in root.rglob("metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["run_dir"] = str(metrics_path.parent)
        summaries.append(summary)
        transition_path = metrics_path.parent / "transitions.json"
        if transition_path.exists():
            with transition_path.open("r", encoding="utf-8") as handle:
                for item in json.load(handle):
                    item["run_dir"] = str(metrics_path.parent)
                    item["parameterization"] = summary["parameterization"]
                    item["rank"] = summary["rank"]
                    item["lifecycle"] = summary["lifecycle"]
                    item["optimizer"] = summary["optimizer"]
                    transitions.append(item)
    return summaries, transitions


def method_label(record: dict) -> str:
    rank = f"-r{record['rank']}" if record.get("rank") is not None else ""
    span = record.get("rotation_span")
    span_tag = f"/span{span:g}" if isinstance(span, (int, float)) else ""
    return f"{record['parameterization']}{rank}/{record['lifecycle']}/{record['optimizer']}{span_tag}"


def export_summary(records: list[dict], path: Path) -> None:
    fields = [
        "run_dir", "parameterization", "rank", "lifecycle", "optimizer", "seed",
        "num_tasks", "rotation_span",
        "final_average_accuracy", "average_anytime_accuracy", "backward_transfer",
        "average_forgetting", "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def plot_final_accuracy(records: list[dict], path: Path) -> None:
    groups = defaultdict(list)
    for record in records:
        groups[method_label(record)].append(record["final_average_accuracy"])
    labels = sorted(groups)
    means = [np.mean(groups[label]) for label in labels]
    errors = [
        np.std(groups[label], ddof=1) / math.sqrt(len(groups[label])) if len(groups[label]) > 1 else 0
        for label in labels
    ]
    fig, axis = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 5))
    axis.bar(np.arange(len(labels)), means, yerr=errors, capsize=3)
    axis.set_ylabel("Final average accuracy")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_curvature_relation(records: list[dict], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 5))
    for optimizer in sorted({record["optimizer"] for record in records}):
        subset = [record for record in records if record["optimizer"] == optimizer]
        x = [record["quadratic_term"] for record in subset]
        y = [record["actual_loss_change"] for record in subset]
        axis.scatter(x, y, s=18, alpha=0.55, label=optimizer)
    axis.set_xlabel(r"Old-task quadratic term $\frac{1}{2}\Delta^T H\Delta$")
    axis.set_ylabel("Actual old-task loss change")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_example_matrix(record: dict, path: Path) -> None:
    matrix = np.array(
        [[np.nan if value is None else value for value in row] for row in record["accuracy_matrix"]]
    )
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axis.set_xlabel("Evaluation task")
    axis.set_ylabel("After training task")
    axis.set_title(method_label(record))
    fig.colorbar(image, ax=axis, label="Accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and plot small-model experiment outputs")
    parser.add_argument("--input", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    summaries, transitions = load_records(args.input)
    if not summaries:
        raise SystemExit(f"No metrics.json files found below {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    export_summary(summaries, args.output / "summary.csv")
    plot_final_accuracy(summaries, args.output / "final_average_accuracy.png")
    plot_example_matrix(summaries[0], args.output / "example_accuracy_matrix.png")
    if transitions:
        plot_curvature_relation(transitions, args.output / "curvature_vs_old_loss_change.png")
    print(f"wrote tables and figures to {args.output}")


if __name__ == "__main__":
    main()
