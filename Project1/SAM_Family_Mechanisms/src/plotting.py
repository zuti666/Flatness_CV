from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load_pyplot(output_dir: Path):
    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _label(key: str) -> str:
    return (
        key.replace("gam_probe_direction", "GAM probe direction")
        .replace("gam_probe_increment", "GAM probe increment")
        .replace("gam", "GAM final regularizer")
        .replace("ms_sam", "MS-SAM")
        .replace("lookbehind", "Lookbehind")
        .replace("matched_sam", "Matched-SAM")
        .replace("fixed_step", "step")
        .replace("fixed_budget", "budget")
        .replace("sam", "SAM")
    )


def _quality_label(key: str) -> str:
    return _label(key)


def _nonempty_object_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in summaries
        if row["key"] != "sgd" and float(row["object_norm"]) > 0
    ]


def make_quadratic_plots(
    output_dir: Path,
    spectral_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    plt = _load_pyplot(output_dir)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in spectral_rows:
        if row["key"] != "sgd" and float(row["gain"]) > 0:
            grouped[str(row["key"])].append(row)
    fig, axis = plt.subplots(figsize=(9.2, 6.2))
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda item: float(item["eigenvalue"]))
        axis.loglog(
            [float(item["eigenvalue"]) for item in rows],
            [float(item["gain"]) for item in rows],
            marker="o",
            markersize=2.6,
            linewidth=1.2,
            label=_label(key),
        )
    axis.set_xlabel("Hessian eigenvalue")
    axis.set_ylabel("spectral gain")
    axis.set_title("Operator-level Hessian spectral gain")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "spectral_gain.png", dpi=180)
    plt.close(fig)

    rows = _nonempty_object_rows(summaries)
    labels = [_label(str(row["key"])) for row in rows]
    positions = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(10.0, 0.62 * len(rows)), 8.0), sharex=True)
    width = 0.25
    for offset, field, name in (
        (-width, "top_energy_1", "E1"),
        (0.0, "top_energy_5", "E5"),
        (width, "top_energy_10", "E10"),
    ):
        axes[0].bar(
            positions + offset,
            [float(row[field]) for row in rows],
            width=width,
            label=name,
        )
    axes[0].set_ylabel("top-subspace energy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(positions, [float(row["positive_rayleigh"]) for row in rows])
    axes[1].set_ylabel("positive Rayleigh quotient")
    axes[1].set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Top Hessian exposure of analyzed method objects")
    fig.tight_layout()
    fig.savefig(output_dir / "top_subspace_curvature.png", dpi=180)
    plt.close(fig)

    fit_rows = [
        row
        for row in summaries
        if row["key"] != "sgd" and row["delta_r2_1"] is not None
    ]
    labels = [_label(str(row["key"])) for row in fit_rows]
    positions = np.arange(len(fit_rows))
    fig, axis = plt.subplots(figsize=(max(10.0, 0.62 * len(fit_rows)), 5.6))
    bottom = np.zeros(len(fit_rows), dtype=np.float64)
    for field, name in (
        ("delta_r2_1", "H g-hat"),
        ("delta_r2_2", "+ H^2 g-hat"),
        ("delta_r2_3", "+ H^3 g-hat"),
    ):
        values = np.asarray([float(row[field]) for row in fit_rows])
        axis.bar(positions, values, bottom=bottom, label=name)
        bottom += values
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("incremental explained energy")
    axis.set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
    axis.set_title("Nested Hessian-power fit after QR orthogonalization")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "hp_fit.png", dpi=180)
    plt.close(fig)

    quality_rows = [
        row for row in summaries if row["q0"] is not None and row["key"] != "sgd"
    ]
    fig, axis = plt.subplots(figsize=(10.6, 6.5))
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(quality_rows)))
    for row, color in zip(quality_rows, colors):
        axis.scatter(
            float(row["q0"]),
            float(row["q1"]),
            s=48,
            color=color,
            label=_quality_label(str(row["key"])),
        )
    axis.axline((0.0, 0.0), slope=1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axis.set_xlabel("Q0: zero-order inner quality")
    axis.set_ylabel("Q1: first-order inner quality")
    axis.set_title("Inner maximization quality at matched path budgets")
    axis.grid(alpha=0.25)
    axis.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "inner_quality.png", dpi=180)
    plt.close(fig)
