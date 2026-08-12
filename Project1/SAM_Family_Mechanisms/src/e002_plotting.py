from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _pyplot(output_dir: Path):
    cache = output_dir / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _short(method: str) -> str:
    return (
        method.replace("gam_exact_hvp_same_batch_alpha1", "GAM-exact-ref")
        .replace("multistep_sam_k2_fixed_budget", "MS-SAM-k2")
        .replace("lookbehind_path_mean_surrogate_k2", "LB-mean-k2")
        .replace("lookbehind_faithful_k2_alpha0.5", "LB-faithful-k2")
        .replace("matched_sam_k2_fixed_budget", "Matched-SAM")
        .replace("looksam_age", "LookSAM-age")
        .replace("sam5_nonrefresh", "SAM-5/SGD")
        .replace("sam5_refresh", "SAM-5/refresh")
    )


def make_e002_plots(
    output_dir: Path,
    *,
    taylor_rows: list[dict[str, Any]],
    covariance_rows: list[dict[str, Any]],
    spectral_rows: list[dict[str, Any]],
    path_rows: list[dict[str, Any]],
    temporal_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    primary_eta: float,
) -> list[str]:
    """Write the six preregistered E002-pilot figures."""
    plt = _pyplot(output_dir)
    written: list[str] = []

    primary = [
        row
        for row in taylor_rows
        if abs(float(row["eta"]) - primary_eta) <= 1e-15
        and int(row["seed"]) == min(int(item["seed"]) for item in taylor_rows)
        and int(row["checkpoint_step"])
        == max(int(item["checkpoint_step"]) for item in taylor_rows)
    ]
    primary.sort(key=lambda row: str(row["method"]))
    x = np.arange(len(primary))
    fig, axis = plt.subplots(figsize=(max(10.0, 0.75 * len(primary)), 6.2))
    positive_bottom = np.zeros(len(primary))
    negative_bottom = np.zeros(len(primary))
    for field, label, color in (
        ("t1", "first order", "#4c78a8"),
        ("t2_mean", "mean curvature", "#f58518"),
        ("t2_noise_positive", "noise / positive H", "#54a24b"),
        ("t2_noise_negative", "noise / negative H", "#e45756"),
    ):
        values = np.asarray([float(row[field]) for row in primary])
        bottoms = np.where(values >= 0, positive_bottom, negative_bottom)
        axis.bar(x, values, bottom=bottoms, label=label, color=color)
        positive_bottom += np.where(values >= 0, values, 0.0)
        negative_bottom += np.where(values < 0, values, 0.0)
    axis.scatter(
        x,
        [float(row["true_mean_loss_change"]) for row in primary],
        color="black",
        marker="x",
        s=45,
        label="true mean change",
        zorder=5,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, [_short(str(row["method"])) for row in primary], rotation=50, ha="right")
    axis.set_ylabel("full-train loss change")
    axis.set_title(f"Shared-anchor Taylor decomposition (final checkpoint, eta={primary_eta:g})")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    name = "shared_taylor.png"
    fig.savefig(output_dir / name, dpi=180)
    plt.close(fig)
    written.append(name)

    covariance = [
        row
        for row in covariance_rows
        if int(row["seed"]) == min(int(item["seed"]) for item in covariance_rows)
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in covariance:
        grouped[str(row["method"])].append(row)
    fig, axis = plt.subplots(figsize=(8.5, 6.2))
    for method, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["checkpoint_step"]))
        axis.plot(
            [float(row["direction_trace_sigma"]) for row in rows],
            [float(row["direction_trace_hplus_sigma"]) for row in rows],
            marker="o",
            linewidth=1.2,
            label=_short(method),
        )
    axis.set_xlabel("Tr(Sigma)")
    axis.set_ylabel("Tr(H+ Sigma)")
    axis.set_title("Update-noise strength and positive-curvature alignment")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    name = "covariance_curvature.png"
    fig.savefig(output_dir / name, dpi=180)
    plt.close(fig)
    written.append(name)

    seed0 = min(int(row["seed"]) for row in spectral_rows)
    last = max(int(row["checkpoint_step"]) for row in spectral_rows)
    selected_methods = {
        "sam",
        "gam_exact_hvp_same_batch_alpha1",
        "multistep_sam_k2_fixed_budget",
        "lookbehind_faithful_k2_alpha0.5",
    }
    spectral: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in spectral_rows:
        if (
            int(row["seed"]) == seed0
            and int(row["checkpoint_step"]) == last
            and str(row["method"]) in selected_methods
        ):
            spectral[str(row["method"])].append(row)
    fig, axis = plt.subplots(figsize=(8.8, 6.2))
    for method, rows in sorted(spectral.items()):
        rows.sort(key=lambda row: float(row["eigenvalue"]))
        axis.plot(
            [float(row["eigenvalue"]) for row in rows],
            [float(row["correction_projection"]) for row in rows],
            marker=".",
            linewidth=1.0,
            label=_short(method),
        )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_yscale("symlog", linthresh=1e-7)
    axis.set_xlabel("signed Hessian eigenvalue")
    axis.set_ylabel("signed projection of mean correction")
    axis.set_title("Signed spectral transfer at the final shared anchor")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    name = "signed_spectrum.png"
    fig.savefig(output_dir / name, dpi=180)
    plt.close(fig)
    written.append(name)

    path = [row for row in path_rows if int(row["seed"]) == min(int(item["seed"]) for item in path_rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for method, rows in sorted(_group(path, "method").items()):
        rows.sort(key=lambda row: int(row["checkpoint_step"]))
        steps = [int(row["checkpoint_step"]) for row in rows]
        axes[0].plot(steps, [float(row["path_misalign_mean"]) for row in rows], marker="o", label=_short(method))
        axes[1].plot(steps, [float(row["last_to_mean_variance_ratio"]) for row in rows], marker="o", label=_short(method))
    axes[0].set_ylabel("mean path misalignment")
    axes[1].set_ylabel("Tr Sigma(last) / Tr Sigma(mean)")
    for axis in axes:
        axis.set_xlabel("SGD anchor step")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Lookbehind / multistep path diagnostics")
    fig.tight_layout()
    name = "lookbehind_path.png"
    fig.savefig(output_dir / name, dpi=180)
    plt.close(fig)
    written.append(name)

    temporal = [row for row in temporal_rows if int(row["seed"]) == min(int(item["seed"]) for item in temporal_rows)]
    fig, axis = plt.subplots(figsize=(8.3, 5.6))
    for object_name, rows in sorted(_group(temporal, "object").items()):
        rows.sort(key=lambda row: int(row["lag"]))
        axis.plot(
            [int(row["lag"]) for row in rows],
            [float(row["median_cosine"]) for row in rows],
            marker="o",
            label=object_name,
        )
    axis.set_xlabel("outer-step lag")
    axis.set_ylabel("median cosine on one fixed diagnostic batch")
    axis.set_ylim(-1.05, 1.05)
    axis.set_title("Temporal persistence of the LookSAM orthogonal component")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    name = "looksam_temporal.png"
    fig.savefig(output_dir / name, dpi=180)
    plt.close(fig)
    written.append(name)

    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in training_rows:
        history[str(row["method"])].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for method, rows in sorted(history.items()):
        by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_step[int(row["step"])].append(row)
        steps = sorted(by_step)
        train_loss = [np.mean([float(row["train_loss"]) for row in by_step[step]]) for step in steps]
        test_acc = [np.mean([float(row["test_accuracy"]) for row in by_step[step]]) for step in steps]
        axes[0].plot(steps, train_loss, label=_short(method))
        axes[1].plot(steps, test_acc, label=_short(method))
    axes[0].set_ylabel("train cross-entropy")
    axes[1].set_ylabel("clean test accuracy")
    axes[1].set_ylim(0.45, 1.01)
    for axis in axes:
        axis.set_xlabel("outer step")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("On-policy pilot trajectories (same seed and batch stream)")
    fig.tight_layout()
    name = "on_policy_trajectories.png"
    fig.savefig(output_dir / name, dpi=180)
    plt.close(fig)
    written.append(name)
    return written


def _group(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return grouped
