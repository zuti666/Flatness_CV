#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SEED = "1993"
METRICS_ROOT = (
    ROOT
    / "outputs_logs"
    / "logs_inc_lora"
    / "seqlora"
    / "exp_F_forked_taskwise"
    / "imagenetr"
    / SEED
)
OUT_DIR = ROOT / "outputs_logs" / "exp_F_imagenetr_r16_t20_forked_taskwise_summary" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "sgd_sgd": {
        "label": "SGD -> SGD",
        "color": "#4d4d4d",
        "linestyle": "-",
        "group": "sgd_prefix",
        "prefix": "SGD tasks 0-9",
        "suffix": "SGD tasks 10-19",
    },
    "sgd_sam_factor": {
        "label": "SGD -> SAM-factor",
        "color": "#0072B2",
        "linestyle": "-",
        "group": "sgd_prefix",
        "prefix": "SGD tasks 0-9",
        "suffix": "SAM-factor tasks 10-19",
    },
    "sgd_random_factor": {
        "label": "SGD -> Random-factor",
        "color": "#CC79A7",
        "linestyle": "--",
        "group": "sgd_prefix",
        "prefix": "SGD tasks 0-9",
        "suffix": "Gaussian random-factor tasks 10-19",
    },
    "sam_factor_sgd": {
        "label": "SAM-factor -> SGD",
        "color": "#D55E00",
        "linestyle": "-",
        "group": "sam_prefix",
        "prefix": "SAM-factor tasks 0-9",
        "suffix": "SGD tasks 10-19",
    },
    "sam_factor_sam_factor": {
        "label": "SAM-factor -> SAM-factor",
        "color": "#009E73",
        "linestyle": "-",
        "group": "sam_prefix",
        "prefix": "SAM-factor tasks 0-9",
        "suffix": "SAM-factor tasks 10-19",
    },
}


def metrics_path(variant: str) -> Path:
    prefix = f"exp_F_imagenetr_r16_t20_fork_{variant}"
    return METRICS_ROOT / prefix / "exp_run" / "10" / f"{prefix}_vit_base_patch16_224_cl_metrics.json"


def load_variant(variant: str) -> dict:
    path = metrics_path(variant)
    with path.open("r", encoding="utf-8") as f:
        book = json.load(f)
    final = book["cnn"]["final"]
    vectors = final["vectors"]
    matrix = np.asarray(book["cnn"]["matrices"]["final"], dtype=float)
    seen_avg = np.asarray(vectors["prefix_mean_t"], dtype=float)
    aaa_curve = []
    for task in range(matrix.shape[0]):
        vals = []
        for row in range(task + 1):
            row_vals = matrix[row, : row + 1]
            vals.extend(row_vals[~np.isnan(row_vals)].tolist())
        aaa_curve.append(float(np.nanmean(vals)) if vals else float("nan"))
    aaa_curve = np.asarray(aaa_curve, dtype=float)
    return {
        "variant": variant,
        "path": path,
        "final": final,
        "matrix": matrix,
        "current_acc": np.asarray(vectors["CA"], dtype=float),
        "seen_avg": seen_avg,
        "aaa_curve": aaa_curve,
        "bwt_t": np.asarray(vectors["BWT_t"], dtype=float),
        "forget_per_task": np.asarray(vectors["Forget_per_task"], dtype=float),
        "bwt_final_per_task": np.asarray(vectors["BWT_final_per_task"], dtype=float),
    }


def mark_fork(ax) -> None:
    ax.axvline(9.5, color="#999999", lw=1.0, ls=":")
    ymin, ymax = ax.get_ylim()
    ax.text(
        9.55,
        ymax - 0.05 * (ymax - ymin),
        "fork after task 9",
        fontsize=8,
        color="#666666",
        va="top",
    )


def plot_time_curves(data: dict[str, dict], variants: list[str], filename: str, title: str) -> None:
    tasks = np.arange(20)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), constrained_layout=True)
    panels = [
        ("seen_avg", "Seen-task avg accuracy", "Accuracy (%)"),
        ("bwt_t", "BWT after each task", "BWT (%)"),
        ("current_acc", "Current-task accuracy", "Accuracy (%)"),
    ]
    for ax, (key, panel_title, ylabel) in zip(axes, panels):
        for variant in variants:
            spec = VARIANTS[variant]
            y = data[variant][key]
            ax.plot(
                tasks,
                y,
                label=spec["label"],
                color=spec["color"],
                linestyle=spec["linestyle"],
                marker="o",
                ms=3,
                lw=1.8,
            )
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("Task index")
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(0, 20, 2))
        ax.grid(True, alpha=0.25, linewidth=0.7)
        mark_fork(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(title, fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{filename}.{suffix}", dpi=220)
    plt.close(fig)


def plot_final_forgetting(data: dict[str, dict], variants: list[str], filename: str, title: str) -> None:
    tasks = np.arange(19)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    for variant in variants:
        spec = VARIANTS[variant]
        ax.plot(
            tasks,
            data[variant]["forget_per_task"],
            label=spec["label"],
            color=spec["color"],
            linestyle=spec["linestyle"],
            marker="o",
            ms=3,
            lw=1.8,
        )
    ax.axvspan(0, 9, color="#E5E5E5", alpha=0.25, label="prefix tasks")
    ax.axvspan(10, 18, color="#F2F2F2", alpha=0.25, label="suffix old tasks")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Evaluated old task index")
    ax.set_ylabel("Final per-task forgetting (%)")
    ax.set_xticks(np.arange(0, 19, 2))
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, loc="best")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{filename}.{suffix}", dpi=220)
    plt.close(fig)


def plot_aaa_curve(data: dict[str, dict], variants: list[str], filename: str, title: str) -> None:
    tasks = np.arange(20)
    fig, ax = plt.subplots(figsize=(8.2, 4.3), constrained_layout=True)
    for variant in variants:
        spec = VARIANTS[variant]
        ax.plot(
            tasks,
            data[variant]["aaa_curve"],
            label=spec["label"],
            color=spec["color"],
            linestyle=spec["linestyle"],
            marker="o",
            ms=3,
            lw=1.9,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Task index")
    ax.set_ylabel("AAA up to current task (%)")
    ax.set_xticks(np.arange(0, 20, 2))
    ax.grid(True, alpha=0.25, linewidth=0.7)
    mark_fork(ax)
    ax.legend(frameon=False, fontsize=8, loc="best")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{filename}.{suffix}", dpi=220)
    plt.close(fig)


def write_long_csv(data: dict[str, dict]) -> None:
    path = OUT_DIR / "per_task_curves_long.csv"
    fieldnames = [
        "variant",
        "label",
        "task",
        "current_acc",
        "seen_avg",
        "aaa_curve",
        "bwt_t",
        "final_forgetting",
        "final_bwt_per_task",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for variant, item in data.items():
            spec = VARIANTS[variant]
            for task in range(20):
                writer.writerow(
                    {
                        "variant": variant,
                        "label": spec["label"],
                        "task": task,
                        "current_acc": item["current_acc"][task],
                        "seen_avg": item["seen_avg"][task],
                        "aaa_curve": item["aaa_curve"][task],
                        "bwt_t": item["bwt_t"][task],
                        "final_forgetting": item["forget_per_task"][task]
                        if task < len(item["forget_per_task"])
                        else "",
                        "final_bwt_per_task": item["bwt_final_per_task"][task]
                        if task < len(item["bwt_final_per_task"])
                        else "",
                    }
                )
    print(path)


def main() -> None:
    data = {variant: load_variant(variant) for variant in VARIANTS}

    plot_time_curves(
        data,
        ["sgd_sgd", "sgd_sam_factor", "sgd_random_factor"],
        "sgd_prefix_time_curves",
        "Same SGD prefix checkpoint, different suffix optimizers",
    )
    plot_time_curves(
        data,
        ["sam_factor_sgd", "sam_factor_sam_factor"],
        "sam_prefix_time_curves",
        "Same SAM-factor prefix checkpoint, different suffix optimizers",
    )
    plot_time_curves(
        data,
        list(VARIANTS.keys()),
        "all_variants_time_curves",
        "ImageNet-R r16 T20 forked taskwise trajectories",
    )

    plot_final_forgetting(
        data,
        ["sgd_sgd", "sgd_sam_factor", "sgd_random_factor"],
        "sgd_prefix_final_forgetting",
        "Final per-task forgetting: same SGD prefix checkpoint",
    )
    plot_final_forgetting(
        data,
        ["sam_factor_sgd", "sam_factor_sam_factor"],
        "sam_prefix_final_forgetting",
        "Final per-task forgetting: same SAM-factor prefix checkpoint",
    )
    plot_final_forgetting(
        data,
        list(VARIANTS.keys()),
        "all_variants_final_forgetting",
        "Final per-task forgetting: all forked variants",
    )
    plot_aaa_curve(
        data,
        ["sgd_sgd", "sgd_sam_factor", "sgd_random_factor"],
        "sgd_prefix_aaa_curve",
        "AAA curve: same SGD prefix checkpoint",
    )
    plot_aaa_curve(
        data,
        ["sam_factor_sgd", "sam_factor_sam_factor"],
        "sam_prefix_aaa_curve",
        "AAA curve: same SAM-factor prefix checkpoint",
    )
    plot_aaa_curve(
        data,
        list(VARIANTS.keys()),
        "all_variants_aaa_curve",
        "ImageNet-R r16 T20 forked taskwise AAA curves",
    )
    write_long_csv(data)
    print(f"figures: {OUT_DIR}")


if __name__ == "__main__":
    main()
