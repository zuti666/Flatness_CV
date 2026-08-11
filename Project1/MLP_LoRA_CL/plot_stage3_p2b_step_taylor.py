from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRANSFORMS = ["identity", "scalar_0.5", "scalar_2"]
LABELS = ["Identity", "Scalar .5", "Scalar 2"]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot(analysis: Path, output: Path, name: str) -> None:
    rows = _read(analysis / "summary.csv")
    orders = sorted({row["order"] for row in rows})
    fig, axes = plt.subplots(len(orders), 2, figsize=(10.5, 4.0 * len(orders)), squeeze=False)
    x = np.arange(len(TRANSFORMS))
    width = 0.36
    colors = {"sgd": "#3b6fb6", "sam": "#d55e00"}
    for order_index, order in enumerate(orders):
        for method_index, method in enumerate(("sgd", "sam")):
            cells = [
                next(
                    row for row in rows
                    if row["order"] == order
                    and row["transformation"] == transform
                    and row["method"] == method
                )
                for transform in TRANSFORMS
            ]
            offset = (method_index - 0.5) * width
            coarse = np.asarray([float(row["coarse_abs_R_mean"]) for row in cells])
            fine = np.asarray([float(row["fine_abs_R_mean"]) for row in cells])
            axes[order_index, 0].bar(
                x + offset,
                coarse,
                width,
                color=colors[method],
                alpha=0.35,
                label=f"{method.upper()} coarse",
            )
            axes[order_index, 0].scatter(
                x + offset,
                fine,
                marker="D",
                color=colors[method],
                label=f"{method.upper()} fine",
                zorder=3,
            )
            axes[order_index, 1].bar(
                x + offset,
                [float(row["clean_prediction_relative_mae_mean"]) for row in cells],
                width,
                color=colors[method],
                label=method.upper(),
            )
        axes[order_index, 0].set_xticks(x, LABELS)
        axes[order_index, 0].set_ylabel("Absolute Taylor residual")
        axes[order_index, 0].set_title(f"4 epoch segments vs ~32 fine segments: {order}")
        axes[order_index, 0].legend(frameon=False, fontsize=8, ncol=2)
        axes[order_index, 1].set_xticks(x, LABELS)
        axes[order_index, 1].set_ylabel("Relative MAE of one-step interference")
        axes[order_index, 1].set_title(f"Clean pullback + bilinear audit: {order}")
        axes[order_index, 1].legend(frameon=False)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot P2b fine Taylor diagnostics")
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", default=Path("figures"), type=Path)
    parser.add_argument("--name", default="stage3_p2b_step_taylor")
    args = parser.parse_args()
    plot(args.analysis, args.output, args.name)


if __name__ == "__main__":
    main()
