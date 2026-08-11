#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "outputs_logs" / "exp_E_imagenetr_r16_t20_support_direction_summary" / "summary_by_variant.csv"
OUT_DIR = ROOT / "outputs_logs" / "exp_E_imagenetr_r16_t20_support_direction_summary" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTS = ["factor", "full", "delta", "all", "frozen"]
SUPPORT_LABELS = {
    "factor": "Factor",
    "full": "Effective full",
    "delta": "Delta tangent",
    "all": "Raw all-param",
    "frozen": "Frozen-only",
}


def load_rows() -> dict[str, dict]:
    with SUMMARY.open("r", encoding="utf-8") as f:
        return {row["variant"]: row for row in csv.DictReader(f)}


def f(row: dict, key: str) -> float:
    return float(row[key])


def write_pairwise(rows: dict[str, dict]) -> Path:
    out = OUT_DIR / "support_direction_pairwise.csv"
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "support",
                "sam_FAA",
                "random_FAA",
                "delta_FAA",
                "sam_BWT",
                "random_BWT",
                "delta_BWT",
                "sam_forget",
                "random_forget",
                "delta_forget",
            ],
        )
        writer.writeheader()
        for support in SUPPORTS:
            sam = rows[f"sam_{support}"]
            random = rows[f"random_{support}"]
            writer.writerow(
                {
                    "support": support,
                    "sam_FAA": f(sam, "CNN_FAA"),
                    "random_FAA": f(random, "CNN_FAA"),
                    "delta_FAA": f(sam, "CNN_FAA") - f(random, "CNN_FAA"),
                    "sam_BWT": f(sam, "CNN_BWT"),
                    "random_BWT": f(random, "CNN_BWT"),
                    "delta_BWT": f(sam, "CNN_BWT") - f(random, "CNN_BWT"),
                    "sam_forget": f(sam, "CNN_Forget"),
                    "random_forget": f(random, "CNN_Forget"),
                    "delta_forget": f(sam, "CNN_Forget") - f(random, "CNN_Forget"),
                }
            )
    return out


def plot(rows: dict[str, dict]) -> None:
    x = np.arange(len(SUPPORTS))
    width = 0.34
    sam_faa = [f(rows[f"sam_{s}"], "CNN_FAA") for s in SUPPORTS]
    rnd_faa = [f(rows[f"random_{s}"], "CNN_FAA") for s in SUPPORTS]
    sam_bwt = [f(rows[f"sam_{s}"], "CNN_BWT") for s in SUPPORTS]
    rnd_bwt = [f(rows[f"random_{s}"], "CNN_BWT") for s in SUPPORTS]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    axes[0].bar(x - width / 2, sam_faa, width, label="SAM direction", color="#0072B2")
    axes[0].bar(x + width / 2, rnd_faa, width, label="Gaussian random direction", color="#999999")
    axes[0].axhline(f(rows["sgd"], "CNN_FAA"), color="#333333", lw=1.0, ls=":", label="SGD")
    axes[0].set_title("Final average accuracy")
    axes[0].set_ylabel("FAA (%)")

    axes[1].bar(x - width / 2, sam_bwt, width, label="SAM direction", color="#0072B2")
    axes[1].bar(x + width / 2, rnd_bwt, width, label="Gaussian random direction", color="#999999")
    axes[1].axhline(f(rows["sgd"], "CNN_BWT"), color="#333333", lw=1.0, ls=":", label="SGD")
    axes[1].set_title("Backward transfer")
    axes[1].set_ylabel("BWT (%)")

    labels = [SUPPORT_LABELS[s] for s in SUPPORTS]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"support_direction_faa_bwt.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    pairwise = write_pairwise(rows)
    plot(rows)
    print(pairwise)
    print(OUT_DIR / "support_direction_faa_bwt.png")
    print(OUT_DIR / "support_direction_faa_bwt.pdf")


if __name__ == "__main__":
    main()
