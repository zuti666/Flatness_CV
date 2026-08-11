#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SEED = "1993"
BASE = ROOT / "outputs_logs" / "logs_inc_lora" / "seqlora" / "exp_F_forked_taskwise" / "imagenetr" / SEED
OUT_DIR = ROOT / "outputs_logs" / "exp_F_imagenetr_r16_t20_forked_taskwise_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "sgd_sgd",
    "sgd_sam_factor",
    "sgd_random_factor",
    "sam_factor_sgd",
    "sam_factor_sam_factor",
]

LABELS = {
    "sgd_sgd": "SGD -> SGD",
    "sgd_sam_factor": "SGD -> SAM-factor",
    "sgd_random_factor": "SGD -> Random-factor",
    "sam_factor_sgd": "SAM-factor -> SGD",
    "sam_factor_sam_factor": "SAM-factor -> SAM-factor",
}


def load_matrix(variant: str) -> np.ndarray | None:
    prefix = f"exp_F_imagenetr_r16_t20_fork_{variant}"
    path = BASE / prefix / "exp_run" / "10" / f"{prefix}_vit_base_patch16_224_cl_metrics.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        book = json.load(f)
    matrix = book.get("cnn", {}).get("matrices", {}).get("final")
    if matrix is None:
        return None
    return np.asarray(matrix, dtype=float)


def sequence_metrics(matrix: np.ndarray, prefix_tasks: int = 10) -> dict[str, float]:
    t = matrix.shape[0]
    final = matrix[t - 1, :t]
    diag = np.asarray([matrix[i, i] for i in range(t)], dtype=float)
    lower_mask = np.tril(np.ones_like(matrix, dtype=bool), k=0)
    aaa = np.nanmean(matrix[lower_mask])
    bwt = np.nanmean(matrix[t - 1, : t - 1] - diag[: t - 1]) if t > 1 else float("nan")
    if t > 1:
        max_per_task = np.nanmax(matrix[:, : t - 1], axis=0)
        forget = np.nanmean(max_per_task - matrix[t - 1, : t - 1])
    else:
        forget = float("nan")
    prefix_forget = np.nanmean([matrix[i, i] - matrix[t - 1, i] for i in range(min(prefix_tasks, t))])
    suffix_acc = np.nanmean(final[prefix_tasks:t])
    return {
        "FAA": float(np.nanmean(final)),
        "AAA": float(aaa),
        "BWT": float(bwt),
        "forget": float(forget),
        "prefix_retention": float(np.nanmean(final[:prefix_tasks])),
        "prefix_diag": float(np.nanmean(diag[:prefix_tasks])),
        "prefix_forget": float(prefix_forget),
        "suffix_final_acc": float(suffix_acc),
    }


def write_prefix_full_table(rows: list[dict[str, float]]) -> None:
    table_rows = []
    matrices = {row["variant"]: load_matrix(row["variant"]) for row in rows}
    for variant in VARIANTS:
        matrix = matrices.get(variant)
        if matrix is None:
            continue
        prefix_metrics = sequence_metrics(matrix[:10, :10], prefix_tasks=10)
        full_metrics = sequence_metrics(matrix, prefix_tasks=10)
        table_rows.append(
            {
                "Method": LABELS.get(variant, variant),
                "t0_t10_AAA": prefix_metrics["AAA"],
                "t0_t10_FAA": prefix_metrics["FAA"],
                "t0_t10_BWT": prefix_metrics["BWT"],
                "t0_t20_AAA": full_metrics["AAA"],
                "t0_t20_FAA": full_metrics["FAA"],
                "t0_t20_BWT": full_metrics["BWT"],
            }
        )

    if not table_rows:
        return

    out_csv = OUT_DIR / "prefix_full_aaa_faa_bwt_table.csv"
    fieldnames = list(table_rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)
    print(out_csv)

    out_tex = OUT_DIR / "prefix_full_aaa_faa_bwt_table.tex"
    with out_tex.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Method & \\multicolumn{3}{c}{t0--t10} & \\multicolumn{3}{c}{t0--t20} \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
        f.write(" & AAA $\\uparrow$ & FAA $\\uparrow$ & BWT $\\uparrow$ & AAA $\\uparrow$ & FAA $\\uparrow$ & BWT $\\uparrow$ \\\\\n")
        f.write("\\midrule\n")
        for row in table_rows:
            f.write(
                f"{row['Method']} & "
                f"{row['t0_t10_AAA']:.2f} & {row['t0_t10_FAA']:.2f} & {row['t0_t10_BWT']:.2f} & "
                f"{row['t0_t20_AAA']:.2f} & {row['t0_t20_FAA']:.2f} & {row['t0_t20_BWT']:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(out_tex)


def main() -> None:
    rows = []
    for variant in VARIANTS:
        matrix = load_matrix(variant)
        if matrix is None:
            continue
        row = {"variant": variant}
        row.update(sequence_metrics(matrix))
        rows.append(row)

    out_csv = OUT_DIR / "summary_by_variant.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(out_csv)
        write_prefix_full_table(rows)
        for row in rows:
            print(
                f"{row['variant']:24s} FAA={row['FAA']:.2f} AAA={row['AAA']:.2f} "
                f"BWT={row['BWT']:.2f} prefix_forget={row['prefix_forget']:.2f} "
                f"suffix_acc={row['suffix_final_acc']:.2f}"
            )
    else:
        print("No completed Exp F forked metrics found.")


if __name__ == "__main__":
    main()
