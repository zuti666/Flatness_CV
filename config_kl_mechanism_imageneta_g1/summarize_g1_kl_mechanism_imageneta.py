#!/usr/bin/env python3
"""Summarize ImageNet-A gamma=1.0 KL mechanism diagnostics."""

from __future__ import annotations

import glob
import json
import math
import os
from typing import Any


RUNS = {
    "g0.99_lam2000_current_best": (
        "outputs_logs/config_Expand_ewc_gamma099/**/"
        "as2normfisher_lam2000_ewcg099_imageneta_*_cl_metrics.json"
    ),
    "g1.00_lam2000_direct": (
        "outputs_logs/config_kl_mechanism_imageneta_g1/**/"
        "klmetric_as2normfisher_lam2000_ewcg100_imageneta_*_cl_metrics.json"
    ),
    "g1.00_lam1900_recommended": (
        "outputs_logs/config_kl_mechanism_imageneta_g1/**/"
        "klmetric_as2normfisher_lam1900_ewcg100_imageneta_*_cl_metrics.json"
    ),
}

MECH_KEYS = [
    "grad_cos_flat_fisher_mean",
    "grad_fisher_to_clean_norm_ratio_mean",
    "grad_flat_to_clean_norm_ratio_mean",
    "kl_iso_delta",
    "kl_fisher_delta",
    "kl_fisher_delta_weighted",
    "kl_fisher_to_iso_ratio",
    "high_fisher_weighted_energy_ratio",
    "ewc_penalty_value",
]


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not math.isnan(float(v))


def _mean(xs: list[float]) -> float | None:
    return None if not xs else sum(xs) / len(xs)


def _load_one(pattern: str) -> tuple[str | None, dict[str, Any] | None]:
    paths = sorted(set(glob.glob(pattern, recursive=True)))
    if not paths:
        return None, None
    path = paths[-1]
    with open(path, "r", encoding="utf-8") as f:
        return path, json.load(f)


def _last_row(data: dict[str, Any]) -> list[float]:
    matrix = data.get("cnn", {}).get("matrices", {}).get("final")
    if matrix:
        return matrix[-1]
    return []


def _mechanism_means(data: dict[str, Any]) -> dict[str, float | None]:
    steps = data.get("mechanism", {}).get("steps", {})
    values = []
    if isinstance(steps, dict):
        for step, metrics in steps.items():
            try:
                step_idx = int(str(step).lstrip("t"))
            except ValueError:
                step_idx = -1
            if step_idx <= 0:
                continue
            if metrics.get("delta_drift_available", 1) == 0:
                continue
            values.append(metrics)
    out: dict[str, float | None] = {}
    for key in MECH_KEYS:
        out[key] = _mean([float(v[key]) for v in values if _finite(v.get(key))])
    return out


def main() -> None:
    print("run,path,Task0_final,FAA,AAA,Forget_avg,cos_flat_fisher,fisher_clean,flat_clean,kl_iso,kl_fisher,kl_fisher_weighted,kl_fisher_iso_ratio,high_fisher_weighted_ratio,ewc_penalty")
    for name, pattern in RUNS.items():
        path, data = _load_one(pattern)
        if data is None:
            print(f"{name},MISSING,,,,,,,,,,,,,")
            continue
        final = data.get("cnn", {}).get("final", {})
        row = _last_row(data)
        mech = _mechanism_means(data)
        vals = [
            name,
            os.path.relpath(path or "."),
            row[0] if row else "",
            final.get("FAA", ""),
            final.get("AAA", ""),
            final.get("Forget_avg", ""),
            mech["grad_cos_flat_fisher_mean"],
            mech["grad_fisher_to_clean_norm_ratio_mean"],
            mech["grad_flat_to_clean_norm_ratio_mean"],
            mech["kl_iso_delta"],
            mech["kl_fisher_delta"],
            mech["kl_fisher_delta_weighted"],
            mech["kl_fisher_to_iso_ratio"],
            mech["high_fisher_weighted_energy_ratio"],
            mech["ewc_penalty_value"],
        ]
        print(",".join("" if v is None else str(v) for v in vals))


if __name__ == "__main__":
    main()
