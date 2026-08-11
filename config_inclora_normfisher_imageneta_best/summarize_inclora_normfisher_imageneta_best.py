#!/usr/bin/env python3
"""Summarize the IncLoRA-FS ImageNet-A best-params run."""

from __future__ import annotations

import glob
import json
import math
import os
from typing import Any


PATTERNS = {
    "seqlora_fs_best_g099_lam2000": (
        "outputs_logs/config_Expand_ewc_gamma099/**/"
        "as2normfisher_lam2000_ewcg099_imageneta_*_cl_metrics.json"
    ),
    "inclora_fs_best_g099_lam2000": (
        "outputs_logs/config_inclora_normfisher_imageneta_best/**/"
        "inclora_fs_lam2000_ewcg099_imageneta_*_cl_metrics.json"
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
    "ewc_penalty_value",
]


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not math.isnan(float(v))


def _load_latest(pattern: str) -> tuple[str | None, dict[str, Any] | None]:
    paths = sorted(set(glob.glob(pattern, recursive=True)))
    if not paths:
        return None, None
    path = paths[-1]
    with open(path, "r", encoding="utf-8") as f:
        return path, json.load(f)


def _last_row(data: dict[str, Any]) -> list[float]:
    matrix = data.get("cnn", {}).get("matrices", {}).get("final")
    return matrix[-1] if matrix else []


def _mech_means(data: dict[str, Any]) -> dict[str, float | None]:
    steps = data.get("mechanism", {}).get("steps", {})
    vals = []
    if isinstance(steps, dict):
        for step, metrics in steps.items():
            try:
                idx = int(str(step).lstrip("t"))
            except ValueError:
                idx = -1
            if idx <= 0 or metrics.get("delta_drift_available", 1) == 0:
                continue
            vals.append(metrics)
    out = {}
    for key in MECH_KEYS:
        xs = [float(v[key]) for v in vals if _finite(v.get(key))]
        out[key] = None if not xs else sum(xs) / len(xs)
    return out


def main() -> None:
    print("run,path,Task0_final,FAA,AAA,Forget_avg,cos_flat_fisher,fisher_clean,flat_clean,kl_iso,kl_fisher,kl_fisher_weighted,kl_fisher_iso_ratio,ewc_penalty")
    for name, pattern in PATTERNS.items():
        path, data = _load_latest(pattern)
        if data is None:
            print(f"{name},MISSING,,,,,,,,,,,,")
            continue
        final = data.get("cnn", {}).get("final", {})
        row = _last_row(data)
        mech = _mech_means(data)
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
            mech["ewc_penalty_value"],
        ]
        print(",".join("" if v is None else str(v) for v in vals))


if __name__ == "__main__":
    main()
