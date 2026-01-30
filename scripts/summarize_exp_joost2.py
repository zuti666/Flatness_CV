#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_ROOT = (
    "logs_inc_lora/seqlora/sgd/imagenetr/1993/"
    "seqlora_curvloc_imagenetr_vitb16_r16_task1_joost/exp_joost2"
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _maybe_float(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    return val


def _safe_get(dct: Dict[str, Any], *keys: str) -> Any:
    cur: Any = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _compute_cumsum(ratios: Optional[List[float]]) -> List[float]:
    if not ratios:
        return []
    out: List[float] = []
    total = 0.0
    for r in ratios:
        total += float(r)
        out.append(total)
    return out


def _pick_cumsum(cumsum: List[float], m: int) -> Optional[float]:
    if m <= 0 or len(cumsum) < m:
        return None
    return cumsum[m - 1]


def _summarize_run(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    flat_dir = run_dir / "flatness"
    if not flat_dir.is_dir():
        return rows

    cl_metrics_path = next(run_dir.glob("*_cl_metrics.json"), None)
    cl_metrics = _load_json(cl_metrics_path) if cl_metrics_path else {}

    for metrics_path in sorted(flat_dir.glob("*_metrics.json")):
        metrics = _load_json(metrics_path)
        ratios = metrics.get("curv_proj_ratios")
        cumsum = metrics.get("curv_proj_ratio_cumsum")
        if cumsum is None:
            cumsum = _compute_cumsum(ratios)

        row = {
            "run": run_dir.name,
            "metrics_file": metrics_path.name,
            "base_loss": _maybe_float(metrics.get("base_loss")),
            "lambda_max_power": _maybe_float(metrics.get("lambda_max_power")),
            "emp_fisher_lambda_max_power": _maybe_float(metrics.get("emp_fisher_lambda_max_power")),
            "emp_fisher_lambda_max_lanczos": _maybe_float(metrics.get("emp_fisher_lambda_max_lanczos")),
            "emp_fisher_trace": _maybe_float(metrics.get("emp_fisher_trace")),
            "curv_proj_ratio_top1": _maybe_float(metrics.get("curv_proj_ratio_top1")),
            "curv_proj_ratio_cumsum_m1": _maybe_float(_pick_cumsum(cumsum, 1)),
            "curv_proj_ratio_cumsum_m2": _maybe_float(_pick_cumsum(cumsum, 2)),
            "curv_proj_ratio_cumsum_m5": _maybe_float(_pick_cumsum(cumsum, 5)),
            "curv_proj_ratio_cumsum_m10": _maybe_float(_pick_cumsum(cumsum, 10)),
            "cnn_acc": _maybe_float(_safe_get(cl_metrics, "cnn", "final", "Acc")),
            "nme_acc": _maybe_float(_safe_get(cl_metrics, "nme", "final", "Acc")),
            "probe_final": _maybe_float(
                _safe_get(cl_metrics, "probe_softmax_joint_seen_all", "final", "final_model")
            ),
            "probe_base": _maybe_float(
                _safe_get(cl_metrics, "probe_softmax_joint_seen_all", "final", "base_model")
            ),
        }
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize exp_joost2 results into a CSV table.")
    parser.add_argument(
        "root",
        nargs="?",
        default=DEFAULT_ROOT,
        help=f"Root directory for runs (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output CSV path. Defaults to stdout.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "checkpoints"):
        rows.extend(_summarize_run(run_dir))

    if not rows:
        raise SystemExit(f"No metrics found under {root}")

    fieldnames = list(rows[0].keys())
    out_path = Path(args.out) if args.out else None
    out_fh = out_path.open("w", encoding="utf-8", newline="") if out_path else None
    try:
        writer = csv.DictWriter(out_fh or sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if out_fh:
            out_fh.close()


if __name__ == "__main__":
    main()
