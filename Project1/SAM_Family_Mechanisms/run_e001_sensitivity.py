#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sensitivity import (  # noqa: E402
    DEFAULT_CONDITION_NUMBERS,
    DEFAULT_DIMENSIONS,
    DEFAULT_INNER_STEPS,
    DEFAULT_RHO_VALUES,
    DEFAULT_STRENGTH_TARGETS,
    run_sensitivity_analysis,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run E001-S: one-factor sensitivity and strict native-radius "
            "correction-strength matching on the exact quadratic model."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "e001_sensitivity",
    )
    parser.add_argument("--rho-values", type=float, nargs="+")
    parser.add_argument("--condition-numbers", type=float, nargs="+")
    parser.add_argument("--dimensions", type=int, nargs="+")
    parser.add_argument("--inner-steps", type=int, nargs="+")
    parser.add_argument("--strength-targets", type=float, nargs="+")
    parser.add_argument("--initialization-samples", type=int)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a two-point grid per factor for CI/smoke validation.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.quick:
        defaults = {
            "rho_values": (1e-3, 1e-2),
            "condition_numbers": (4.0, 100.0),
            "dimensions": (10, 20),
            "inner_steps": (1, 5),
            "strength_targets": (0.1,),
            "initialization_samples": 8,
        }
    else:
        defaults = {
            "rho_values": DEFAULT_RHO_VALUES,
            "condition_numbers": DEFAULT_CONDITION_NUMBERS,
            "dimensions": DEFAULT_DIMENSIONS,
            "inner_steps": DEFAULT_INNER_STEPS,
            "strength_targets": DEFAULT_STRENGTH_TARGETS,
            "initialization_samples": 100,
        }
    for name in defaults:
        supplied = getattr(args, name)
        if supplied is not None:
            defaults[name] = supplied
    result = run_sensitivity_analysis(
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
        **defaults,
    )
    manifest = result["manifest"]
    maximum_matching_error = max(
        float(row["matching_relative_error"])
        for row in result["strength_matched_rows"]
    )
    print(f"E001-S complete: {result['output_dir']}")
    print(
        f"OFAT cases: {manifest['case_count']}; "
        f"strict strength-match rows: {manifest['strength_matched_row_count']}"
    )
    print(f"Maximum correction-strength matching error: {maximum_matching_error:.3e}")
    print("Scope: robustness within the PSD quadratic family only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
