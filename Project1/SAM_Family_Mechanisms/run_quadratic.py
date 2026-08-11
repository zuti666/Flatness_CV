#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import run_quadratic_experiment  # noqa: E402


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError(
                "Non-JSON YAML requires PyYAML; the bundled config is JSON-compatible YAML"
            ) from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run E001: exact operator diagnostics on a 20D quadratic model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "quadratic.yaml",
        help="JSON or YAML configuration file",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--lambda-min", type=float)
    parser.add_argument("--lambda-max", type=float)
    parser.add_argument("--rho-scales", type=float, nargs="+")
    parser.add_argument("--primary-rho-scale", type=float)
    parser.add_argument("--inner-steps", type=int, nargs="+")
    parser.add_argument(
        "--path-protocols",
        choices=("fixed_step", "fixed_budget"),
        nargs="+",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = _load_config(args.config)
    overrides = {
        "dimension": args.dimension,
        "lambda_min": args.lambda_min,
        "lambda_max": args.lambda_max,
        "rho_scales": args.rho_scales,
        "primary_rho_scale": args.primary_rho_scale,
        "inner_steps": args.inner_steps,
        "path_protocols": args.path_protocols,
        "seed": args.seed,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    if args.no_plots:
        config["make_plots"] = False
    result = run_quadratic_experiment(config, output_dir=args.output_dir)
    checks = result["metrics"]["checks"]
    print(f"E001 complete: {result['output_dir']}")
    print(f"SAM identity relative error: {checks['sam_identity_relative_error']:.3e}")
    print(f"GAM probe/H^2 cosine: {checks['gam_probe_h2_cosine']:.12f}")
    print("Scope: operator-level evidence only; E002/E003 remain planned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

