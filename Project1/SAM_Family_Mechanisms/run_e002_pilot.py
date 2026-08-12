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

from src.e002_experiment import run_e002_pilot  # noqa: E402


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("Non-JSON YAML requires PyYAML") from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the E002 Two-Moons shared-anchor and on-policy GPU pilot."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "e002_pilot.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "e002_gpu5_pilot",
    )
    parser.add_argument("--device", choices=("cuda:0", "cpu"))
    parser.add_argument("--probe-batches", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--quick", action="store_true", help="Small CPU/GPU smoke protocol")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = _load_config(args.config)
    for name in ("device", "probe_batches", "bootstrap_replicates"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.no_plots:
        config["make_plots"] = False
    result = run_e002_pilot(config, output_dir=args.output_dir, quick=args.quick)
    gates = result["metrics"]["gates"]
    print(f"E002 pilot complete: {result['output_dir']}")
    print(f"GPU identity verified: {gates['gpu_identity_verified']}")
    print(f"Method/numerical fidelity passed: {gates['method_fidelity_passed']}")
    print(
        "Taylor gate (primary/minimum eta): "
        f"{gates['primary_eta_taylor_passed']}/{gates['minimum_eta_taylor_passed']}"
    )
    print("Scope: calibration evidence; no causal or generalization ranking claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
