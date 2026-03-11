"""Unified entry for incremental and all-data training.

This module is a thin wrapper over existing trainer scripts so current logic
keeps working while providing a stable `python -m flatness_cil` interface.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is on sys.path so imports like `utils` work when running src/main.py directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    # Package-relative (preferred when using `python -m src.main`)
    from .trainer import train as train_inc
    from .trainer_allData import train_all
except ImportError:  # fallback when invoked as a script
    from trainer import train as train_inc  # type: ignore
    from trainer_allData import train_all  # type: ignore
from utils.config import load_config


def _parse_override_pairs(pairs) -> Dict[str, object]:
    if not pairs:
        return {}

    def _convert(value: str):
        lower = value.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower in {"null", "none"}:
            return None
        # allow JSON-style list/dict literals
        if value.startswith("[") or value.startswith("{"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        try:
            if value.startswith("[") or value.startswith("{"):
                return json.loads(value)
        except json.JSONDecodeError:
            pass
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
        return value

    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' is not in key=value format")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override key in '{pair}'")
        overrides[key] = _convert(value.strip())
    return overrides


def setup_parser():
    parser = argparse.ArgumentParser(description="Flatness continual learning runner")
    parser.add_argument("--config", type=str, default="./exps/simplecil.json", help="Settings file (.json/.yaml/.yml)")
    parser.add_argument("--mode", type=str, choices=["inc", "all"], help="Training mode; default derives from config all_or_inc")
    parser.add_argument("--override", nargs="+", help="Override config entries via key=value pairs")
    return parser


def main(argv=None):
    ns = setup_parser().parse_args(argv)
    cfg = load_config(ns.config)

    # merge CLI flags
    cli_dict = vars(ns)
    override_pairs = cli_dict.pop("override", None)
    for k, v in cli_dict.items():
        if k == "config":
            continue
        if v is not None:
            cfg[k] = v
    cfg.update(_parse_override_pairs(override_pairs))
    cfg["config"] = ns.config

    mode = cfg.get("mode", cfg.get("all_or_inc", "inc"))
    if str(mode) == "inc":
        train_inc(cfg)
    else:
        train_all(cfg)


if __name__ == "__main__":
    main()
