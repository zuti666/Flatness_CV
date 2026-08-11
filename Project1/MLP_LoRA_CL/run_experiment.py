#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from small_cl.config import load_config
from small_cl.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one controlled MLP Dense/LoRA CL experiment")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    run_experiment(load_config(args.config), project_root)


if __name__ == "__main__":
    main()
