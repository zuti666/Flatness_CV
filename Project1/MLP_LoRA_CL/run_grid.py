#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

from small_cl.config import load_config, resolve_path
from small_cl.data import prepare_dataset
from small_cl.experiment import _run_name, run_experiment


GRID_KEYS = {
    "parameterizations": "model.parameterization",
    "ranks": "model.rank",
    "factor_gauge_scales": "model.factor_gauge_scale",
    "lifecycles": "model.lifecycle",
    "optimizers": "training.optimizer",
    "optimizer_schedules": "training.optimizer_schedule",
    "learning_rates": "training.lr",
    "task_b_learning_rates": "training.task_b_lr",
    "sam_radii": "training.sam_rho",
    "gam_radii": "training.gam_radius",
    "angle_pairs": "data.angles",
    "rotation_spans": "data.rotation_span",
    "target_ranks": "data.teacher.target_rank",
    "principal_angles": "data.teacher.principal_angle_degrees",
    "seeds": "seed",
}


def freeze_identity(value):
    if isinstance(value, list):
        return tuple(freeze_identity(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, freeze_identity(item)) for key, item in value.items()))
    return value


def set_nested(config: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    target = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def variants(config: dict):
    grid = config.pop("grid", {})
    keys = [key for key in GRID_KEYS if key in grid]
    value_lists = [grid[key] for key in keys]
    seen_jobs = set()
    for values in itertools.product(*value_lists):
        variant = copy.deepcopy(config)
        labels = []
        for key, value in zip(keys, values):
            set_nested(variant, GRID_KEYS[key], value)
            labels.append(f"{key}={value}")
        # Collapse grid axes that do not affect the resolved method.
        ignored = set()
        if variant["model"]["parameterization"] == "dense":
            ignored.add("ranks")
        optimizer = variant["training"]["optimizer"]
        schedule = variant["training"].get("optimizer_schedule") or []
        used_optimizers = {str(optimizer).lower(), *(str(item).lower() for item in schedule)}
        if "sam" not in used_optimizers:
            ignored.add("sam_radii")
        if not used_optimizers.intersection({"gam_fd", "gam_exact"}):
            ignored.add("gam_radii")
        identity = tuple(
            (key, freeze_identity(value))
            for key, value in zip(keys, values)
            if key not in ignored
        )
        if identity in seen_jobs:
            continue
        seen_jobs.add(identity)
        yield ", ".join(labels), variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand and run the MLP Dense/LoRA experiment grid")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-data-only",
        action="store_true",
        help="Download/check the configured external dataset, then exit",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many expanded jobs (use with --limit for disjoint workers)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Do not rerun jobs whose output directory already contains metrics.json",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_config(args.config)
    if args.prepare_data_only:
        print(f"dataset ready: {prepare_dataset(config, project_root)}")
        return
    if not args.dry_run:
        print(f"dataset ready: {prepare_dataset(config, project_root)}")
    jobs = list(variants(config))
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    jobs = jobs[args.offset:]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    print(f"expanded {len(jobs)} runs")
    for index, (label, variant) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {label}")
        if not args.dry_run:
            metrics_path = (
                resolve_path(variant["output_root"], project_root)
                / _run_name(variant)
                / "metrics.json"
            )
            if args.skip_completed and metrics_path.exists():
                print(f"skipping completed run: {metrics_path.parent}")
                continue
            run_experiment(variant, project_root)


if __name__ == "__main__":
    main()
