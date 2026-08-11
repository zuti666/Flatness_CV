from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "experiment_name": "rotated_mnist_mlp_lora",
    "seed": 0,
    "device": "cuda",
    "deterministic": True,
    "output_root": "outputs",
    "data": {
        "name": "rotated_mnist",
        "root": "data",
        "download": True,
        "num_tasks": 5,
        "rotation_span": 60.0,
        "angles": None,
        "include_angles_in_run_name": False,
        "train_subset": None,
        "test_subset": None,
        "num_workers": 2,
    },
    "model": {
        "input_dim": 784,
        "hidden_dim": 32,
        "num_classes": 10,
        "activation": "gelu",
        "parameterization": "factor_lora",
        # Optional causal control: train task A with this parameterization,
        # then rebase every target parameterization at the exact same effective
        # weight before task B.  Currently intended for two-task experiments.
        "task_a_parameterization": None,
        "rank": 4,
        "subspace_dim": None,
        "lora_alpha": "rank",
        "factor_gauge_scale": 1.0,
        "lifecycle": "persistent",
        "subspace_seed": "run_seed",
        "balance_every": "epoch",
    },
    "base": {
        "mode": "pretrained_mnist",
        "checkpoint": "checkpoints/mlp_base_gelu_h32.pt",
        "pretrain_angle": 0.0,
        "seed": 3407,
        "epochs": 5,
        "batch_size": 256,
        "lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 0.0,
    },
    "training": {
        "epochs_per_task": 5,
        "batch_size": 128,
        "optimizer": "sgd",
        "optimizer_schedule": None,
        "lr": 0.03,
        "momentum": 0.9,
        "task_a_lr": None,
        "task_a_momentum": None,
        "task_b_lr": None,
        "parameterization_lrs": None,
        "weight_decay": 0.0,
        "sam_rho": 0.05,
        "parameterization_sam_rhos": None,
        "gam_radius": 0.01,
        "gam_weight": 0.05,
        "perturbation_metric": "parameter",
        "grad_clip": None,
    },
    "diagnostics": {
        "enabled": True,
        "max_samples_per_task": 512,
        "save_hvp_tensors": False,
        "pathwise": False,
        "path_granularity": "epoch",
        "pathwise_immediate_only": True,
        "hvp_fd_radii": [],
        "hessian_lanczos_steps": 0,
        "geometry_enabled": True,
        "geometry_immediate_only": True,
        "prospective_enabled": False,
        "trajectory_eval": False,
        "ggn_damping": 0.01,
        "safe_relative_threshold": 0.01,
        "ggn_overlap_top_k": 16,
    },
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    return _deep_update(copy.deepcopy(DEFAULTS), supplied)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def resolve_path(path: str | Path, project_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else project_root / value
