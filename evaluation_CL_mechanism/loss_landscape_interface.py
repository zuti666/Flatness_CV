"""
loss_landscape_interface.py
---------------------------
Thin wrapper that exposes loss-landscape artifacts from evaluation_CL_mechanism.

The goal is not to duplicate the large flatness pipeline, only to let the
mechanism evaluator request small 1D/2D landscapes on old/new/all slices and
serialize them in a stable format that visualize.py can render later.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import numpy as np
import torch

from evaluation_weight_sharpness.loss_landscape import _loss_landscape_1d, _loss_landscape_2d

logger = logging.getLogger(__name__)


def _artifact_dir(save_root: str) -> str:
    return os.path.join(save_root, "loss_landscape")


def _collect_trainable_params(network) -> List[torch.nn.Parameter]:
    return [p for p in network.parameters() if p.requires_grad]


def run_loss_landscape_suite(
    network,
    loaders: Dict[str, Any],
    *,
    device: torch.device,
    save_root: str,
    task_idx: int,
    args: dict,
) -> Dict[str, Any]:
    """
    Optionally compute 1D/2D random-direction landscapes on selected slices.
    """
    if not bool(args.get("cl_eval_lossland", False)):
        return {}

    targets = str(args.get("cl_eval_lossland_targets", "old,new,all")).split(",")
    targets = [t.strip().lower() for t in targets if t.strip()]
    do_1d = bool(args.get("cl_eval_lossland_1d", True))
    do_2d = bool(args.get("cl_eval_lossland_2d", False))
    radius = float(args.get("cl_eval_lossland_radius", 0.25))
    points = int(args.get("cl_eval_lossland_points", 15))
    max_batches = int(args.get("cl_eval_lossland_max_batches", args.get("cl_eval_max_batches", 5)))
    filter_norm = bool(args.get("cl_eval_lossland_filter_norm", True))
    seed_base = int(args.get("seed", 0) if not isinstance(args.get("seed"), list) else args.get("seed")[0])

    params = _collect_trainable_params(network)
    if not params:
        return {"lossland_status": "skipped_no_trainable_params"}

    out_dir = _artifact_dir(save_root)
    os.makedirs(out_dir, exist_ok=True)
    metrics: Dict[str, Any] = {}

    for offset, target in enumerate(targets):
        loader = loaders.get(target)
        if loader is None:
            metrics[f"lossland_{target}_status"] = "skipped_no_loader"
            continue

        if do_1d:
            try:
                res1d = _loss_landscape_1d(
                    network,
                    loader,
                    device,
                    params,
                    radius=radius,
                    num_points=points,
                    max_batches=max_batches,
                    filter_norm=filter_norm,
                )
                path1d = os.path.join(out_dir, f"task{task_idx:02d}_{target}_1d.npz")
                np.savez_compressed(path1d, **res1d)
                metrics[f"lossland_{target}_1d_file"] = path1d
                base = float(np.min(res1d["loss"])) if res1d["loss"].size > 0 else 0.0
                metrics[f"lossland_{target}_1d_min"] = float(np.min(res1d["loss"]))
                metrics[f"lossland_{target}_1d_max"] = float(np.max(res1d["loss"]))
                metrics[f"lossland_{target}_1d_delta_max"] = float(np.max(res1d["loss"]) - base)
            except Exception as exc:
                logger.warning("[lossland] 1D %s failed at task %d: %s", target, task_idx, exc)
                metrics[f"lossland_{target}_1d_status"] = f"failed:{exc}"

        if do_2d:
            try:
                torch.manual_seed(seed_base + 17 * (task_idx + 1) + offset)
                res2d = _loss_landscape_2d(
                    network,
                    loader,
                    device,
                    params,
                    radius=radius,
                    num_points=points,
                    max_batches=max_batches,
                    filter_norm=filter_norm,
                )
                path2d = os.path.join(out_dir, f"task{task_idx:02d}_{target}_2d.npz")
                np.savez_compressed(path2d, **res2d)
                metrics[f"lossland_{target}_2d_file"] = path2d
                metrics[f"lossland_{target}_2d_min"] = float(np.min(res2d["loss"]))
                metrics[f"lossland_{target}_2d_max"] = float(np.max(res2d["loss"]))
            except Exception as exc:
                logger.warning("[lossland] 2D %s failed at task %d: %s", target, task_idx, exc)
                metrics[f"lossland_{target}_2d_status"] = f"failed:{exc}"

    return metrics
