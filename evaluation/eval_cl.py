"""Offline continual learning evaluation for LoRA-based methods.

This script reconstructs task-wise models from saved LoRA adapters and
classifier heads, reports CL metrics (CA/BWT/FG as well as ACA/ABWT/AFG/FAA),
and optionally evaluates parameter- and feature-space flatness on each task's
training data. It mirrors the bookkeeping performed inside ``trainer.py`` while
avoiding any additional training.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import os
from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import compute_sequence_metrics
from evaluation.probe import evaluate_linear_probe
from eval_flat.eval_flatness_weight_Loss import FlatnessConfig, evaluate_flatness_metrics
from eval_flat.eval_flat_feature import (
    FeatureFlatnessConfig,
    evaluate_feature_metrics,
)
from trainer import _set_device, _set_random, print_args
from utils.data_manager import DataManager
from utils.factory import get_model

try:
    from torch.backends.cuda import sdp_kernel
except (ImportError, AttributeError):  # pragma: no cover - dependent on torch version
    sdp_kernel = None


def _default_checkpoint_dir(cfg: dict) -> str:
    init_cls = 0 if cfg["init_cls"] == cfg["increment"] else cfg["init_cls"]
    return os.path.join(
        "logs",
        cfg["model_name"],
        cfg["dataset"],
        str(init_cls),
        "checkpoints",
    )


def _build_loader(
    data_manager: DataManager,
    class_range: Sequence[int],
    batch_size: int,
    num_workers: int,
    source: str = "test",
    mode: str = "test",
    shuffle: bool = False,
) -> DataLoader:
    dataset = data_manager.get_dataset(np.arange(class_range[0], class_range[1]), source=source, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _parse_task_list(task_spec: str | None, nb_tasks: int) -> List[int]:
    if not task_spec or task_spec.lower() == "all":
        return list(range(nb_tasks))
    indices = []
    for chunk in task_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value < 0 or value >= nb_tasks:
            raise ValueError(f"Task index {value} outside valid range [0, {nb_tasks - 1}]")
        indices.append(value)
    return sorted(set(indices))


def run_evaluation(cfg: dict, args: argparse.Namespace) -> None:
    seed = args.seed if args.seed is not None else cfg["seed"][0]
    cfg_eval = copy.deepcopy(cfg)
    cfg_eval["seed"] = [seed]

    _set_device(cfg_eval)
    _set_random(seed)

    logging.info("Loaded config for evaluation (seed=%d)", seed)
    print_args(cfg_eval)

    data_manager = DataManager(
        cfg_eval["dataset"],
        cfg_eval["shuffle"],
        seed,
        cfg_eval["init_cls"],
        cfg_eval["increment"],
        cfg_eval,
    )

    nb_tasks = data_manager.nb_tasks
    class_ranges = [data_manager.get_task_class_range(idx) for idx in range(nb_tasks)]

    task_indices = _parse_task_list(args.tasks, nb_tasks)
    logging.info("Evaluating tasks: %s", task_indices)

    # Switches aligned with trainer.py
    head_eval = bool(cfg.get("head_eval", False))
    linear_probe_eval = bool(cfg.get("linear_probe_eval", False))
    linear_probe_eval_base = bool(cfg.get("linear_probe_eval_base", True))
    flat_eval = bool(cfg.get("flat_eval", False)) or bool(getattr(args, "flat_eval", False))
    feature_flat_eval = bool(cfg.get("feature_flat_eval", False)) or bool(
        getattr(args, "feature_flat_eval", False)
    )

    head_R = np.full((nb_tasks, nb_tasks), np.nan, dtype=float) if head_eval else None
    probe_R = np.full((nb_tasks, nb_tasks), np.nan, dtype=float) if linear_probe_eval else None

    feature_save_path = args.feature_flat_save_path or cfg_eval.get("feature_flat_save_path")

    # Optional: baseline linear-probe before any task training (like trainer.py)
    if linear_probe_eval and linear_probe_eval_base:
        base_learner = get_model(cfg_eval["model_name"], copy.deepcopy(cfg_eval))

        # Ensure baseline network is on the target device before feature extraction
        if hasattr(base_learner, "_prepare_network"):
            base_learner._prepare_network()
        elif hasattr(base_learner, "_network"):
            device = getattr(base_learner, "_device", None)
            if device is not None:
                base_learner._network.to(device)

        def _base_loader(class_range, source="test", mode="test", shuffle=False):
            return _build_loader(
                data_manager,
                class_range,
                batch_size=cfg_eval.get("eval_batch_size", cfg_eval.get("batch_size", 128)),
                num_workers=cfg_eval.get("eval_num_workers", 2),
                source=source,
                mode=mode,
                shuffle=shuffle,
            )

        train_mode = cfg_eval.get("probe_train_mode", "test")
        test_mode = cfg_eval.get("probe_test_mode", "test")
        max_train_batches = cfg_eval.get("probe_train_max_batches", None)
        max_test_batches = cfg_eval.get("probe_test_max_batches", None)
        l2_reg = cfg_eval.get("probe_ridge_lambda", 1e-3)

        base_row = np.full(nb_tasks, np.nan, dtype=float)
        for j, (start, end) in enumerate(class_ranges):
            tr_loader = _base_loader(class_ranges[j], source="train", mode=train_mode)
            te_loader = _base_loader(class_ranges[j], source="test", mode=test_mode)
            acc = evaluate_linear_probe(
                base_learner._network,
                tr_loader,
                te_loader,
                class_offset=start,
                num_classes=end - start,
                device=base_learner._device,
                l2_reg=l2_reg,
                max_train_batches=max_train_batches,
                max_test_batches=max_test_batches,
            )
            base_row[j] = acc
        logging.info(
            "Linear probe baseline accuracies (pre-training): %s",
            np.array2string(
                base_row, precision=2, formatter={"float_kind": lambda x: f"{x:.2f}"}
            ),
        )

    for task_idx in task_indices:
        task_args = copy.deepcopy(cfg_eval)
        learner = get_model(task_args["model_name"], task_args)

        if not hasattr(learner, "restore_task_snapshot"):
            raise AttributeError(
                f"Model {task_args['model_name']} does not expose 'restore_task_snapshot'; offline evaluation is unsupported."
            )

        learner.restore_task_snapshot(data_manager, task_idx)

        cnn_accy, nme_accy = learner.eval_task()
        logging.info(
            "Task %d => CNN top1 %.2f | top5 %.2f",
            task_idx,
            cnn_accy["top1"],
            cnn_accy.get("top5", float("nan")),
        )
        if nme_accy is not None:
            logging.info(
                "Task %d => NME top1 %.2f | top5 %.2f",
                task_idx,
                nme_accy["top1"],
                nme_accy.get("top5", float("nan")),
            )

        # Populate accuracy matrix row using per-task test splits (gated by head_eval)
        if head_eval and head_R is not None:
            row = np.full(nb_tasks, np.nan, dtype=float)
            for j in range(task_idx + 1):
                loader = _build_loader(
                    data_manager,
                    class_ranges[j],
                    batch_size=cfg_eval.get("eval_batch_size", cfg_eval.get("batch_size", 128)),
                    num_workers=cfg_eval.get("eval_num_workers", 2),
                )
                row[j] = learner._compute_accuracy(learner._network, loader)
            head_R[task_idx, :] = row

        # Linear probe evaluation mirroring trainer.py (if enabled)
        if linear_probe_eval and probe_R is not None:
            row = np.full(nb_tasks, np.nan, dtype=float)
            train_mode = cfg_eval.get("probe_train_mode", "test")
            test_mode = cfg_eval.get("probe_test_mode", "test")
            max_train_batches = cfg_eval.get("probe_train_max_batches", None)
            max_test_batches = cfg_eval.get("probe_test_max_batches", None)
            l2_reg = cfg_eval.get("probe_ridge_lambda", 1e-3)

            def _ldr(crange, source, mode):
                return _build_loader(
                    data_manager,
                    crange,
                    batch_size=cfg_eval.get("eval_batch_size", cfg_eval.get("batch_size", 128)),
                    num_workers=cfg_eval.get("eval_num_workers", 2),
                    source=source,
                    mode=mode,
                )

            for j, (start, end) in enumerate(class_ranges):
                tr_loader = _ldr(class_ranges[j], source="train", mode=train_mode)
                te_loader = _ldr(class_ranges[j], source="test", mode=test_mode)
                acc = evaluate_linear_probe(
                    learner._network,
                    tr_loader,
                    te_loader,
                    class_offset=start,
                    num_classes=end - start,
                    device=learner._device,
                    l2_reg=l2_reg,
                    max_train_batches=max_train_batches,
                    max_test_batches=max_test_batches,
                )
                row[j] = acc
            probe_R[task_idx, :] = row

        if flat_eval:
            try:
                dataset_fraction = cfg_eval.get("flat_eval_dataset_fraction", getattr(args, "flat_eval_dataset_fraction", None))
                dataset_fraction_seed = cfg_eval.get("flat_eval_dataset_fraction_seed", getattr(args, "flat_eval_dataset_fraction_seed", None))
                if dataset_fraction_seed is None:
                    base_seed = cfg_eval.get("seed", getattr(args, "seed", None))
                    if isinstance(base_seed, (list, tuple)) and base_seed:
                        dataset_fraction_seed = base_seed[0]
                    elif isinstance(base_seed, int):
                        dataset_fraction_seed = base_seed
                flat_cfg = FlatnessConfig(
                    sharpness_radius=cfg_eval.get("flat_eval_sharpness_radius", cfg_eval.get("flat_eval_rho", getattr(args, "flat_rho", 0.05))),
                    esh_num_samples=cfg_eval.get("flat_eval_esh_samples", cfg_eval.get("flat_eval_num_samples", getattr(args, "flat_samples", 10))),
                    esh_gaussian_std=cfg_eval.get("flat_eval_esh_gaussian_std", cfg_eval.get("flat_eval_gaussian_std", getattr(args, "flat_gaussian_std", None))),
                    loss_eval_max_batches=cfg_eval.get("flat_eval_loss_max_batches", cfg_eval.get("flat_eval_max_batches", getattr(args, "flat_max_batches", 1))),
                    hessian_power_iters=cfg_eval.get("flat_eval_hessian_power_iters", cfg_eval.get("flat_eval_power_iters", getattr(args, "flat_power_iters", 5))),
                    hessian_trace_samples=cfg_eval.get("flat_eval_hessian_trace_samples", cfg_eval.get("flat_eval_trace_samples", getattr(args, "flat_trace_samples", 5))),
                    first_order_grad_batches=cfg_eval.get("flat_eval_first_order_grad_batches", cfg_eval.get("flat_eval_grad_batches", getattr(args, "flat_grad_batches", 1))),
                    dataset_fraction=dataset_fraction,
                    dataset_fraction_seed=dataset_fraction_seed,
                )
                hvp_context = contextlib.nullcontext()
                if sdp_kernel is not None and torch.cuda.is_available():
                    hvp_context = sdp_kernel(
                        enable_flash=False,
                        enable_mem_efficient=False,
                        enable_math=True,
                    )

                with hvp_context:
                    flat_metrics = evaluate_flatness_metrics(
                        learner._network,
                        learner.train_loader,
                        device=learner._device,
                        config=flat_cfg,
                    )
                logging.info("Flatness (task %d): %s", task_idx, flat_metrics)
            except Exception as err:  # pylint: disable=broad-except
                logging.exception("Flatness eval failed at task %d: %s", task_idx, err)

        if feature_flat_eval:
            try:
                device_override = learner._device
                if isinstance(device_override, str):
                    device_override = torch.device(device_override)
                feature_cfg = FeatureFlatnessConfig(
                    max_batches=cfg_eval.get(
                        "feature_flat_max_batches", getattr(args, "feature_flat_max_batches", None)
                    ),
                    topk_eigen=cfg_eval.get(
                        "feature_flat_topk", getattr(args, "feature_flat_topk", 5)
                    ),
                    eps=cfg_eval.get("feature_flat_eps", getattr(args, "feature_flat_eps", 1e-12)),
                    rank_tol=cfg_eval.get(
                        "feature_flat_rank_tol", getattr(args, "feature_flat_rank_tol", 1e-6)
                    ),
                    save_matrix_path=feature_save_path,
                    save_prefix=f"task{task_idx}",
                    device_override=device_override,
                )
                feature_metrics = evaluate_feature_metrics(
                    learner._network,
                    learner.train_loader,
                    config=feature_cfg,
                )
                logging.info("Feature flatness (task %d): %s", task_idx, feature_metrics)
            except Exception as err:  # pylint: disable=broad-except
                logging.exception("Feature flatness eval failed at task %d: %s", task_idx, err)

    if head_eval and head_R is not None:
        local_T = len(task_indices)
        evaluated_rows = np.full((local_T, local_T), np.nan, dtype=float)
        for local_i, global_i in enumerate(task_indices):
            for local_j, global_j in enumerate(task_indices[: local_i + 1]):
                evaluated_rows[local_i, local_j] = head_R[global_i, global_j]

        metrics = compute_sequence_metrics(evaluated_rows)
        logging.info(
            "Head summary => ACA=%.2f | ABWT=%s | AFG=%s | FAA=%.2f",
            metrics["ACA"],
            f"{metrics['ABWT']:.2f}" if not np.isnan(metrics["ABWT"]) else "nan",
            f"{metrics['AFG']:.2f}" if not np.isnan(metrics["AFG"]) else "nan",
            metrics["FAA"],
        )

    if linear_probe_eval and probe_R is not None:
        probe_metrics = compute_sequence_metrics(probe_R)
        logging.info(
            "Linear probe summary => ACA=%.2f | ABWT=%s | AFG=%s | FAA=%.2f",
            probe_metrics["ACA"],
            f"{probe_metrics['ABWT']:.2f}" if not np.isnan(probe_metrics["ABWT"]) else "nan",
            f"{probe_metrics['AFG']:.2f}" if not np.isnan(probe_metrics["AFG"]) else "nan",
            probe_metrics["FAA"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate continual-learning checkpoints for LoRA models")
    parser.add_argument("--config", required=True, help="Path to the training config JSON used for the run.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing saved LoRA adapters / classifier heads. Defaults to logs/.../checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to evaluate (defaults to the first entry in the config).",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated task indices to evaluate, or 'all'.",
    )

    # Flatness options
    parser.add_argument("--flat-eval", action="store_true", help="Enable parameter-space flatness evaluation.")
    parser.add_argument("--flat-rho", type=float, default=0.05)
    parser.add_argument("--flat-samples", type=int, default=10)
    parser.add_argument("--flat-gaussian-std", type=float, default=None)
    parser.add_argument("--flat-max-batches", type=int, default=1)
    parser.add_argument("--flat-power-iters", type=int, default=5)
    parser.add_argument("--flat-trace-samples", type=int, default=5)
    parser.add_argument("--flat-grad-batches", type=int, default=1)

    # Feature flatness options
    parser.add_argument("--feature-flat-eval", action="store_true", help="Enable feature-space flatness evaluation.")
    parser.add_argument("--feature-flat-max-batches", type=int, default=None)
    parser.add_argument("--feature-flat-topk", type=int, default=5)
    parser.add_argument("--feature-flat-eps", type=float, default=1e-12)
    parser.add_argument("--feature-flat-rank-tol", type=float, default=1e-6)
    parser.add_argument(
        "--feature-flat-save-path",
        default=None,
        help="Optional directory to store per-task empirical feature matrices.",
    )

    parser.add_argument("--log-level", default="INFO", help="Logging verbosity level (e.g., INFO, DEBUG).")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    checkpoint_dir = args.checkpoint_dir or _default_checkpoint_dir(cfg)
    os.makedirs(checkpoint_dir, exist_ok=True)
    cfg["filepath"] = checkpoint_dir if checkpoint_dir.endswith(os.sep) else checkpoint_dir + os.sep
    cfg.setdefault("feature_flat_save_path", checkpoint_dir)

    run_evaluation(cfg, args)


if __name__ == "__main__":
    main()
