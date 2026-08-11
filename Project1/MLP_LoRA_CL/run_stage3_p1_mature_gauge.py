from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from small_cl.config import load_config, resolve_path, save_config
from small_cl.data import TaskData, build_task_datasets, prepare_dataset
from small_cl.diagnostics import (
    collect_diagnostic_batches,
    effective_tangent_basis,
    factor_parameterization_diagnostics,
    pathwise_diagnostics,
    weight_gradient,
)
from small_cl.experiment import (
    _loader,
    _make_adaptive_model,
    _reparameterize_at_effective_weight,
    _weight_sha256,
    evaluate,
    prepare_base_model,
    resolve_device,
    set_seed,
)
from small_cl.gauge import (
    actual_factor_step_metrics,
    apply_factor_gauge_,
    make_gauge_matrix,
    pullback_gradient_metrics,
    tangent_projector_distance,
)
from small_cl.models import AdaptiveMLP
from small_cl.optimizers import make_sgd, train_batch


PROJECT_ROOT = Path(__file__).resolve().parent


def _angle_label(value: float) -> str:
    return ("m" if value < 0 else "p") + f"{abs(float(value)):g}"


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, tuple):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_finite_or_none(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def _mean_or_none(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _aggregate_step_geometry(rows: list[dict[str, float]]) -> dict[str, float | None]:
    keys = rows[0].keys() if rows else []
    result: dict[str, float | None] = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        result[f"mean_{key}"] = _mean_or_none(values)
        result[f"max_{key}"] = max(values) if values else None
    return result


def _resolved_lora_alpha(config: dict[str, Any]) -> float:
    value = config["model"]["lora_alpha"]
    return float(config["model"]["rank"] if str(value).lower() == "rank" else value)


def _resolved_subspace_seed(config: dict[str, Any]) -> int:
    value = config["model"].get("subspace_seed", "run_seed")
    return int(config["seed"] if str(value).lower() == "run_seed" else value)


def _train_epochs(
    model: AdaptiveMLP,
    dataset,
    *,
    epochs: int,
    method: str,
    training_config: dict[str, Any],
    loader_seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    optimizer = make_sgd(model.trainable_parameters(), training_config)
    loader = _loader(
        dataset,
        int(training_config["batch_size"]),
        True,
        int(loader_seed),
        int(config["data"]["num_workers"]),
        device,
    )
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        rows = []
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            rows.append(
                train_batch(
                    model, optimizer, criterion, inputs, targets, method, training_config
                )
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean([row["loss"] for row in rows])),
                "perturbed_loss": _mean_or_none([row["perturbed_loss"] for row in rows]),
                "curvature_correction_norm": _mean_or_none(
                    [row["curvature_correction_norm"] for row in rows]
                ),
                "effective_perturbation_norm": _mean_or_none(
                    [row["effective_perturbation_norm"] for row in rows]
                ),
            }
        )
    return history


def _branch_name(method: str, rho: float | None) -> str:
    return method if method == "sgd" else f"{method}_rho{float(rho):g}"


def _branch_specs(config: dict[str, Any]) -> list[tuple[str, float | None]]:
    p1 = config["mature_gauge"]
    result: list[tuple[str, float | None]] = []
    for method in p1["continuation_methods"]:
        key = str(method).lower()
        if key == "sgd":
            result.append((key, None))
        elif key == "sam":
            result.extend((key, float(rho)) for rho in p1["sam_radii"])
        else:
            raise ValueError(f"P1 currently supports SGD and SAM, got {method}")
    return result


def _run_continuation(
    source_model: AdaptiveMLP,
    *,
    method: str,
    rho: float | None,
    transform_name: str,
    transform_audit: dict[str, Any],
    prefix: dict[str, Any],
    task_data: TaskData,
    test_loaders: list,
    diagnostic_batches: list[list[tuple[torch.Tensor, torch.Tensor]]],
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> None:
    model = copy.deepcopy(source_model)
    training = dict(config["training"])
    transformation_lrs = config["mature_gauge"].get("transformation_lrs", {})
    training["lr"] = float(
        transformation_lrs.get(
            transform_name, config["mature_gauge"]["task_b_lr"]
        )
    )
    training["momentum"] = float(config["mature_gauge"].get("task_b_momentum", 0.0))
    if rho is not None:
        training["sam_rho"] = float(rho)
    optimizer = make_sgd(model.trainable_parameters(), training)
    loader = _loader(
        task_data.train[1],
        int(training["batch_size"]),
        True,
        int(config["seed"]) + int(config["mature_gauge"]["continuation_loader_seed_offset"]),
        int(config["data"]["num_workers"]),
        device,
    )
    criterion = nn.CrossEntropyLoss()
    start_weight = model.effective_weight().detach().clone()
    epoch_weights = [start_weight]
    step_path_length = 0.0
    step_rows: list[dict[str, float]] = []
    history: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    dynamic_geometry: list[dict[str, Any]] = []
    checkpoint_interval = int(
        config["mature_gauge"].get("pathwise_checkpoint_interval_steps", 0)
    )
    fine_weights = [start_weight] if checkpoint_interval > 0 else []
    sampled_step_alignment: list[dict[str, Any]] = []
    global_step = 0

    def record_dynamic_geometry(epoch: int) -> None:
        if not bool(config["mature_gauge"].get("dynamic_diagnostics", False)):
            return
        weight = model.effective_weight().detach().clone()
        factor_diagnostics = factor_parameterization_diagnostics(
            model,
            diagnostic_batches[1][0],
            weight,
            model.lora_a.detach().clone(),
            model.lora_b.detach().clone(),
            float(config["mature_gauge"]["diagnostic_radius"]),
        )
        for batch_id, (old_batch, new_batch) in enumerate(
            zip(diagnostic_batches[0], diagnostic_batches[1])
        ):
            old_gradient = weight_gradient(model, old_batch[0], old_batch[1], weight)
            new_gradient = weight_gradient(model, new_batch[0], new_batch[1], weight)
            dynamic_geometry.append(
                {
                    "epoch": epoch,
                    "diagnostic_batch": batch_id,
                    "factor_a_norm": float(model.lora_a.norm()),
                    "factor_b_norm": float(model.lora_b.norm()),
                    **pullback_gradient_metrics(model, old_gradient, new_gradient),
                    **factor_diagnostics,
                }
            )

    def record_trajectory(epoch: int) -> None:
        old = evaluate(model, test_loaders[0], device)
        current = evaluate(model, test_loaders[1], device)
        trajectory.append(
            {
                "epoch": epoch,
                "old_loss": old["loss"],
                "old_accuracy": old["accuracy"],
                "current_loss": current["loss"],
                "current_accuracy": current["accuracy"],
                "continuation_old_loss_damage": old["loss"] - prefix["mature_old_loss"],
                "old_loss_damage_from_task_a": old["loss"] - prefix["task_a_old_loss"],
                "effective_drift": float((model.effective_weight() - start_weight).norm()),
                "cumulative_step_path_length": step_path_length,
            }
        )

    record_trajectory(0)
    record_dynamic_geometry(0)
    continuation_epochs = int(config["mature_gauge"]["continuation_epochs"])
    for epoch in range(continuation_epochs):
        model.train()
        batch_rows = []
        for step, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            before_a = model.lora_a.detach().clone()
            before_b = model.lora_b.detach().clone()
            before_weight = model.effective_weight().detach().clone()
            sample_step = checkpoint_interval > 0 and global_step % checkpoint_interval == 0
            if sample_step:
                old_gradient_for_step = weight_gradient(
                    model,
                    diagnostic_batches[0][0][0],
                    diagnostic_batches[0][0][1],
                    before_weight,
                )
                new_gradient_for_step = weight_gradient(
                    model, inputs, targets, before_weight
                )
                clean_pullback = pullback_gradient_metrics(
                    model, old_gradient_for_step, new_gradient_for_step
                )
            batch_result = train_batch(
                model, optimizer, criterion, inputs, targets, method, training
            )
            geometry = actual_factor_step_metrics(model, before_a, before_b, before_weight)
            if sample_step:
                delta_a = model.lora_a.detach() - before_a
                delta_b = model.lora_b.detach() - before_b
                bilinear_update = model.lora_scale * (delta_b @ delta_a)
                actual_update = model.effective_weight().detach() - before_weight
                actual_interference = float(
                    (old_gradient_for_step * actual_update).sum()
                )
                clean_linear_interference = float(training["lr"]) * float(
                    clean_pullback["predicted_sgd_old_interference"]
                )
                bilinear_interference = float(
                    (old_gradient_for_step * bilinear_update).sum()
                )
                clean_prediction = clean_linear_interference + bilinear_interference
                sampled_step_alignment.append(
                    {
                        "epoch": epoch + 1,
                        "step": step,
                        "global_step": global_step,
                        "actual_step_interference": actual_interference,
                        "clean_pullback_linear_prediction": clean_linear_interference,
                        "actual_bilinear_interference": bilinear_interference,
                        "clean_plus_bilinear_prediction": clean_prediction,
                        "clean_prediction_error": actual_interference - clean_prediction,
                        "clean_prediction_relative_error": abs(
                            actual_interference - clean_prediction
                        )
                        / max(abs(actual_interference), 1e-20),
                        "curvature_correction_norm": batch_result[
                            "curvature_correction_norm"
                        ],
                        **clean_pullback,
                    }
                )
            step_path_length += geometry["effective_step_norm"]
            geometry.update({"epoch": float(epoch + 1), "step": float(step)})
            step_rows.append(geometry)
            batch_rows.append(batch_result)
            global_step += 1
            if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                fine_weights.append(model.effective_weight().detach().clone())
        epoch_weights.append(model.effective_weight().detach().clone())
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean([row["loss"] for row in batch_rows])),
                "perturbed_loss": _mean_or_none(
                    [row["perturbed_loss"] for row in batch_rows]
                ),
                "curvature_correction_norm": _mean_or_none(
                    [row["curvature_correction_norm"] for row in batch_rows]
                ),
                "effective_perturbation_norm": _mean_or_none(
                    [row["effective_perturbation_norm"] for row in batch_rows]
                ),
            }
        )
        record_trajectory(epoch + 1)
        record_dynamic_geometry(epoch + 1)

    final_old = evaluate(model, test_loaders[0], device)
    final_current = evaluate(model, test_loaders[1], device)
    final_weight = model.effective_weight().detach().clone()
    if checkpoint_interval > 0 and not torch.equal(fine_weights[-1], final_weight):
        fine_weights.append(final_weight)
    pathwise = pathwise_diagnostics(
        model,
        diagnostic_batches[0][0][0],
        diagnostic_batches[0][0][1],
        epoch_weights,
    )
    pathwise_replicates: list[dict[str, Any]] = []
    if bool(config["mature_gauge"].get("dynamic_diagnostics", False)):
        pathwise_replicates.append({"diagnostic_batch": 0, "pathwise": pathwise})
        pathwise_replicates.extend(
            {
                "diagnostic_batch": batch_id,
                "pathwise": pathwise_diagnostics(model, inputs, targets, epoch_weights),
            }
            for batch_id, (inputs, targets) in enumerate(
                diagnostic_batches[0][1:], start=1
            )
        )
    metrics = {
        "seed": int(config["seed"]),
        "angles": [float(value) for value in task_data.angles],
        "transformation": transform_name,
        "optimizer_method": method,
        "sam_rho": rho,
        "continuation_lr": float(training["lr"]),
        "warmup_epochs": int(config["mature_gauge"]["warmup_epochs"]),
        "continuation_epochs": continuation_epochs,
        "task_a_old_loss": prefix["task_a_old_loss"],
        "task_a_old_accuracy": prefix["task_a_old_accuracy"],
        "mature_old_loss": prefix["mature_old_loss"],
        "mature_old_accuracy": prefix["mature_old_accuracy"],
        "mature_current_loss": prefix["mature_current_loss"],
        "mature_current_accuracy": prefix["mature_current_accuracy"],
        "final_old_loss": final_old["loss"],
        "final_old_accuracy": final_old["accuracy"],
        "final_current_loss": final_current["loss"],
        "final_current_accuracy": final_current["accuracy"],
        "continuation_old_loss_damage": final_old["loss"] - prefix["mature_old_loss"],
        "old_loss_damage_from_task_a": final_old["loss"] - prefix["task_a_old_loss"],
        "current_loss_reduction": prefix["mature_current_loss"] - final_current["loss"],
        "effective_drift": float((final_weight - start_weight).norm()),
        "cumulative_step_path_length": step_path_length,
        "effective_weight_sha256": _weight_sha256(final_weight),
        "pathwise": pathwise,
        "step_geometry": _aggregate_step_geometry(step_rows),
        "transform_audit": transform_audit,
        "start_pullback": transform_audit["start_pullback"],
        "start_factor_diagnostics": transform_audit["start_factor_diagnostics"],
    }
    _dump(metrics, run_dir / "metrics.json")
    _dump(history, run_dir / "training_history.json")
    _dump(trajectory, run_dir / "trajectory_history.json")
    _dump(step_rows, run_dir / "step_geometry.json")
    if dynamic_geometry:
        _dump(dynamic_geometry, run_dir / "dynamic_geometry.json")
        _dump(pathwise_replicates, run_dir / "pathwise_replicates.json")
    if checkpoint_interval > 0:
        fine_pathwise = pathwise_diagnostics(
            model,
            diagnostic_batches[0][0][0],
            diagnostic_batches[0][0][1],
            fine_weights,
        )
        _dump(fine_pathwise, run_dir / "fine_pathwise.json")
        _dump(sampled_step_alignment, run_dir / "sampled_step_alignment.json")
    save_config(config, run_dir / "config_resolved.yaml")
    torch.save(model.state_dict(), run_dir / "model_final.pt")


def run_prefix_job(
    base_config: dict[str, Any],
    angle_pair: list[float],
    seed: int,
    *,
    skip_completed: bool,
) -> Path:
    config = copy.deepcopy(base_config)
    config["seed"] = int(seed)
    config["data"]["angles"] = [float(value) for value in angle_pair]
    device = resolve_device(str(config["device"]))
    set_seed(int(seed), bool(config["deterministic"]))
    base = prepare_base_model(config, PROJECT_ROOT, device)
    task_data = build_task_datasets(config, PROJECT_ROOT, base)
    set_seed(int(seed), bool(config["deterministic"]))

    output_root = resolve_path(config["output_root"], PROJECT_ROOT)
    angle_tag = "angles-" + "-".join(_angle_label(value) for value in angle_pair)
    prefix_dir = output_root / config["experiment_name"] / angle_tag / f"seed_{seed}"
    expected = len(config["mature_gauge"]["transformations"]) * len(_branch_specs(config))
    if skip_completed:
        completed = len(list(prefix_dir.glob("transform_*/**/metrics.json")))
        if completed == expected:
            print(f"skip completed prefix: {prefix_dir}", flush=True)
            return prefix_dir
    prefix_dir.mkdir(parents=True, exist_ok=True)

    lora_alpha = _resolved_lora_alpha(config)
    subspace_seed = _resolved_subspace_seed(config)
    dense = _make_adaptive_model(
        base, config["model"], "dense", lora_alpha, subspace_seed
    ).to(device)
    task_a_training = dict(config["training"])
    task_a_training["lr"] = float(config["mature_gauge"]["task_a_lr"])
    task_a_training["momentum"] = float(config["mature_gauge"]["task_a_momentum"])
    task_a_history = _train_epochs(
        dense,
        task_data.train[0],
        epochs=int(config["mature_gauge"]["task_a_epochs"]),
        method="sgd",
        training_config=task_a_training,
        loader_seed=int(seed) + int(config["mature_gauge"]["task_a_loader_seed_offset"]),
        config=config,
        device=device,
    )

    test_loaders = [
        _loader(
            dataset,
            int(config["training"]["batch_size"]),
            False,
            int(seed) + 20_000 + task_id,
            int(config["data"]["num_workers"]),
            device,
        )
        for task_id, dataset in enumerate(task_data.test)
    ]
    task_a_old = evaluate(dense, test_loaders[0], device)
    task_a_weight = dense.effective_weight().detach().clone()
    factor = _reparameterize_at_effective_weight(
        dense, config, "factor_lora", lora_alpha, subspace_seed
    )
    warmup_training = dict(config["training"])
    warmup_training["lr"] = float(config["mature_gauge"]["task_b_lr"])
    warmup_training["momentum"] = float(config["mature_gauge"].get("task_b_momentum", 0.0))
    warmup_history = _train_epochs(
        factor,
        task_data.train[1],
        epochs=int(config["mature_gauge"]["warmup_epochs"]),
        method="sgd",
        training_config=warmup_training,
        loader_seed=int(seed) + int(config["mature_gauge"]["warmup_loader_seed_offset"]),
        config=config,
        device=device,
    )
    mature_weight = factor.effective_weight().detach().clone()
    if float(factor.lora_a.norm()) <= 1e-10 or float(factor.lora_b.norm()) <= 1e-10:
        raise RuntimeError("Warmup did not produce a mature nonzero factor state")
    mature_old = evaluate(factor, test_loaders[0], device)
    mature_current = evaluate(factor, test_loaders[1], device)
    diagnostic_batches = [
        collect_diagnostic_batches(
            dataset,
            int(config["diagnostics"]["max_samples_per_task"]),
            int(config["diagnostics"].get("num_nonoverlap_batches", 1)),
            device,
        )
        for dataset in task_data.test
    ]
    with torch.no_grad():
        reference_logits = factor(diagnostic_batches[0][0][0][:32]).detach().clone()
    reference_basis = effective_tangent_basis(factor)
    old_gradient = weight_gradient(
        factor, diagnostic_batches[0][0][0], diagnostic_batches[0][0][1], mature_weight
    )
    new_gradient = weight_gradient(
        factor, diagnostic_batches[1][0][0], diagnostic_batches[1][0][1], mature_weight
    )
    prefix = {
        "task_a_old_loss": task_a_old["loss"],
        "task_a_old_accuracy": task_a_old["accuracy"],
        "mature_old_loss": mature_old["loss"],
        "mature_old_accuracy": mature_old["accuracy"],
        "mature_current_loss": mature_current["loss"],
        "mature_current_accuracy": mature_current["accuracy"],
    }
    prefix_audit = {
        "seed": seed,
        "angles": task_data.angles,
        "task_a_effective_weight_sha256": _weight_sha256(task_a_weight),
        "mature_effective_weight_sha256": _weight_sha256(mature_weight),
        "task_a_to_mature_drift": float((mature_weight - task_a_weight).norm()),
        "mature_factor_a_norm": float(factor.lora_a.norm()),
        "mature_factor_b_norm": float(factor.lora_b.norm()),
        "mature_factor_product_rank": int(torch.linalg.matrix_rank(factor.lora_b @ factor.lora_a)),
        **prefix,
        "task_a_history": task_a_history,
        "warmup_history": warmup_history,
    }
    _dump(prefix_audit, prefix_dir / "shared_prefix.json")
    save_config(config, prefix_dir / "config_resolved.yaml")
    torch.save(factor.state_dict(), prefix_dir / "mature_prefix.pt")

    gauge_seed = int(seed) + int(config["mature_gauge"]["gauge_seed_offset"])
    for transform_name in config["mature_gauge"]["transformations"]:
        transformed = copy.deepcopy(factor)
        matrix = make_gauge_matrix(
            str(transform_name),
            transformed.rank,
            gauge_seed,
            device=device,
            dtype=transformed.lora_a.dtype,
        )
        apply_factor_gauge_(transformed, matrix)
        transformed_weight = transformed.effective_weight().detach().clone()
        transformed_basis = effective_tangent_basis(transformed)
        with torch.no_grad():
            transformed_logits = transformed(diagnostic_batches[0][0][0][:32])
        product_error = float((transformed_weight - mature_weight).norm())
        logits_error = float((transformed_logits - reference_logits).abs().max())
        projector_error = tangent_projector_distance(reference_basis, transformed_basis)
        if product_error > 1e-5 or logits_error > 1e-5 or projector_error > 1e-3:
            raise RuntimeError(
                f"Gauge audit failed for {transform_name}: weight={product_error}, "
                f"logits={logits_error}, projector={projector_error}"
            )
        start_pullback = pullback_gradient_metrics(
            transformed, old_gradient, new_gradient
        )
        start_factor = factor_parameterization_diagnostics(
            transformed,
            diagnostic_batches[1][0],
            transformed_weight,
            transformed.lora_a.detach().clone(),
            transformed.lora_b.detach().clone(),
            float(config["mature_gauge"]["diagnostic_radius"]),
        )
        transform_audit = {
            "gauge_matrix": matrix.detach().cpu().tolist(),
            "gauge_matrix_condition": float(torch.linalg.cond(matrix)),
            "effective_weight_error": product_error,
            "max_logit_error": logits_error,
            "tangent_projector_distance": projector_error,
            "tangent_dimension": int(transformed_basis.shape[1]),
            "start_factor_a_norm": float(transformed.lora_a.norm()),
            "start_factor_b_norm": float(transformed.lora_b.norm()),
            "start_pullback": start_pullback,
            "start_factor_diagnostics": start_factor,
        }
        transform_dir = prefix_dir / f"transform_{transform_name}"
        _dump(transform_audit, transform_dir / "transform_audit.json")
        for method, rho in _branch_specs(config):
            branch_dir = transform_dir / _branch_name(method, rho)
            metrics_path = branch_dir / "metrics.json"
            if skip_completed and metrics_path.exists():
                continue
            start_time = time.time()
            _run_continuation(
                transformed,
                method=method,
                rho=rho,
                transform_name=str(transform_name),
                transform_audit=transform_audit,
                prefix=prefix,
                task_data=task_data,
                test_loaders=test_loaders,
                diagnostic_batches=diagnostic_batches,
                config=config,
                device=device,
                run_dir=branch_dir,
            )
            print(
                f"completed angles={angle_pair} seed={seed} transform={transform_name} "
                f"branch={_branch_name(method, rho)} seconds={time.time() - start_time:.1f}",
                flush=True,
            )
    return prefix_dir


def _jobs(config: dict[str, Any]) -> list[tuple[list[float], int]]:
    grid = config["grid"]
    return [
        ([float(value) for value in pair], int(seed))
        for pair in grid["angle_pairs"]
        for seed in grid["seeds"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3 P1 mature-factor gauge experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--prepare-data-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.prepare_data_only:
        print(json.dumps(prepare_dataset(config, PROJECT_ROOT), indent=2))
        return
    jobs = _jobs(config)[args.offset :]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    for angle_pair, seed in jobs:
        run_prefix_job(config, angle_pair, seed, skip_completed=args.skip_completed)


if __name__ == "__main__":
    main()
