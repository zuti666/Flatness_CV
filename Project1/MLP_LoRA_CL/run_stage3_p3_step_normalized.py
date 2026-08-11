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

from run_stage3_p1_mature_gauge import (
    _angle_label,
    _dump,
    _resolved_lora_alpha,
    _resolved_subspace_seed,
    _train_epochs,
)
from small_cl.config import load_config, resolve_path, save_config
from small_cl.data import TaskData, build_task_datasets, prepare_dataset
from small_cl.diagnostics import (
    collect_diagnostic_batches,
    effective_tangent_basis,
    exact_directional_diagnostics,
    pathwise_diagnostics,
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
    tangent_projector_distance,
)
from small_cl.models import AdaptiveMLP
from small_cl.normalized_steps import (
    apply_factor_step_,
    make_factor_step_candidate,
    resolve_factor_candidate,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _profile_key(angle_pair: list[float]) -> str:
    return "-".join(_angle_label(value) for value in angle_pair)


def _load_target_profile(config: dict[str, Any], angle_pair: list[float]) -> list[float]:
    path = resolve_path(config["step_normalization"]["target_profile"], PROJECT_ROOT)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    key = _profile_key(angle_pair)
    try:
        values = payload["profiles"][key]["median_step_norms"]
    except KeyError as error:
        raise KeyError(f"No P3 target profile for {key} in {path}") from error
    profile = [float(value) for value in values]
    if not profile or min(profile) <= 0:
        raise ValueError(f"Invalid target profile for {key}")
    return profile


def _branch_specs(config: dict[str, Any]) -> list[tuple[str, float | None]]:
    specifications = [("raw", None)]
    specifications.extend(
        ("normalized", float(scale))
        for scale in config["step_normalization"]["pilot_scales"]
    )
    return specifications


def _mode_name(mode: str, scale: float | None) -> str:
    return "mode_raw" if mode == "raw" else f"mode_normalized_q{float(scale):g}"


def _method_name(method: str, rho: float | None) -> str:
    return method if method == "sgd" else f"{method}_rho{float(rho):g}"


def _method_specs(config: dict[str, Any]) -> list[tuple[str, float | None]]:
    output = []
    for method in config["mature_gauge"]["continuation_methods"]:
        key = str(method).lower()
        if key == "sgd":
            output.append((key, None))
        elif key == "sam":
            output.extend(
                (key, float(radius))
                for radius in config["mature_gauge"]["sam_radii"]
            )
        else:
            raise ValueError(f"P3 supports SGD and SAM, got {method}")
    return output


def _training_config(config: dict[str, Any], rho: float | None) -> dict[str, Any]:
    training = dict(config["training"])
    training["lr"] = float(config["mature_gauge"]["task_b_lr"])
    training["momentum"] = 0.0
    training["weight_decay"] = 0.0
    training["grad_clip"] = None
    if rho is not None:
        training["sam_rho"] = float(rho)
    return training


def _loss_change(
    model: AdaptiveMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    start_weight: torch.Tensor,
    delta_weight: torch.Tensor,
) -> float:
    with torch.no_grad():
        start = nn.functional.cross_entropy(
            model.forward_with_middle_weight(inputs, start_weight), targets
        )
        end = nn.functional.cross_entropy(
            model.forward_with_middle_weight(inputs, start_weight + delta_weight), targets
        )
    return float(end - start)


def _step_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    q = np.asarray([float(row["effective_step_norm"]) for row in rows], dtype=float)
    q_squared = float(np.square(q).sum())
    return {
        "num_steps": int(q.size),
        "cumulative_step_path_length": float(q.sum()),
        "quadratic_step_budget": q_squared,
        "maximum_step_norm": float(q.max()),
        "minimum_step_norm": float(q.min()),
        "effective_step_count": float(q.sum() ** 2 / max(q_squared, 1e-20)),
        "maximum_target_relative_error": float(
            max(float(row["target_relative_error"]) for row in rows)
        ),
        "maximum_alpha": float(max(float(row["alpha"]) for row in rows)),
        "minimum_raw_to_resolved_cosine": float(
            min(float(row["raw_to_resolved_cosine"]) for row in rows)
        ),
        "mean_sam_raw_correction_ratio": float(
            np.mean([float(row["sam_raw_correction_ratio"]) for row in rows])
        ),
        "mean_sam_resolved_correction_ratio": float(
            np.mean([float(row["sam_resolved_correction_ratio"]) for row in rows])
        ),
        "mean_sam_raw_update_angle": float(
            np.mean([1.0 - float(row["sam_raw_update_cosine"]) for row in rows])
        ),
        "mean_sam_resolved_update_angle": float(
            np.mean([1.0 - float(row["sam_resolved_update_cosine"]) for row in rows])
        ),
    }


def _run_continuation(
    source_model: AdaptiveMLP,
    *,
    method: str,
    rho: float | None,
    update_mode: str,
    target_scale: float | None,
    target_profile: list[float],
    transform_name: str,
    transform_audit: dict[str, Any],
    prefix: dict[str, Any],
    task_data: TaskData,
    progress_loaders: list,
    diagnostic_batches: list[list[tuple[torch.Tensor, torch.Tensor]]],
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> None:
    model = copy.deepcopy(source_model)
    training = _training_config(config, rho)
    loader = _loader(
        task_data.train[1],
        int(training["batch_size"]),
        True,
        int(config["seed"])
        + int(config["mature_gauge"]["continuation_loader_seed_offset"]),
        int(config["data"]["num_workers"]),
        device,
    )
    progress_interval = int(config["step_normalization"]["progress_interval_steps"])
    direction_interval = int(
        config["step_normalization"]["direction_diagnostic_interval_steps"]
    )
    maximum_alpha = float(config["step_normalization"]["maximum_alpha"])
    maximum_steps = int(
        config["step_normalization"].get("maximum_steps", len(target_profile))
    )
    expected_steps = min(len(target_profile), maximum_steps)
    start_weight = model.effective_weight().detach().clone()
    epoch_weights = [start_weight]
    step_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    global_step = 0

    def record_progress() -> None:
        old = evaluate(model, progress_loaders[0], device)
        current = evaluate(model, progress_loaders[1], device)
        summary = _step_summary(step_rows) if step_rows else {
            "cumulative_step_path_length": 0.0,
            "quadratic_step_budget": 0.0,
            "maximum_step_norm": 0.0,
            "effective_step_count": 0.0,
        }
        progress_rows.append(
            {
                "global_step": global_step,
                "old_loss": old["loss"],
                "old_accuracy": old["accuracy"],
                "current_loss": current["loss"],
                "current_accuracy": current["accuracy"],
                "old_loss_damage": old["loss"] - prefix["mature_old_loss"],
                "effective_drift": float(
                    (model.effective_weight() - start_weight).norm()
                ),
                "cumulative_step_path_length": summary[
                    "cumulative_step_path_length"
                ],
                "quadratic_step_budget": summary["quadratic_step_budget"],
                "maximum_step_norm": summary["maximum_step_norm"],
                "effective_step_count": summary["effective_step_count"],
            }
        )

    record_progress()
    continuation_epochs = int(config["mature_gauge"]["continuation_epochs"])
    for epoch in range(continuation_epochs):
        batch_rows = []
        for step, (inputs, targets) in enumerate(loader):
            if global_step >= expected_steps:
                break
            if global_step >= len(target_profile):
                raise RuntimeError("P3 target profile is shorter than the training run")
            inputs, targets = inputs.to(device), targets.to(device)
            before_a = model.lora_a.detach().clone()
            before_b = model.lora_b.detach().clone()
            before_weight = model.effective_weight().detach().clone()
            candidate = make_factor_step_candidate(
                model, inputs, targets, method, training
            )
            target = (
                float(target_scale) * float(target_profile[global_step])
                if update_mode == "normalized"
                else None
            )
            delta_a, delta_b, predicted_delta, audit = resolve_factor_candidate(
                model,
                candidate,
                update_mode,
                target,
                maximum_alpha=maximum_alpha,
            )
            apply_factor_step_(model, delta_a, delta_b)
            actual_delta = model.effective_weight().detach() - before_weight
            closure = float(
                (actual_delta - predicted_delta).norm()
                / actual_delta.norm().clamp_min(1e-20)
            )
            geometry = actual_factor_step_metrics(
                model, before_a, before_b, before_weight
            )
            row = {
                "epoch": epoch + 1,
                "step": step,
                "global_step": global_step,
                "clean_loss": candidate.clean_loss,
                "perturbed_loss": candidate.perturbed_loss,
                "candidate_closure_relative_error": closure,
                **audit,
                **geometry,
            }
            step_rows.append(row)
            batch_rows.append(row)

            if direction_interval > 0 and global_step % direction_interval == 0:
                for batch_id, (old_inputs, old_targets) in enumerate(
                    diagnostic_batches[0]
                ):
                    diagnostic = exact_directional_diagnostics(
                        model,
                        old_inputs,
                        old_targets,
                        before_weight,
                        before_weight + actual_delta,
                    )
                    q = max(float(diagnostic["delta_norm"]), 1e-20)
                    direction_rows.append(
                        {
                            "epoch": epoch + 1,
                            "step": step,
                            "global_step": global_step,
                            "diagnostic_batch": batch_id,
                            **diagnostic,
                            "unit_interference": float(
                                diagnostic["interference_I"] / q
                            ),
                            "unit_hessian_curvature": float(
                                diagnostic["directional_curvature"] / (q * q)
                            ),
                            "unit_quadratic_term": float(
                                diagnostic["directional_curvature_C"] / (q * q)
                            ),
                        }
                    )
            global_step += 1
            if progress_interval > 0 and global_step % progress_interval == 0:
                record_progress()
        epoch_weights.append(model.effective_weight().detach().clone())
        history.append(
            {
                "epoch": epoch + 1,
                "mean_clean_loss": float(
                    np.mean([float(row["clean_loss"]) for row in batch_rows])
                ),
                "mean_perturbed_loss": float(
                    np.nanmean([float(row["perturbed_loss"]) for row in batch_rows])
                )
                if method == "sam"
                else None,
                "steps": len(batch_rows),
            }
        )
        if global_step >= expected_steps:
            break
    if global_step != expected_steps:
        raise RuntimeError(
            f"P3 expected {expected_steps} steps but run used {global_step}"
        )
    if progress_rows[-1]["global_step"] != global_step:
        record_progress()

    final_weight = model.effective_weight().detach().clone()
    final_old = evaluate(model, progress_loaders[0], device)
    final_current = evaluate(model, progress_loaders[1], device)
    step_summary = _step_summary(step_rows)
    pathwise = pathwise_diagnostics(
        model,
        diagnostic_batches[0][0][0],
        diagnostic_batches[0][0][1],
        epoch_weights,
    )
    metrics = {
        "seed": int(config["seed"]),
        "angles": [float(value) for value in task_data.angles],
        "transformation": transform_name,
        "optimizer_method": method,
        "sam_rho": rho,
        "update_mode": update_mode,
        "target_scale": target_scale,
        "continuation_lr": float(training["lr"]),
        "task_a_old_loss": prefix["task_a_old_loss"],
        "mature_old_loss": prefix["mature_old_loss"],
        "mature_current_loss": prefix["mature_current_loss"],
        "final_old_loss": final_old["loss"],
        "final_old_accuracy": final_old["accuracy"],
        "final_current_loss": final_current["loss"],
        "final_current_accuracy": final_current["accuracy"],
        "continuation_old_loss_damage": final_old["loss"]
        - prefix["mature_old_loss"],
        "current_loss_reduction": prefix["mature_current_loss"]
        - final_current["loss"],
        "effective_drift": float((final_weight - start_weight).norm()),
        "effective_weight_sha256": _weight_sha256(final_weight),
        "pathwise": pathwise,
        "step_summary": step_summary,
        "transform_audit": transform_audit,
    }
    _dump(metrics, run_dir / "metrics.json")
    _dump(step_rows, run_dir / "step_geometry.json")
    _dump(direction_rows, run_dir / "direction_diagnostics.json")
    _dump(progress_rows, run_dir / "progress_history.json")
    _dump(history, run_dir / "training_history.json")
    save_config(config, run_dir / "config_resolved.yaml")
    torch.save(model.state_dict(), run_dir / "model_final.pt")


def _run_shared_state_counterfactuals(
    source_model: AdaptiveMLP,
    *,
    target_profile: list[float],
    task_data: TaskData,
    diagnostic_batches: list[list[tuple[torch.Tensor, torch.Tensor]]],
    config: dict[str, Any],
    device: torch.device,
    prefix_dir: Path,
) -> None:
    interval = int(config["step_normalization"]["counterfactual_interval_steps"])
    maximum_alpha = float(config["step_normalization"]["maximum_alpha"])
    maximum_steps = int(
        config["step_normalization"].get("maximum_steps", len(target_profile))
    )
    expected_steps = min(len(target_profile), maximum_steps)
    gauge_seed = int(config["seed"]) + int(config["mature_gauge"]["gauge_seed_offset"])
    transformations = list(config["mature_gauge"]["transformations"])
    for target_scale in config["step_normalization"]["pilot_scales"]:
        output_path = prefix_dir / f"counterfactual_q{float(target_scale):g}.json"
        if output_path.exists():
            continue
        reference = copy.deepcopy(source_model)
        loader = _loader(
            task_data.train[1],
            int(config["training"]["batch_size"]),
            True,
            int(config["seed"])
            + int(config["mature_gauge"]["continuation_loader_seed_offset"]),
            int(config["data"]["num_workers"]),
            device,
        )
        rows: list[dict[str, Any]] = []
        global_step = 0
        for epoch in range(int(config["mature_gauge"]["continuation_epochs"])):
            for step, (inputs, targets) in enumerate(loader):
                if global_step >= expected_steps:
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                target = float(target_scale) * float(target_profile[global_step])
                if interval > 0 and global_step % interval == 0:
                    reference_weight = reference.effective_weight().detach().clone()
                    for transform_name in transformations:
                        transformed = copy.deepcopy(reference)
                        matrix = make_gauge_matrix(
                            str(transform_name),
                            transformed.rank,
                            gauge_seed,
                            device=device,
                            dtype=transformed.lora_a.dtype,
                        )
                        apply_factor_gauge_(transformed, matrix)
                        weight_error = float(
                            (transformed.effective_weight() - reference_weight).norm()
                        )
                        for method, rho in _method_specs(config):
                            training = _training_config(config, rho)
                            candidate = make_factor_step_candidate(
                                transformed, inputs, targets, method, training
                            )
                            _, _, delta_weight, audit = resolve_factor_candidate(
                                transformed,
                                candidate,
                                "normalized",
                                target,
                                maximum_alpha=maximum_alpha,
                            )
                            for batch_id, (old_batch, new_batch) in enumerate(
                                zip(diagnostic_batches[0], diagnostic_batches[1])
                            ):
                                old_diagnostic = exact_directional_diagnostics(
                                    transformed,
                                    old_batch[0],
                                    old_batch[1],
                                    reference_weight,
                                    reference_weight + delta_weight,
                                )
                                q = max(float(old_diagnostic["delta_norm"]), 1e-20)
                                rows.append(
                                    {
                                        "seed": int(config["seed"]),
                                        "angles": [
                                            float(value) for value in task_data.angles
                                        ],
                                        "target_scale": float(target_scale),
                                        "epoch": epoch + 1,
                                        "step": step,
                                        "global_step": global_step,
                                        "diagnostic_batch": batch_id,
                                        "transformation": transform_name,
                                        "method": method,
                                        "gauge_weight_error": weight_error,
                                        "old_loss_change": old_diagnostic[
                                            "actual_loss_change"
                                        ],
                                        "new_loss_change": _loss_change(
                                            transformed,
                                            new_batch[0],
                                            new_batch[1],
                                            reference_weight,
                                            delta_weight,
                                        ),
                                        "interference_I": old_diagnostic[
                                            "interference_I"
                                        ],
                                        "curvature_C": old_diagnostic[
                                            "directional_curvature_C"
                                        ],
                                        "taylor_residual": old_diagnostic[
                                            "taylor_residual"
                                        ],
                                        "unit_interference": float(
                                            old_diagnostic["interference_I"] / q
                                        ),
                                        "unit_hessian_curvature": float(
                                            old_diagnostic["directional_curvature"]
                                            / (q * q)
                                        ),
                                        **audit,
                                    }
                                )
                reference_candidate = make_factor_step_candidate(
                    reference,
                    inputs,
                    targets,
                    "sgd",
                    _training_config(config, None),
                )
                delta_a, delta_b, _, _ = resolve_factor_candidate(
                    reference,
                    reference_candidate,
                    "normalized",
                    target,
                    maximum_alpha=maximum_alpha,
                )
                apply_factor_step_(reference, delta_a, delta_b)
                global_step += 1
            if global_step >= expected_steps:
                break
        if global_step != expected_steps:
            raise RuntimeError("Counterfactual reference did not consume target profile")
        _dump(rows, output_path)


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
    target_profile = _load_target_profile(config, angle_pair)

    output_root = resolve_path(config["output_root"], PROJECT_ROOT)
    angle_tag = "angles-" + "-".join(_angle_label(value) for value in angle_pair)
    prefix_dir = output_root / config["experiment_name"] / angle_tag / f"seed_{seed}"
    expected = (
        len(config["mature_gauge"]["transformations"])
        * len(_method_specs(config))
        * len(_branch_specs(config))
    )
    if skip_completed:
        completed = len(list(prefix_dir.glob("transform_*/mode_*/*/metrics.json")))
        counterfactuals = len(
            list(prefix_dir.glob("counterfactual_q*.json"))
        )
        if completed == expected and counterfactuals == len(
            config["step_normalization"]["pilot_scales"]
        ):
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
    progress_batch_size = int(
        config["step_normalization"].get(
            "progress_batch_size", config["training"]["batch_size"]
        )
    )
    progress_loaders = [
        _loader(
            dataset,
            progress_batch_size,
            False,
            int(seed) + 20_000 + task_id,
            int(config["data"]["num_workers"]),
            device,
        )
        for task_id, dataset in enumerate(task_data.test)
    ]
    task_a_old = evaluate(dense, progress_loaders[0], device)
    task_a_weight = dense.effective_weight().detach().clone()
    factor = _reparameterize_at_effective_weight(
        dense, config, "factor_lora", lora_alpha, subspace_seed
    )
    warmup_training = dict(config["training"])
    warmup_training["lr"] = float(config["mature_gauge"]["task_b_lr"])
    warmup_training["momentum"] = 0.0
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
    mature_old = evaluate(factor, progress_loaders[0], device)
    mature_current = evaluate(factor, progress_loaders[1], device)
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
        "target_profile_steps": len(target_profile),
        "target_profile_path": float(sum(target_profile)),
        **prefix,
        "task_a_history": task_a_history,
        "warmup_history": warmup_history,
    }
    _dump(prefix_audit, prefix_dir / "shared_prefix.json")
    save_config(config, prefix_dir / "config_resolved.yaml")
    torch.save(factor.state_dict(), prefix_dir / "mature_prefix.pt")

    _run_shared_state_counterfactuals(
        factor,
        target_profile=target_profile,
        task_data=task_data,
        diagnostic_batches=diagnostic_batches,
        config=config,
        device=device,
        prefix_dir=prefix_dir,
    )

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
        transform_audit = {
            "gauge_matrix": matrix.detach().cpu().tolist(),
            "gauge_matrix_condition": float(torch.linalg.cond(matrix)),
            "effective_weight_error": float((transformed_weight - mature_weight).norm()),
            "max_logit_error": float((transformed_logits - reference_logits).abs().max()),
            "tangent_projector_distance": tangent_projector_distance(
                reference_basis, transformed_basis
            ),
        }
        if (
            transform_audit["effective_weight_error"] > 1e-5
            or transform_audit["max_logit_error"] > 1e-5
            or transform_audit["tangent_projector_distance"] > 1e-3
        ):
            raise RuntimeError(f"P3 gauge audit failed: {transform_audit}")
        transform_dir = prefix_dir / f"transform_{transform_name}"
        _dump(transform_audit, transform_dir / "transform_audit.json")
        for update_mode, target_scale in _branch_specs(config):
            mode_dir = transform_dir / _mode_name(update_mode, target_scale)
            for method, rho in _method_specs(config):
                branch_dir = mode_dir / _method_name(method, rho)
                if skip_completed and (branch_dir / "metrics.json").exists():
                    continue
                start_time = time.time()
                _run_continuation(
                    transformed,
                    method=method,
                    rho=rho,
                    update_mode=update_mode,
                    target_scale=target_scale,
                    target_profile=target_profile,
                    transform_name=str(transform_name),
                    transform_audit=transform_audit,
                    prefix=prefix,
                    task_data=task_data,
                    progress_loaders=progress_loaders,
                    diagnostic_batches=diagnostic_batches,
                    config=config,
                    device=device,
                    run_dir=branch_dir,
                )
                print(
                    f"completed angles={angle_pair} seed={seed} transform={transform_name} "
                    f"mode={_mode_name(update_mode, target_scale)} "
                    f"method={_method_name(method, rho)} "
                    f"seconds={time.time() - start_time:.1f}",
                    flush=True,
                )
    return prefix_dir


def _jobs(config: dict[str, Any]) -> list[tuple[list[float], int]]:
    return [
        ([float(value) for value in pair], int(seed))
        for pair in config["grid"]["angle_pairs"]
        for seed in config["grid"]["seeds"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 exact effective-step normalization")
    parser.add_argument("--config", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument(
        "--transformation",
        action="append",
        help="Run only the named gauge transformation; repeat to select several.",
    )
    parser.add_argument("--prepare-data-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.transformation:
        configured = set(config["mature_gauge"]["transformations"])
        requested = list(dict.fromkeys(args.transformation))
        unknown = sorted(set(requested) - configured)
        if unknown:
            parser.error(
                f"Unknown transformation(s) {unknown}; configured choices are "
                f"{sorted(configured)}"
            )
        config["mature_gauge"]["transformations"] = requested
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
