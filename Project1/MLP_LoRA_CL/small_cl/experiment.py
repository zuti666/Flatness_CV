from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import resolve_path, save_config
from .data import TaskData, build_task_datasets, build_unrotated_base_datasets
from .diagnostics import (
    collect_diagnostic_batch,
    effective_tangent_basis,
    exact_directional_diagnostics,
    factor_parameterization_diagnostics,
    ggn_safe_route_diagnostics,
    pathwise_diagnostics,
    prospective_reachable_direction_diagnostics,
    reachable_coverage,
    subspace_overlap,
)
from .metrics import continual_metrics
from .models import AdaptiveMLP, FullMLP
from .optimizers import make_sgd, train_batch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    count = 0
    loss_sum = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss_sum += float(criterion(logits, targets))
        correct += int((logits.argmax(dim=1) == targets).sum())
        count += targets.numel()
    return {"accuracy": correct / count, "loss": loss_sum / count}


def _build_full_mlp(config: dict) -> FullMLP:
    model_config = config["model"]
    return FullMLP(
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_classes=int(model_config["num_classes"]),
        activation=model_config["activation"],
    )


def _make_adaptive_model(
    base: FullMLP,
    model_config: dict,
    parameterization: str,
    lora_alpha: float,
    subspace_seed: int,
) -> AdaptiveMLP:
    return AdaptiveMLP(
        base,
        parameterization=parameterization,
        rank=int(model_config["rank"]),
        lora_alpha=lora_alpha,
        factor_gauge_scale=float(model_config.get("factor_gauge_scale", 1.0)),
        subspace_seed=subspace_seed,
        subspace_dim=model_config.get("subspace_dim"),
    )


def _reparameterize_at_effective_weight(
    model: AdaptiveMLP,
    config: dict,
    parameterization: str,
    lora_alpha: float,
    subspace_seed: int,
) -> AdaptiveMLP:
    """Create a zero-update parameterization at exactly the model's current W."""
    device = model.base_weight.device
    rebased = _build_full_mlp(config).to(device)
    with torch.no_grad():
        rebased.fc1.weight.copy_(model.fc1.weight)
        rebased.fc1.bias.copy_(model.fc1.bias)
        rebased.middle.weight.copy_(model.effective_weight())
        rebased.middle.bias.copy_(model.middle_bias)
        rebased.fc3.weight.copy_(model.fc3.weight)
        rebased.fc3.bias.copy_(model.fc3.bias)
    return _make_adaptive_model(
        rebased, model_config=config["model"], parameterization=parameterization,
        lora_alpha=lora_alpha, subspace_seed=subspace_seed,
    ).to(device)


def _weight_sha256(weight: torch.Tensor) -> str:
    raw = weight.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def prepare_base_model(config: dict, project_root: Path, device: torch.device) -> FullMLP:
    """Load or train the common base. Every compared run must use this checkpoint."""
    base_config = config["base"]
    checkpoint = resolve_path(base_config["checkpoint"], project_root)
    set_seed(int(base_config["seed"]), bool(config["deterministic"]))
    model = _build_full_mlp(config).to(device)
    if base_config["mode"] == "random":
        return model
    if base_config["mode"] != "pretrained_mnist":
        raise ValueError("base.mode must be 'random' or 'pretrained_mnist'")
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        expected_metadata = {
            "activation": config["model"]["activation"],
            "hidden_dim": config["model"]["hidden_dim"],
            "dataset": config["data"]["name"],
            "pretrain_angle": float(base_config.get("pretrain_angle", 0.0)),
        }
        for key, expected in expected_metadata.items():
            if key in payload and payload[key] != expected:
                raise ValueError(
                    f"Base checkpoint {checkpoint} has {key}={payload[key]!r}, expected {expected!r}"
                )
        state_dict = payload["state_dict"] if "state_dict" in payload else payload
        model.load_state_dict(state_dict)
        return model

    train_dataset, test_dataset = build_unrotated_base_datasets(config, project_root)
    train_loader = _loader(
        train_dataset,
        int(base_config["batch_size"]),
        True,
        int(base_config["seed"]),
        int(config["data"]["num_workers"]),
        device,
    )
    test_loader = _loader(
        test_dataset,
        int(base_config["batch_size"]),
        False,
        int(base_config["seed"]),
        int(config["data"]["num_workers"]),
        device,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(base_config["lr"]),
        momentum=float(base_config["momentum"]),
        weight_decay=float(base_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(int(base_config["epochs"])):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        result = evaluate(model, test_loader, device)
        print(f"base epoch={epoch + 1} accuracy={result['accuracy']:.4f}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "activation": config["model"]["activation"],
            "hidden_dim": config["model"]["hidden_dim"],
            "dataset": config["data"]["name"],
            "pretrain_angle": float(base_config.get("pretrain_angle", 0.0)),
            "seed": base_config["seed"],
        },
        checkpoint,
    )
    return model


def _run_name(config: dict) -> str:
    model = config["model"]
    training = config["training"]
    data = config["data"]
    rank = "full" if model["parameterization"] == "dense" else f"r{model['rank']}"
    if model["parameterization"] in {"factor_lora", "balanced_lora", "lora"}:
        gauge = float(model.get("factor_gauge_scale", 1.0))
        if gauge != 1.0:
            rank += f"_gauge{gauge:g}"
    if data["name"].lower() == "teacher_student":
        teacher = data["teacher"]
        angle_tag = f"targetr{teacher['target_rank']}_angle{float(teacher['principal_angle_degrees']):g}"
    else:
        if data.get("angles") is not None and data.get("include_angles_in_run_name", False):
            def angle_label(value: float) -> str:
                value = float(value)
                prefix = "m" if value < 0 else "p"
                return f"{prefix}{abs(value):g}"
            angle_tag = "angles-" + "-".join(angle_label(value) for value in data["angles"])
        else:
            angle_tag = "angles" if data.get("angles") is not None else f"span{float(data['rotation_span']):g}"
    schedule = training.get("optimizer_schedule")
    optimizer = str(training["optimizer"]).lower()
    if schedule:
        methods = [str(method).lower() for method in schedule]
        optimizer_tag = f"schedule-{'-'.join(methods)}_lr{float(training['lr']):g}"
    else:
        methods = [optimizer]
        optimizer_tag = f"{optimizer}_lr{float(training['lr']):g}"
    if "sam" in methods:
        sam_rho = (training.get("parameterization_sam_rhos") or {}).get(
            model["parameterization"], training["sam_rho"]
        )
        optimizer_tag += f"_rho{float(sam_rho):g}"
    task_b_lr = (training.get("parameterization_lrs") or {}).get(
        model["parameterization"], training.get("task_b_lr")
    )
    if task_b_lr is not None:
        optimizer_tag += f"_taskblr{float(task_b_lr):g}"
    if set(methods).intersection({"gam_fd", "gam_exact"}):
        optimizer_tag += (
            f"_radius{float(training['gam_radius']):g}"
            f"_weight{float(training['gam_weight']):g}"
        )
    return (
        f"{config['experiment_name']}/{data['name']}_t{data['num_tasks']}_{angle_tag}/"
        f"{model['parameterization']}_{rank}/{model['lifecycle']}/{optimizer_tag}/"
        f"seed_{config['seed']}"
    )


def _resolve_optimizer_schedule(train_config: dict, task_count: int) -> list[str]:
    supplied = train_config.get("optimizer_schedule")
    if supplied is None:
        methods = [str(train_config["optimizer"]).lower()] * task_count
    else:
        if not isinstance(supplied, list) or len(supplied) != task_count:
            raise ValueError(
                "training.optimizer_schedule must be a list with one method per task"
            )
        methods = [str(method).lower() for method in supplied]
    supported = {"sgd", "sam", "gam_fd", "gam_exact", "random_perturb"}
    invalid = [method for method in methods if method not in supported]
    if invalid:
        raise ValueError(f"Unsupported optimizer schedule methods: {invalid}")
    return methods


def _task_training_config(
    train_config: dict, task_id: int, parameterization: str
) -> dict:
    resolved = dict(train_config)
    if task_id == 0:
        if train_config.get("task_a_lr") is not None:
            resolved["lr"] = float(train_config["task_a_lr"])
        if train_config.get("task_a_momentum") is not None:
            resolved["momentum"] = float(train_config["task_a_momentum"])
    elif task_id == 1:
        mapping = train_config.get("parameterization_lrs") or {}
        if parameterization in mapping:
            resolved["lr"] = float(mapping[parameterization])
        elif train_config.get("task_b_lr") is not None:
            resolved["lr"] = float(train_config["task_b_lr"])
        radius_mapping = train_config.get("parameterization_sam_rhos") or {}
        if parameterization in radius_mapping:
            resolved["sam_rho"] = float(radius_mapping[parameterization])
    return resolved


def _json_dump(value: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _write_accuracy_matrix(matrix: list[list[float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["after_task"] + [f"eval_task_{index}" for index in range(len(matrix))])
        for task_id, row in enumerate(matrix):
            writer.writerow([task_id] + ["" if np.isnan(value) else value for value in row])


def run_experiment(config: dict, project_root: Path) -> Path:
    set_seed(int(config["seed"]), bool(config["deterministic"]))
    device = resolve_device(config["device"])
    base = prepare_base_model(config, project_root, device)
    task_data: TaskData = build_task_datasets(config, project_root, base)
    # Restore the run seed after common-base creation so every variant gets controlled randomness.
    set_seed(int(config["seed"]), bool(config["deterministic"]))
    model_config = config["model"]
    alpha_value = model_config["lora_alpha"]
    lora_alpha = int(model_config["rank"]) if str(alpha_value).lower() == "rank" else float(alpha_value)
    seed_value = model_config.get("subspace_seed", "run_seed")
    subspace_seed = int(config["seed"]) if str(seed_value).lower() == "run_seed" else int(seed_value)
    target_parameterization = str(model_config["parameterization"]).lower()
    task_a_parameterization = model_config.get("task_a_parameterization")
    if task_a_parameterization is not None:
        task_a_parameterization = str(task_a_parameterization).lower()
        if len(task_data.train) != 2:
            raise ValueError("model.task_a_parameterization currently requires exactly two tasks")
    initial_parameterization = task_a_parameterization or target_parameterization
    model = _make_adaptive_model(
        base,
        model_config=model_config,
        parameterization=initial_parameterization,
        lora_alpha=lora_alpha,
        subspace_seed=subspace_seed,
    ).to(device)
    lifecycle = model_config["lifecycle"].lower()
    if lifecycle not in {"persistent", "merge_reset"}:
        raise ValueError("model.lifecycle must be 'persistent' or 'merge_reset'")

    output_root = resolve_path(config["output_root"], project_root)
    run_dir = output_root / _run_name(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config_resolved.yaml")

    train_config = config["training"]
    data_config = config["data"]
    diagnostic_config = config["diagnostics"]
    test_loaders = [
        _loader(
            dataset,
            int(train_config["batch_size"]),
            False,
            int(config["seed"]) + 20_000 + task_id,
            int(data_config["num_workers"]),
            device,
        )
        for task_id, dataset in enumerate(task_data.test)
    ]
    diagnostic_batches = []
    if diagnostic_config["enabled"]:
        diagnostic_batches = [
            collect_diagnostic_batch(
                dataset,
                int(diagnostic_config["max_samples_per_task"]),
                device,
            )
            for dataset in task_data.test
        ]

    task_count = len(task_data.train)
    optimizer_schedule = _resolve_optimizer_schedule(train_config, task_count)
    accuracy_matrix = [[float("nan")] * task_count for _ in range(task_count)]
    loss_matrix = [[float("nan")] * task_count for _ in range(task_count)]
    training_history: list[dict[str, Any]] = []
    trajectory_history: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    previous_start_basis: torch.Tensor | None = None
    task_geometry: list[dict[str, Any]] = []
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    for task_id, train_dataset in enumerate(task_data.train):
        task_start_metadata: dict[str, Any] = {}
        if task_id == 1 and task_a_parameterization is not None:
            common_weight = model.effective_weight().detach().clone()
            if diagnostic_batches:
                probe_inputs = diagnostic_batches[0][0][:32]
            else:
                probe_inputs, _ = next(iter(test_loaders[0]))
                probe_inputs = probe_inputs[:32].to(device)
            with torch.no_grad():
                logits_before = model(probe_inputs)
            model = _reparameterize_at_effective_weight(
                model,
                config,
                parameterization=target_parameterization,
                lora_alpha=lora_alpha,
                subspace_seed=subspace_seed,
            )
            with torch.no_grad():
                weight_error = float((model.effective_weight() - common_weight).abs().max())
                logits_error = float((model(probe_inputs) - logits_before).abs().max())
            if weight_error > 1e-7 or logits_error > 1e-6:
                raise RuntimeError(
                    "Common-endpoint reparameterization changed the model "
                    f"(weight error={weight_error}, logits error={logits_error})"
                )
            task_start_metadata = {
                "common_task_a_parameterization": task_a_parameterization,
                "task_b_parameterization": target_parameterization,
                "common_start_weight_sha256": _weight_sha256(common_weight),
                "common_reparameterization_weight_error": weight_error,
                "common_reparameterization_logit_error": logits_error,
            }
        task_method = optimizer_schedule[task_id]
        task_train_config = _task_training_config(
            train_config, task_id, model.parameterization
        )
        start_weight = model.effective_weight().detach().clone()

        def record_trajectory_point(epoch: int) -> None:
            if not diagnostic_config.get("trajectory_eval", False):
                return
            current_result = evaluate(model, test_loaders[task_id], device)
            old_result = evaluate(model, test_loaders[task_id - 1], device) if task_id > 0 else None
            if epoch == 0:
                record_trajectory_point.old_reference = old_result
            old_reference = getattr(record_trajectory_point, "old_reference", None)
            trajectory_history.append(
                {
                    "task": task_id,
                    "angle": task_data.angles[task_id],
                    "epoch": epoch,
                    "optimizer_method": task_method,
                    "current_loss": current_result["loss"],
                    "current_accuracy": current_result["accuracy"],
                    "old_task": task_id - 1 if task_id > 0 else None,
                    "old_loss": old_result["loss"] if old_result else None,
                    "old_accuracy": old_result["accuracy"] if old_result else None,
                    "old_loss_damage": (
                        old_result["loss"] - old_reference["loss"]
                        if old_result and old_reference
                        else None
                    ),
                    "old_accuracy_damage": (
                        old_reference["accuracy"] - old_result["accuracy"]
                        if old_result and old_reference
                        else None
                    ),
                    "effective_drift": float(
                        (model.effective_weight().detach() - start_weight).norm()
                    ),
                }
            )

        record_trajectory_point(0)
        start_basis = effective_tangent_basis(model)
        factor_geometry_at_start = model.factor_geometry()
        factor_state_at_start = (
            (model.lora_a.detach().clone(), model.lora_b.detach().clone())
            if model.is_factorized
            else None
        )
        basis_dimension = start_basis.shape[1]
        basis_overlap = None
        if task_id > 0:
            basis_overlap = subspace_overlap(previous_start_basis, start_basis, start_weight.numel())

        optimizer = make_sgd(model.trainable_parameters(), task_train_config)
        train_loader = _loader(
            train_dataset,
            int(train_config["batch_size"]),
            True,
            int(config["seed"]) + task_id,
            int(data_config["num_workers"]),
            device,
        )
        path_weights = [start_weight]
        path_granularity = diagnostic_config.get("path_granularity", "epoch").lower()
        if path_granularity not in {"epoch", "step"}:
            raise ValueError("diagnostics.path_granularity must be 'epoch' or 'step'")
        for epoch in range(int(train_config["epochs_per_task"])):
            model.train()
            losses = []
            perturbed_losses = []
            correction_norms = []
            effective_perturbation_norms = []
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                batch_result = train_batch(
                    model,
                    optimizer,
                    criterion,
                    inputs,
                    targets,
                    task_method,
                    task_train_config,
                )
                if model_config.get("balance_every") == "step":
                    model.rebalance_factors(optimizer)
                losses.append(batch_result["loss"])
                if np.isfinite(batch_result["perturbed_loss"]):
                    perturbed_losses.append(batch_result["perturbed_loss"])
                if np.isfinite(batch_result["curvature_correction_norm"]):
                    correction_norms.append(batch_result["curvature_correction_norm"])
                if np.isfinite(batch_result["effective_perturbation_norm"]):
                    effective_perturbation_norms.append(
                        batch_result["effective_perturbation_norm"]
                    )
                if diagnostic_config["pathwise"] and path_granularity == "step":
                    path_weights.append(model.effective_weight().detach().clone())
            if model_config.get("balance_every") == "epoch":
                model.rebalance_factors(optimizer)
            if not diagnostic_config["pathwise"] or path_granularity == "epoch":
                path_weights.append(model.effective_weight().detach().clone())
            training_history.append(
                {
                    "task": task_id,
                    "angle": task_data.angles[task_id],
                    "epoch": epoch + 1,
                    "optimizer_method": task_method,
                    "train_loss": float(np.mean(losses)),
                    "perturbed_loss": float(np.mean(perturbed_losses)) if perturbed_losses else None,
                    "curvature_correction_norm": (
                        float(np.mean(correction_norms)) if correction_norms else None
                    ),
                    "effective_perturbation_norm": (
                        float(np.mean(effective_perturbation_norms))
                        if effective_perturbation_norms
                        else None
                    ),
                }
            )
            record_trajectory_point(epoch + 1)
            print(
                f"task={task_id} method={task_method} "
                f"angle={task_data.angles[task_id]:g} epoch={epoch + 1} "
                f"loss={np.mean(losses):.4f}"
            )

        end_weight = model.effective_weight().detach().clone()
        transition_delta = end_weight - start_weight
        transition_common = {
            "from_task": task_id - 1,
            "to_task": task_id,
            "angle": task_data.angles[task_id],
            "angle_gap_from_previous": (
                None if task_id == 0 else task_data.angles[task_id] - task_data.angles[task_id - 1]
            ),
            "optimizer_method": task_method,
            "previous_optimizer_method": (
                None if task_id == 0 else optimizer_schedule[task_id - 1]
            ),
            "delta_norm": float(transition_delta.norm()),
            "start_tangent_dimension": int(basis_dimension),
            "reachable_coverage": reachable_coverage(start_basis, transition_delta),
            "previous_current_tangent_overlap": basis_overlap,
            "actual_parameterization": model.parameterization,
        }
        transition_common.update(task_start_metadata)
        if factor_geometry_at_start is not None:
            transition_common.update(
                {f"start_{key}": value for key, value in factor_geometry_at_start.items()}
            )
        end_factor_geometry = model.factor_geometry()
        if end_factor_geometry is not None:
            transition_common.update(
                {f"end_{key}": value for key, value in end_factor_geometry.items()}
            )
        for eval_task in range(task_id + 1):
            result = evaluate(model, test_loaders[eval_task], device)
            accuracy_matrix[task_id][eval_task] = result["accuracy"]
            loss_matrix[task_id][eval_task] = result["loss"]

        if diagnostic_config["enabled"] and task_id > 0:
            for old_task in range(task_id):
                inputs, targets = diagnostic_batches[old_task]
                tensor_path = None
                if diagnostic_config["save_hvp_tensors"]:
                    tensor_path = run_dir / "hvp_tensors" / f"transition_{task_id}_old_{old_task}.pt"
                record = exact_directional_diagnostics(
                    model,
                    inputs,
                    targets,
                    start_weight,
                    end_weight,
                    tensor_path,
                    diagnostic_config.get("hvp_fd_radii"),
                    int(diagnostic_config.get("hessian_lanczos_steps", 0))
                    if old_task == task_id - 1
                    else 0,
                )
                record.update(transition_common)
                record["old_task"] = old_task
                record["old_angle"] = task_data.angles[old_task]
                record["is_immediate_old_task"] = old_task == task_id - 1
                record["accuracy_change"] = (
                    accuracy_matrix[task_id][old_task] - accuracy_matrix[task_id - 1][old_task]
                )
                record["accuracy_forgetting"] = -record["accuracy_change"]
                if diagnostic_config.get("prospective_enabled", False) and (
                    old_task == task_id - 1
                ):
                    record.update(
                        prospective_reachable_direction_diagnostics(
                            model,
                            diagnostic_batches[old_task],
                            diagnostic_batches[task_id],
                            start_weight,
                            start_basis,
                        )
                    )
                geometry_requested = diagnostic_config.get("geometry_enabled", True)
                geometry_immediate_only = diagnostic_config.get("geometry_immediate_only", True)
                if geometry_requested and (not geometry_immediate_only or old_task == task_id - 1):
                    record.update(
                        ggn_safe_route_diagnostics(
                            model,
                            diagnostic_batches[old_task],
                            diagnostic_batches[task_id],
                            start_weight,
                            end_weight,
                            start_basis,
                            float(diagnostic_config["ggn_damping"]),
                            float(diagnostic_config["safe_relative_threshold"]),
                            int(diagnostic_config["ggn_overlap_top_k"]),
                        )
                    )
                    if factor_state_at_start is not None:
                        if task_method in {"sam", "random_perturb"}:
                            mechanism_radius = float(task_train_config["sam_rho"])
                        else:
                            mechanism_radius = float(task_train_config["gam_radius"])
                        record.update(
                            factor_parameterization_diagnostics(
                                model,
                                diagnostic_batches[task_id],
                                start_weight,
                                factor_state_at_start[0],
                                factor_state_at_start[1],
                                mechanism_radius,
                            )
                        )
                if diagnostic_config["pathwise"] and (
                    not diagnostic_config.get("pathwise_immediate_only", True)
                    or old_task == task_id - 1
                ):
                    record["pathwise"] = pathwise_diagnostics(
                        model, inputs, targets, path_weights
                    )
                transitions.append(record)

        # Save before reset for reproducibility; then test that merging is function preserving.
        torch.save(model.state_dict(), run_dir / f"model_after_task_{task_id}_premerge.pt")
        if lifecycle == "merge_reset":
            probe_inputs, _ = diagnostic_batches[task_id] if diagnostic_batches else next(iter(test_loaders[task_id]))
            probe_inputs = probe_inputs[:32].to(device)
            with torch.no_grad():
                logits_before = model(probe_inputs)
                model.merge_and_reset()
                logits_after = model(probe_inputs)
            merge_error = float((logits_before - logits_after).abs().max())
            if merge_error > 1e-5:
                raise RuntimeError(f"merge-and-reset changed logits (max error {merge_error})")
            transition_common["merge_function_error"] = merge_error
        task_geometry.append(transition_common)
        previous_start_basis = start_basis

    summary = continual_metrics(accuracy_matrix)
    summary.update(
        {
            "angles": task_data.angles,
            "num_tasks": task_count,
            "rotation_span": data_config["rotation_span"],
            "parameterization": target_parameterization,
            "task_a_parameterization": task_a_parameterization,
            "rank": model.rank if model.parameterization != "dense" else None,
            "factor_gauge_scale": model.factor_gauge_scale if model.is_factorized else None,
            "lifecycle": lifecycle,
            "optimizer": train_config["optimizer"],
            "optimizer_schedule": optimizer_schedule,
            "learning_rate": train_config["lr"],
            "task_learning_rates": [
                _task_training_config(train_config, task_id, target_parameterization)["lr"]
                for task_id in range(task_count)
            ],
            "task_sam_rhos": [
                _task_training_config(train_config, task_id, target_parameterization)["sam_rho"]
                for task_id in range(task_count)
            ],
            "sam_rho": train_config["sam_rho"],
            "gam_radius": train_config["gam_radius"],
            "gam_weight": train_config["gam_weight"],
            "seed": config["seed"],
            "elapsed_seconds": time.time() - start_time,
            "data_metadata": task_data.metadata,
            "accuracy_matrix": [
                [None if np.isnan(value) else value for value in row] for row in accuracy_matrix
            ],
            "loss_matrix": [
                [None if np.isnan(value) else value for value in row] for row in loss_matrix
            ],
        }
    )
    _json_dump(summary, run_dir / "metrics.json")
    _json_dump(training_history, run_dir / "training_history.json")
    _json_dump(trajectory_history, run_dir / "trajectory_history.json")
    _json_dump(transitions, run_dir / "transitions.json")
    _json_dump(
        [record for record in transitions if record["is_immediate_old_task"]],
        run_dir / "immediate_transitions.json",
    )
    _json_dump(task_geometry, run_dir / "task_geometry.json")
    _write_accuracy_matrix(accuracy_matrix, run_dir / "accuracy_matrix.csv")
    print(f"completed run: {run_dir}")
    return run_dir
