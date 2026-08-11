from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from small_cl.config import load_config
from small_cl.data import build_task_datasets
from small_cl.diagnostics import (
    collect_diagnostic_batch,
    exact_directional_diagnostics,
    weight_gradient,
)
from small_cl.experiment import (
    _loader,
    _make_adaptive_model,
    prepare_base_model,
    resolve_device,
    set_seed,
)
from small_cl.gauge import apply_factor_gauge_, make_gauge_matrix, pullback_gradient_metrics
from small_cl.optimizers import make_sgd, train_batch


PROJECT_ROOT = Path(__file__).resolve().parent


def _write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _prefix_dirs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob("angles-*/seed_*/shared_prefix.json"))


def analyze(input_root: Path, output: Path, offset: int = 0, limit: int | None = None) -> None:
    rows = []
    prefixes = _prefix_dirs(input_root)[offset:]
    if limit is not None:
        prefixes = prefixes[:limit]
    for prefix_dir in prefixes:
        config = load_config(prefix_dir / "config_resolved.yaml")
        seed = int(config["seed"])
        device = resolve_device(str(config["device"]))
        set_seed(seed, bool(config["deterministic"]))
        base = prepare_base_model(config, PROJECT_ROOT, device)
        task_data = build_task_datasets(config, PROJECT_ROOT, base)
        alpha_value = config["model"]["lora_alpha"]
        alpha = float(
            config["model"]["rank"]
            if str(alpha_value).lower() == "rank"
            else alpha_value
        )
        seed_value = config["model"].get("subspace_seed", "run_seed")
        subspace_seed = seed if str(seed_value).lower() == "run_seed" else int(seed_value)
        model = _make_adaptive_model(
            base, config["model"], "factor_lora", alpha, subspace_seed
        ).to(device)
        model.load_state_dict(
            torch.load(prefix_dir / "mature_prefix.pt", map_location=device, weights_only=True)
        )
        old_batch = collect_diagnostic_batch(
            task_data.test[0], int(config["diagnostics"]["max_samples_per_task"]), device
        )
        train_loader = _loader(
            task_data.train[1],
            int(config["training"]["batch_size"]),
            True,
            seed + int(config["mature_gauge"]["continuation_loader_seed_offset"]),
            int(config["data"]["num_workers"]),
            device,
        )
        train_inputs, train_targets = next(iter(train_loader))
        train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
        gauge_seed = seed + int(config["mature_gauge"]["gauge_seed_offset"])
        for transformation in config["mature_gauge"]["transformations"]:
            branch = _make_adaptive_model(
                base, config["model"], "factor_lora", alpha, subspace_seed
            ).to(device)
            branch.load_state_dict(model.state_dict())
            matrix = make_gauge_matrix(
                transformation,
                branch.rank,
                gauge_seed,
                device=device,
                dtype=branch.lora_a.dtype,
            )
            apply_factor_gauge_(branch, matrix)
            start_weight = branch.effective_weight().detach().clone()
            old_gradient = weight_gradient(
                branch, old_batch[0], old_batch[1], start_weight
            )
            new_gradient = weight_gradient(
                branch, train_inputs, train_targets, start_weight
            )
            pullback = pullback_gradient_metrics(branch, old_gradient, new_gradient)
            learning_rate = float(config["mature_gauge"]["task_b_lr"])
            a = branch.lora_a.detach().clone()
            b = branch.lora_b.detach().clone()
            delta_a = -learning_rate * branch.lora_scale * (b.T @ new_gradient)
            delta_b = -learning_rate * branch.lora_scale * (new_gradient @ a.T)
            predicted_linear = -learning_rate * branch.lora_scale**2 * (
                b @ b.T @ new_gradient + new_gradient @ a.T @ a
            )
            predicted_bilinear = branch.lora_scale * delta_b @ delta_a
            predicted_total = predicted_linear + predicted_bilinear
            training = dict(config["training"])
            training["lr"] = learning_rate
            training["momentum"] = float(config["mature_gauge"]["task_b_momentum"])
            optimizer = make_sgd(branch.trainable_parameters(), training)
            train_batch(
                branch,
                optimizer,
                nn.CrossEntropyLoss(),
                train_inputs,
                train_targets,
                "sgd",
                training,
            )
            end_weight = branch.effective_weight().detach().clone()
            actual_delta = end_weight - start_weight
            directional = exact_directional_diagnostics(
                branch,
                old_batch[0],
                old_batch[1],
                start_weight,
                end_weight,
            )
            rows.append(
                {
                    "order": f"{task_data.angles[0]:g}->{task_data.angles[1]:g}",
                    "seed": seed,
                    "transformation": transformation,
                    "gauge_condition": float(torch.linalg.cond(matrix)),
                    "actual_step_norm": float(actual_delta.norm()),
                    "predicted_linear_norm": float(predicted_linear.norm()),
                    "predicted_bilinear_norm": float(predicted_bilinear.norm()),
                    "linear_relative_error": float(
                        (actual_delta - predicted_linear).norm()
                        / actual_delta.norm().clamp_min(1e-20)
                    ),
                    "linear_plus_bilinear_relative_error": float(
                        (actual_delta - predicted_total).norm()
                        / actual_delta.norm().clamp_min(1e-20)
                    ),
                    "predicted_linear_interference": float(
                        (old_gradient * predicted_linear).sum()
                    ),
                    "predicted_total_interference": float(
                        (old_gradient * predicted_total).sum()
                    ),
                    "actual_interference": directional["interference_I"],
                    "actual_curvature": directional["directional_curvature_C"],
                    "actual_old_loss_change": directional["actual_loss_change"],
                    "taylor_residual": directional["taylor_residual"],
                    **pullback,
                }
            )
    if not rows:
        raise RuntimeError(f"No mature prefixes found below {input_root}")
    _write(rows, output)
    audit = {
        "rows": len(rows),
        "max_linear_plus_bilinear_relative_error": max(
            row["linear_plus_bilinear_relative_error"] for row in rows
        ),
        "max_interference_prediction_error": max(
            abs(row["predicted_total_interference"] - row["actual_interference"])
            for row in rows
        ),
        "max_linear_only_interference_error": max(
            abs(row["predicted_linear_interference"] - row["actual_interference"])
            for row in rows
        ),
        "predicted_total_actual_interference_correlation": float(
            np.corrcoef(
                [row["predicted_total_interference"] for row in rows],
                [row["actual_interference"] for row in rows],
            )[0, 1]
        ),
        "median_bilinear_fraction_of_step": float(
            np.median(
                [
                    row["predicted_bilinear_norm"]
                    / max(row["actual_step_norm"], 1e-20)
                    for row in rows
                ]
            )
        ),
    }
    with output.with_suffix(".audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(f"P1 first-step audit wrote {len(rows)} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact P1 first-step pullback audit")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    analyze(Path(args.input), Path(args.output), args.offset, args.limit)


if __name__ == "__main__":
    main()
