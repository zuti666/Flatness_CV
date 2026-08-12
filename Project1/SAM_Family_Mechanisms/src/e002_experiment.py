from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .e002_core import (
    E002Trace,
    FlatTanhMLP,
    full_hessian,
    gradient,
    hessian_vector_product,
    idealized_gam_direction,
    lookbehind_plain_sgd_delta,
    loss_value,
    make_two_moons,
    orthogonal_component,
    recompose_looksam_direction,
    sam_direction,
    sgd_direction,
    unit,
)
from .e002_plotting import make_e002_plots


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_id": "E002_two_moons_shared_anchor_pilot",
    "seeds": [3407, 3408],
    "train_samples": 512,
    "validation_samples": 1024,
    "test_samples": 4096,
    "data_noise": 0.15,
    "train_label_flip": 0.10,
    "hidden_width": 16,
    "batch_size": 32,
    "training_steps": 800,
    "learning_rate": 0.05,
    "checkpoint_steps": [80, 240, 480, 720, 800],
    "probe_batches": 64,
    "rho": 0.05,
    "rho_scan_checkpoint": 240,
    "rho_scan": [0.01, 0.03, 0.05],
    "inner_steps": 2,
    "path_protocol": "fixed_budget",
    "lookbehind_alpha": 0.5,
    "looksam_period": 5,
    "looksam_alpha": 0.7,
    "taylor_etas": [0.00625, 0.0125, 0.025, 0.05],
    "primary_taylor_eta": 0.05,
    "temporal_lags": [1, 2, 5, 10, 20],
    "temporal_stride": 5,
    "bootstrap_replicates": 500,
    "dtype": "float64",
    "device": "cuda:0",
    "physical_gpu_index": 5,
    "expected_gpu_uuid": "GPU-6eac7f06-8173-e1a3-c938-239f6f6eb19e",
    "make_plots": True,
}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    number = _finite(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _resolved_config(overrides: dict[str, Any] | None, quick: bool) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        config.update(copy.deepcopy(overrides))
    if quick:
        config.update(
            {
                "seeds": [int(config["seeds"][0])],
                "train_samples": 128,
                "validation_samples": 128,
                "test_samples": 256,
                "training_steps": 20,
                "checkpoint_steps": [5, 10, 20],
                "rho_scan_checkpoint": 10,
                "probe_batches": 4,
                "bootstrap_replicates": 20,
                "temporal_lags": [1, 2],
                "temporal_stride": 2,
            }
        )
    integer_fields = (
        "train_samples",
        "validation_samples",
        "test_samples",
        "hidden_width",
        "batch_size",
        "training_steps",
        "probe_batches",
        "rho_scan_checkpoint",
        "inner_steps",
        "looksam_period",
        "temporal_stride",
        "bootstrap_replicates",
        "physical_gpu_index",
    )
    for name in integer_fields:
        config[name] = _integer(config[name], name)
    for name in (
        "data_noise",
        "train_label_flip",
        "learning_rate",
        "rho",
        "lookbehind_alpha",
        "looksam_alpha",
        "primary_taylor_eta",
    ):
        config[name] = _finite(config[name], name)
    for name in ("seeds", "checkpoint_steps", "rho_scan", "taylor_etas", "temporal_lags"):
        if not isinstance(config.get(name), (list, tuple)) or not config[name]:
            raise ValueError(f"{name} must be a nonempty list")
    config["seeds"] = [_integer(value, f"seeds[{i}]") for i, value in enumerate(config["seeds"])]
    config["checkpoint_steps"] = [
        _integer(value, f"checkpoint_steps[{i}]")
        for i, value in enumerate(config["checkpoint_steps"])
    ]
    config["temporal_lags"] = [
        _integer(value, f"temporal_lags[{i}]")
        for i, value in enumerate(config["temporal_lags"])
    ]
    config["rho_scan"] = [_finite(value, f"rho_scan[{i}]") for i, value in enumerate(config["rho_scan"])]
    config["taylor_etas"] = [
        _finite(value, f"taylor_etas[{i}]")
        for i, value in enumerate(config["taylor_etas"])
    ]
    if config["dtype"] != "float64":
        raise ValueError("E002 pilot intentionally fixes float64")
    if config["device"] not in {"cuda:0", "cpu"}:
        raise ValueError("device must be cuda:0 or cpu")
    if config["path_protocol"] != "fixed_budget":
        raise ValueError("the preregistered E002 pilot uses fixed_budget")
    if config["hidden_width"] != 16:
        raise ValueError("the preregistered E002 model is exactly 2-16-2")
    if config["batch_size"] <= 1 or config["batch_size"] > config["train_samples"]:
        raise ValueError("invalid batch_size")
    if config["training_steps"] < 1 or config["probe_batches"] < 2:
        raise ValueError("training_steps must be positive and probe_batches >=2")
    if sorted(set(config["checkpoint_steps"])) != config["checkpoint_steps"]:
        raise ValueError("checkpoint_steps must be sorted and unique")
    if config["checkpoint_steps"][-1] != config["training_steps"]:
        raise ValueError("the final checkpoint must equal training_steps")
    if config["rho_scan_checkpoint"] not in config["checkpoint_steps"]:
        raise ValueError("rho_scan_checkpoint must be a registered checkpoint")
    if any(value <= 0 for value in config["rho_scan"] + config["taylor_etas"]):
        raise ValueError("rho_scan and taylor_etas must be positive")
    if config["rho"] <= 0 or config["learning_rate"] <= 0:
        raise ValueError("rho and learning_rate must be positive")
    if not 0 <= config["train_label_flip"] < 1 or config["data_noise"] < 0:
        raise ValueError("invalid data noise/label flip")
    if not 0 <= config["lookbehind_alpha"] <= 1 or config["looksam_alpha"] < 0:
        raise ValueError("invalid Lookbehind/LookSAM alpha")
    if len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("seeds must be unique")
    if config["make_plots"] not in (True, False, 0, 1):
        raise ValueError("make_plots must be boolean")
    config["make_plots"] = bool(config["make_plots"])
    config["quick"] = bool(quick)
    return config


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite JSON value: {number}")
        return number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"unsupported JSON type: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for key in fields:
                value = row.get(key)
                if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
                    raise ValueError(f"non-finite CSV field {key}")
                clean[key] = value
            writer.writerow(clean)


def _hash_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _query_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_used_mib": int(parts[3]),
                "memory_total_mib": int(parts[4]),
                "utilization_percent": int(parts[5]),
                "temperature_c": int(parts[6]),
            }
        )
    return rows


def _runtime_device(config: dict[str, Any]) -> tuple[torch.device, dict[str, Any]]:
    requested = str(config["device"])
    physical = int(config["physical_gpu_index"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    preflight = _query_gpus()
    physical_row = next((row for row in preflight if row["index"] == physical), None)
    if requested == "cpu":
        return torch.device("cpu"), {
            "requested": requested,
            "physical_gpu_index": physical,
            "cuda_visible_devices": visible,
            "identity_verified": False,
            "preflight": physical_row,
        }
    if visible != str(physical):
        raise RuntimeError(
            f"E002 must be launched with CUDA_VISIBLE_DEVICES={physical}; got {visible!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the requested single visible CUDA device is unavailable")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    expected_uuid = str(config.get("expected_gpu_uuid", ""))
    if physical_row is None:
        raise RuntimeError("could not audit the physical GPU through nvidia-smi")
    if expected_uuid and physical_row["uuid"] != expected_uuid:
        raise RuntimeError(
            f"physical GPU UUID mismatch: {physical_row['uuid']} != {expected_uuid}"
        )
    torch.cuda.reset_peak_memory_stats(0)
    return torch.device("cuda:0"), {
        "requested": requested,
        "logical_device": "cuda:0",
        "physical_gpu_index": physical,
        "cuda_visible_devices": visible,
        "identity_verified": True,
        "uuid": physical_row["uuid"],
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "preflight": physical_row,
    }


def _tensor(array: np.ndarray, device: torch.device, *, target: bool = False) -> torch.Tensor:
    dtype = torch.long if target else torch.float64
    return torch.as_tensor(array, dtype=dtype, device=device)


def _accuracy(model: FlatTanhMLP, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        prediction = model.forward(theta, x).argmax(dim=1)
        return float((prediction == y).to(torch.float64).mean().cpu())


def _loss_float(model: FlatTanhMLP, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
    return float(loss_value(model, theta, x, y).detach().cpu())


def _candidate_losses(
    model: FlatTanhMLP,
    candidates: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Vectorized full-data CE for a stack of flat 2-16-2 parameter vectors."""
    if candidates.ndim != 2 or candidates.shape[1] != model.parameter_count:
        raise ValueError("candidate matrix has the wrong shape")
    h = model.hidden_width
    cursor = 0
    w1 = candidates[:, cursor : cursor + 2 * h].reshape(-1, h, 2)
    cursor += 2 * h
    b1 = candidates[:, cursor : cursor + h]
    cursor += h
    w2 = candidates[:, cursor : cursor + 2 * h].reshape(-1, 2, h)
    cursor += 2 * h
    b2 = candidates[:, cursor : cursor + 2]
    hidden = torch.tanh(torch.einsum("ni,mhi->mnh", inputs, w1) + b1[:, None, :])
    logits = torch.einsum("mnh,moh->mno", hidden, w2) + b2[:, None, :]
    log_probabilities = logits - torch.logsumexp(logits, dim=2, keepdim=True)
    gather = targets[None, :, None].expand(candidates.shape[0], -1, 1)
    return -torch.gather(log_probabilities, 2, gather).squeeze(2).mean(dim=1)


def _cosine(first: np.ndarray, second: np.ndarray, epsilon: float = 1e-15) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= epsilon:
        return None
    return float(np.dot(first, second) / denominator)


def _trace_from_direction(method: str, direction: torch.Tensor, clean: torch.Tensor) -> E002Trace:
    return E002Trace(
        method=method,
        direction=direction.detach(),
        clean_gradient=clean.detach(),
        correction=(direction - clean).detach(),
        perturbation=torch.zeros_like(clean),
    )


def _shared_traces(
    model: FlatTanhMLP,
    theta: torch.Tensor,
    history: list[torch.Tensor],
    checkpoint_step: int,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: dict[str, Any],
) -> tuple[dict[str, E002Trace], E002Trace]:
    rho = float(config["rho"])
    k = int(config["inner_steps"])
    rho_step = rho / k
    alpha_lb = float(config["lookbehind_alpha"])
    learning_rate = float(config["learning_rate"])
    sgd = sgd_direction(model, theta, inputs, targets)
    sam = sam_direction(model, theta, inputs, targets, rho)
    gam = idealized_gam_direction(model, theta, inputs, targets, rho, alpha=1.0)
    gam.method = "gam_exact_hvp_same_batch_alpha1"
    lookbehind = lookbehind_plain_sgd_delta(
        model,
        theta,
        inputs,
        targets,
        rho_step,
        k,
        learning_rate,
        alpha_lb,
    )
    lookbehind.method = f"lookbehind_faithful_k{k}_alpha{alpha_lb:g}"
    assert lookbehind.path_gradients is not None
    multistep = _trace_from_direction(
        f"multistep_sam_k{k}_fixed_budget",
        lookbehind.path_gradients[-1],
        sgd.direction,
    )
    multistep.path_points = lookbehind.path_points
    multistep.path_gradients = lookbehind.path_gradients
    assert lookbehind.path_mean_surrogate is not None
    surrogate = _trace_from_direction(
        f"lookbehind_path_mean_surrogate_k{k}",
        lookbehind.path_mean_surrogate,
        sgd.direction,
    )
    surrogate.path_points = lookbehind.path_points
    surrogate.path_gradients = lookbehind.path_gradients
    rho_eff = 0.5 * (k + 1) * rho_step
    matched = sam_direction(model, theta, inputs, targets, rho_eff)
    matched.method = f"matched_sam_k{k}_fixed_budget"
    traces: dict[str, E002Trace] = {
        "sgd": sgd,
        "sam": sam,
        gam.method: gam,
        multistep.method: multistep,
        surrogate.method: surrogate,
        lookbehind.method: lookbehind,
        matched.method: matched,
        "sam5_refresh": _trace_from_direction("sam5_refresh", sam.direction, sgd.direction),
        "sam5_nonrefresh": _trace_from_direction("sam5_nonrefresh", sgd.direction, sgd.direction),
    }
    for age in range(int(config["looksam_period"])):
        method = f"looksam_age{age}"
        if age == 0:
            direction = sam.direction
        else:
            old_theta = history[max(0, checkpoint_step - age)]
            old_clean = gradient(model, old_theta, inputs, targets)
            old_sam = sam_direction(model, old_theta, inputs, targets, rho)
            cached = orthogonal_component(old_sam.direction, old_clean)
            direction = recompose_looksam_direction(
                sgd.direction,
                cached,
                alpha=float(config["looksam_alpha"]),
            )
        traces[method] = _trace_from_direction(method, direction, sgd.direction)
    return traces, lookbehind


def _covariance(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    result = centered.T @ centered / (values.shape[0] - 1)
    return 0.5 * (result + result.T)


def _bootstrap_trace_ci(
    values: np.ndarray,
    matrix: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    sample_count = values.shape[0]
    norm_sq = np.einsum("mi,mi->m", values, values)
    quadratic = np.einsum("mi,ij,mj->m", values, matrix, values)
    trace_samples = np.empty(replicates, dtype=np.float64)
    matrix_samples = np.empty(replicates, dtype=np.float64)
    factor = sample_count / (sample_count - 1)
    for index in range(replicates):
        chosen = rng.integers(0, sample_count, size=sample_count)
        current = values[chosen]
        mean = current.mean(axis=0)
        trace_samples[index] = factor * (norm_sq[chosen].mean() - np.dot(mean, mean))
        matrix_samples[index] = factor * (
            quadratic[chosen].mean() - float(mean @ matrix @ mean)
        )
    trace_low, trace_high = np.quantile(trace_samples, (0.025, 0.975))
    matrix_low, matrix_high = np.quantile(matrix_samples, (0.025, 0.975))
    return float(trace_low), float(trace_high), float(matrix_low), float(matrix_high)


def _covariance_row(
    *,
    seed: int,
    checkpoint: int,
    method: str,
    directions: np.ndarray,
    corrections: np.ndarray,
    hessian: np.ndarray,
    hplus: np.ndarray,
    hminus_positive: np.ndarray,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], np.ndarray]:
    sigma = _covariance(directions)
    correction_sigma = _covariance(corrections)
    trace_sigma = float(np.trace(sigma))
    trace_hplus = float(np.trace(hplus @ sigma))
    trace_hminus = float(np.trace(hminus_positive @ sigma))
    trace_h = float(np.trace(hessian @ sigma))
    hplus_trace = float(np.trace(hplus))
    dimension = directions.shape[1]
    nha = None
    if trace_sigma > 1e-20 and hplus_trace > 1e-20:
        nha = dimension * trace_hplus / (hplus_trace * trace_sigma)
    ci = _bootstrap_trace_ci(
        directions,
        hplus,
        bootstrap_replicates,
        seed + 17 * checkpoint + sum(method.encode("utf-8")),
    )
    half_width = 0.5 * (ci[3] - ci[2])
    relative_half_width = None if abs(trace_hplus) <= 1e-15 else half_width / abs(trace_hplus)
    eigen_scale = max(1.0, float(np.linalg.norm(sigma, ord=2)))
    minimum_eigenvalue = float(np.linalg.eigvalsh(sigma)[0])
    return {
        "seed": seed,
        "checkpoint_step": checkpoint,
        "method": method,
        "probe_batches": int(directions.shape[0]),
        "direction_trace_sigma": trace_sigma,
        "direction_trace_hplus_sigma": trace_hplus,
        "direction_trace_hminus_sigma": trace_hminus,
        "direction_trace_h_sigma": trace_h,
        "direction_nha_positive": nha,
        "correction_trace_sigma": float(np.trace(correction_sigma)),
        "correction_trace_hplus_sigma": float(np.trace(hplus @ correction_sigma)),
        "correction_trace_hminus_sigma": float(np.trace(hminus_positive @ correction_sigma)),
        "covariance_min_eigenvalue": minimum_eigenvalue,
        "covariance_psd_tolerance": 1e-10 * eigen_scale,
        "trace_sigma_ci_low": ci[0],
        "trace_sigma_ci_high": ci[1],
        "trace_hplus_sigma_ci_low": ci[2],
        "trace_hplus_sigma_ci_high": ci[3],
        "trace_hplus_relative_ci_half_width": relative_half_width,
        "underpowered_relative_ci": (
            None if relative_half_width is None else bool(relative_half_width > 0.25)
        ),
    }, sigma


def _path_cosine_metrics(gradients: np.ndarray) -> tuple[float, float]:
    pair_values = []
    for first in range(gradients.shape[0]):
        for second in range(first + 1, gradients.shape[0]):
            value = _cosine(gradients[first], gradients[second])
            if value is not None:
                pair_values.append(value)
    path_misalign = 0.0 if not pair_values else 1.0 - float(np.mean(pair_values))
    average = gradients.mean(axis=0)
    last_average = _cosine(gradients[-1], average)
    return path_misalign, 0.0 if last_average is None else 1.0 - last_average


def _checkpoint_probe(
    *,
    model: FlatTanhMLP,
    theta: torch.Tensor,
    history: list[torch.Tensor],
    checkpoint: int,
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    probe_indices: np.ndarray,
    config: dict[str, Any],
    raw_arrays: dict[str, np.ndarray],
    hessian_rows: list[dict[str, Any]],
    covariance_rows: list[dict[str, Any]],
    taylor_rows: list[dict[str, Any]],
    spectral_rows: list[dict[str, Any]],
    path_rows: list[dict[str, Any]],
    fidelity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    full_gradient = gradient(model, theta, x_train, y_train).cpu().numpy()
    hessian_tensor = full_hessian(model, theta, x_train, y_train)
    hessian = hessian_tensor.cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    hplus = (eigenvectors * np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    hminus_positive = (eigenvectors * np.clip(-eigenvalues, 0.0, None)) @ eigenvectors.T
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    sign_tolerance = max(1e-10, 1e-8 * float(np.max(np.abs(eigenvalues))))
    symmetry_error = float(np.linalg.norm(hessian - hessian.T) / max(np.linalg.norm(hessian), 1e-30))
    reconstruction = (eigenvectors * eigenvalues) @ eigenvectors.T
    reconstruction_error = float(
        np.linalg.norm(hessian - reconstruction) / max(np.linalg.norm(hessian), 1e-30)
    )
    fidelity_rows.extend(
        [
            {
                "seed": seed,
                "checkpoint_step": checkpoint,
                "check": "hessian_symmetry_relative_error",
                "value": symmetry_error,
                "threshold": 1e-10,
                "passed": symmetry_error <= 1e-10,
                "details": "full-train flipped-label mean cross-entropy",
            },
            {
                "seed": seed,
                "checkpoint_step": checkpoint,
                "check": "hessian_eigen_reconstruction_relative_error",
                "value": reconstruction_error,
                "threshold": 1e-8,
                "passed": reconstruction_error <= 1e-8,
                "details": "torch Hessian -> numpy eigh",
            },
        ]
    )
    raw_arrays[f"seed{seed}__step{checkpoint}__theta"] = theta.detach().cpu().numpy()
    raw_arrays[f"seed{seed}__step{checkpoint}__hessian"] = hessian
    raw_arrays[f"seed{seed}__step{checkpoint}__eigenvalues"] = eigenvalues
    raw_arrays[f"seed{seed}__step{checkpoint}__eigenvectors"] = eigenvectors
    for index in range(len(eigenvalues)):
        hessian_rows.append(
            {
                "seed": seed,
                "checkpoint_step": checkpoint,
                "eigen_index_ascending": index + 1,
                "eigenvalue": float(eigenvalues[index]),
                "sign": (
                    "positive"
                    if eigenvalues[index] > sign_tolerance
                    else "negative"
                    if eigenvalues[index] < -sign_tolerance
                    else "near_zero"
                ),
                "sign_tolerance": sign_tolerance,
            }
        )
    if checkpoint == int(config["rho_scan_checkpoint"]):
        g_tensor = torch.as_tensor(full_gradient, dtype=torch.float64, device=theta.device)
        exact = hessian_vector_product(model, theta, x_train, y_train, unit(g_tensor)).cpu().numpy()
        for rho in config["rho_scan"]:
            perturbed = gradient(
                model,
                theta + float(rho) * unit(g_tensor),
                x_train,
                y_train,
            ).cpu().numpy()
            estimate = (perturbed - full_gradient) / float(rho)
            cosine = _cosine(estimate, exact)
            relative = float(np.linalg.norm(estimate - exact) / max(np.linalg.norm(exact), 1e-30))
            fidelity_rows.append(
                {
                    "seed": seed,
                    "checkpoint_step": checkpoint,
                    "check": "sam_finite_difference_hvp",
                    "value": relative,
                    "threshold": 0.01 if float(rho) == min(config["rho_scan"]) else None,
                    "passed": (
                        bool(relative <= 0.01 and cosine is not None and cosine >= 0.999)
                        if float(rho) == min(config["rho_scan"])
                        else True
                    ),
                    "details": f"rho={rho:g}; cosine={cosine}",
                }
            )

    direction_lists: dict[str, list[np.ndarray]] = defaultdict(list)
    correction_lists: dict[str, list[np.ndarray]] = defaultdict(list)
    clean_batches: list[np.ndarray] = []
    path_misalign: list[float] = []
    last_mean_difference: list[float] = []
    last_gradients: list[np.ndarray] = []
    mean_gradients: list[np.ndarray] = []
    maximum_lb_equivalence_error = 0.0
    maximum_refresh_error = 0.0
    maximum_sam5_error = 0.0
    for batch_indices in probe_indices:
        batch_x = x_train[torch.as_tensor(batch_indices, device=x_train.device)]
        batch_y = y_train[torch.as_tensor(batch_indices, device=y_train.device)]
        traces, lb = _shared_traces(
            model, theta, history, checkpoint, batch_x, batch_y, config
        )
        clean = traces["sgd"].direction.cpu().numpy()
        clean_batches.append(clean)
        for method, trace in traces.items():
            direction = trace.direction.cpu().numpy()
            correction = direction - clean
            direction_lists[method].append(direction)
            correction_lists[method].append(correction)
        assert lb.path_gradients is not None and lb.path_mean_surrogate is not None
        path = lb.path_gradients.cpu().numpy()
        current_path_misalign, current_last_mean = _path_cosine_metrics(path)
        path_misalign.append(current_path_misalign)
        last_mean_difference.append(current_last_mean)
        last_gradients.append(path[-1])
        mean_gradients.append(path.mean(axis=0))
        faithful = traces[f"lookbehind_faithful_k{config['inner_steps']}_alpha{config['lookbehind_alpha']:g}"]
        maximum_lb_equivalence_error = max(
            maximum_lb_equivalence_error,
            float(
                torch.linalg.vector_norm(faithful.direction - lb.path_mean_surrogate)
                / max(float(torch.linalg.vector_norm(lb.path_mean_surrogate)), 1e-30)
            ),
        )
        maximum_refresh_error = max(
            maximum_refresh_error,
            float(torch.linalg.vector_norm(traces["looksam_age0"].direction - traces["sam"].direction)),
        )
        maximum_sam5_error = max(
            maximum_sam5_error,
            float(torch.linalg.vector_norm(traces["sam5_nonrefresh"].direction - traces["sgd"].direction)),
        )
    fidelity_rows.extend(
        [
            {
                "seed": seed,
                "checkpoint_step": checkpoint,
                "check": "lookbehind_alpha_half_equals_path_mean_k2",
                "value": maximum_lb_equivalence_error,
                "threshold": 1e-12,
                "passed": maximum_lb_equivalence_error <= 1e-12,
                "details": "faithful slow direction remains separately labelled",
            },
            {
                "seed": seed,
                "checkpoint_step": checkpoint,
                "check": "looksam_age0_equals_sam",
                "value": maximum_refresh_error,
                "threshold": 1e-12,
                "passed": maximum_refresh_error <= 1e-12,
                "details": "refresh contract",
            },
            {
                "seed": seed,
                "checkpoint_step": checkpoint,
                "check": "sam5_nonrefresh_equals_sgd",
                "value": maximum_sam5_error,
                "threshold": 1e-12,
                "passed": maximum_sam5_error <= 1e-12,
                "details": "periodic-SAM control",
            },
        ]
    )
    clean_matrix = np.stack(clean_batches)
    base_loss = _loss_float(model, theta, x_train, y_train)
    ghat = full_gradient / max(np.linalg.norm(full_gradient), 1e-30)
    ghat_coordinates = eigenvectors.T @ ghat
    mask_threshold = max(1e-10, 1e-6 * float(np.max(np.abs(ghat_coordinates))))
    summaries: dict[str, Any] = {}
    for method in sorted(direction_lists):
        directions = np.stack(direction_lists[method])
        corrections = np.stack(correction_lists[method])
        raw_arrays[f"seed{seed}__step{checkpoint}__{method}__directions"] = directions
        raw_arrays[f"seed{seed}__step{checkpoint}__{method}__corrections"] = corrections
        covariance_row, sigma = _covariance_row(
            seed=seed,
            checkpoint=checkpoint,
            method=method,
            directions=directions,
            corrections=corrections,
            hessian=hessian,
            hplus=hplus,
            hminus_positive=hminus_positive,
            bootstrap_replicates=int(config["bootstrap_replicates"]),
        )
        covariance_rows.append(covariance_row)
        mu = directions.mean(axis=0)
        mean_correction = corrections.mean(axis=0)
        correction_coordinates = eigenvectors.T @ mean_correction
        for index, eigenvalue in enumerate(eigenvalues):
            masked = abs(ghat_coordinates[index]) < mask_threshold
            spectral_rows.append(
                {
                    "seed": seed,
                    "checkpoint_step": checkpoint,
                    "method": method,
                    "eigen_index_ascending": index + 1,
                    "eigenvalue": float(eigenvalue),
                    "gradient_unit_projection": float(ghat_coordinates[index]),
                    "correction_projection": float(correction_coordinates[index]),
                    "denominator_masked": bool(masked),
                    "signed_transfer": (
                        None
                        if masked
                        else float(correction_coordinates[index] / ghat_coordinates[index])
                    ),
                }
            )
        direction_tensor = torch.as_tensor(directions, dtype=torch.float64, device=theta.device)
        quadratic_each = np.einsum("mi,ij,mj->m", directions, hessian, directions)
        for eta in config["taylor_etas"]:
            candidates = theta[None, :] - float(eta) * direction_tensor
            true_each = _candidate_losses(model, candidates, x_train, y_train).detach().cpu().numpy() - base_loss
            first_each = -float(eta) * (directions @ full_gradient)
            second_each = 0.5 * float(eta) ** 2 * quadratic_each
            predicted_each = first_each + second_each
            absolute_residual = np.abs(true_each - predicted_each)
            component_scale = np.abs(first_each) + np.abs(second_each) + 1e-15
            component_error = absolute_residual / component_scale
            true_error = absolute_residual / (np.abs(true_each) + 1e-15)
            factor = (directions.shape[0] - 1) / directions.shape[0]
            t1 = -float(eta) * float(np.dot(full_gradient, mu))
            t2_mean = 0.5 * float(eta) ** 2 * float(mu @ hessian @ mu)
            t2_positive = 0.5 * float(eta) ** 2 * factor * float(np.trace(hplus @ sigma))
            t2_negative = -0.5 * float(eta) ** 2 * factor * float(
                np.trace(hminus_positive @ sigma)
            )
            predicted_mean = t1 + t2_mean + t2_positive + t2_negative
            taylor_rows.append(
                {
                    "seed": seed,
                    "checkpoint_step": checkpoint,
                    "method": method,
                    "eta": float(eta),
                    "t1": t1,
                    "t2_mean": t2_mean,
                    "t2_noise_positive": t2_positive,
                    "t2_noise_negative": t2_negative,
                    "predicted_mean_loss_change": predicted_mean,
                    "true_mean_loss_change": float(true_each.mean()),
                    "mean_absolute_residual": float(absolute_residual.mean()),
                    "component_normalized_error_median": float(np.median(component_error)),
                    "component_normalized_error_p90": float(np.quantile(component_error, 0.9)),
                    "true_normalized_error_median": float(np.median(true_error)),
                    "true_normalized_error_p90": float(np.quantile(true_error, 0.9)),
                    "finite_sample_covariance_factor": factor,
                }
            )
        summaries[method] = {
            "mean_direction_norm": float(np.linalg.norm(mu)),
            "mean_correction_norm": float(np.linalg.norm(mean_correction)),
            "trace_sigma": covariance_row["direction_trace_sigma"],
            "trace_hplus_sigma": covariance_row["direction_trace_hplus_sigma"],
            "trace_hminus_sigma": covariance_row["direction_trace_hminus_sigma"],
        }
    last_sigma = float(np.trace(_covariance(np.stack(last_gradients))))
    mean_sigma = float(np.trace(_covariance(np.stack(mean_gradients))))
    for method in (
        f"multistep_sam_k{config['inner_steps']}_fixed_budget",
        f"lookbehind_path_mean_surrogate_k{config['inner_steps']}",
        f"lookbehind_faithful_k{config['inner_steps']}_alpha{config['lookbehind_alpha']:g}",
    ):
        path_rows.append(
            {
                "seed": seed,
                "checkpoint_step": checkpoint,
                "method": method,
                "inner_steps": int(config["inner_steps"]),
                "protocol": "fixed_budget",
                "rho": float(config["rho"]),
                "rho_step": float(config["rho"]) / int(config["inner_steps"]),
                "lookbehind_alpha": float(config["lookbehind_alpha"]),
                "path_misalign_mean": float(np.mean(path_misalign)),
                "last_average_difference_mean": float(np.mean(last_mean_difference)),
                "last_gradient_trace_sigma": last_sigma,
                "mean_gradient_trace_sigma": mean_sigma,
                "last_to_mean_variance_ratio": last_sigma / max(mean_sigma, 1e-30),
            }
        )
    return {
        "base_loss": base_loss,
        "gradient_norm": float(np.linalg.norm(full_gradient)),
        "lambda_min": float(eigenvalues[0]),
        "lambda_max": float(eigenvalues[-1]),
        "negative_eigenvalues": int(np.count_nonzero(eigenvalues < -sign_tolerance)),
        "positive_eigenvalues": int(np.count_nonzero(eigenvalues > sign_tolerance)),
        "near_zero_eigenvalues": int(np.count_nonzero(np.abs(eigenvalues) <= sign_tolerance)),
        "methods": summaries,
    }


def _temporal_rows(
    model: FlatTanhMLP,
    history: list[torch.Tensor],
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = y_train.detach().cpu().numpy()
    per_class = min(64, int(np.count_nonzero(labels == 0)), int(np.count_nonzero(labels == 1)))
    indices = np.concatenate((np.flatnonzero(labels == 0)[:per_class], np.flatnonzero(labels == 1)[:per_class]))
    x = x_train[torch.as_tensor(indices, device=x_train.device)]
    y = y_train[torch.as_tensor(indices, device=y_train.device)]
    base_times = list(range(0, len(history), int(config["temporal_stride"])))
    needed = set(base_times)
    for time_index in base_times:
        for lag in config["temporal_lags"]:
            if time_index + lag < len(history):
                needed.add(time_index + lag)
    objects: dict[int, dict[str, np.ndarray]] = {}
    for time_index in sorted(needed):
        theta = history[time_index]
        clean = gradient(model, theta, x, y)
        sam = sam_direction(model, theta, x, y, float(config["rho"]))
        orthogonal = orthogonal_component(sam.direction, clean)
        objects[time_index] = {
            "gradient": clean.cpu().numpy(),
            "sam_direction": sam.direction.cpu().numpy(),
            "orthogonal_correction": orthogonal.cpu().numpy(),
        }
    rows = []
    for lag in config["temporal_lags"]:
        for object_name in ("gradient", "sam_direction", "orthogonal_correction"):
            cosines = []
            drifts = []
            for start in base_times:
                end = start + lag
                if end not in objects:
                    continue
                first = objects[start][object_name]
                second = objects[end][object_name]
                cosine = _cosine(first, second)
                norm = float(np.linalg.norm(first))
                if cosine is None or norm <= 1e-15:
                    continue
                cosines.append(cosine)
                drifts.append(float(np.linalg.norm(second - first) / norm))
            if not cosines:
                continue
            rows.append(
                {
                    "seed": seed,
                    "object": object_name,
                    "lag": int(lag),
                    "base_stride": int(config["temporal_stride"]),
                    "pair_count": len(cosines),
                    "median_cosine": float(np.median(cosines)),
                    "median_relative_drift": float(np.median(drifts)),
                }
            )
    return rows


def _train_shared_anchor(
    model: FlatTanhMLP,
    initial: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    stream: np.ndarray,
    config: dict[str, Any],
) -> list[torch.Tensor]:
    theta = initial.detach().clone()
    history = [theta.detach().clone()]
    for batch_indices in stream:
        indices = torch.as_tensor(batch_indices, device=x_train.device)
        direction = gradient(model, theta, x_train[indices], y_train[indices])
        theta = (theta - float(config["learning_rate"]) * direction).detach()
        history.append(theta.detach().clone())
    return history


def _on_policy_direction(
    method: str,
    model: FlatTanhMLP,
    theta: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    outer_index: int,
    cache: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    rho = float(config["rho"])
    if method == "sgd":
        return sgd_direction(model, theta, inputs, targets).direction, cache
    if method == "sam":
        return sam_direction(model, theta, inputs, targets, rho).direction, cache
    if method == f"multistep_sam_k{config['inner_steps']}_fixed_budget":
        trace = lookbehind_plain_sgd_delta(
            model,
            theta,
            inputs,
            targets,
            rho / int(config["inner_steps"]),
            int(config["inner_steps"]),
            float(config["learning_rate"]),
            float(config["lookbehind_alpha"]),
        )
        assert trace.path_gradients is not None
        return trace.path_gradients[-1], cache
    if method == f"lookbehind_faithful_k{config['inner_steps']}_alpha{config['lookbehind_alpha']:g}":
        trace = lookbehind_plain_sgd_delta(
            model,
            theta,
            inputs,
            targets,
            rho / int(config["inner_steps"]),
            int(config["inner_steps"]),
            float(config["learning_rate"]),
            float(config["lookbehind_alpha"]),
        )
        return trace.direction, cache
    if method == f"looksam_k{config['looksam_period']}_alpha{config['looksam_alpha']:g}":
        clean = gradient(model, theta, inputs, targets)
        refresh = outer_index % int(config["looksam_period"]) == 0 or "orthogonal" not in cache
        if refresh:
            sam = sam_direction(model, theta, inputs, targets, rho)
            cache = {
                "orthogonal": orthogonal_component(sam.direction, clean),
                "phase": outer_index % int(config["looksam_period"]),
            }
            return sam.direction, cache
        return recompose_looksam_direction(
            clean,
            cache["orthogonal"],
            alpha=float(config["looksam_alpha"]),
        ), cache
    if method == "sam5":
        if outer_index % int(config["looksam_period"]) == 0:
            return sam_direction(model, theta, inputs, targets, rho).direction, cache
        return sgd_direction(model, theta, inputs, targets).direction, cache
    raise ValueError(f"unsupported on-policy method {method}")


def _train_on_policy(
    *,
    model: FlatTanhMLP,
    initial: torch.Tensor,
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    stream: np.ndarray,
    config: dict[str, Any],
    checkpoint_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    methods = [
        "sgd",
        "sam",
        f"multistep_sam_k{config['inner_steps']}_fixed_budget",
        f"lookbehind_faithful_k{config['inner_steps']}_alpha{config['lookbehind_alpha']:g}",
        f"looksam_k{config['looksam_period']}_alpha{config['looksam_alpha']:g}",
        "sam5",
    ]
    report_interval = max(1, int(config["training_steps"]) // 20)
    report_steps = set(range(0, int(config["training_steps"]) + 1, report_interval))
    report_steps.update(config["checkpoint_steps"])
    history_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    for method in methods:
        theta = initial.detach().clone()
        cache: dict[str, Any] = {}
        initial_loss = _loss_float(model, theta, x_train, y_train)
        initial_accuracy = _accuracy(model, theta, x_test, y_test)
        for step in range(0, int(config["training_steps"]) + 1):
            if step in report_steps:
                history_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "step": step,
                        "train_loss": _loss_float(model, theta, x_train, y_train),
                        "train_accuracy": _accuracy(model, theta, x_train, y_train),
                        "test_accuracy": _accuracy(model, theta, x_test, y_test),
                        "cache_present": bool("orthogonal" in cache),
                    }
                )
            if step in config["checkpoint_steps"]:
                target = checkpoint_directory / f"seed{seed}__{method}__step{step}.pt"
                torch.save(
                    {
                        "theta": theta.detach().cpu(),
                        "looksam_cache": {
                            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                            for key, value in cache.items()
                        },
                    },
                    target,
                )
            if step == int(config["training_steps"]):
                break
            batch_indices = torch.as_tensor(stream[step], device=x_train.device)
            direction, cache = _on_policy_direction(
                method,
                model,
                theta,
                x_train[batch_indices],
                y_train[batch_indices],
                step,
                cache,
                config,
            )
            theta = (theta - float(config["learning_rate"]) * direction).detach()
        final_loss = _loss_float(model, theta, x_train, y_train)
        final_accuracy = _accuracy(model, theta, x_test, y_test)
        endpoint_rows.append(
            {
                "seed": seed,
                "method": method,
                "initial_train_loss": initial_loss,
                "final_train_loss": final_loss,
                "train_loss_ratio": final_loss / initial_loss,
                "initial_test_accuracy": initial_accuracy,
                "final_test_accuracy": final_accuracy,
                "finite": bool(np.isfinite(final_loss) and np.isfinite(final_accuracy)),
            }
        )
    return history_rows, endpoint_rows


def _fingerprint() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    files = [
        "run_e002_pilot.py",
        "configs/e002_pilot.yaml",
        "src/e002_core.py",
        "src/e002_experiment.py",
        "src/e002_plotting.py",
    ]
    aggregate = hashlib.sha256()
    details = {}
    for relative in files:
        content = (root / relative).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        details[relative] = digest
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(content)
    return {"sha256": aggregate.hexdigest(), "files": details}


def _integrity(destination: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.name == "integrity.json" or ".matplotlib" in path.parts:
            continue
        content = path.read_bytes()
        files[str(path.relative_to(destination))] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return {"schema_version": "1.0", "file_count": len(files), "files": files}


def run_e002_pilot(
    overrides: dict[str, Any] | None = None,
    *,
    output_dir: str | Path,
    quick: bool = False,
) -> dict[str, Any]:
    config = _resolved_config(overrides, quick)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "checkpoints" / "shared").mkdir(parents=True, exist_ok=True)
    on_policy_directory = destination / "checkpoints" / "on_policy"
    on_policy_directory.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "resolved_config.json", config)

    os.environ.setdefault("PYTHONHASHSEED", str(config["seeds"][0]))
    torch.manual_seed(int(config["seeds"][0]))
    np.random.seed(int(config["seeds"][0]))
    torch.use_deterministic_algorithms(True)
    device, device_record = _runtime_device(config)
    started = time.time()
    model = FlatTanhMLP(hidden_width=int(config["hidden_width"]))

    data_payload: dict[str, np.ndarray] = {}
    batch_payload: dict[str, np.ndarray] = {}
    initial_states: dict[str, torch.Tensor] = {}
    raw_arrays: dict[str, np.ndarray] = {}
    training_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    hessian_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    taylor_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    checkpoint_metrics: dict[str, Any] = {}

    for seed in config["seeds"]:
        torch.manual_seed(int(seed))
        data = make_two_moons(
            train_samples=int(config["train_samples"]),
            validation_samples=int(config["validation_samples"]),
            test_samples=int(config["test_samples"]),
            noise=float(config["data_noise"]),
            label_flip=float(config["train_label_flip"]),
            seed=int(seed),
        )
        for name in (
            "x_train",
            "y_train",
            "x_validation",
            "y_validation",
            "x_test",
            "y_test",
            "flipped_indices",
        ):
            data_payload[f"seed{seed}__{name}"] = getattr(data, name)
        x_train = _tensor(data.x_train, device)
        y_train = _tensor(data.y_train, device, target=True)
        x_test = _tensor(data.x_test, device)
        y_test = _tensor(data.y_test, device, target=True)
        rng = np.random.default_rng(int(seed) + 101)
        training_stream = np.stack(
            [
                rng.choice(
                    int(config["train_samples"]),
                    size=int(config["batch_size"]),
                    replace=False,
                )
                for _ in range(int(config["training_steps"]))
            ]
        ).astype(np.int64)
        probe_indices = np.stack(
            [
                rng.choice(
                    int(config["train_samples"]),
                    size=int(config["batch_size"]),
                    replace=False,
                )
                for _ in range(int(config["probe_batches"]))
            ]
        ).astype(np.int64)
        batch_payload[f"seed{seed}__training_stream"] = training_stream
        batch_payload[f"seed{seed}__probe_indices"] = probe_indices
        initial = model.init_parameters(int(seed), device=device, dtype=torch.float64)
        initial_states[f"seed{seed}"] = initial.detach().cpu()
        history = _train_shared_anchor(
            model, initial, x_train, y_train, training_stream, config
        )
        checkpoint_metrics[str(seed)] = {}
        for checkpoint in config["checkpoint_steps"]:
            theta = history[int(checkpoint)]
            torch.save(
                {"theta": theta.detach().cpu(), "step": int(checkpoint), "seed": int(seed)},
                destination / "checkpoints" / "shared" / f"seed{seed}__step{checkpoint}.pt",
            )
            checkpoint_metrics[str(seed)][str(checkpoint)] = _checkpoint_probe(
                model=model,
                theta=theta,
                history=history,
                checkpoint=int(checkpoint),
                seed=int(seed),
                x_train=x_train,
                y_train=y_train,
                probe_indices=probe_indices,
                config=config,
                raw_arrays=raw_arrays,
                hessian_rows=hessian_rows,
                covariance_rows=covariance_rows,
                taylor_rows=taylor_rows,
                spectral_rows=spectral_rows,
                path_rows=path_rows,
                fidelity_rows=fidelity_rows,
            )
        temporal_rows.extend(
            _temporal_rows(model, history, int(seed), x_train, y_train, config)
        )
        current_training, current_endpoints = _train_on_policy(
            model=model,
            initial=initial,
            seed=int(seed),
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            stream=training_stream,
            config=config,
            checkpoint_directory=on_policy_directory,
        )
        training_rows.extend(current_training)
        endpoint_rows.extend(current_endpoints)

    np.savez_compressed(destination / "data.npz", **data_payload)
    np.savez_compressed(destination / "batch_indices.npz", **batch_payload)
    np.savez_compressed(destination / "batch_probes.npz", **raw_arrays)
    torch.save(initial_states, destination / "initial_state.pt")
    _write_csv(destination / "training_history.csv", training_rows)
    _write_csv(destination / "method_fidelity.csv", fidelity_rows)
    _write_csv(destination / "hessian_spectrum.csv", hessian_rows)
    _write_csv(destination / "covariance_summary.csv", covariance_rows)
    _write_csv(destination / "taylor_summary.csv", taylor_rows)
    _write_csv(destination / "signed_spectral_transfer.csv", spectral_rows)
    _write_csv(destination / "path_metrics.csv", path_rows)
    _write_csv(destination / "temporal_reuse.csv", temporal_rows)
    _write_csv(destination / "endpoint_summary.csv", endpoint_rows)

    fidelity_passed = all(bool(row["passed"]) for row in fidelity_rows)
    covariance_psd = all(
        float(row["covariance_min_eigenvalue"]) >= -float(row["covariance_psd_tolerance"])
        for row in covariance_rows
    )
    primary_rows = [
        row
        for row in taylor_rows
        if abs(float(row["eta"]) - float(config["primary_taylor_eta"])) <= 1e-15
    ]
    minimum_eta = min(float(value) for value in config["taylor_etas"])
    minimum_rows = [row for row in taylor_rows if abs(float(row["eta"]) - minimum_eta) <= 1e-15]
    primary_taylor = all(
        float(row["component_normalized_error_median"]) <= 0.10
        and float(row["component_normalized_error_p90"]) <= 0.25
        for row in primary_rows
    )
    minimum_taylor = all(
        float(row["component_normalized_error_median"]) <= 0.02
        and float(row["component_normalized_error_p90"]) <= 0.05
        for row in minimum_rows
    )
    sgd_endpoints = [row for row in endpoint_rows if row["method"] == "sgd"]
    sgd_smoke = all(
        bool(row["finite"])
        and float(row["train_loss_ratio"]) < 0.7
        and float(row["final_test_accuracy"]) >= 0.80
        for row in sgd_endpoints
    )
    any_underpowered = any(row["underpowered_relative_ci"] is True for row in covariance_rows)
    gates = {
        "gpu_identity_verified": bool(device_record["identity_verified"]),
        "method_fidelity_passed": fidelity_passed,
        "covariance_psd_passed": covariance_psd,
        "primary_eta_taylor_passed": primary_taylor,
        "minimum_eta_taylor_passed": minimum_taylor,
        "sgd_training_smoke_passed": sgd_smoke,
        "some_covariance_ci_underpowered": any_underpowered,
        "engineering_passed": bool(fidelity_passed and covariance_psd and sgd_smoke),
        "formal_e002_ready": bool(
            fidelity_passed
            and covariance_psd
            and primary_taylor
            and minimum_taylor
            and sgd_smoke
            and not any_underpowered
        ),
    }
    metrics = {
        "schema_version": "2.0-pilot",
        "experiment_id": config["experiment_id"],
        "gates": gates,
        "checkpoint_metrics": checkpoint_metrics,
        "endpoint_summary": endpoint_rows,
        "interpretation_scope": {
            "supports": [
                "method and numerical calibration on a 2-16-2 Two-Moons MLP",
                "shared-anchor descriptive covariance and one-step Taylor decomposition",
                "signed positive/negative Hessian and LookSAM temporal diagnostics",
            ],
            "does_not_support": [
                "causal attribution of final performance",
                "optimizer generalization ranking from two pilot seeds",
                "FashionMNIST or large-model extrapolation",
            ],
        },
    }
    _write_json(destination / "metrics.json", metrics)

    plot_files: list[str] = []
    if config["make_plots"]:
        plot_files = make_e002_plots(
            destination,
            taylor_rows=taylor_rows,
            covariance_rows=covariance_rows,
            spectral_rows=spectral_rows,
            path_rows=path_rows,
            temporal_rows=temporal_rows,
            training_rows=training_rows,
            primary_eta=float(config["primary_taylor_eta"]),
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
        device_record["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
        device_record["peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
        device_record["postflight"] = next(
            (row for row in _query_gpus() if row["index"] == int(config["physical_gpu_index"])),
            None,
        )
    manifest = {
        "schema_version": "2.0-pilot",
        "experiment_id": config["experiment_id"],
        "implementation_status": "E002_pilot_executed",
        "protocol": {
            "shared_anchor": "SGD anchors; paired virtual updates on identical probe batches",
            "on_policy": "same initialization and batch stream; trajectory sanity only",
            "gam": "paper Algorithm-1 exact-HVP same-batch alpha=1 reference; not accelerated GAM",
            "lookbehind": "faithful plain-SGD slow delta and path-mean surrogate are distinct labels",
            "looksam": "refresh is SAM; nonrefresh reuses cached orthogonal component",
        },
        "config": config,
        "code_fingerprint": _fingerprint(),
        "runtime": {
            "command": sys.argv,
            "elapsed_seconds": time.time() - started,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "numpy": np.__version__,
            "dtype": "float64",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "device": device_record,
        },
        "data_hashes": {key: _hash_array(value) for key, value in data_payload.items()},
        "batch_hashes": {key: _hash_array(value) for key, value in batch_payload.items()},
        "method_compute_semantics": {
            "sgd": {"loss_forwards": 1, "first_reverse": 1},
            "sam": {"loss_forwards": 2, "first_reverse": 2, "same_batch": True},
            "gam_exact_hvp_same_batch_alpha1": {
                "first_gradients": 2,
                "hvp_reverse": 2,
                "same_batch": True,
            },
            "lookbehind_faithful_k2_alpha0.5": {
                "fast_steps": 2,
                "paper_reference_forward_backward": 4,
                "batch_draws": 1,
            },
            "looksam_k5_alpha0.7": {
                "refresh_forward_backward": 2,
                "nonrefresh_forward_backward": 1,
                "amortized_forward_backward": 1.2,
            },
        },
        "products": {
            "tables": [
                "training_history.csv",
                "method_fidelity.csv",
                "hessian_spectrum.csv",
                "covariance_summary.csv",
                "taylor_summary.csv",
                "signed_spectral_transfer.csv",
                "path_metrics.csv",
                "temporal_reuse.csv",
                "endpoint_summary.csv",
            ],
            "figures": plot_files,
        },
    }
    _write_json(destination / "manifest.json", manifest)
    _write_json(destination / "integrity.json", _integrity(destination))
    return {"output_dir": str(destination), "metrics": metrics, "manifest": manifest}
