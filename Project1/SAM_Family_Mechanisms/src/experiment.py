from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .diagnostics import (
    cosine,
    descent_metrics,
    eigensystem_descending,
    hvp_scan,
    inner_problem_quality,
    nested_hessian_fit,
    path_metrics,
    path_novelty,
    signed_curvature_metrics,
    spectral_gain,
    top_subspace_energy,
)
from .operators import (
    Array,
    OperatorTrace,
    gam_trace,
    lookbehind_trace,
    matched_sam_trace,
    multistep_sam_trace,
    sam_trace,
    sgd_trace,
    unit,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_id": "E001_quadratic_operator",
    "seed": 3407,
    "dimension": 20,
    "lambda_min": 0.1,
    "lambda_max": 10.0,
    "rho_scales": [1e-4, 1e-3, 1e-2, 1e-1],
    "primary_rho_scale": 1e-2,
    "inner_steps": [2, 5],
    "path_protocols": ["fixed_step", "fixed_budget"],
    "dtype": "float64",
    "epsilon": 1e-15,
    "output_dir": "outputs/quadratic_operator",
    "make_plots": True,
}


SUMMARY_FIELDS = [
    "key",
    "method",
    "object_kind",
    "is_update",
    "protocol",
    "inner_steps",
    "rho_scale",
    "rho",
    "rho_step",
    "paired_path_rho_step",
    "oracle_radius",
    "gradient_evaluations",
    "hvp_evaluations",
    "backward_equivalents",
    "gradient_norm",
    "direction_norm",
    "object_norm",
    "correction_norm",
    "correction_gradient_ratio",
    "top_energy_1",
    "top_energy_5",
    "top_energy_10",
    "positive_curvature_quadratic",
    "negative_curvature_quadratic",
    "positive_rayleigh",
    "negative_rayleigh",
    "r2_1",
    "delta_r2_1",
    "r2_2",
    "delta_r2_2",
    "r2_3",
    "delta_r2_3",
    "q0",
    "q1",
    "q0_numerator",
    "q0_denominator",
    "q1_numerator",
    "q1_denominator",
    "quality_perturbation_kind",
    "associated_perturbation_radius",
    "endpoint_radius",
    "path_radius",
    "path_budget",
    "path_misalign",
    "last_average_difference",
    "path_novelty",
    "h1_residual",
    "matched_direction_cosine",
    "matched_correction_cosine",
    "rho_fit",
    "rho_fit_over_rho_eff",
    "matched_correction_norm_ratio",
    "matched_correction_relative_error",
    "direction_gradient_cosine",
    "descent_per_unit_norm",
    "positive_curvature_exposure",
    "safe_descent",
]


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _exact_int(value: Any, name: str) -> int:
    number = _finite_float(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


def _validate_config(config: dict[str, Any]) -> None:
    if str(config.get("dtype", "float64")) != "float64":
        raise ValueError("E001 intentionally supports float64 only")
    if float(config.get("epsilon", 1e-15)) != 1e-15:
        raise ValueError("E001 currently fixes epsilon at 1e-15 for schema-stable diagnostics")
    if int(config["dimension"]) < 3:
        raise ValueError("dimension must be at least three for H^1/H^2/H^3 fitting")
    if float(config["lambda_min"]) <= 0:
        raise ValueError("lambda_min must be positive in E001")
    if float(config["lambda_max"]) <= float(config["lambda_min"]):
        raise ValueError("lambda_max must exceed lambda_min")
    numeric = np.finfo(np.float64)
    if float(config["lambda_min"]) < 32.0 * numeric.tiny**0.5:
        raise ValueError("lambda_min is too small for stable float64 squared-norm diagnostics")
    if float(config["lambda_max"]) > numeric.max**0.5 / 32.0:
        raise ValueError("lambda_max is too large for stable H^2 float64 diagnostics")
    if float(config["primary_rho_scale"]) <= 0:
        raise ValueError("primary_rho_scale must be positive")
    rho_scales = [float(value) for value in config["rho_scales"]]
    if not rho_scales or any(value <= 0 for value in rho_scales):
        raise ValueError("rho_scales must contain positive values")
    minimum_resolvable_radius = 32.0 * np.finfo(np.float64).eps
    if float(config["primary_rho_scale"]) < minimum_resolvable_radius:
        raise ValueError("primary_rho_scale is below stable float64 resolution")
    if any(value < minimum_resolvable_radius for value in rho_scales):
        raise ValueError("rho_scales contain a radius below stable float64 resolution")
    if len(rho_scales) != len(set(rho_scales)):
        raise ValueError("rho_scales must not contain duplicates")
    inner_steps = [int(value) for value in config["inner_steps"]]
    if not inner_steps or any(value < 1 for value in inner_steps):
        raise ValueError("inner_steps must contain positive integers")
    protocols = list(config["path_protocols"])
    if not protocols:
        raise ValueError("path_protocols must not be empty")
    unsupported = set(protocols) - {"fixed_step", "fixed_budget"}
    if unsupported:
        raise ValueError(f"Unsupported path protocols: {sorted(unsupported)}")
    if len(protocols) != len(set(protocols)):
        raise ValueError("path_protocols must not contain duplicates")
    if len(inner_steps) != len(set(inner_steps)):
        raise ValueError("inner_steps must not contain duplicates")
    maximum_radius = max(
        max(rho_scales),
        float(config["primary_rho_scale"]) * max(inner_steps),
    )
    maximum_log_h_delta = math.log(float(config["lambda_max"])) + math.log(maximum_radius)
    if maximum_log_h_delta > 0.5 * math.log(np.finfo(np.float64).max) - math.log(32.0):
        raise ValueError("rho and lambda_max imply unstable float64 quadratic diagnostics")


def resolved_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        config.update(copy.deepcopy(overrides))
    config["dimension"] = _exact_int(config["dimension"], "dimension")
    config["lambda_min"] = _finite_float(config["lambda_min"], "lambda_min")
    config["lambda_max"] = _finite_float(config["lambda_max"], "lambda_max")
    for field in ("rho_scales", "inner_steps", "path_protocols"):
        if not isinstance(config.get(field), (list, tuple)):
            raise ValueError(f"{field} must be a list")
    config["rho_scales"] = [
        _finite_float(value, f"rho_scales[{index}]")
        for index, value in enumerate(config["rho_scales"])
    ]
    config["primary_rho_scale"] = _finite_float(
        config["primary_rho_scale"], "primary_rho_scale"
    )
    config["inner_steps"] = [
        _exact_int(value, f"inner_steps[{index}]")
        for index, value in enumerate(config["inner_steps"])
    ]
    config["path_protocols"] = [str(value) for value in config["path_protocols"]]
    config["dtype"] = str(config.get("dtype", "float64"))
    config["seed"] = _exact_int(config["seed"], "seed")
    config["epsilon"] = _finite_float(config.get("epsilon", 1e-15), "epsilon")
    config["make_plots"] = _strict_bool(config.get("make_plots", True), "make_plots")
    if not str(config.get("experiment_id", "")).strip():
        raise ValueError("experiment_id must be a non-empty string")
    if not str(config.get("output_dir", "")).strip():
        raise ValueError("output_dir must be a non-empty path")
    _validate_config(config)
    return config


def build_quadratic_problem(config: dict[str, Any]) -> tuple[Array, Array, Array]:
    """Construct the diagonal spectrum and equal-gradient-amplitude start."""
    log_eigenvalues = np.linspace(
        np.log(float(config["lambda_min"])),
        np.log(float(config["lambda_max"])),
        int(config["dimension"]),
        dtype=np.float64,
    )
    eigenvalues = np.exp(log_eigenvalues)
    hessian = np.diag(eigenvalues)
    # Subtract the largest log-weight before exponentiating.  This preserves
    # w_i proportional to 1/lambda_i without overflowing at wide spectra.
    log_weights = -log_eigenvalues
    weights = np.exp(log_weights - float(log_weights.max()))
    weights /= np.linalg.norm(weights)
    gradient = hessian @ weights
    if (
        not np.all(np.isfinite(hessian))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(gradient))
        or np.any(gradient == 0.0)
        or float(np.linalg.norm(gradient)) == 0.0
    ):
        raise ValueError("Derived quadratic problem is not representable stably in float64")
    return hessian, weights, gradient


def build_traces(
    hessian: Array,
    weights: Array,
    config: dict[str, Any],
) -> tuple[list[OperatorTrace], dict[str, float]]:
    weight_norm = float(np.linalg.norm(weights))
    rho = float(config["primary_rho_scale"]) * weight_norm
    traces = [sgd_trace(hessian, weights), sam_trace(hessian, weights, rho), gam_trace(hessian, weights, rho)]
    comparison_radii = {trace.key: rho for trace in traces}

    for protocol in config["path_protocols"]:
        for inner_steps in config["inner_steps"]:
            rho_step = rho if protocol == "fixed_step" else rho / int(inner_steps)
            path_budget = int(inner_steps) * rho_step
            multistep = multistep_sam_trace(
                hessian, weights, rho_step, int(inner_steps), protocol
            )
            lookbehind = lookbehind_trace(
                hessian, weights, rho_step, int(inner_steps), protocol
            )
            matched = matched_sam_trace(
                hessian, weights, rho_step, int(inner_steps), protocol
            )
            traces.extend([multistep, lookbehind, matched])
            comparison_radii[multistep.key] = path_budget
            comparison_radii[lookbehind.key] = path_budget
            # Use the paired Lookbehind path budget for the matched-SAM Q0/Q1
            # comparison.  Its actual perturbation radius remains separately
            # recorded as endpoint_radius.
            comparison_radii[matched.key] = path_budget
    return traces, comparison_radii


def _record_objects(traces: list[OperatorTrace]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trace in traces:
        object_kind = "final_regularizer" if trace.method == "gam" else "update_correction"
        records.append(
            {
                "key": trace.key,
                "method": trace.method,
                "object_kind": object_kind,
                "is_update": True,
                "is_correction": True,
                "vector": trace.correction,
                "quality_perturbation": None if trace.method == "gam" else trace.perturbation,
                "quality_perturbation_kind": None if trace.method == "gam" else (
                    "path_endpoint"
                    if trace.method in {"multistep_sam", "lookbehind"}
                    else "zero_perturbation"
                    if trace.method == "sgd"
                    else "sam_gradient_perturbation"
                ),
                "trace": trace,
            }
        )
        if trace.method == "gam":
            if trace.probe_direction is None or trace.probe_increment is None:
                raise RuntimeError("GAM trace is missing probe objects")
            records.extend(
                [
                    {
                        "key": "gam_probe_direction",
                        "method": "gam",
                        "object_kind": "probe_direction",
                        "is_update": False,
                        "is_correction": False,
                        "vector": trace.probe_direction,
                        "quality_perturbation": trace.perturbation,
                        "quality_perturbation_kind": "gam_probe_perturbation",
                        "trace": trace,
                    },
                    {
                        "key": "gam_probe_increment",
                        "method": "gam",
                        "object_kind": "probe_increment",
                        "is_update": False,
                        "is_correction": False,
                        "vector": trace.probe_increment,
                        "quality_perturbation": None,
                        "quality_perturbation_kind": None,
                        "trace": trace,
                    },
                ]
            )
    return records


def _summary_rows(
    records: list[dict[str, Any]],
    traces: list[OperatorTrace],
    comparison_radii: dict[str, float],
    hessian: Array,
    weights: Array,
    gradient: Array,
    eigenvectors: Array,
    rho_scale: float,
    rho: float,
) -> list[dict[str, Any]]:
    traces_by_key = {trace.key: trace for trace in traces}
    summaries = []
    for record in records:
        trace: OperatorTrace = record["trace"]
        vector: Array = record["vector"]
        is_update = bool(record["is_update"])
        is_correction = bool(record["is_correction"])
        path = path_metrics(trace)
        fit = nested_hessian_fit(vector, hessian, gradient, max_order=3)
        curvature = signed_curvature_metrics(vector, hessian)
        object_norm = float(np.linalg.norm(vector))
        correction_norm = object_norm if is_correction else None
        h1_residual = path_novelty(vector, hessian, gradient)
        quality_perturbation = record["quality_perturbation"]
        paired_path_rho_step = None
        if trace.method == "matched_sam":
            paired_path_rho_step = float(trace.metadata["rho_step"])
        row: dict[str, Any] = {
            "key": record["key"],
            "method": record["method"],
            "object_kind": record["object_kind"],
            "is_update": is_update,
            "protocol": trace.protocol,
            "inner_steps": trace.inner_steps,
            "rho_scale": float(rho_scale),
            "rho": float(rho),
            "rho_step": trace.rho_step,
            "paired_path_rho_step": paired_path_rho_step,
            "oracle_radius": comparison_radii[trace.key] if quality_perturbation is not None else None,
            "gradient_evaluations": trace.gradient_evaluations,
            "hvp_evaluations": trace.hvp_evaluations,
            "backward_equivalents": trace.backward_equivalents,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "direction_norm": float(np.linalg.norm(trace.direction)) if is_update else None,
            "object_norm": object_norm,
            "correction_norm": correction_norm,
            "correction_gradient_ratio": (
                correction_norm / float(np.linalg.norm(gradient))
                if correction_norm is not None
                else None
            ),
            "top_energy_1": top_subspace_energy(vector, eigenvectors, 1),
            "top_energy_5": top_subspace_energy(vector, eigenvectors, 5),
            "top_energy_10": top_subspace_energy(vector, eigenvectors, 10),
            **curvature,
            **fit,
            "q0": None,
            "q1": None,
            "q0_numerator": None,
            "q0_denominator": None,
            "q1_numerator": None,
            "q1_denominator": None,
            "quality_perturbation_kind": record["quality_perturbation_kind"],
            "associated_perturbation_radius": float(np.linalg.norm(trace.perturbation)),
            "endpoint_radius": path["endpoint_radius"] if is_update else None,
            "path_radius": path["path_radius"] if is_update else None,
            "path_budget": comparison_radii[trace.key] if (is_update or quality_perturbation is not None) else None,
            "path_misalign": path["path_misalign"] if is_update else None,
            "last_average_difference": path["last_average_difference"] if is_update else None,
            "path_novelty": (
                h1_residual
                if is_update and trace.method in {"multistep_sam", "lookbehind"}
                else None
            ),
            "h1_residual": h1_residual,
            "matched_direction_cosine": None,
            "matched_correction_cosine": None,
            "rho_fit": None,
            "rho_fit_over_rho_eff": None,
            "matched_correction_norm_ratio": None,
            "matched_correction_relative_error": None,
            "direction_gradient_cosine": None,
            "descent_per_unit_norm": None,
            "positive_curvature_exposure": None,
            "safe_descent": None,
        }
        if quality_perturbation is not None:
            quality = inner_problem_quality(
                hessian,
                weights,
                quality_perturbation,
                comparison_radii[trace.key],
            )
            for name in (
                "q0",
                "q1",
                "q0_numerator",
                "q0_denominator",
                "q1_numerator",
                "q1_denominator",
            ):
                row[name] = quality[name]
        if is_update:
            row.update(descent_metrics(trace.direction, gradient, hessian))

        if trace.method == "lookbehind" and is_update:
            matched_key = f"matched_sam_k{trace.inner_steps}_{trace.protocol}"
            matched = traces_by_key[matched_key]
            row["matched_direction_cosine"] = cosine(trace.direction, matched.direction)
            row["matched_correction_cosine"] = cosine(trace.correction, matched.correction)
            h1_reference = hessian @ unit(gradient)
            rho_fit = float(
                np.dot(trace.correction, h1_reference)
                / np.dot(h1_reference, h1_reference)
            )
            rho_eff = 0.5 * (int(trace.inner_steps) + 1) * float(trace.rho_step)
            row["rho_fit"] = rho_fit
            row["rho_fit_over_rho_eff"] = rho_fit / rho_eff
            matched_norm = float(np.linalg.norm(matched.correction))
            row["matched_correction_norm_ratio"] = correction_norm / matched_norm
            row["matched_correction_relative_error"] = float(
                np.linalg.norm(trace.correction - matched.correction) / correction_norm
            )
        summaries.append(row)
    return summaries


def _spectral_rows(
    records: list[dict[str, Any]],
    eigenvalues: Array,
    eigenvectors: Array,
    gradient: Array,
    rho_scale: float,
    rho: float,
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        trace: OperatorTrace = record["trace"]
        vector = np.asarray(record["vector"], dtype=np.float64)
        correction_projections = eigenvectors.T @ vector
        ghat_projections = eigenvectors.T @ unit(gradient)
        gains = spectral_gain(vector, gradient, eigenvectors)
        correction_is_zero = float(np.linalg.norm(vector)) <= 1e-30
        for index, (eigenvalue, gain, correction_projection, ghat_projection) in enumerate(
            zip(eigenvalues, gains, correction_projections, ghat_projections), start=1
        ):
            rows.append(
                {
                    "key": record["key"],
                    "method": record["method"],
                    "object_kind": record["object_kind"],
                    "is_update": bool(record["is_update"]),
                    "is_correction": bool(record["is_correction"]),
                    "protocol": trace.protocol,
                    "inner_steps": trace.inner_steps,
                    "rho_scale": float(rho_scale),
                    "rho": float(rho),
                    "eigen_index": index,
                    "eigenvalue": float(eigenvalue),
                    "object_projection": float(correction_projection),
                    "correction_projection": (
                        float(correction_projection) if record["is_correction"] else None
                    ),
                    "ghat_projection": float(ghat_projection),
                    "gain": None if correction_is_zero else float(gain),
                }
            )
    return rows


def _safe_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")


def _code_fingerprint() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    relative_paths = [
        "run_quadratic.py",
        "src/operators.py",
        "src/diagnostics.py",
        "src/experiment.py",
        "src/plotting.py",
    ]
    files = {}
    aggregate = hashlib.sha256()
    for relative in relative_paths:
        content = (package_root / relative).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        files[relative] = digest
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(content)
    return {"sha256": aggregate.hexdigest(), "files": files}


def _array_payload(
    hessian: Array,
    weights: Array,
    gradient: Array,
    eigenvalues: Array,
    eigenvectors: Array,
    traces: list[OperatorTrace],
) -> dict[str, Array]:
    payload: dict[str, Array] = {
        "hessian": hessian,
        "weights": weights,
        "gradient": gradient,
        "eigenvalues_descending": eigenvalues,
        "eigenvectors_descending": eigenvectors,
    }
    for trace in traces:
        prefix = _safe_key(trace.key)
        payload[f"{prefix}__direction"] = trace.direction
        payload[f"{prefix}__correction"] = trace.correction
        payload[f"{prefix}__perturbation"] = trace.perturbation
        if trace.path_points is not None:
            payload[f"{prefix}__path_points"] = trace.path_points
        if trace.path_gradients is not None:
            payload[f"{prefix}__path_gradients"] = trace.path_gradients
        if trace.probe_direction is not None:
            payload[f"{prefix}__probe_direction"] = trace.probe_direction
        if trace.probe_increment is not None:
            payload[f"{prefix}__probe_increment"] = trace.probe_increment
        if trace.final_regularizer is not None:
            payload[f"{prefix}__final_regularizer"] = trace.final_regularizer
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"Defined metric is non-finite: {number!r}")
        return number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            selected = {field: row.get(field) for field in fields}
            for field, value in selected.items():
                if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
                    raise ValueError(f"Non-finite CSV value for {field}: {value!r}")
            writer.writerow(selected)


def run_quadratic_experiment(
    overrides: dict[str, Any] | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = resolved_config(overrides)
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    destination = Path(config["output_dir"]).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    hessian, weights, gradient = build_quadratic_problem(config)
    eigenvalues, eigenvectors = eigensystem_descending(hessian)
    traces, comparison_radii = build_traces(hessian, weights, config)
    records = _record_objects(traces)
    rho_scale = float(config["primary_rho_scale"])
    rho = rho_scale * float(np.linalg.norm(weights))
    summaries = _summary_rows(
        records,
        traces,
        comparison_radii,
        hessian,
        weights,
        gradient,
        eigenvectors,
        rho_scale,
        rho,
    )
    spectrum = _spectral_rows(
        records, eigenvalues, eigenvectors, gradient, rho_scale, rho
    )
    hvp_rows = hvp_scan(hessian, weights, config["rho_scales"])

    sam = next(trace for trace in traces if trace.key == "sam")
    gam = next(trace for trace in traces if trace.key == "gam")
    sam_truth = rho * (hessian @ unit(gradient))
    gam_probe_truth = rho * (hessian @ unit(hessian @ unit(gradient)))
    if gam.probe_increment is None or gam.final_regularizer is None:
        raise RuntimeError("Incomplete GAM trace")
    h1_reference = hessian @ unit(gradient)
    checks = {
        "equal_gradient_coordinate_spread": float(np.ptp(np.abs(gradient))),
        "sam_identity_relative_error": float(
            np.linalg.norm(sam.correction - sam_truth) / np.linalg.norm(sam_truth)
        ),
        "gam_probe_h2_cosine": cosine(gam.probe_increment, gam_probe_truth),
        "gam_probe_h2_relative_error": float(
            np.linalg.norm(gam.probe_increment - gam_probe_truth) / np.linalg.norm(gam_probe_truth)
        ),
        "gam_final_h1_cosine": cosine(gam.final_regularizer, h1_reference),
        "gam_probe_final_cosine": cosine(gam.probe_increment, gam.final_regularizer),
    }
    metrics = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "problem": {
            "dimension": config["dimension"],
            "lambda_min": float(eigenvalues[-1]),
            "lambda_max": float(eigenvalues[0]),
            "weight_norm": float(np.linalg.norm(weights)),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "primary_rho": rho,
        },
        "checks": checks,
        "hvp_scan": hvp_rows,
        "method_summary": summaries,
    }
    manifest = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "implementation_status": "E001_implemented",
        "code_fingerprint": _code_fingerprint(),
        "config": config,
        "runtime": {
            "dtype": "float64",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "command": sys.argv,
        },
        "compute_budget": [
            {
                "key": trace.key,
                "method": trace.method,
                "gradient_evaluations": trace.gradient_evaluations,
                "hvp_evaluations": trace.hvp_evaluations,
                "backward_equivalents": trace.backward_equivalents,
                "accounting": "algorithmic equivalent; E001 itself uses analytic NumPy operators",
            }
            for trace in traces
        ],
        "objects": [
            {
                "key": record["key"],
                "method": record["method"],
                "object_kind": record["object_kind"],
                "is_update": record["is_update"],
                "is_correction": record["is_correction"],
                "quality_perturbation_kind": record["quality_perturbation_kind"],
            }
            for record in records
        ],
        "definitions": {
            "gradient": "g = H w",
            "direction": "d is the method's actual outer descent direction",
            "correction": "c = d - g",
            "gam_probe_increment": "grad(w + rho u_GAM) - g; H^2-type response",
            "gam_final_regularizer": "rho H grad_hat(w_adv); lowest-order H-type response",
            "fixed_step": "rho_step = primary rho, so path budget grows with k",
            "fixed_budget": "rho_step = primary rho / k, so path budget is fixed",
        },
        "files": {
            "arrays": "arrays.npz",
            "hvp_scan": "hvp_scan.csv",
            "spectral_gain": "spectral_gain.csv",
            "method_summary": "method_summary.csv",
            "metrics": "metrics.json",
            "figures": [
                "spectral_gain.png",
                "top_subspace_curvature.png",
                "hp_fit.png",
                "inner_quality.png",
            ] if config["make_plots"] else [],
        },
        "scope_boundary": (
            "E001 validates operator-level identities only; it cannot establish "
            "trajectory covariance, basin selection, generalization, or endpoint performance."
        ),
    }

    _write_json(destination / "manifest.json", manifest)
    _write_json(destination / "metrics.json", metrics)
    np.savez(
        destination / "arrays.npz",
        **_array_payload(hessian, weights, gradient, eigenvalues, eigenvectors, traces),
    )
    _write_csv(
        destination / "hvp_scan.csv",
        hvp_rows,
        [
            "rho_scale",
            "rho",
            "cos_hvp",
            "relative_error_hvp",
            "estimate_norm",
            "truth_norm",
        ],
    )
    _write_csv(
        destination / "spectral_gain.csv",
        spectrum,
        [
            "key",
            "method",
            "object_kind",
            "is_update",
            "is_correction",
            "protocol",
            "inner_steps",
            "rho_scale",
            "rho",
            "eigen_index",
            "eigenvalue",
            "object_projection",
            "correction_projection",
            "ghat_projection",
            "gain",
        ],
    )
    _write_csv(destination / "method_summary.csv", summaries, SUMMARY_FIELDS)

    if config["make_plots"]:
        from .plotting import make_quadratic_plots

        make_quadratic_plots(destination, spectrum, summaries)
    return {
        "output_dir": str(destination),
        "config": config,
        "metrics": metrics,
        "manifest": manifest,
    }
