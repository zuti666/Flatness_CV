from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .diagnostics import (
    cosine,
    descent_metrics,
    eigensystem_descending,
    inner_problem_quality,
    nested_hessian_fit,
    path_metrics,
    path_novelty,
    signed_curvature_metrics,
    spectral_gain,
    top_subspace_energy,
)
from .experiment import (
    build_quadratic_problem,
    build_traces,
    resolved_config,
    run_quadratic_experiment,
)
from .operators import (
    Array,
    OperatorTrace,
    gam_trace,
    lookbehind_trace,
    matched_sam_trace,
    multistep_sam_trace,
    sam_trace,
    unit,
)


DEFAULT_RHO_VALUES = (1e-4, 1e-3, 1e-2, 5e-2, 1e-1)
DEFAULT_CONDITION_NUMBERS = (4.0, 100.0, 10_000.0)
DEFAULT_DIMENSIONS = (10, 20, 50, 100)
DEFAULT_INNER_STEPS = (1, 2, 3, 5, 10)
DEFAULT_STRENGTH_TARGETS = (0.1, 0.25, 0.5)


SENSITIVITY_FIELDS = [
    "case_id",
    "factor",
    "factor_value",
    "dimension",
    "lambda_min",
    "lambda_max",
    "condition_number",
    "primary_rho",
    "sam_reference_ratio",
    "key",
    "method",
    "object_kind",
    "is_update",
    "protocol",
    "inner_steps",
    "spectral_log_slope",
    "spectral_log_r2",
    "correction_gradient_ratio",
    "path_budget",
    "local_path_strength",
    "path_novelty",
    "h1_residual",
    "matched_correction_cosine",
    "q0",
    "q1",
    "r2_1",
    "delta_r2_2",
    "direction_gradient_cosine",
    "descent_per_unit_norm",
]


STRENGTH_FIELDS = [
    "key",
    "method",
    "object_kind",
    "protocol",
    "inner_steps",
    "target_correction_gradient_ratio",
    "achieved_correction_gradient_ratio",
    "matching_relative_error",
    "native_correction_gradient_ratio",
    "matched_primary_rho",
    "native_primary_rho",
    "radius_multiplier",
    "rho_step",
    "associated_perturbation_radius",
    "endpoint_radius",
    "path_radius",
    "path_budget",
    "spectral_log_slope",
    "spectral_log_r2",
    "correction_gradient_cosine",
    "correction_h1_cosine",
    "top_energy_1",
    "top_energy_5",
    "positive_rayleigh",
    "r2_1",
    "delta_r2_2",
    "path_novelty",
    "h1_residual",
    "matched_direction_cosine",
    "matched_correction_cosine",
    "q0",
    "q1",
    "direction_gradient_cosine",
    "descent_per_unit_norm",
    "positive_curvature_exposure",
    "safe_descent",
]


EFFECTIVE_RADIUS_FIELDS = [
    "key",
    "method",
    "protocol",
    "inner_steps",
    "effective_rho",
    "rho_step",
    "path_budget",
    "correction_gradient_ratio",
    "spectral_log_slope",
    "spectral_log_r2",
    "path_novelty",
    "matched_direction_cosine",
    "matched_correction_cosine",
    "q0",
    "q1",
    "direction_gradient_cosine",
    "descent_per_unit_norm",
]


QUALITY_FIELDS = [
    "key",
    "method",
    "quality_perturbation_kind",
    "protocol",
    "inner_steps",
    "oracle_radius",
    "candidate_radius",
    "radius_utilization",
    "raw_q0",
    "raw_q1",
    "boundary_q0",
    "boundary_q1",
    "own_radius_q0",
    "own_radius_q1",
]


INITIALIZATION_SAMPLE_FIELDS = [
    "initialization",
    "sample_index",
    "rho",
    "gradient_norm",
    "worst_case_radius_strength",
    "sam_correction_gradient_ratio",
    "sam_q0",
    "sam_q1",
    "gam_correction_gradient_ratio",
    "gam_probe_q0",
    "gam_probe_q1",
    "lookbehind_correction_gradient_ratio",
    "lookbehind_path_novelty",
    "lookbehind_matched_correction_cosine",
]


INITIALIZATION_SUMMARY_FIELDS = [
    "initialization",
    "count",
    "metric",
    "mean",
    "median",
    "q10",
    "q90",
]


@dataclass(frozen=True)
class SweepCase:
    case_id: str
    factor: str
    factor_value: float | int
    overrides: dict[str, Any]


def _slug_number(value: float | int) -> str:
    return format(float(value), ".8g").replace("-", "m").replace(".", "p").replace("+", "")


def build_sweep_cases(
    *,
    rho_values: Iterable[float] = DEFAULT_RHO_VALUES,
    condition_numbers: Iterable[float] = DEFAULT_CONDITION_NUMBERS,
    dimensions: Iterable[int] = DEFAULT_DIMENSIONS,
    inner_steps: Iterable[int] = DEFAULT_INNER_STEPS,
) -> list[SweepCase]:
    """Build four one-factor-at-a-time branches around the E001 baseline."""
    cases: list[SweepCase] = []
    for value in rho_values:
        value = float(value)
        cases.append(
            SweepCase(
                case_id=f"rho_{_slug_number(value)}",
                factor="rho",
                factor_value=value,
                overrides={"primary_rho_scale": value},
            )
        )
    for value in condition_numbers:
        value = float(value)
        if not math.isfinite(value) or value <= 1.0:
            raise ValueError("condition numbers must be finite and greater than one")
        root = math.sqrt(value)
        cases.append(
            SweepCase(
                case_id=f"condition_{_slug_number(value)}",
                factor="condition",
                factor_value=value,
                overrides={"lambda_min": 1.0 / root, "lambda_max": root},
            )
        )
    for value in dimensions:
        if isinstance(value, bool) or int(value) != value or int(value) < 3:
            raise ValueError("dimensions must contain integers of at least three")
        value = int(value)
        cases.append(
            SweepCase(
                case_id=f"dimension_{value}",
                factor="dimension",
                factor_value=value,
                overrides={"dimension": value},
            )
        )
    for value in inner_steps:
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError("inner steps must contain positive integers")
        value = int(value)
        cases.append(
            SweepCase(
                case_id=f"inner_steps_{value}",
                factor="inner_steps",
                factor_value=value,
                overrides={"inner_steps": [value]},
            )
        )
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("sensitivity case identifiers must be unique")
    return cases


def loglog_spectral_fit(rows: Iterable[dict[str, Any]]) -> dict[str, float | None]:
    pairs = []
    for row in rows:
        eigenvalue = row.get("eigenvalue")
        gain = row.get("gain")
        if eigenvalue in (None, "") or gain in (None, ""):
            continue
        eigenvalue = float(eigenvalue)
        gain = float(gain)
        if math.isfinite(eigenvalue) and math.isfinite(gain) and eigenvalue > 0 and gain > 0:
            pairs.append((eigenvalue, gain))
    if len(pairs) < 2:
        return {"spectral_log_slope": None, "spectral_log_r2": None}
    x = np.log(np.asarray([item[0] for item in pairs], dtype=np.float64))
    y = np.log(np.asarray([item[1] for item in pairs], dtype=np.float64))
    design = np.column_stack([x, np.ones_like(x)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coefficients
    residual = float(np.dot(y - prediction, y - prediction))
    centered = y - float(np.mean(y))
    total = float(np.dot(centered, centered))
    r_squared = 1.0 if total <= 1e-30 and residual <= 1e-30 else 1.0 - residual / total
    return {
        "spectral_log_slope": float(coefficients[0]),
        "spectral_log_r2": float(min(1.0, max(0.0, r_squared))),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _spectral_fits(path: Path) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _read_csv(path):
        grouped.setdefault(row["key"], []).append(row)
    return {key: loglog_spectral_fit(rows) for key, rows in grouped.items()}


def _trace_at_primary_rho(
    template: OperatorTrace,
    hessian: Array,
    weights: Array,
    primary_rho: float,
) -> OperatorTrace:
    if template.method == "sam":
        return sam_trace(hessian, weights, primary_rho)
    if template.method == "gam":
        return gam_trace(hessian, weights, primary_rho)
    if template.inner_steps is None or template.protocol is None:
        raise ValueError(f"path metadata missing for {template.key}")
    inner_steps = int(template.inner_steps)
    protocol = str(template.protocol)
    rho_step = primary_rho if protocol == "fixed_step" else primary_rho / inner_steps
    if template.method == "multistep_sam":
        return multistep_sam_trace(hessian, weights, rho_step, inner_steps, protocol)
    if template.method == "lookbehind":
        return lookbehind_trace(hessian, weights, rho_step, inner_steps, protocol)
    if template.method == "matched_sam":
        return matched_sam_trace(hessian, weights, rho_step, inner_steps, protocol)
    raise ValueError(f"unsupported strength-matching method: {template.method}")


def _correction_ratio(trace: OperatorTrace, gradient_norm: float) -> float:
    return float(np.linalg.norm(trace.correction)) / gradient_norm


def match_primary_rho(
    template: OperatorTrace,
    hessian: Array,
    weights: Array,
    target_ratio: float,
    *,
    initial_rho: float,
    relative_tolerance: float = 1e-10,
    max_iterations: int = 160,
) -> tuple[float, OperatorTrace]:
    """Solve for the method's native radius at a common correction strength."""
    target_ratio = float(target_ratio)
    if not math.isfinite(target_ratio) or target_ratio <= 0:
        raise ValueError("target correction ratio must be positive and finite")
    if not math.isfinite(initial_rho) or initial_rho <= 0:
        raise ValueError("initial rho must be positive and finite")
    gradient_norm = float(np.linalg.norm(hessian @ weights))
    low = 0.0
    high = float(initial_rho)
    high_trace = _trace_at_primary_rho(template, hessian, weights, high)
    high_ratio = _correction_ratio(high_trace, gradient_norm)
    previous_ratio = 0.0
    for _ in range(100):
        if high_ratio >= target_ratio:
            break
        if high_ratio + 1e-14 < previous_ratio:
            raise RuntimeError(f"non-monotone correction strength while bracketing {template.key}")
        previous_ratio = high_ratio
        high *= 2.0
        if not math.isfinite(high):
            raise RuntimeError(f"could not bracket correction strength for {template.key}")
        high_trace = _trace_at_primary_rho(template, hessian, weights, high)
        high_ratio = _correction_ratio(high_trace, gradient_norm)
    else:
        raise RuntimeError(f"could not bracket correction strength for {template.key}")

    best_rho = high
    best_trace = high_trace
    best_error = abs(high_ratio - target_ratio)
    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        trace = _trace_at_primary_rho(template, hessian, weights, middle)
        ratio = _correction_ratio(trace, gradient_norm)
        error = abs(ratio - target_ratio)
        if error < best_error:
            best_rho, best_trace, best_error = middle, trace, error
        if error <= relative_tolerance * target_ratio:
            return middle, trace
        if ratio < target_ratio:
            low = middle
        else:
            high = middle
    if best_error > relative_tolerance * target_ratio * 10.0:
        raise RuntimeError(f"correction-strength match did not converge for {template.key}")
    return best_rho, best_trace


def _trace_spectral_fit(
    trace: OperatorTrace,
    gradient: Array,
    eigenvalues: Array,
    eigenvectors: Array,
) -> dict[str, float | None]:
    gains = spectral_gain(trace.correction, gradient, eigenvectors)
    return loglog_spectral_fit(
        {"eigenvalue": value, "gain": gain}
        for value, gain in zip(eigenvalues, gains)
    )


def strength_matched_rows(
    config: dict[str, Any],
    targets: Iterable[float] = DEFAULT_STRENGTH_TARGETS,
) -> list[dict[str, Any]]:
    """Re-run every update operator at exactly matched ``||c||/||g||``."""
    config = resolved_config(config)
    hessian, weights, gradient = build_quadratic_problem(config)
    eigenvalues, eigenvectors = eigensystem_descending(hessian)
    templates, _ = build_traces(hessian, weights, config)
    templates = [trace for trace in templates if trace.method != "sgd"]
    gradient_norm = float(np.linalg.norm(gradient))
    native_primary_rho = float(config["primary_rho_scale"]) * float(np.linalg.norm(weights))
    native_ratios = {
        trace.key: _correction_ratio(trace, gradient_norm) for trace in templates
    }
    rows: list[dict[str, Any]] = []
    for target in targets:
        target = float(target)
        matched: dict[str, OperatorTrace] = {}
        radii: dict[str, float] = {}
        for template in templates:
            primary_rho, trace = match_primary_rho(
                template,
                hessian,
                weights,
                target,
                initial_rho=native_primary_rho,
            )
            matched[template.key] = trace
            radii[template.key] = primary_rho
        for template in templates:
            trace = matched[template.key]
            primary_rho = radii[template.key]
            achieved = _correction_ratio(trace, gradient_norm)
            path = path_metrics(trace)
            fit = nested_hessian_fit(trace.correction, hessian, gradient, max_order=3)
            spectral_fit = _trace_spectral_fit(
                trace, gradient, eigenvalues, eigenvectors
            )
            curvature = signed_curvature_metrics(trace.correction, hessian)
            descent = descent_metrics(trace.direction, gradient, hessian)
            inner_steps = trace.inner_steps
            if trace.protocol == "fixed_step" and inner_steps is not None:
                path_budget = float(inner_steps) * primary_rho
            elif trace.protocol == "fixed_budget":
                path_budget = primary_rho
            else:
                path_budget = primary_rho
            quality: dict[str, float | None] = {"q0": None, "q1": None}
            if trace.method != "gam":
                quality = inner_problem_quality(
                    hessian, weights, trace.perturbation, path_budget
                )
            matched_direction = None
            matched_correction = None
            if trace.method == "lookbehind":
                control_key = f"matched_sam_k{inner_steps}_{trace.protocol}"
                control = matched[control_key]
                matched_direction = cosine(trace.direction, control.direction)
                matched_correction = cosine(trace.correction, control.correction)
            rows.append(
                {
                    "key": template.key,
                    "method": trace.method,
                    "object_kind": (
                        "final_regularizer" if trace.method == "gam" else "update_correction"
                    ),
                    "protocol": trace.protocol,
                    "inner_steps": trace.inner_steps,
                    "target_correction_gradient_ratio": target,
                    "achieved_correction_gradient_ratio": achieved,
                    "matching_relative_error": abs(achieved - target) / target,
                    "native_correction_gradient_ratio": native_ratios[template.key],
                    "matched_primary_rho": primary_rho,
                    "native_primary_rho": native_primary_rho,
                    "radius_multiplier": primary_rho / native_primary_rho,
                    "rho_step": trace.rho_step,
                    "associated_perturbation_radius": float(np.linalg.norm(trace.perturbation)),
                    "endpoint_radius": path["endpoint_radius"],
                    "path_radius": path["path_radius"],
                    "path_budget": path_budget,
                    **spectral_fit,
                    "correction_gradient_cosine": cosine(trace.correction, gradient),
                    "correction_h1_cosine": cosine(
                        trace.correction, hessian @ unit(gradient)
                    ),
                    "top_energy_1": top_subspace_energy(trace.correction, eigenvectors, 1),
                    "top_energy_5": top_subspace_energy(trace.correction, eigenvectors, 5),
                    "positive_rayleigh": curvature["positive_rayleigh"],
                    "r2_1": fit["r2_1"],
                    "delta_r2_2": fit["delta_r2_2"],
                    "path_novelty": (
                        path_novelty(trace.correction, hessian, gradient)
                        if trace.method in {"multistep_sam", "lookbehind"}
                        else None
                    ),
                    "h1_residual": path_novelty(
                        trace.correction, hessian, gradient
                    ),
                    "matched_direction_cosine": matched_direction,
                    "matched_correction_cosine": matched_correction,
                    "q0": quality["q0"],
                    "q1": quality["q1"],
                    **descent,
                }
            )
    return rows


def fixed_effective_radius_rows(
    config: dict[str, Any],
    inner_steps_values: Iterable[int] = DEFAULT_INNER_STEPS,
    *,
    effective_rho: float | None = None,
) -> list[dict[str, Any]]:
    """Vary path resolution while holding Lookbehind's first-order radius fixed.

    The protocol sets ``rho_step = 2 * rho_eff / (k + 1)``.  Therefore the
    matched-SAM control is the same SAM operator for every ``k``.  This is a
    complementary control to fixed path budget, not a replacement for it.
    """
    config = resolved_config(config)
    hessian, weights, gradient = build_quadratic_problem(config)
    eigenvalues, eigenvectors = eigensystem_descending(hessian)
    if effective_rho is None:
        effective_rho = float(config["primary_rho_scale"]) * float(np.linalg.norm(weights))
    effective_rho = float(effective_rho)
    if not math.isfinite(effective_rho) or effective_rho <= 0:
        raise ValueError("effective rho must be positive and finite")
    rows: list[dict[str, Any]] = []
    for inner_steps in inner_steps_values:
        if isinstance(inner_steps, bool) or int(inner_steps) != inner_steps or int(inner_steps) < 1:
            raise ValueError("inner steps must contain positive integers")
        inner_steps = int(inner_steps)
        rho_step = 2.0 * effective_rho / (inner_steps + 1)
        path_budget = inner_steps * rho_step
        multistep = multistep_sam_trace(
            hessian, weights, rho_step, inner_steps, "fixed_effective_radius"
        )
        lookbehind = lookbehind_trace(
            hessian, weights, rho_step, inner_steps, "fixed_effective_radius"
        )
        matched = matched_sam_trace(
            hessian, weights, rho_step, inner_steps, "fixed_effective_radius"
        )
        for trace in (multistep, lookbehind, matched):
            spectral_fit = _trace_spectral_fit(
                trace, gradient, eigenvalues, eigenvectors
            )
            quality = inner_problem_quality(
                hessian, weights, trace.perturbation, path_budget
            )
            descent = descent_metrics(trace.direction, gradient, hessian)
            rows.append(
                {
                    "key": trace.key,
                    "method": trace.method,
                    "protocol": trace.protocol,
                    "inner_steps": inner_steps,
                    "effective_rho": effective_rho,
                    "rho_step": rho_step,
                    "path_budget": path_budget,
                    "correction_gradient_ratio": (
                        float(np.linalg.norm(trace.correction))
                        / float(np.linalg.norm(gradient))
                    ),
                    **spectral_fit,
                    "path_novelty": path_novelty(
                        trace.correction, hessian, gradient
                    ),
                    "matched_direction_cosine": (
                        cosine(trace.direction, matched.direction)
                        if trace.method == "lookbehind"
                        else None
                    ),
                    "matched_correction_cosine": (
                        cosine(trace.correction, matched.correction)
                        if trace.method == "lookbehind"
                        else None
                    ),
                    "q0": quality["q0"],
                    "q1": quality["q1"],
                    **descent,
                }
            )
    return rows


def quality_decomposition_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Separate candidate direction quality from radial budget utilization."""
    config = resolved_config(config)
    hessian, weights, _ = build_quadratic_problem(config)
    traces, comparison_radii = build_traces(hessian, weights, config)
    rows: list[dict[str, Any]] = []
    for trace in traces:
        if trace.method == "sgd":
            continue
        key = trace.key
        quality_kind = (
            "gam_probe_perturbation" if trace.method == "gam"
            else "path_endpoint" if trace.method in {"multistep_sam", "lookbehind"}
            else "sam_gradient_perturbation"
        )
        if trace.method == "gam":
            key = "gam_probe_direction"
        perturbation = trace.perturbation
        candidate_radius = float(np.linalg.norm(perturbation))
        oracle_radius = float(comparison_radii[trace.key])
        raw = inner_problem_quality(
            hessian, weights, perturbation, oracle_radius
        )
        boundary_perturbation = oracle_radius * unit(perturbation)
        boundary = inner_problem_quality(
            hessian, weights, boundary_perturbation, oracle_radius
        )
        own = inner_problem_quality(
            hessian, weights, perturbation, candidate_radius
        )
        rows.append(
            {
                "key": key,
                "method": trace.method,
                "quality_perturbation_kind": quality_kind,
                "protocol": trace.protocol,
                "inner_steps": trace.inner_steps,
                "oracle_radius": oracle_radius,
                "candidate_radius": candidate_radius,
                "radius_utilization": candidate_radius / oracle_radius,
                "raw_q0": raw["q0"],
                "raw_q1": raw["q1"],
                "boundary_q0": boundary["q0"],
                "boundary_q1": boundary["q1"],
                "own_radius_q0": own["q0"],
                "own_radius_q1": own["q1"],
            }
        )
    return rows


def initialization_ensemble_rows(
    config: dict[str, Any],
    *,
    sample_count: int = 100,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare the engineered start with random parameter/gradient envelopes."""
    config = resolved_config(config)
    if isinstance(sample_count, bool) or int(sample_count) != sample_count or sample_count < 1:
        raise ValueError("initialization sample count must be a positive integer")
    sample_count = int(sample_count)
    if seed is None:
        seed = int(config["seed"])
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("initialization seed must be an integer")
    rng = np.random.default_rng(int(seed))
    hessian, engineered_weights, _ = build_quadratic_problem(config)
    dimension = hessian.shape[0]
    primary_scale = float(config["primary_rho_scale"])
    lambda_max = float(np.linalg.eigvalsh(hessian)[-1])

    weights_by_kind: dict[str, list[Array]] = {"equal_gradient": [engineered_weights]}
    random_weights = []
    random_gradients = []
    for _ in range(sample_count):
        random_weights.append(unit(rng.normal(size=dimension)))
        requested_gradient = rng.normal(size=dimension)
        weights_from_gradient = np.linalg.solve(hessian, requested_gradient)
        random_gradients.append(unit(weights_from_gradient))
    weights_by_kind["random_w"] = random_weights
    weights_by_kind["random_g"] = random_gradients

    rows: list[dict[str, Any]] = []
    for initialization, weight_samples in weights_by_kind.items():
        for index, weights in enumerate(weight_samples):
            gradient = hessian @ weights
            gradient_norm = float(np.linalg.norm(gradient))
            rho = primary_scale * float(np.linalg.norm(weights))
            sam = sam_trace(hessian, weights, rho)
            gam = gam_trace(hessian, weights, rho)
            inner_steps = 5
            lookbehind = lookbehind_trace(
                hessian,
                weights,
                rho / inner_steps,
                inner_steps,
                "fixed_budget",
            )
            matched = matched_sam_trace(
                hessian,
                weights,
                rho / inner_steps,
                inner_steps,
                "fixed_budget",
            )
            sam_quality = inner_problem_quality(hessian, weights, sam.perturbation, rho)
            gam_quality = inner_problem_quality(hessian, weights, gam.perturbation, rho)
            rows.append(
                {
                    "initialization": initialization,
                    "sample_index": index,
                    "rho": rho,
                    "gradient_norm": gradient_norm,
                    "worst_case_radius_strength": rho * lambda_max / gradient_norm,
                    "sam_correction_gradient_ratio": (
                        float(np.linalg.norm(sam.correction)) / gradient_norm
                    ),
                    "sam_q0": sam_quality["q0"],
                    "sam_q1": sam_quality["q1"],
                    "gam_correction_gradient_ratio": (
                        float(np.linalg.norm(gam.correction)) / gradient_norm
                    ),
                    "gam_probe_q0": gam_quality["q0"],
                    "gam_probe_q1": gam_quality["q1"],
                    "lookbehind_correction_gradient_ratio": (
                        float(np.linalg.norm(lookbehind.correction)) / gradient_norm
                    ),
                    "lookbehind_path_novelty": path_novelty(
                        lookbehind.correction, hessian, gradient
                    ),
                    "lookbehind_matched_correction_cosine": cosine(
                        lookbehind.correction, matched.correction
                    ),
                }
            )

    summaries: list[dict[str, Any]] = []
    metric_fields = [
        field
        for field in INITIALIZATION_SAMPLE_FIELDS
        if field not in {"initialization", "sample_index"}
    ]
    for initialization in weights_by_kind:
        selected = [row for row in rows if row["initialization"] == initialization]
        for metric in metric_fields:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            summaries.append(
                {
                    "initialization": initialization,
                    "count": len(values),
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "q10": float(np.quantile(values, 0.1)),
                    "q90": float(np.quantile(values, 0.9)),
                }
            )
    return rows, summaries


def _case_summary_rows(
    case: SweepCase,
    result: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    config = result["config"]
    summaries = result["metrics"]["method_summary"]
    fits = _spectral_fits(run_dir / "spectral_gain.csv")
    sam_reference = next(
        float(row["correction_gradient_ratio"])
        for row in summaries
        if row["key"] == "sam"
    )
    rows = []
    for summary in summaries:
        path_budget = summary["path_budget"]
        local_path_strength = None
        if path_budget is not None:
            primary_rho = float(config["primary_rho_scale"])
            local_path_strength = float(path_budget) * sam_reference / primary_rho
        rows.append(
            {
                "case_id": case.case_id,
                "factor": case.factor,
                "factor_value": case.factor_value,
                "dimension": config["dimension"],
                "lambda_min": config["lambda_min"],
                "lambda_max": config["lambda_max"],
                "condition_number": float(config["lambda_max"]) / float(config["lambda_min"]),
                "primary_rho": float(config["primary_rho_scale"]),
                "sam_reference_ratio": sam_reference,
                "key": summary["key"],
                "method": summary["method"],
                "object_kind": summary["object_kind"],
                "is_update": summary["is_update"],
                "protocol": summary["protocol"],
                "inner_steps": summary["inner_steps"],
                **fits[summary["key"]],
                "correction_gradient_ratio": summary["correction_gradient_ratio"],
                "path_budget": path_budget,
                "local_path_strength": local_path_strength,
                "path_novelty": summary["path_novelty"],
                "h1_residual": summary["h1_residual"],
                "matched_correction_cosine": summary["matched_correction_cosine"],
                "q0": summary["q0"],
                "q1": summary["q1"],
                "r2_1": summary["r2_1"],
                "delta_r2_2": summary["delta_r2_2"],
                "direction_gradient_cosine": summary["direction_gradient_cosine"],
                "descent_per_unit_norm": summary["descent_per_unit_norm"],
            }
        )
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"cannot serialize non-finite value {value!r}")
        return value
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"unsupported JSON value {type(value)!r}")


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
            writer.writerow({field: row.get(field) for field in fields})


def _fingerprint() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    paths = [
        "run_e001_sensitivity.py",
        "src/sensitivity.py",
        "src/experiment.py",
        "src/operators.py",
        "src/diagnostics.py",
    ]
    aggregate = hashlib.sha256()
    files: dict[str, str] = {}
    for relative in paths:
        content = (package_root / relative).read_bytes()
        files[relative] = hashlib.sha256(content).hexdigest()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(content)
    return {"sha256": aggregate.hexdigest(), "files": files}


def _make_plots(
    output_dir: Path,
    sensitivity_rows: list[dict[str, Any]],
    strength_rows: list[dict[str, Any]],
    effective_radius_rows: list[dict[str, Any]],
    initialization_rows: list[dict[str, Any]],
) -> None:
    import os

    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = {
        "sam": "SAM",
        "gam": "GAM final",
        "lookbehind_k5_fixed_step": "LB k=5 step",
        "lookbehind_k5_fixed_budget": "LB k=5 budget",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    for key, label in selected.items():
        rows = [
            row for row in sensitivity_rows
            if row["factor"] == "rho" and row["key"] == key
        ]
        if rows:
            rows.sort(key=lambda row: float(row["factor_value"]))
            axes[0, 0].semilogx(
                [float(row["factor_value"]) for row in rows],
                [float(row["spectral_log_slope"]) for row in rows],
                marker="o",
                label=label,
            )
    axes[0, 0].set(title="Radius: spectral slope", xlabel="primary rho", ylabel="log-log slope")
    axes[0, 0].legend(fontsize=8)

    for key, label in selected.items():
        rows = [
            row for row in sensitivity_rows
            if row["factor"] == "condition" and row["key"] == key
        ]
        if rows and all(row["correction_gradient_ratio"] is not None for row in rows):
            rows.sort(key=lambda row: float(row["factor_value"]))
            axes[0, 1].loglog(
                [float(row["factor_value"]) for row in rows],
                [float(row["correction_gradient_ratio"]) for row in rows],
                marker="o",
                label=label,
            )
    axes[0, 1].set(title="Conditioning: correction strength", xlabel="condition number", ylabel="||c|| / ||g||")
    axes[0, 1].legend(fontsize=8)

    for protocol, label in (("fixed_step", "fixed step"), ("fixed_budget", "fixed budget")):
        rows = [
            row for row in sensitivity_rows
            if row["factor"] == "inner_steps"
            and row["method"] == "lookbehind"
            and row["protocol"] == protocol
        ]
        rows.sort(key=lambda row: float(row["factor_value"]))
        axes[1, 0].plot(
            [float(row["factor_value"]) for row in rows],
            [float(row["path_novelty"]) for row in rows],
            marker="o",
            label=label,
        )
    effective_rows = [
        row for row in effective_radius_rows if row["method"] == "lookbehind"
    ]
    effective_rows.sort(key=lambda row: int(row["inner_steps"]))
    axes[1, 0].plot(
        [int(row["inner_steps"]) for row in effective_rows],
        [float(row["path_novelty"]) for row in effective_rows],
        marker="o",
        label="fixed effective radius",
    )
    axes[1, 0].set(title="Lookbehind: path novelty", xlabel="inner steps k", ylabel="novelty")
    axes[1, 0].legend(fontsize=8)

    for key, label in selected.items():
        rows = [
            row for row in sensitivity_rows
            if row["factor"] == "dimension" and row["key"] == key
        ]
        if rows:
            rows.sort(key=lambda row: float(row["factor_value"]))
            axes[1, 1].plot(
                [float(row["factor_value"]) for row in rows],
                [float(row["spectral_log_slope"]) for row in rows],
                marker="o",
                label=label,
            )
    axes[1, 1].set(title="Dimension robustness", xlabel="dimension", ylabel="log-log slope")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "sensitivity_overview.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.2, 6.0))
    selected_strength = {
        "sam": "SAM",
        "gam": "GAM final",
        "ms_sam_k5_fixed_budget": "MS k=5 budget",
        "lookbehind_k5_fixed_budget": "LB k=5 budget",
        "lookbehind_k5_fixed_step": "LB k=5 step",
    }
    for key, label in selected_strength.items():
        rows = [row for row in strength_rows if row["key"] == key]
        if rows:
            rows.sort(key=lambda row: float(row["target_correction_gradient_ratio"]))
            axis.plot(
                [float(row["target_correction_gradient_ratio"]) for row in rows],
                [float(row["descent_per_unit_norm"]) for row in rows],
                marker="o",
                label=label,
            )
    axis.set_xlabel("matched correction strength ||c|| / ||g||")
    axis.set_ylabel("descent per unit update norm")
    axis.set_title("Native-radius matching separates direction from correction magnitude")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "strength_matched.png", dpi=180)
    plt.close(fig)

    metrics = [
        ("sam_q1", "SAM normalized gradient-gain quality"),
        ("gam_probe_q0", "GAM probe zero-order quality"),
        ("lookbehind_path_novelty", "LB k=5 budget H1 residual"),
    ]
    kinds = ["equal_gradient", "random_w", "random_g"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))
    for axis, (metric, title) in zip(axes, metrics):
        values = [
            [
                float(row[metric])
                for row in initialization_rows
                if row["initialization"] == kind
            ]
            for kind in kinds
        ]
        axis.boxplot(
            values,
            tick_labels=["equal-grad", "random-w", "random-g"],
            showfliers=False,
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("The equal-gradient start is a diagnostic fixture, not a typicality claim")
    fig.tight_layout()
    fig.savefig(output_dir / "initialization_sensitivity.png", dpi=180)
    plt.close(fig)


def run_sensitivity_analysis(
    *,
    output_dir: str | Path,
    baseline_overrides: dict[str, Any] | None = None,
    rho_values: Iterable[float] = DEFAULT_RHO_VALUES,
    condition_numbers: Iterable[float] = DEFAULT_CONDITION_NUMBERS,
    dimensions: Iterable[int] = DEFAULT_DIMENSIONS,
    inner_steps: Iterable[int] = DEFAULT_INNER_STEPS,
    strength_targets: Iterable[float] = DEFAULT_STRENGTH_TARGETS,
    initialization_samples: int = 100,
    make_plots: bool = True,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rho_values = tuple(float(value) for value in rho_values)
    condition_numbers = tuple(float(value) for value in condition_numbers)
    dimensions = tuple(dimensions)
    inner_steps = tuple(inner_steps)
    strength_targets = tuple(float(value) for value in strength_targets)
    baseline_input = {} if baseline_overrides is None else dict(baseline_overrides)
    baseline_input["make_plots"] = False
    baseline = resolved_config(baseline_input)
    cases = build_sweep_cases(
        rho_values=rho_values,
        condition_numbers=condition_numbers,
        dimensions=dimensions,
        inner_steps=inner_steps,
    )
    rows: list[dict[str, Any]] = []
    run_manifests = []
    for case in cases:
        overrides = dict(baseline)
        overrides.update(case.overrides)
        overrides["experiment_id"] = f"E001S_{case.case_id}"
        overrides["make_plots"] = False
        run_dir = destination / "runs" / case.case_id
        result = run_quadratic_experiment(overrides, output_dir=run_dir)
        rows.extend(_case_summary_rows(case, result, run_dir))
        run_manifests.append(
            {
                "case_id": case.case_id,
                "factor": case.factor,
                "factor_value": case.factor_value,
                "output_dir": str(run_dir.relative_to(destination)),
                "code_sha256": result["manifest"]["code_fingerprint"]["sha256"],
            }
        )

    strength_rows = strength_matched_rows(baseline, strength_targets)
    effective_rows = fixed_effective_radius_rows(baseline, inner_steps)
    quality_rows = quality_decomposition_rows(baseline)
    initialization_rows, initialization_summaries = initialization_ensemble_rows(
        baseline, sample_count=initialization_samples
    )
    _write_csv(destination / "sensitivity_summary.csv", rows, SENSITIVITY_FIELDS)
    _write_csv(destination / "strength_matched.csv", strength_rows, STRENGTH_FIELDS)
    _write_csv(
        destination / "fixed_effective_radius.csv",
        effective_rows,
        EFFECTIVE_RADIUS_FIELDS,
    )
    _write_csv(
        destination / "quality_decomposition.csv", quality_rows, QUALITY_FIELDS
    )
    _write_csv(
        destination / "initialization_samples.csv",
        initialization_rows,
        INITIALIZATION_SAMPLE_FIELDS,
    )
    _write_csv(
        destination / "initialization_summary.csv",
        initialization_summaries,
        INITIALIZATION_SUMMARY_FIELDS,
    )
    if make_plots:
        _make_plots(
            destination,
            rows,
            strength_rows,
            effective_rows,
            initialization_rows,
        )
    manifest = {
        "schema_version": "1.0",
        "experiment_id": "E001S_quadratic_sensitivity",
        "implementation_status": "implemented",
        "analysis_type": "one_factor_at_a_time_plus_native_radius_strength_matching",
        "code_fingerprint": _fingerprint(),
        "baseline_config": baseline,
        "grids": {
            "rho": [float(value) for value in rho_values],
            "condition_number": [float(value) for value in condition_numbers],
            "dimension": [int(value) for value in dimensions],
            "inner_steps": [int(value) for value in inner_steps],
            "strength_targets": [float(value) for value in strength_targets],
            "initialization_samples_per_random_family": int(initialization_samples),
        },
        "runtime": {
            "dtype": "float64",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "command": sys.argv,
        },
        "case_count": len(cases),
        "sensitivity_row_count": len(rows),
        "strength_matched_row_count": len(strength_rows),
        "fixed_effective_radius_row_count": len(effective_rows),
        "quality_decomposition_row_count": len(quality_rows),
        "initialization_sample_row_count": len(initialization_rows),
        "runs": run_manifests,
        "definitions": {
            "ofat": "Each branch changes only its named factor from the E001 baseline; repeated baseline settings across branches are intentional.",
            "condition": "lambda_min=1/sqrt(kappa), lambda_max=sqrt(kappa), so geometric-mean curvature stays one.",
            "sam_reference_ratio": "rho * ||H g_hat|| / ||g||, equal to the exact SAM correction ratio on E001.",
            "local_path_strength": "path_budget * ||H g_hat|| / ||g||; a dimensionless local curvature-radius diagnostic.",
            "strength_matching": "For each method and target r_c, bisection re-runs the native operator at a method-specific primary rho until ||c||/||g|| matches; vectors are not post-hoc rescaled.",
            "fixed_effective_radius": "rho_step=2*rho_eff/(k+1), holding Lookbehind's first-order matched-SAM radius fixed while k changes.",
            "quality_decomposition": "raw Q uses the registered oracle radius; boundary Q radially projects the same candidate direction to that boundary; own-radius Q changes the oracle ball to the candidate norm.",
            "initialization_ensemble": "equal_gradient is one engineered fixture; random_w samples isotropic normalized parameters; random_g samples isotropic requested gradients and maps them through H^-1 before normalizing w.",
        },
        "files": {
            "sensitivity_summary": "sensitivity_summary.csv",
            "strength_matched": "strength_matched.csv",
            "fixed_effective_radius": "fixed_effective_radius.csv",
            "quality_decomposition": "quality_decomposition.csv",
            "initialization_samples": "initialization_samples.csv",
            "initialization_summary": "initialization_summary.csv",
            "figures": [
                "sensitivity_overview.png",
                "strength_matched.png",
                "initialization_sensitivity.png",
            ] if make_plots else [],
        },
        "scope_boundary": (
            "This analysis tests robustness inside the same deterministic PSD quadratic family. "
            "It cannot validate nonconvex trajectories, stochastic covariance, basin selection, or generalization."
        ),
    }
    _write_json(destination / "manifest.json", manifest)
    return {
        "output_dir": str(destination),
        "manifest": manifest,
        "sensitivity_rows": rows,
        "strength_matched_rows": strength_rows,
        "fixed_effective_radius_rows": effective_rows,
        "quality_decomposition_rows": quality_rows,
        "initialization_rows": initialization_rows,
        "initialization_summary_rows": initialization_summaries,
    }
