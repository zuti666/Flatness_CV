from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


Array = np.ndarray


def unit(vector: Array, eps: float = 1e-15) -> Array:
    """Return a numerically safe L2-normalized copy of ``vector``."""
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        raise ValueError("Cannot normalize a zero vector")
    return np.asarray(vector, dtype=np.float64) / norm


def _validate_problem(hessian: Array, weights: Array) -> tuple[Array, Array]:
    hessian = np.asarray(hessian, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be a square matrix")
    if weights.shape != (hessian.shape[0],):
        raise ValueError("weights must have one entry per Hessian dimension")
    if not np.allclose(hessian, hessian.T, rtol=0.0, atol=1e-12):
        raise ValueError("hessian must be symmetric")
    return hessian, weights


@dataclass
class OperatorTrace:
    """Everything needed to separate a method's base gradient and correction."""

    key: str
    method: str
    gradient: Array
    direction: Array
    correction: Array
    perturbation: Array
    protocol: str | None = None
    inner_steps: int | None = None
    rho_step: float | None = None
    path_points: Array | None = None
    path_gradients: Array | None = None
    probe_direction: Array | None = None
    probe_increment: Array | None = None
    final_regularizer: Array | None = None
    gradient_evaluations: int = 1
    hvp_evaluations: int = 0
    backward_equivalents: int = 1
    metadata: dict[str, float | int | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("gradient", "direction", "correction", "perturbation"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            setattr(self, name, value)
        if self.path_points is not None:
            self.path_points = np.asarray(self.path_points, dtype=np.float64)
        if self.path_gradients is not None:
            self.path_gradients = np.asarray(self.path_gradients, dtype=np.float64)
        for name in ("probe_direction", "probe_increment", "final_regularizer"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value, dtype=np.float64))


def sgd_trace(hessian: Array, weights: Array) -> OperatorTrace:
    hessian, weights = _validate_problem(hessian, weights)
    gradient = hessian @ weights
    zeros = np.zeros_like(weights)
    return OperatorTrace(
        key="sgd",
        method="sgd",
        gradient=gradient,
        direction=gradient.copy(),
        correction=zeros.copy(),
        perturbation=zeros.copy(),
        rho_step=None,
    )


def sam_trace(
    hessian: Array,
    weights: Array,
    rho: float,
    *,
    key: str = "sam",
    method: str = "sam",
    protocol: str | None = None,
    inner_steps: int | None = None,
    metadata: dict[str, float | int | str | bool] | None = None,
) -> OperatorTrace:
    """Exact SAM direction for a quadratic loss."""
    hessian, weights = _validate_problem(hessian, weights)
    if rho <= 0:
        raise ValueError("rho must be positive")
    gradient = hessian @ weights
    perturbation = float(rho) * unit(gradient)
    direction = hessian @ (weights + perturbation)
    return OperatorTrace(
        key=key,
        method=method,
        gradient=gradient,
        direction=direction,
        correction=direction - gradient,
        perturbation=perturbation,
        protocol=protocol,
        inner_steps=inner_steps,
        rho_step=float(rho),
        gradient_evaluations=2,
        backward_equivalents=2,
        metadata={} if metadata is None else dict(metadata),
    )


def gam_trace(hessian: Array, weights: Array, rho: float) -> OperatorTrace:
    """Idealized exact-HVP GAM trace used by the operator-level experiment.

    The trace deliberately keeps three distinct objects:

    * ``probe_direction = normalize(H normalize(g))``;
    * ``probe_increment = grad(w + rho * probe_direction) - g``;
    * ``final_regularizer = rho * H normalize(grad(w_adv))``.

    Only the probe increment has the local ``H^2`` spectral response.  The
    update is defined as ``g + final_regularizer`` so the final regularizer is
    never mislabeled as the probe increment.
    """
    hessian, weights = _validate_problem(hessian, weights)
    if rho <= 0:
        raise ValueError("rho must be positive")
    gradient = hessian @ weights
    current_hvp = hessian @ unit(gradient)
    probe_direction = unit(current_hvp)
    perturbation = float(rho) * probe_direction
    adversarial_gradient = hessian @ (weights + perturbation)
    probe_increment = adversarial_gradient - gradient
    final_regularizer = float(rho) * (hessian @ unit(adversarial_gradient))
    direction = gradient + final_regularizer
    return OperatorTrace(
        key="gam",
        method="gam",
        gradient=gradient,
        direction=direction,
        correction=final_regularizer.copy(),
        perturbation=perturbation,
        rho_step=float(rho),
        probe_direction=probe_direction,
        probe_increment=probe_increment,
        final_regularizer=final_regularizer,
        gradient_evaluations=2,
        hvp_evaluations=2,
        backward_equivalents=4,
        metadata={"definition": "g_plus_rho_hessian_at_probe_normalized_gradient"},
    )


def ascent_path(
    hessian: Array,
    weights: Array,
    rho_step: float,
    inner_steps: int,
) -> tuple[Array, Array]:
    """Build ``z_0,...,z_k`` and post-ascent gradients ``g_1,...,g_k``."""
    hessian, weights = _validate_problem(hessian, weights)
    if rho_step <= 0:
        raise ValueError("rho_step must be positive")
    if inner_steps < 1:
        raise ValueError("inner_steps must be at least one")
    point = weights.copy()
    gradient = hessian @ point
    points = [point.copy()]
    gradients: list[Array] = []
    for _ in range(int(inner_steps)):
        point = point + float(rho_step) * unit(gradient)
        gradient = hessian @ point
        points.append(point.copy())
        gradients.append(gradient.copy())
    return np.stack(points), np.stack(gradients)


def multistep_sam_trace(
    hessian: Array,
    weights: Array,
    rho_step: float,
    inner_steps: int,
    protocol: str,
) -> OperatorTrace:
    hessian, weights = _validate_problem(hessian, weights)
    points, gradients = ascent_path(hessian, weights, rho_step, inner_steps)
    clean_gradient = hessian @ weights
    direction = gradients[-1]
    return OperatorTrace(
        key=f"ms_sam_k{inner_steps}_{protocol}",
        method="multistep_sam",
        gradient=clean_gradient,
        direction=direction,
        correction=direction - clean_gradient,
        perturbation=points[-1] - weights,
        protocol=protocol,
        inner_steps=int(inner_steps),
        rho_step=float(rho_step),
        path_points=points,
        path_gradients=gradients,
        gradient_evaluations=int(inner_steps) + 1,
        backward_equivalents=int(inner_steps) + 1,
    )


def lookbehind_trace(
    hessian: Array,
    weights: Array,
    rho_step: float,
    inner_steps: int,
    protocol: str,
) -> OperatorTrace:
    hessian, weights = _validate_problem(hessian, weights)
    points, gradients = ascent_path(hessian, weights, rho_step, inner_steps)
    clean_gradient = hessian @ weights
    direction = gradients.mean(axis=0)
    return OperatorTrace(
        key=f"lookbehind_k{inner_steps}_{protocol}",
        method="lookbehind",
        gradient=clean_gradient,
        direction=direction,
        correction=direction - clean_gradient,
        perturbation=points[-1] - weights,
        protocol=protocol,
        inner_steps=int(inner_steps),
        rho_step=float(rho_step),
        path_points=points,
        path_gradients=gradients,
        gradient_evaluations=int(inner_steps) + 1,
        backward_equivalents=int(inner_steps) + 1,
        metadata={"aggregation": "mean_post_ascent_gradients", "alpha": 1.0},
    )


def matched_sam_trace(
    hessian: Array,
    weights: Array,
    rho_step: float,
    inner_steps: int,
    protocol: str,
) -> OperatorTrace:
    """SAM with the first-order Lookbehind-equivalent radius."""
    rho_effective = 0.5 * (int(inner_steps) + 1) * float(rho_step)
    return sam_trace(
        hessian,
        weights,
        rho_effective,
        key=f"matched_sam_k{inner_steps}_{protocol}",
        method="matched_sam",
        protocol=protocol,
        inner_steps=int(inner_steps),
        metadata={
            "matched_to": f"lookbehind_k{inner_steps}_{protocol}",
            "rho_effective": rho_effective,
            "rho_step": float(rho_step),
            "inner_steps": int(inner_steps),
        },
    )
