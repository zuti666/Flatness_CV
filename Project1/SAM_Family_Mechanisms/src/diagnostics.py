from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .operators import Array, OperatorTrace, unit


def cosine(first: Array, second: Array, eps: float = 1e-15) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= eps:
        return None
    return float(np.dot(first, second) / denominator)


def eigensystem_descending(hessian: Array) -> tuple[Array, Array]:
    values, vectors = np.linalg.eigh(np.asarray(hessian, dtype=np.float64))
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def spectral_gain(
    correction: Array,
    gradient: Array,
    eigenvectors: Array,
    eps: float = 1e-15,
) -> Array:
    numerator = np.abs(eigenvectors.T @ correction)
    denominator = np.abs(eigenvectors.T @ unit(gradient)) + float(eps)
    return numerator / denominator


def top_subspace_energy(vector: Array, eigenvectors: Array, q: int) -> float | None:
    denominator = float(np.dot(vector, vector))
    if denominator <= 1e-30:
        return None
    q = min(max(1, int(q)), eigenvectors.shape[1])
    projection = eigenvectors[:, :q].T @ vector
    return float(np.dot(projection, projection) / denominator)


def signed_curvature_metrics(vector: Array, hessian: Array) -> dict[str, float | None]:
    norm_sq = float(np.dot(vector, vector))
    if norm_sq <= 1e-30:
        return {
            "positive_curvature_quadratic": None,
            "negative_curvature_quadratic": None,
            "positive_rayleigh": None,
            "negative_rayleigh": None,
        }
    values, vectors = np.linalg.eigh(hessian)
    coordinates_sq = np.square(vectors.T @ vector)
    positive = float(np.dot(np.clip(values, 0.0, None), coordinates_sq))
    negative = float(np.dot(np.clip(-values, 0.0, None), coordinates_sq))
    return {
        "positive_curvature_quadratic": positive,
        "negative_curvature_quadratic": negative,
        "positive_rayleigh": positive / norm_sq,
        "negative_rayleigh": negative / norm_sq,
    }


def nested_hessian_fit(
    correction: Array,
    hessian: Array,
    gradient: Array,
    max_order: int = 3,
) -> dict[str, float | None]:
    """Fit nested spans of H g-hat, H^2 g-hat, ... after QR."""
    target = np.asarray(correction, dtype=np.float64)
    target_norm_sq = float(np.dot(target, target))
    output: dict[str, float | None] = {}
    if target_norm_sq <= 1e-30:
        for order in range(1, max_order + 1):
            output[f"r2_{order}"] = None
            output[f"delta_r2_{order}"] = None
        return output

    basis_columns = []
    column = unit(gradient)
    for _ in range(max_order):
        column = hessian @ column
        basis_columns.append(column.copy())
    basis = np.column_stack(basis_columns)
    q_matrix, r_matrix = np.linalg.qr(basis, mode="reduced")
    diagonal = np.abs(np.diag(r_matrix))
    tolerance = max(basis.shape) * np.finfo(np.float64).eps * max(1.0, float(diagonal.max()))
    previous = 0.0
    for order in range(1, max_order + 1):
        rank = int(np.count_nonzero(diagonal[:order] > tolerance))
        if rank == 0:
            r_squared = 0.0
        else:
            coefficients = q_matrix[:, :rank].T @ target
            r_squared = float(np.dot(coefficients, coefficients) / target_norm_sq)
        r_squared = min(1.0, max(previous, r_squared))
        output[f"r2_{order}"] = r_squared
        output[f"delta_r2_{order}"] = r_squared - previous
        previous = r_squared
    return output


def maximize_quadratic_on_ball(
    matrix: Array,
    linear: Array,
    radius: float,
    *,
    tolerance: float = 1e-14,
    max_iterations: int = 300,
) -> Array:
    """Globally maximize b^T x + 1/2 x^T A x on an L2 ball.

    The experiment uses positive-semidefinite ``A``.  The implementation also
    handles the standard hard case in which ``b`` has no component in the top
    eigenspace.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    linear = np.asarray(linear, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if linear.shape != (matrix.shape[0],):
        raise ValueError("linear term has the wrong shape")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(linear)):
        raise ValueError("matrix and linear term must be finite")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("matrix must be symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    psd_tolerance = tolerance * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues[0]) < -psd_tolerance:
        raise ValueError("maximize_quadratic_on_ball currently requires a PSD matrix")
    coefficients = eigenvectors.T @ linear
    top = float(eigenvalues[-1])
    scale = max(1.0, abs(top), float(np.linalg.norm(linear)))
    gap = max(tolerance * scale, np.finfo(np.float64).eps * scale * 32.0)

    if float(np.linalg.norm(linear)) <= tolerance:
        return float(radius) * eigenvectors[:, -1]

    def coordinates(multiplier: float) -> Array:
        return coefficients / (multiplier - eigenvalues)

    low = top + gap
    at_low = coordinates(low)
    low_norm = float(np.linalg.norm(at_low))
    if low_norm < radius:
        # Hard case: solve in the non-top subspace at lambda_max and use the
        # remaining norm in a top eigenvector.  Its sign is immaterial because
        # the corresponding linear coefficient is zero up to tolerance.
        denominators = top - eigenvalues
        base_coordinates = np.zeros_like(coefficients)
        non_top = denominators > gap
        base_coordinates[non_top] = coefficients[non_top] / denominators[non_top]
        remaining_sq = max(0.0, float(radius) ** 2 - float(np.dot(base_coordinates, base_coordinates)))
        base_coordinates[-1] = remaining_sq**0.5
        return eigenvectors @ base_coordinates

    high = max(top + 1.0, low * 2.0 if low > 0 else 1.0)
    while float(np.linalg.norm(coordinates(high))) > radius:
        high = top + 2.0 * (high - top)
        if not np.isfinite(high):
            raise RuntimeError("Could not bracket trust-region multiplier")
    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        if float(np.linalg.norm(coordinates(middle))) > radius:
            low = middle
        else:
            high = middle
        if high - low <= tolerance * max(1.0, abs(high)):
            break
    solution = eigenvectors @ coordinates(high)
    # Remove the tiny inward bias introduced by bisection without changing the
    # direction at reported precision.
    solution *= float(radius) / float(np.linalg.norm(solution))
    return solution


def quadratic_increment(hessian: Array, gradient: Array, perturbation: Array) -> float:
    return float(np.dot(gradient, perturbation) + 0.5 * perturbation @ hessian @ perturbation)


def inner_problem_quality(
    hessian: Array,
    weights: Array,
    perturbation: Array,
    radius: float,
    eps: float = 1e-15,
) -> dict[str, float]:
    gradient = hessian @ weights
    perturbation_norm = float(np.linalg.norm(perturbation))
    if perturbation_norm > radius * (1.0 + 1e-10) + 1e-12:
        raise ValueError(
            f"perturbation norm {perturbation_norm:.6g} exceeds comparison radius {radius:.6g}"
        )
    delta_zero = maximize_quadratic_on_ball(hessian, gradient, radius)
    squared_gradient_hessian = hessian @ hessian
    delta_first = maximize_quadratic_on_ball(
        squared_gradient_hessian,
        hessian @ gradient,
        radius,
    )
    numerator_zero = quadratic_increment(hessian, gradient, perturbation)
    denominator_zero = quadratic_increment(hessian, gradient, delta_zero)
    base_gradient_norm = float(np.linalg.norm(gradient))
    numerator_first = float(np.linalg.norm(gradient + hessian @ perturbation)) - base_gradient_norm
    denominator_first = float(np.linalg.norm(gradient + hessian @ delta_first)) - base_gradient_norm
    q_zero = numerator_zero / (denominator_zero + eps)
    q_first = numerator_first / (denominator_first + eps)
    return {
        "q0": float(q_zero),
        "q1": float(q_first),
        "q0_numerator": numerator_zero,
        "q0_denominator": denominator_zero,
        "q1_numerator": numerator_first,
        "q1_denominator": denominator_first,
        "oracle_radius": float(radius),
        "delta0_star_norm": float(np.linalg.norm(delta_zero)),
        "delta1_star_norm": float(np.linalg.norm(delta_first)),
    }


def path_metrics(trace: OperatorTrace) -> dict[str, float | None]:
    if trace.path_points is None or trace.path_gradients is None:
        return {
            "path_misalign": None,
            "last_average_difference": None,
            "endpoint_radius": float(np.linalg.norm(trace.perturbation)),
            "path_radius": float(np.linalg.norm(trace.perturbation)),
        }
    gradients = trace.path_gradients
    pair_cosines: list[float] = []
    for first in range(len(gradients)):
        for second in range(first + 1, len(gradients)):
            value = cosine(gradients[first], gradients[second])
            if value is not None:
                pair_cosines.append(value)
    path_misalign = None if not pair_cosines else 1.0 - float(np.mean(pair_cosines))
    last_average_cosine = cosine(gradients[-1], gradients.mean(axis=0))
    differences = np.diff(trace.path_points, axis=0)
    return {
        "path_misalign": path_misalign,
        "last_average_difference": None if last_average_cosine is None else 1.0 - last_average_cosine,
        "endpoint_radius": float(np.linalg.norm(trace.path_points[-1] - trace.path_points[0])),
        "path_radius": float(np.linalg.norm(differences, axis=1).sum()),
    }


def path_novelty(correction: Array, hessian: Array, gradient: Array) -> float | None:
    correction_norm = float(np.linalg.norm(correction))
    if correction_norm <= 1e-15:
        return None
    reference = hessian @ unit(gradient)
    coefficient = float(np.dot(correction, reference) / np.dot(reference, reference))
    residual = correction - coefficient * reference
    return float(np.linalg.norm(residual) / correction_norm)


def descent_metrics(direction: Array, gradient: Array, hessian: Array) -> dict[str, float | None]:
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-15:
        return {
            "direction_gradient_cosine": None,
            "descent_per_unit_norm": None,
            "positive_curvature_exposure": None,
            "safe_descent": None,
        }
    signed = signed_curvature_metrics(direction, hessian)
    positive_quadratic = signed["positive_curvature_quadratic"]
    dot_product = float(np.dot(gradient, direction))
    safe = None
    if positive_quadratic is not None:
        safe = dot_product / (float(positive_quadratic) ** 0.5 + 1e-15)
    return {
        "direction_gradient_cosine": cosine(direction, gradient),
        "descent_per_unit_norm": dot_product / direction_norm,
        "positive_curvature_exposure": signed["positive_rayleigh"],
        "safe_descent": safe,
    }


def hvp_scan(
    hessian: Array,
    weights: Array,
    rho_scales: Iterable[float],
) -> list[dict[str, float]]:
    gradient = hessian @ weights
    truth = hessian @ unit(gradient)
    weight_norm = float(np.linalg.norm(weights))
    rows = []
    for scale in rho_scales:
        rho = float(scale) * weight_norm
        perturbed_gradient = hessian @ (weights + rho * unit(gradient))
        estimate = (perturbed_gradient - gradient) / rho
        alignment = cosine(estimate, truth)
        rows.append(
            {
                "rho_scale": float(scale),
                "rho": rho,
                "cos_hvp": 0.0 if alignment is None else alignment,
                "relative_error_hvp": float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth)),
                "estimate_norm": float(np.linalg.norm(estimate)),
                "truth_norm": float(np.linalg.norm(truth)),
            }
        )
    return rows
