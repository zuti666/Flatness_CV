from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .models import AdaptiveMLP
from .optimizers import _make_perturbations


def collect_diagnostic_batch(
    dataset: Dataset,
    max_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=max_samples, shuffle=False, num_workers=0)
    inputs, targets = next(iter(loader))
    return inputs.to(device), targets.to(device)


def collect_diagnostic_batches(
    dataset: Dataset,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Collect deterministic, non-overlapping diagnostic batches.

    The batches are consecutive slices of the dataset (``shuffle=False``).  They
    are repeated measurements within an experimental seed, not extra independent
    seeds, and are therefore kept separate in downstream analyses.
    """
    if batch_size <= 0:
        raise ValueError("Diagnostic batch_size must be positive")
    if num_batches <= 0:
        raise ValueError("Diagnostic num_batches must be positive")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = []
    for inputs, targets in loader:
        batches.append((inputs.to(device), targets.to(device)))
        if len(batches) == num_batches:
            break
    if len(batches) != num_batches:
        raise ValueError(
            f"Requested {num_batches} diagnostic batches of size {batch_size}, "
            f"but the dataset only provided {len(batches)}"
        )
    return batches


def _hessian_lanczos_max(
    gradient: torch.Tensor,
    weight: torch.Tensor,
    steps: int,
) -> float | None:
    if steps <= 0:
        return None
    size = weight.numel()
    q = torch.arange(1, size + 1, device=weight.device, dtype=weight.dtype).sin().view_as(weight)
    q = q / q.norm()
    q_previous = torch.zeros_like(q)
    beta_previous = weight.new_zeros(())
    vectors: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []
    for iteration in range(min(steps, size)):
        hvp = torch.autograd.grad((gradient * q).sum(), weight, retain_graph=True)[0]
        residual = hvp - beta_previous * q_previous
        alpha = (q * residual).sum()
        residual = residual - alpha * q
        for previous in vectors:
            residual = residual - (previous * residual).sum() * previous
        beta = residual.norm()
        vectors.append(q)
        alphas.append(alpha.detach())
        if iteration == steps - 1 or beta < 1e-10:
            break
        betas.append(beta.detach())
        q_previous, q = q, residual / beta
        beta_previous = beta
    tridiagonal = torch.diag(torch.stack(alphas))
    if betas:
        off_diagonal = torch.stack(betas[: len(alphas) - 1])
        indices = torch.arange(len(off_diagonal), device=weight.device)
        tridiagonal[indices, indices + 1] = off_diagonal
        tridiagonal[indices + 1, indices] = off_diagonal
    return float(torch.linalg.eigvalsh(tridiagonal).max())


def exact_directional_diagnostics(
    model: AdaptiveMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    start_weight: torch.Tensor,
    end_weight: torch.Tensor,
    save_tensor_path: Path | None = None,
    finite_difference_radii: list[float] | None = None,
    hessian_lanczos_steps: int = 0,
) -> dict[str, Any]:
    """Exact HΔ and old-loss Taylor decomposition in effective-W coordinates."""
    criterion = nn.CrossEntropyLoss()
    start = start_weight.detach().clone().requires_grad_(True)
    delta = (end_weight - start_weight).detach()
    start_loss = criterion(model.forward_with_middle_weight(inputs, start), targets)
    gradient = torch.autograd.grad(start_loss, start, create_graph=True)[0]
    hvp = torch.autograd.grad((gradient * delta).sum(), start, retain_graph=True)[0]
    lambda_max = _hessian_lanczos_max(gradient, start, hessian_lanczos_steps)
    with torch.no_grad():
        end_loss = criterion(model.forward_with_middle_weight(inputs, end_weight), targets)
        interference = (gradient.detach() * delta).sum()
        directional_curvature = (delta * hvp).sum()
        delta_norm = delta.norm()
        normalized_curvature = directional_curvature / delta_norm.square().clamp_min(1e-20)
        taylor_change = interference + 0.5 * directional_curvature
        taylor_residual = end_loss - start_loss.detach() - taylor_change
        result: dict[str, Any] = {
            "start_loss": float(start_loss.detach()),
            "end_loss": float(end_loss),
            "actual_loss_change": float(end_loss - start_loss.detach()),
            "interference_I": float(interference),
            "linear_term": float(interference),
            "directional_curvature_C": float(0.5 * directional_curvature),
            "directional_curvature": float(directional_curvature),
            "quadratic_term": float(0.5 * directional_curvature),
            "normalized_hessian_curvature": float(normalized_curvature),
            "pathwise_taylor_prediction": float(taylor_change),
            "second_order_prediction": float(start_loss.detach() + taylor_change),
            "taylor_residual": float(taylor_residual),
            "delta_norm": float(delta_norm),
            "delta_norm_squared": float(delta_norm.square()),
            "gradient_norm": float(gradient.detach().norm()),
            "hvp_norm": float(hvp.norm()),
            "hessian_lambda_max": lambda_max,
        }
    if finite_difference_radii:
        unit_direction = delta / delta.norm().clamp_min(1e-20)
        exact_unit_hvp = hvp.detach() / delta.norm().clamp_min(1e-20)
        absolute_errors = {}
        relative_errors = {}
        forward_absolute_errors = {}
        forward_relative_errors = {}
        forward_angle_errors = {}
        for radius_value in finite_difference_radii:
            radius = float(radius_value)
            if radius <= 0:
                raise ValueError("HVP finite-difference radii must be positive")
            plus = (start.detach() + radius * unit_direction).requires_grad_(True)
            minus = (start.detach() - radius * unit_direction).requires_grad_(True)
            plus_loss = criterion(model.forward_with_middle_weight(inputs, plus), targets)
            minus_loss = criterion(model.forward_with_middle_weight(inputs, minus), targets)
            plus_gradient = torch.autograd.grad(plus_loss, plus)[0]
            minus_gradient = torch.autograd.grad(minus_loss, minus)[0]
            estimate = (plus_gradient - minus_gradient) / (2.0 * radius)
            absolute_error = (estimate - exact_unit_hvp).norm()
            relative_error = absolute_error / exact_unit_hvp.norm().clamp_min(1e-20)
            forward_estimate = (plus_gradient - gradient.detach()) / radius
            forward_absolute_error = (forward_estimate - exact_unit_hvp).norm()
            forward_relative_error = (
                forward_absolute_error / exact_unit_hvp.norm().clamp_min(1e-20)
            )
            forward_cosine = torch.nn.functional.cosine_similarity(
                forward_estimate.flatten(), exact_unit_hvp.flatten(), dim=0, eps=1e-20
            )
            absolute_errors[f"{radius:g}"] = float(absolute_error)
            relative_errors[f"{radius:g}"] = float(relative_error)
            forward_absolute_errors[f"{radius:g}"] = float(forward_absolute_error)
            forward_relative_errors[f"{radius:g}"] = float(forward_relative_error)
            forward_angle_errors[f"{radius:g}"] = float(1.0 - forward_cosine)
        result["hvp_fd_absolute_error"] = absolute_errors
        result["hvp_fd_relative_error"] = relative_errors
        result["hvp_fd_central_relative_error"] = relative_errors
        result["hvp_fd_forward_absolute_error"] = forward_absolute_errors
        result["hvp_fd_forward_relative_error"] = forward_relative_errors
        result["hvp_fd_forward_angle_error"] = forward_angle_errors
    if save_tensor_path is not None:
        save_tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "delta": delta.cpu(),
                "gradient": gradient.detach().cpu(),
                "hvp": hvp.detach().cpu(),
            },
            save_tensor_path,
        )
    return result


def trainable_coordinate_hvp_diagnostics(
    model: AdaptiveMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    effective_radius: float,
) -> dict[str, float]:
    """GAM-aligned forward FD versus exact HVP in actual trainable coordinates."""
    parameters = model.trainable_parameters()
    loss = nn.functional.cross_entropy(model(inputs), targets)
    gradients = torch.autograd.grad(loss, parameters, create_graph=True)
    gradient_norm = torch.sqrt(
        sum(gradient.square().sum() for gradient in gradients)
    ).clamp_min(1e-20)
    directions = [gradient.detach() / gradient_norm.detach() for gradient in gradients]
    directional_gradient = sum(
        (gradient * direction).sum() for gradient, direction in zip(gradients, directions)
    )
    exact = [value.detach() for value in torch.autograd.grad(directional_gradient, parameters)]
    perturbations, raw_step, achieved_radius = _make_perturbations(
        model,
        parameters,
        [gradient.detach() for gradient in gradients],
        float(effective_radius),
        "effective_weight",
    )
    with torch.no_grad():
        for parameter, perturbation in zip(parameters, perturbations):
            parameter.add_(perturbation)
    perturbed_loss = nn.functional.cross_entropy(model(inputs), targets)
    perturbed_gradients = torch.autograd.grad(perturbed_loss, parameters)
    with torch.no_grad():
        for parameter, perturbation in zip(parameters, perturbations):
            parameter.sub_(perturbation)
    finite_difference = [
        (perturbed.detach() - clean.detach()) / raw_step
        for perturbed, clean in zip(perturbed_gradients, gradients)
    ]
    exact_flat = torch.cat([value.flatten() for value in exact])
    fd_flat = torch.cat([value.flatten() for value in finite_difference])
    absolute_error = (fd_flat - exact_flat).norm()
    relative_error = absolute_error / exact_flat.norm().clamp_min(1e-20)
    angle_error = 1.0 - nn.functional.cosine_similarity(
        fd_flat, exact_flat, dim=0, eps=1e-20
    )
    exact_weight = model.map_trainable_direction_to_weight(exact)
    fd_weight = model.map_trainable_direction_to_weight(finite_difference)
    weight_absolute_error = (fd_weight - exact_weight).norm()
    weight_relative_error = weight_absolute_error / exact_weight.norm().clamp_min(1e-20)
    weight_angle_error = 1.0 - nn.functional.cosine_similarity(
        fd_weight.flatten(), exact_weight.flatten(), dim=0, eps=1e-20
    )
    return {
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "raw_coordinate_step": raw_step,
        "effective_weight_radius": achieved_radius,
        "exact_hvp_norm": float(exact_flat.norm()),
        "fd_hvp_norm": float(fd_flat.norm()),
        "fd_absolute_error": float(absolute_error),
        "fd_relative_error": float(relative_error),
        "fd_angle_error": float(angle_error),
        "pushed_forward_exact_norm": float(exact_weight.norm()),
        "pushed_forward_fd_norm": float(fd_weight.norm()),
        "pushed_forward_absolute_error": float(weight_absolute_error),
        "pushed_forward_relative_error": float(weight_relative_error),
        "pushed_forward_angle_error": float(weight_angle_error),
    }


def effective_tangent_basis(model: AdaptiveMLP) -> torch.Tensor:
    return model.effective_subspace_basis().detach()


def subspace_overlap(first: torch.Tensor, second: torch.Tensor, ambient_dim: int) -> float:
    del ambient_dim
    denominator = max(1, min(first.shape[1], second.shape[1]))
    return float((first.T @ second).square().sum() / denominator)


def reachable_coverage(basis: torch.Tensor, delta: torch.Tensor) -> float:
    vector = delta.detach().flatten()
    denominator = vector.square().sum().clamp_min(1e-20)
    return float((basis.T @ vector).square().sum() / denominator)


def _activation_derivative(name: str, preactivation: torch.Tensor) -> torch.Tensor:
    if name.lower() == "softplus":
        return torch.sigmoid(preactivation)
    if name.lower() == "gelu":
        normal_cdf = 0.5 * (1.0 + torch.erf(preactivation / 2.0**0.5))
        normal_pdf = torch.exp(-0.5 * preactivation.square()) / (2.0 * torch.pi) ** 0.5
        return normal_cdf + preactivation * normal_pdf
    raise ValueError(f"Unsupported activation: {name}")


def weighted_logit_jacobian(
    model: AdaptiveMLP,
    inputs: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """K with KᵀK equal to the empirical cross-entropy GGN in W-space."""
    with torch.no_grad():
        hidden = model.hidden_before_middle(inputs)
        preactivation = torch.nn.functional.linear(hidden, weight, model.middle_bias)
        derivative = _activation_derivative(model.activation_name, preactivation)
        logits = model.forward_with_middle_weight(inputs, weight)
        probabilities = logits.softmax(dim=1)
        output_sensitivity = model.fc3.weight.unsqueeze(0) * derivative.unsqueeze(1)
        jacobian = torch.einsum("ncm,nk->ncmk", output_sensitivity, hidden).flatten(2)
        covariance = torch.diag_embed(probabilities) - probabilities.unsqueeze(2) * probabilities.unsqueeze(1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        sqrt_covariance = (
            eigenvectors
            @ torch.diag_embed(eigenvalues.clamp_min(0).sqrt())
            @ eigenvectors.transpose(1, 2)
        )
        weighted = torch.bmm(sqrt_covariance, jacobian)
        return weighted.flatten(0, 1) / inputs.shape[0] ** 0.5


def weight_gradient(
    model: AdaptiveMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    variable = weight.detach().clone().requires_grad_(True)
    loss = nn.functional.cross_entropy(model.forward_with_middle_weight(inputs, variable), targets)
    return torch.autograd.grad(loss, variable)[0].detach()


def prospective_reachable_direction_diagnostics(
    model: AdaptiveMLP,
    old_batch: tuple[torch.Tensor, torch.Tensor],
    new_batch: tuple[torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    reachable_basis: torch.Tensor,
) -> dict[str, float]:
    """Old-task sensitivity along the reachable new-task gradient at task start.

    This is measured before the new-task optimizer takes a step, so it isolates
    robustness imprinted while learning the protected task from trajectory changes
    made by the next-task optimizer.
    """
    old_inputs, old_targets = old_batch
    new_inputs, new_targets = new_batch
    old_variable = weight.detach().clone().requires_grad_(True)
    old_loss = nn.functional.cross_entropy(
        model.forward_with_middle_weight(old_inputs, old_variable), old_targets
    )
    old_gradient = torch.autograd.grad(old_loss, old_variable, create_graph=True)[0]
    new_gradient = weight_gradient(model, new_inputs, new_targets, weight).flatten()
    projected = reachable_basis @ (reachable_basis.T @ new_gradient)
    projected_norm = projected.norm()
    direction = projected / projected_norm.clamp_min(1e-20)
    direction_matrix = direction.view_as(old_variable)
    old_hvp = torch.autograd.grad(
        (old_gradient * direction_matrix).sum(), old_variable
    )[0]
    old_k = weighted_logit_jacobian(model, old_inputs, weight)
    ggn_curvature = (old_k @ direction).square().sum()
    reachable_fraction = projected_norm.square() / new_gradient.square().sum().clamp_min(1e-20)
    old_alignment = (old_gradient.detach().flatten() * direction).sum()
    old_gradient_norm = old_gradient.detach().norm()
    old_gradient_cosine = old_alignment / old_gradient_norm.clamp_min(1e-20)
    return {
        "prospective_old_gradient_norm": float(old_gradient_norm),
        "prospective_new_gradient_norm": float(new_gradient.norm()),
        "prospective_projected_gradient_norm": float(projected_norm),
        "prospective_reachable_gradient_fraction": float(reachable_fraction),
        "prospective_old_gradient_alignment": float(old_alignment),
        "prospective_old_gradient_cosine": float(old_gradient_cosine),
        "prospective_hessian_curvature": float((direction_matrix * old_hvp).sum()),
        "prospective_ggn_curvature": float(ggn_curvature),
    }


def _regularized_inverse_quadratic(matrix: torch.Tensor, vector: torch.Tensor, damping: float) -> float:
    """Compute vᵀ(λI+MᵀM)⁻¹v using the smaller primal/dual system."""
    rows, columns = matrix.shape
    if columns <= rows:
        system = matrix.T @ matrix + damping * torch.eye(
            columns, device=matrix.device, dtype=matrix.dtype
        )
        solution = torch.linalg.solve(system, vector)
    else:
        identity = torch.eye(rows, device=matrix.device, dtype=matrix.dtype)
        middle = torch.linalg.solve(identity + matrix @ matrix.T / damping, matrix @ vector)
        solution = vector / damping - matrix.T @ middle / damping**2
    return float(vector @ solution)


def ggn_safe_route_diagnostics(
    model: AdaptiveMLP,
    old_batch: tuple[torch.Tensor, torch.Tensor],
    new_batch: tuple[torch.Tensor, torch.Tensor],
    start_weight: torch.Tensor,
    end_weight: torch.Tensor,
    reachable_basis: torch.Tensor,
    damping: float,
    safe_relative_threshold: float,
    overlap_top_k: int,
) -> dict[str, float]:
    """Compute κ, P_r(λ), τ_r, gradient cosine and measured GGN overlap."""
    old_inputs, old_targets = old_batch
    new_inputs, new_targets = new_batch
    delta = (end_weight - start_weight).detach().flatten()
    old_gradient = weight_gradient(model, old_inputs, old_targets, start_weight).flatten()
    new_gradient = weight_gradient(model, new_inputs, new_targets, start_weight).flatten()
    old_k = weighted_logit_jacobian(model, old_inputs, start_weight)
    new_k = weighted_logit_jacobian(model, new_inputs, start_weight)
    old_singular, old_vh = torch.linalg.svd(old_k, full_matrices=False)[1:]
    new_singular, new_vh = torch.linalg.svd(new_k, full_matrices=False)[1:]
    old_eigenvalues = old_singular.square()
    lambda_max = old_eigenvalues[0] if len(old_eigenvalues) else old_k.new_zeros(())

    projected_gradient = reachable_basis.T @ new_gradient
    restricted_jacobian = old_k @ reachable_basis
    safe_plasticity = _regularized_inverse_quadratic(
        restricted_jacobian, projected_gradient, float(damping)
    )
    ggn_cost = (old_k @ delta).square().sum()
    delta_norm_squared = delta.square().sum().clamp_min(1e-20)

    unsafe_count = int((old_eigenvalues > safe_relative_threshold * lambda_max).sum().item())
    unsafe_basis = old_vh[:unsafe_count].T
    unsafe_overlap = (
        (unsafe_basis.T @ reachable_basis).square().sum()
        if unsafe_count
        else old_k.new_zeros(())
    )
    reachable_safe_dimension = reachable_basis.shape[1] - unsafe_overlap
    total_safe_dimension = max(1, start_weight.numel() - unsafe_count)
    tau_r = reachable_safe_dimension / total_safe_dimension
    tau_quality = reachable_safe_dimension / max(1, reachable_basis.shape[1])

    top_k = min(overlap_top_k, old_vh.shape[0], new_vh.shape[0])
    ggn_overlap = (
        (old_vh[:top_k] @ new_vh[:top_k].T).square().sum() / max(1, top_k)
        if top_k
        else old_k.new_zeros(())
    )
    gradient_cosine = (old_gradient @ new_gradient) / (
        old_gradient.norm() * new_gradient.norm()
    ).clamp_min(1e-20)
    gradient_reachable_fraction = projected_gradient.square().sum() / new_gradient.square().sum().clamp_min(1e-20)
    return {
        "ggn_directional_cost": float(ggn_cost),
        "kappa_G": float(ggn_cost / delta_norm_squared),
        "P_r_lambda": safe_plasticity,
        "tau_r": float(tau_r),
        "tau_r_quality": float(tau_quality),
        "reachable_dimension": int(reachable_basis.shape[1]),
        "new_gradient_reachable_fraction": float(gradient_reachable_fraction),
        "old_new_gradient_cosine": float(gradient_cosine),
        "old_new_ggn_top_overlap": float(ggn_overlap),
        "ggn_lambda_max": float(lambda_max),
        "ggn_unsafe_dimension": unsafe_count,
    }


def _factor_jacobian_stats(
    a: torch.Tensor,
    b: torch.Tensor,
    lora_scale: float,
) -> tuple[float, float, int]:
    """Exact spectrum of the factor Jacobian without forming the large matrix.

    For ``dW = s(B dA + dB A)``, ``J J^T`` is the Kronecker sum of
    ``s^2 B B^T`` and ``s^2 A^T A``.  Its eigenvalues are therefore all
    pairwise sums of the two small spectra.  This is algebraically equivalent
    to the explicit SVD but makes time-resolved P2 diagnostics inexpensive.
    """
    output_dim, rank = b.shape
    input_dim = a.shape[1]
    b_eigenvalues = torch.linalg.svdvals(b).square()
    a_eigenvalues = torch.linalg.svdvals(a).square()
    if output_dim > rank:
        b_eigenvalues = torch.cat(
            [b_eigenvalues, b.new_zeros(output_dim - rank)]
        )
    if input_dim > rank:
        a_eigenvalues = torch.cat(
            [a_eigenvalues, a.new_zeros(input_dim - rank)]
        )
    singular_values = abs(float(lora_scale)) * torch.sqrt(
        (b_eigenvalues[:, None] + a_eigenvalues[None, :]).flatten().clamp_min(0)
    )
    jacobian_shape = (output_dim * input_dim, rank * (output_dim + input_dim))
    tolerance = (
        singular_values.max()
        * max(jacobian_shape)
        * torch.finfo(singular_values.dtype).eps
    )
    nonzero = singular_values[singular_values > tolerance]
    if len(nonzero) == 0:
        return 0.0, float("inf"), 0
    return float(nonzero.max()), float(nonzero.max() / nonzero.min()), int(len(nonzero))


def factor_parameterization_diagnostics(
    model: AdaptiveMLP,
    new_batch: tuple[torch.Tensor, torch.Tensor],
    start_weight: torch.Tensor,
    start_a: torch.Tensor,
    start_b: torch.Tensor,
    perturbation_radius: float,
) -> dict[str, float]:
    """Appendix diagnostics for Jacobian conditioning, scaling and bilinear drift."""
    new_inputs, new_targets = new_batch
    gradient_w = weight_gradient(model, new_inputs, new_targets, start_weight)
    gradient_a = model.lora_scale * start_b.T @ gradient_w
    gradient_b = model.lora_scale * gradient_w @ start_a.T
    norm = torch.sqrt(gradient_a.square().sum() + gradient_b.square().sum()).clamp_min(1e-20)
    direction_a, direction_b = gradient_a / norm, gradient_b / norm
    tangent = start_b @ direction_a + direction_b @ start_a
    bilinear = direction_b @ direction_a
    bilinear_ratio = (
        float(perturbation_radius) * bilinear.norm() / tangent.norm().clamp_min(1e-20)
    )
    operator_norm, condition, numerical_rank = _factor_jacobian_stats(
        start_a, start_b, model.lora_scale
    )
    rescaled_operator_norms = []
    rescaled_conditions = []
    for factor in (0.1, 1.0, 10.0):
        op_norm, cond, _ = _factor_jacobian_stats(
            factor * start_a, start_b / factor, model.lora_scale
        )
        rescaled_operator_norms.append(op_norm)
        rescaled_conditions.append(cond)
    finite_conditions = [value for value in rescaled_conditions if torch.isfinite(torch.tensor(value))]
    return {
        "factor_bilinear_ratio": float(bilinear_ratio),
        "factor_jacobian_operator_norm": operator_norm,
        "factor_jacobian_condition": condition,
        "factor_jacobian_numerical_rank": numerical_rank,
        "factor_rescaling_operator_sensitivity": (
            max(rescaled_operator_norms) / max(min(rescaled_operator_norms), 1e-20)
        ),
        "factor_rescaling_condition_sensitivity": (
            max(finite_conditions) / max(min(finite_conditions), 1e-20)
            if finite_conditions
            else float("inf")
        ),
    }


def pathwise_diagnostics(
    model: AdaptiveMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: list[torch.Tensor],
) -> dict[str, Any]:
    segments = []
    for index, (start, end) in enumerate(zip(weights[:-1], weights[1:])):
        item = exact_directional_diagnostics(model, inputs, targets, start, end)
        delta = (end - start).detach().flatten()
        weighted_jacobian = weighted_logit_jacobian(model, inputs, start)
        ggn_cost = (weighted_jacobian @ delta).square().sum()
        item["ggn_directional_cost"] = float(ggn_cost)
        item["kappa_G"] = float(ggn_cost / delta.square().sum().clamp_min(1e-20))
        item["segment"] = index
        segments.append(item)
    return {
        "num_segments": len(segments),
        "actual_loss_change_sum": sum(item["actual_loss_change"] for item in segments),
        "interference_sum": sum(item["interference_I"] for item in segments),
        "linear_term_sum": sum(item["linear_term"] for item in segments),
        "directional_curvature_sum": sum(item["directional_curvature_C"] for item in segments),
        "ggn_directional_cost_sum": sum(item["ggn_directional_cost"] for item in segments),
        "quadratic_term_sum": sum(item["quadratic_term"] for item in segments),
        "pathwise_taylor_prediction": sum(
            item["interference_I"] + item["directional_curvature_C"] for item in segments
        ),
        "taylor_residual_sum": sum(item["taylor_residual"] for item in segments),
        "path_length": sum(item["delta_norm"] for item in segments),
        "segments": segments,
    }
