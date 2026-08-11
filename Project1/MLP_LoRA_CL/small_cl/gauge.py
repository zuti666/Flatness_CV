from __future__ import annotations

import re

import torch

from .models import AdaptiveMLP


def _orthogonal_matrix(
    rank: int,
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Deterministic Haar-like orthogonal matrix with fixed QR signs."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn(rank, rank, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    q = q * signs.unsqueeze(0)
    return q.to(device=device, dtype=dtype)


def make_gauge_matrix(
    name: str,
    rank: int,
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct S for the exact factor gauge B' = B S, A' = S^-1 A.

    ``anisotropic_k`` uses a rotated determinant-one SPD matrix whose spectral
    condition number is k.  This changes coordinate geometry without changing
    BA or the local effective-weight tangent space.
    """
    key = name.lower()
    identity = torch.eye(rank, device=device, dtype=dtype)
    if key == "identity":
        return identity
    if key == "orthogonal":
        return _orthogonal_matrix(rank, seed, device=device, dtype=dtype)
    scalar = re.fullmatch(r"scalar_([0-9]*\.?[0-9]+)", key)
    if scalar:
        value = float(scalar.group(1))
        if value <= 0:
            raise ValueError("A scalar gauge must be positive")
        return value * identity
    anisotropic = re.fullmatch(r"anisotropic_([0-9]*\.?[0-9]+)", key)
    if anisotropic:
        condition = float(anisotropic.group(1))
        if condition < 1:
            raise ValueError("An anisotropic gauge condition must be at least one")
        if rank < 2 and condition != 1:
            raise ValueError("An anisotropic gauge requires rank >= 2")
        q = _orthogonal_matrix(rank, seed + 1, device=device, dtype=dtype)
        diagonal = torch.ones(rank, device=device, dtype=dtype)
        if rank >= 2:
            diagonal[0] = condition**0.5
            diagonal[1] = condition**-0.5
        return q @ torch.diag(diagonal) @ q.T
    raise ValueError(f"Unsupported gauge transformation: {name}")


def apply_factor_gauge_(model: AdaptiveMLP, matrix: torch.Tensor) -> None:
    """Apply an invertible mature-factor gauge in place."""
    if not model.is_factorized:
        raise TypeError("Gauge transformations require a factorized LoRA model")
    if matrix.shape != (model.rank, model.rank):
        raise ValueError(
            f"Expected a {model.rank}x{model.rank} gauge matrix, got {tuple(matrix.shape)}"
        )
    matrix = matrix.to(device=model.lora_a.device, dtype=model.lora_a.dtype)
    with torch.no_grad():
        original_a = model.lora_a.detach().clone()
        original_b = model.lora_b.detach().clone()
        model.lora_b.copy_(original_b @ matrix)
        model.lora_a.copy_(torch.linalg.solve(matrix, original_a))


def tangent_projector_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    """Frobenius distance between projectors, without forming ambient projectors."""
    # Re-orthogonalize in float64: subtracting two ~O(d) traces in float32
    # otherwise creates a spurious ~1e-3 distance even for identical bases.
    first_q = torch.linalg.qr(first.to(dtype=torch.float64), mode="reduced")[0]
    second_q = torch.linalg.qr(second.to(dtype=torch.float64), mode="reduced")[0]
    overlap = (first_q.T @ second_q).square().sum()
    squared = first_q.shape[1] + second_q.shape[1] - 2.0 * overlap
    return float(squared.clamp_min(0).sqrt())


def pullback_gradient_metrics(
    model: AdaptiveMLP,
    old_weight_gradient: torch.Tensor,
    new_weight_gradient: torch.Tensor,
) -> dict[str, float]:
    """Measure the factor-gradient pullback operator M = J J^T in W-space."""
    if not model.is_factorized:
        raise TypeError("Pullback metrics require a factorized LoRA model")
    a = model.lora_a.detach()
    b = model.lora_b.detach()
    scale_squared = model.lora_scale**2

    def apply_metric(gradient: torch.Tensor) -> torch.Tensor:
        return scale_squared * (b @ b.T @ gradient + gradient @ a.T @ a)

    metric_new = apply_metric(new_weight_gradient)
    old_new_inner = (old_weight_gradient * new_weight_gradient).sum()
    old_metric_new_inner = (old_weight_gradient * metric_new).sum()
    new_metric_new_energy = (new_weight_gradient * metric_new).sum()
    cosine = old_metric_new_inner / (
        old_weight_gradient.norm() * metric_new.norm()
    ).clamp_min(1e-20)
    return {
        "weight_gradient_old_norm": float(old_weight_gradient.norm()),
        "weight_gradient_new_norm": float(new_weight_gradient.norm()),
        "weight_gradient_inner": float(old_new_inner),
        "pullback_new_norm": float(metric_new.norm()),
        "pullback_new_energy": float(new_metric_new_energy),
        "pullback_old_new_inner": float(old_metric_new_inner),
        "pullback_old_new_cosine": float(cosine),
        "predicted_sgd_old_interference": float(-old_metric_new_inner),
    }


def actual_factor_step_metrics(
    model: AdaptiveMLP,
    before_a: torch.Tensor,
    before_b: torch.Tensor,
    before_weight: torch.Tensor,
) -> dict[str, float]:
    """Decompose one actual factor step into tangent and bilinear W updates."""
    after_a = model.lora_a.detach()
    after_b = model.lora_b.detach()
    after_weight = model.effective_weight().detach()
    delta_a = after_a - before_a
    delta_b = after_b - before_b
    linear = model.lora_scale * (before_b @ delta_a + delta_b @ before_a)
    bilinear = model.lora_scale * (delta_b @ delta_a)
    actual = after_weight - before_weight
    closure_error = actual - linear - bilinear
    linear_norm = linear.norm()
    bilinear_norm = bilinear.norm()
    actual_norm = actual.norm()
    return {
        "effective_step_norm": float(actual_norm),
        "linear_step_norm": float(linear_norm),
        "bilinear_step_norm": float(bilinear_norm),
        "bilinear_to_linear_ratio": float(bilinear_norm / linear_norm.clamp_min(1e-20)),
        "bilinear_to_step_ratio": float(bilinear_norm / actual_norm.clamp_min(1e-20)),
        "factor_step_closure_relative_error": float(
            closure_error.norm() / actual_norm.clamp_min(1e-20)
        ),
    }
