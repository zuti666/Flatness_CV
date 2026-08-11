from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .models import AdaptiveMLP
from .optimizers import _make_perturbations


@dataclass
class FactorStepCandidate:
    """A factor-coordinate step before it is committed to the model."""

    delta_a: torch.Tensor
    delta_b: torch.Tensor
    clean_delta_a: torch.Tensor
    clean_delta_b: torch.Tensor
    clean_loss: float
    perturbed_loss: float
    effective_perturbation_norm: float
    parameter_gradient_correction_norm: float


def _factor_parameters(model: AdaptiveMLP) -> tuple[nn.Parameter, nn.Parameter]:
    if not model.is_factorized or model.lora_a is None or model.lora_b is None:
        raise TypeError("Step normalization requires a factorized LoRA model")
    return model.lora_a, model.lora_b


def _gradient_step(
    parameters: tuple[nn.Parameter, nn.Parameter],
    gradients: tuple[torch.Tensor, torch.Tensor],
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(
        -float(learning_rate) * (gradient.detach() + float(weight_decay) * parameter.detach())
        for parameter, gradient in zip(parameters, gradients)
    )


def make_factor_step_candidate(
    model: AdaptiveMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    method: str,
    config: dict[str, Any],
) -> FactorStepCandidate:
    """Construct the SGD or SAM factor step without mutating the model.

    P3 deliberately fixes momentum to zero.  Returning both the clean and the
    selected candidate makes the effective-weight SAM correction measurable,
    rather than inferring it from an interference-prediction residual.
    """

    if abs(float(config.get("momentum", 0.0))) > 1e-15:
        raise ValueError("P3 factor candidates require zero momentum")
    if config.get("grad_clip") is not None:
        raise ValueError("P3 factor candidates currently require grad_clip: null")
    parameters = _factor_parameters(model)
    learning_rate = float(config["lr"])
    weight_decay = float(config.get("weight_decay", 0.0))
    criterion = nn.CrossEntropyLoss()
    clean_loss_tensor = criterion(model(inputs), targets)
    clean_gradients = tuple(
        value.detach()
        for value in torch.autograd.grad(clean_loss_tensor, parameters)
    )
    clean_delta_a, clean_delta_b = _gradient_step(
        parameters, clean_gradients, learning_rate, weight_decay
    )

    key = str(method).lower()
    if key == "sgd":
        return FactorStepCandidate(
            delta_a=clean_delta_a,
            delta_b=clean_delta_b,
            clean_delta_a=clean_delta_a,
            clean_delta_b=clean_delta_b,
            clean_loss=float(clean_loss_tensor.detach()),
            perturbed_loss=float("nan"),
            effective_perturbation_norm=float("nan"),
            parameter_gradient_correction_norm=0.0,
        )
    if key != "sam":
        raise ValueError(f"P3 supports SGD and SAM candidates, got {method}")

    perturbations, _, effective_norm = _make_perturbations(
        model,
        list(parameters),
        list(clean_gradients),
        float(config["sam_rho"]),
        str(config.get("perturbation_metric", "effective_weight")),
    )
    with torch.no_grad():
        for parameter, perturbation in zip(parameters, perturbations):
            parameter.add_(perturbation)
    try:
        perturbed_loss_tensor = criterion(model(inputs), targets)
        perturbed_gradients = tuple(
            value.detach()
            for value in torch.autograd.grad(perturbed_loss_tensor, parameters)
        )
    finally:
        with torch.no_grad():
            for parameter, perturbation in zip(parameters, perturbations):
                parameter.sub_(perturbation)

    delta_a, delta_b = _gradient_step(
        parameters, perturbed_gradients, learning_rate, weight_decay
    )
    correction = torch.linalg.vector_norm(
        torch.stack(
            [
                (perturbed - clean).norm()
                for perturbed, clean in zip(perturbed_gradients, clean_gradients)
            ]
        )
    )
    return FactorStepCandidate(
        delta_a=delta_a,
        delta_b=delta_b,
        clean_delta_a=clean_delta_a,
        clean_delta_b=clean_delta_b,
        clean_loss=float(clean_loss_tensor.detach()),
        perturbed_loss=float(perturbed_loss_tensor.detach()),
        effective_perturbation_norm=float(effective_norm),
        parameter_gradient_correction_norm=float(correction),
    )


def factor_effective_delta(
    model: AdaptiveMLP,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Map a jointly scaled factor step to its exact effective-W change."""

    _factor_parameters(model)
    scaled_a = float(alpha) * delta_a
    scaled_b = float(alpha) * delta_b
    return model.lora_scale * (
        model.lora_b.detach() @ scaled_a
        + scaled_b @ model.lora_a.detach()
        + scaled_b @ scaled_a
    )


def solve_effective_step_scale(
    model: AdaptiveMLP,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    target_norm: float,
    *,
    relative_tolerance: float = 1e-6,
    maximum_alpha: float = 64.0,
    iterations: int = 60,
) -> tuple[float, torch.Tensor, float]:
    """Find the smallest bracketed positive scale reaching ``target_norm``.

    The exact norm contains a linear and a quadratic term.  We expand from zero
    and retain the first sampled sign-changing bracket before bisection.  The
    caller audits the raw/normalized direction cosine because changing alpha
    can also change the linear-to-bilinear ratio.
    """

    target = float(target_norm)
    if not target > 0:
        raise ValueError("target_norm must be positive")
    if maximum_alpha <= 0:
        raise ValueError("maximum_alpha must be positive")

    low = 0.0
    high = min(1.0, float(maximum_alpha))
    high_delta = factor_effective_delta(model, delta_a, delta_b, high)
    high_value = float(high_delta.norm())
    while high_value < target and high < maximum_alpha:
        low = high
        high = min(2.0 * high, float(maximum_alpha))
        high_delta = factor_effective_delta(model, delta_a, delta_b, high)
        high_value = float(high_delta.norm())
    if high_value < target:
        raise RuntimeError(
            f"Could not reach target effective step {target:.8g}; "
            f"maximum at alpha={high:g} was {high_value:.8g}"
        )

    for _ in range(int(iterations)):
        middle = 0.5 * (low + high)
        middle_value = float(
            factor_effective_delta(model, delta_a, delta_b, middle).norm()
        )
        if middle_value < target:
            low = middle
        else:
            high = middle
    achieved_delta = factor_effective_delta(model, delta_a, delta_b, high)
    achieved = float(achieved_delta.norm())
    relative_error = abs(achieved - target) / max(target, 1e-20)
    if relative_error > float(relative_tolerance):
        raise RuntimeError(
            f"Step normalization error {relative_error:.3e} exceeds "
            f"tolerance {relative_tolerance:.3e}"
        )
    return float(high), achieved_delta, relative_error


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        nn.functional.cosine_similarity(
            first.flatten(), second.flatten(), dim=0, eps=1e-20
        )
    )


def resolve_factor_candidate(
    model: AdaptiveMLP,
    candidate: FactorStepCandidate,
    update_mode: str,
    target_norm: float | None,
    *,
    maximum_alpha: float = 64.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Resolve a raw or exact-norm factor candidate and return audit metrics."""

    raw_delta = factor_effective_delta(
        model, candidate.delta_a, candidate.delta_b, 1.0
    )
    raw_norm = float(raw_delta.norm())
    key = str(update_mode).lower()
    if key == "raw":
        alpha = 1.0
        effective_delta = raw_delta
        target = raw_norm
        relative_error = 0.0
    elif key == "normalized":
        if target_norm is None:
            raise ValueError("normalized updates require target_norm")
        target = float(target_norm)
        alpha, effective_delta, relative_error = solve_effective_step_scale(
            model,
            candidate.delta_a,
            candidate.delta_b,
            target,
            maximum_alpha=maximum_alpha,
        )
    else:
        raise ValueError(f"Unknown P3 update mode: {update_mode}")

    resolved_a = alpha * candidate.delta_a
    resolved_b = alpha * candidate.delta_b
    clean_raw_delta = factor_effective_delta(
        model, candidate.clean_delta_a, candidate.clean_delta_b, 1.0
    )
    if key == "normalized":
        clean_alpha, clean_effective_delta, _ = solve_effective_step_scale(
            model,
            candidate.clean_delta_a,
            candidate.clean_delta_b,
            target,
            maximum_alpha=maximum_alpha,
        )
    else:
        clean_alpha = 1.0
        clean_effective_delta = clean_raw_delta

    correction_raw = raw_delta - clean_raw_delta
    correction_resolved = effective_delta - clean_effective_delta
    metrics = {
        "alpha": float(alpha),
        "clean_alpha": float(clean_alpha),
        "raw_effective_step_norm": raw_norm,
        "target_effective_step_norm": float(target),
        "effective_step_norm": float(effective_delta.norm()),
        "target_relative_error": float(relative_error),
        "raw_to_resolved_cosine": _cosine(raw_delta, effective_delta),
        "sam_raw_correction_ratio": float(
            correction_raw.norm() / clean_raw_delta.norm().clamp_min(1e-20)
        ),
        "sam_raw_update_cosine": _cosine(raw_delta, clean_raw_delta),
        "sam_resolved_correction_ratio": float(
            correction_resolved.norm()
            / clean_effective_delta.norm().clamp_min(1e-20)
        ),
        "sam_resolved_update_cosine": _cosine(
            effective_delta, clean_effective_delta
        ),
        "parameter_gradient_correction_norm": float(
            candidate.parameter_gradient_correction_norm
        ),
        "effective_perturbation_norm": float(
            candidate.effective_perturbation_norm
        ),
    }
    return resolved_a, resolved_b, effective_delta, metrics


def apply_factor_step_(
    model: AdaptiveMLP, delta_a: torch.Tensor, delta_b: torch.Tensor
) -> None:
    """Commit a previously resolved factor step."""

    parameter_a, parameter_b = _factor_parameters(model)
    with torch.no_grad():
        parameter_a.add_(delta_a)
        parameter_b.add_(delta_b)
