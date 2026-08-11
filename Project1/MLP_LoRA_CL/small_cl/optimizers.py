from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def make_sgd(parameters: Iterable[nn.Parameter], config: dict) -> torch.optim.SGD:
    return torch.optim.SGD(
        parameters,
        lr=float(config["lr"]),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
    )


def _global_norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise RuntimeError("No trainable gradients were produced")
    return torch.linalg.vector_norm(torch.stack([tensor.norm(2) for tensor in tensors]))


def _clip(parameters: list[nn.Parameter], threshold: float | None) -> None:
    if threshold is not None:
        torch.nn.utils.clip_grad_norm_(parameters, float(threshold))


def _effective_weight(model: nn.Module) -> torch.Tensor:
    if not hasattr(model, "effective_weight"):
        raise TypeError("effective-weight perturbations require model.effective_weight()")
    return model.effective_weight()


def _project_directions(model: nn.Module, directions: list[torch.Tensor]) -> list[torch.Tensor]:
    if hasattr(model, "project_trainable_directions"):
        return model.project_trainable_directions(directions)
    return directions


def _set_gradients(parameters: list[nn.Parameter], gradients: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient.detach().clone()


def _finish_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.step()
    if hasattr(model, "retract_trainable_update"):
        model.retract_trainable_update()


def _make_perturbations(
    model: nn.Module,
    parameters: list[nn.Parameter],
    directions: list[torch.Tensor],
    radius: float,
    metric: str,
) -> tuple[list[torch.Tensor], float, float]:
    """Scale one parameter direction in raw or effective-W norm."""
    raw_direction_norm = _global_norm(directions).clamp_min(1e-20)
    unit_directions = [direction / raw_direction_norm for direction in directions]
    metric = metric.lower()
    if metric == "parameter":
        scale = float(radius)
    elif metric == "effective_weight":
        with torch.no_grad():
            reference = _effective_weight(model).detach().clone()

            def mapped_norm(candidate: float) -> float:
                for parameter, direction in zip(parameters, unit_directions):
                    parameter.add_(candidate * direction)
                value = float((_effective_weight(model) - reference).norm())
                for parameter, direction in zip(parameters, unit_directions):
                    parameter.sub_(candidate * direction)
                return value

            low, high = 0.0, max(float(radius), 1e-8)
            for _ in range(30):
                if mapped_norm(high) >= radius:
                    break
                high *= 2.0
            else:
                raise RuntimeError("Could not reach requested effective-weight perturbation radius")
            for _ in range(40):
                middle = 0.5 * (low + high)
                if mapped_norm(middle) < radius:
                    low = middle
                else:
                    high = middle
            scale = high
    else:
        raise ValueError("training.perturbation_metric must be 'parameter' or 'effective_weight'")

    perturbations = [scale * direction for direction in unit_directions]
    with torch.no_grad():
        reference = _effective_weight(model).detach().clone()
        for parameter, perturbation in zip(parameters, perturbations):
            parameter.add_(perturbation)
        effective_norm = float((_effective_weight(model) - reference).norm())
        for parameter, perturbation in zip(parameters, perturbations):
            parameter.sub_(perturbation)
    return perturbations, float(scale), effective_norm


def train_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    method: str,
    config: dict,
) -> dict[str, float]:
    """Run one SGD, SAM, finite-difference GAM, or exact-HVP GAM batch."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    method = method.lower()
    if method == "sgd":
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        gradients = _project_directions(
            model, [parameter.grad.detach().clone() for parameter in parameters]
        )
        _set_gradients(parameters, gradients)
        _clip(parameters, config.get("grad_clip"))
        _finish_step(model, optimizer)
        return {
            "loss": float(loss.detach()),
            "perturbed_loss": float("nan"),
            "curvature_correction_norm": float("nan"),
            "effective_perturbation_norm": float("nan"),
        }

    if method == "sam":
        optimizer.zero_grad(set_to_none=True)
        clean_loss = criterion(model(inputs), targets)
        clean_loss.backward()
        clean_grads = _project_directions(
            model, [parameter.grad.detach().clone() for parameter in parameters]
        )
        perturbations, _, effective_norm = _make_perturbations(
            model,
            parameters,
            clean_grads,
            float(config["sam_rho"]),
            config.get("perturbation_metric", "parameter"),
        )
        with torch.no_grad():
            for parameter, perturbation in zip(parameters, perturbations):
                parameter.add_(perturbation)
        optimizer.zero_grad(set_to_none=True)
        perturbed_loss = criterion(model(inputs), targets)
        perturbed_loss.backward()
        perturbed_grads = [parameter.grad.detach().clone() for parameter in parameters]
        with torch.no_grad():
            for parameter, perturbation in zip(parameters, perturbations):
                parameter.sub_(perturbation)
        perturbed_grads = _project_directions(model, perturbed_grads)
        _set_gradients(parameters, perturbed_grads)
        correction_norm = _global_norm(
            [perturbed - clean for perturbed, clean in zip(perturbed_grads, clean_grads)]
        )
        _clip(parameters, config.get("grad_clip"))
        _finish_step(model, optimizer)
        return {
            "loss": float(clean_loss.detach()),
            "perturbed_loss": float(perturbed_loss.detach()),
            "curvature_correction_norm": float(correction_norm),
            "effective_perturbation_norm": effective_norm,
        }

    if method == "random_perturb":
        optimizer.zero_grad(set_to_none=True)
        clean_loss = criterion(model(inputs), targets)
        clean_loss.backward()
        clean_grads = [parameter.grad.detach().clone() for parameter in parameters]
        random_directions = [torch.randn_like(parameter) for parameter in parameters]
        perturbations, _, effective_norm = _make_perturbations(
            model,
            parameters,
            random_directions,
            float(config["sam_rho"]),
            config.get("perturbation_metric", "parameter"),
        )
        with torch.no_grad():
            for parameter, perturbation in zip(parameters, perturbations):
                parameter.add_(perturbation)
        optimizer.zero_grad(set_to_none=True)
        perturbed_loss = criterion(model(inputs), targets)
        perturbed_loss.backward()
        perturbed_grads = [parameter.grad.detach().clone() for parameter in parameters]
        correction_norm = _global_norm(
            [perturbed - clean for perturbed, clean in zip(perturbed_grads, clean_grads)]
        )
        with torch.no_grad():
            for parameter, perturbation in zip(parameters, perturbations):
                parameter.sub_(perturbation)
        _clip(parameters, config.get("grad_clip"))
        optimizer.step()
        return {
            "loss": float(clean_loss.detach()),
            "perturbed_loss": float(perturbed_loss.detach()),
            "curvature_correction_norm": float(correction_norm),
            "effective_perturbation_norm": effective_norm,
        }

    if method == "gam_fd":
        radius = float(config["gam_radius"])
        if radius <= 0:
            raise ValueError("gam_radius must be positive")
        optimizer.zero_grad(set_to_none=True)
        clean_loss = criterion(model(inputs), targets)
        clean_loss.backward()
        clean_grads = [parameter.grad.detach().clone() for parameter in parameters]
        perturbations, raw_step, effective_norm = _make_perturbations(
            model,
            parameters,
            clean_grads,
            radius,
            config.get("perturbation_metric", "parameter"),
        )
        with torch.no_grad():
            for parameter, perturbation in zip(parameters, perturbations):
                parameter.add_(perturbation)
        optimizer.zero_grad(set_to_none=True)
        perturbed_loss = criterion(model(inputs), targets)
        perturbed_loss.backward()
        perturbed_grads = [parameter.grad.detach().clone() for parameter in parameters]
        with torch.no_grad():
            for parameter, perturbation, clean_gradient, perturbed_gradient in zip(
                parameters, perturbations, clean_grads, perturbed_grads
            ):
                parameter.sub_(perturbation)
                hvp_estimate = (perturbed_gradient - clean_gradient) / raw_step
                parameter.grad = clean_gradient + float(config["gam_weight"]) * hvp_estimate
        _clip(parameters, config.get("grad_clip"))
        optimizer.step()
        return {
            "loss": float(clean_loss.detach()),
            "perturbed_loss": float(perturbed_loss.detach()),
            "curvature_correction_norm": float(
                _global_norm(
                    [(perturbed - clean) / raw_step for perturbed, clean in zip(perturbed_grads, clean_grads)]
                )
            ),
            "effective_perturbation_norm": effective_norm,
        }

    if method == "gam_exact":
        optimizer.zero_grad(set_to_none=True)
        clean_loss = criterion(model(inputs), targets)
        gradients = torch.autograd.grad(clean_loss, parameters, create_graph=True)
        norm = _global_norm(list(gradients)).clamp_min(1e-12)
        directions = [gradient.detach() / norm.detach() for gradient in gradients]
        directional_gradient = sum(
            (gradient * direction).sum() for gradient, direction in zip(gradients, directions)
        )
        hvps = torch.autograd.grad(directional_gradient, parameters)
        with torch.no_grad():
            for parameter, gradient, hvp in zip(parameters, gradients, hvps):
                parameter.grad = gradient.detach() + float(config["gam_weight"]) * hvp.detach()
        _clip(parameters, config.get("grad_clip"))
        optimizer.step()
        return {
            "loss": float(clean_loss.detach()),
            "perturbed_loss": float("nan"),
            "curvature_correction_norm": float(_global_norm([hvp.detach() for hvp in hvps])),
            "effective_perturbation_norm": float("nan"),
        }

    raise ValueError(f"Unsupported optimizer method: {method}")
