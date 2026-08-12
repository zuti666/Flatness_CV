from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass(frozen=True)
class TwoMoonsData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    flipped_indices: np.ndarray


def _moon_split(sample_count: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if sample_count < 2 or not np.isfinite(noise) or noise < 0:
        raise ValueError("sample_count must be >=2 and noise must be finite/nonnegative")
    rng = np.random.default_rng(int(seed))
    first = sample_count // 2
    second = sample_count - first
    outer = np.linspace(0.0, np.pi, first, endpoint=True, dtype=np.float64)
    inner = np.linspace(0.0, np.pi, second, endpoint=True, dtype=np.float64)
    x0 = np.column_stack((np.cos(outer), np.sin(outer)))
    x1 = np.column_stack((1.0 - np.cos(inner), 0.5 - np.sin(inner)))
    features = np.concatenate((x0, x1), axis=0)
    labels = np.concatenate(
        (np.zeros(first, dtype=np.int64), np.ones(second, dtype=np.int64))
    )
    features += rng.normal(0.0, float(noise), size=features.shape)
    order = rng.permutation(sample_count)
    return features[order], labels[order]


def make_two_moons(
    *,
    train_samples: int = 512,
    validation_samples: int = 1024,
    test_samples: int = 4096,
    noise: float = 0.15,
    label_flip: float = 0.10,
    seed: int = 3407,
) -> TwoMoonsData:
    """Create deterministic train/validation/test moons with train-only flips."""
    if not np.isfinite(label_flip) or not 0.0 <= label_flip < 1.0:
        raise ValueError("label_flip must be in [0,1)")
    x_train, y_train = _moon_split(int(train_samples), noise, int(seed) + 11)
    x_validation, y_validation = _moon_split(
        int(validation_samples), noise, int(seed) + 23
    )
    x_test, y_test = _moon_split(int(test_samples), noise, int(seed) + 37)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    if np.any(std <= 0) or not np.all(np.isfinite(std)):
        raise RuntimeError("invalid training feature scale")
    x_train = (x_train - mean) / std
    x_validation = (x_validation - mean) / std
    x_test = (x_test - mean) / std
    flip_count = int(round(float(label_flip) * int(train_samples)))
    rng = np.random.default_rng(int(seed) + 53)
    flipped = np.sort(
        rng.choice(int(train_samples), size=flip_count, replace=False).astype(np.int64)
    )
    y_train = y_train.copy()
    y_train[flipped] = 1 - y_train[flipped]
    return TwoMoonsData(
        x_train=x_train.astype(np.float64, copy=False),
        y_train=y_train,
        x_validation=x_validation.astype(np.float64, copy=False),
        y_validation=y_validation,
        x_test=x_test.astype(np.float64, copy=False),
        y_test=y_test,
        flipped_indices=flipped,
    )


@dataclass(frozen=True)
class FlatTanhMLP:
    hidden_width: int = 16

    @property
    def parameter_count(self) -> int:
        return 5 * int(self.hidden_width) + 2

    def init_parameters(
        self,
        seed: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        if self.hidden_width < 1:
            raise ValueError("hidden_width must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        bound1 = float(np.sqrt(6.0 / (2 + self.hidden_width)))
        bound2 = float(np.sqrt(6.0 / (self.hidden_width + 2)))
        w1 = torch.empty((self.hidden_width, 2), dtype=torch.float64).uniform_(
            -bound1, bound1, generator=generator
        )
        b1 = torch.zeros(self.hidden_width, dtype=torch.float64)
        w2 = torch.empty((2, self.hidden_width), dtype=torch.float64).uniform_(
            -bound2, bound2, generator=generator
        )
        b2 = torch.zeros(2, dtype=torch.float64)
        theta = torch.cat((w1.flatten(), b1, w2.flatten(), b2))
        return theta.to(device=device, dtype=dtype)

    def forward(self, theta: Tensor, inputs: Tensor) -> Tensor:
        _validate_theta(theta, self.parameter_count)
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError("inputs must have shape [batch,2]")
        h = int(self.hidden_width)
        cursor = 0
        w1 = theta[cursor : cursor + 2 * h].reshape(h, 2)
        cursor += 2 * h
        b1 = theta[cursor : cursor + h]
        cursor += h
        w2 = theta[cursor : cursor + 2 * h].reshape(2, h)
        cursor += 2 * h
        b2 = theta[cursor : cursor + 2]
        hidden = torch.tanh(inputs @ w1.T + b1)
        return hidden @ w2.T + b2


def _validate_theta(theta: Tensor, expected: int | None = None) -> None:
    if theta.ndim != 1 or (expected is not None and theta.numel() != expected):
        raise ValueError(f"theta must be flat with {expected or 'the expected number of'} entries")
    if not torch.isfinite(theta).all().item():
        raise ValueError("theta must be finite")


def _validate_batch(inputs: Tensor, targets: Tensor) -> None:
    if inputs.ndim != 2 or inputs.shape[1] != 2:
        raise ValueError("inputs must have shape [batch,2]")
    if targets.ndim != 1 or targets.shape[0] != inputs.shape[0]:
        raise ValueError("targets must have shape [batch]")
    if inputs.device != targets.device:
        raise ValueError("inputs and targets must share a device")
    if not torch.isfinite(inputs).all().item():
        raise ValueError("inputs must be finite")


def loss_value(model: FlatTanhMLP, theta: Tensor, inputs: Tensor, targets: Tensor) -> Tensor:
    _validate_batch(inputs, targets)
    return F.cross_entropy(model.forward(theta, inputs), targets, reduction="mean")


def gradient(
    model: FlatTanhMLP,
    theta: Tensor,
    inputs: Tensor,
    targets: Tensor,
    *,
    create_graph: bool = False,
) -> Tensor:
    point = theta if theta.requires_grad else theta.detach().requires_grad_(True)
    value = loss_value(model, point, inputs, targets)
    result = torch.autograd.grad(value, point, create_graph=create_graph)[0]
    return result if create_graph else result.detach()


def full_hessian(
    model: FlatTanhMLP, theta: Tensor, inputs: Tensor, targets: Tensor
) -> Tensor:
    point = theta.detach().requires_grad_(True)
    matrix = torch.autograd.functional.hessian(
        lambda candidate: loss_value(model, candidate, inputs, targets),
        point,
        vectorize=True,
    ).detach()
    matrix = 0.5 * (matrix + matrix.T)
    if not torch.isfinite(matrix).all().item():
        raise RuntimeError("Hessian is non-finite")
    return matrix


def hessian_vector_product(
    model: FlatTanhMLP,
    theta: Tensor,
    inputs: Tensor,
    targets: Tensor,
    vector: Tensor,
) -> Tensor:
    _validate_theta(vector, theta.numel())
    point = theta.detach().requires_grad_(True)
    first = gradient(model, point, inputs, targets, create_graph=True)
    product = torch.autograd.grad(torch.dot(first, vector.detach()), point)[0].detach()
    if not torch.isfinite(product).all().item():
        raise RuntimeError("HVP is non-finite")
    return product


def unit(vector: Tensor, epsilon: float = 1e-12) -> Tensor:
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm).item():
        raise ValueError("vector norm is non-finite")
    if float(norm) <= float(epsilon):
        return torch.zeros_like(vector)
    return vector / norm


@dataclass
class E002Trace:
    method: str
    direction: Tensor
    clean_gradient: Tensor
    correction: Tensor
    perturbation: Tensor
    path_points: Tensor | None = None
    path_gradients: Tensor | None = None
    probe_direction: Tensor | None = None
    probe_increment: Tensor | None = None
    final_regularizer: Tensor | None = None
    path_mean_surrogate: Tensor | None = None
    effective_direction: Tensor | None = None
    slow_delta: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _trace(method: str, direction: Tensor, clean: Tensor, perturbation: Tensor, **kwargs: Any) -> E002Trace:
    values = (direction, clean, perturbation)
    if any(not torch.isfinite(value).all().item() for value in values):
        raise RuntimeError(f"{method} produced a non-finite tensor")
    return E002Trace(
        method=method,
        direction=direction.detach(),
        clean_gradient=clean.detach(),
        correction=(direction - clean).detach(),
        perturbation=perturbation.detach(),
        **kwargs,
    )


def sgd_direction(model: FlatTanhMLP, theta: Tensor, inputs: Tensor, targets: Tensor) -> E002Trace:
    clean = gradient(model, theta, inputs, targets)
    return _trace("sgd", clean, clean, torch.zeros_like(theta))


def sam_direction(
    model: FlatTanhMLP,
    theta: Tensor,
    inputs: Tensor,
    targets: Tensor,
    rho: float,
    *,
    epsilon: float = 1e-12,
) -> E002Trace:
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    clean = gradient(model, theta, inputs, targets)
    perturbation = float(rho) * unit(clean, epsilon)
    direction = gradient(model, theta + perturbation, inputs, targets)
    return _trace("sam", direction, clean, perturbation)


def idealized_gam_direction(
    model: FlatTanhMLP,
    theta: Tensor,
    inputs: Tensor,
    targets: Tensor,
    rho: float,
    *,
    alpha: float = 1.0,
    epsilon: float = 1e-12,
) -> E002Trace:
    """Paper-Algorithm-1 exact-HVP, same-batch reference (not accelerated GAM)."""
    if not np.isfinite(rho) or rho < 0 or not np.isfinite(alpha):
        raise ValueError("rho/alpha must be finite and rho nonnegative")
    clean = gradient(model, theta, inputs, targets)
    first_hvp = hessian_vector_product(
        model, theta, inputs, targets, unit(clean, epsilon)
    )
    probe_direction = unit(first_hvp, epsilon)
    perturbation = float(rho) * probe_direction
    probe_gradient = gradient(model, theta + perturbation, inputs, targets)
    final_regularizer = float(rho) * hessian_vector_product(
        model,
        theta + perturbation,
        inputs,
        targets,
        unit(probe_gradient, epsilon),
    )
    direction = clean + float(alpha) * final_regularizer
    return _trace(
        "gam_exact_hvp_same_batch_alpha",
        direction,
        clean,
        perturbation,
        probe_direction=probe_direction.detach(),
        probe_increment=(probe_gradient - clean).detach(),
        final_regularizer=final_regularizer.detach(),
        metadata={"alpha": float(alpha), "oracle": "same_batch_mean_cross_entropy"},
    )


def _ascent_path(
    model: FlatTanhMLP,
    theta: Tensor,
    inputs: Tensor,
    targets: Tensor,
    rho_step: float,
    inner_steps: int,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if inner_steps < 1 or not np.isfinite(rho_step) or rho_step < 0:
        raise ValueError("inner_steps must be positive and rho_step finite/nonnegative")
    clean = gradient(model, theta, inputs, targets)
    point = theta.detach().clone()
    points = [point]
    gradients = []
    for _ in range(int(inner_steps)):
        ascent_gradient = gradient(model, point, inputs, targets)
        point = point + float(rho_step) * unit(ascent_gradient, epsilon)
        gradients.append(gradient(model, point, inputs, targets))
        points.append(point)
    return clean, torch.stack(points), torch.stack(gradients)


def multistep_sam_direction(
    model: FlatTanhMLP,
    theta: Tensor,
    inputs: Tensor,
    targets: Tensor,
    rho_step: float,
    inner_steps: int,
    *,
    epsilon: float = 1e-12,
) -> E002Trace:
    clean, points, gradients = _ascent_path(
        model, theta, inputs, targets, rho_step, inner_steps, epsilon
    )
    return _trace(
        "multistep_sam",
        gradients[-1],
        clean,
        points[-1] - theta,
        path_points=points.detach(),
        path_gradients=gradients.detach(),
        metadata={"inner_steps": int(inner_steps), "rho_step": float(rho_step)},
    )


def lookbehind_plain_sgd_delta(
    model: FlatTanhMLP,
    theta: Tensor,
    inputs: Tensor,
    targets: Tensor,
    rho_step: float,
    inner_steps: int,
    learning_rate: float,
    alpha: float,
    *,
    epsilon: float = 1e-12,
) -> E002Trace:
    """Faithful plain-SGD slow delta and the distinct path-mean surrogate."""
    if not np.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("learning_rate must be finite and nonnegative")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    clean, points, gradients = _ascent_path(
        model, theta, inputs, targets, rho_step, inner_steps, epsilon
    )
    surrogate = gradients.mean(dim=0)
    effective = float(alpha) * gradients.sum(dim=0)
    slow_delta = -float(learning_rate) * effective
    return _trace(
        "lookbehind_faithful_plain_sgd",
        effective,
        clean,
        points[-1] - theta,
        path_points=points.detach(),
        path_gradients=gradients.detach(),
        path_mean_surrogate=surrogate.detach(),
        effective_direction=effective.detach(),
        slow_delta=slow_delta.detach(),
        metadata={
            "inner_steps": int(inner_steps),
            "rho_step": float(rho_step),
            "alpha": float(alpha),
            "fast_steps": int(inner_steps),
        },
    )


def orthogonal_component(sam_gradient: Tensor, clean_gradient: Tensor, epsilon: float = 1e-12) -> Tensor:
    norm_sq = torch.dot(clean_gradient, clean_gradient)
    if float(norm_sq) <= float(epsilon) ** 2:
        return torch.zeros_like(clean_gradient)
    parallel = torch.dot(sam_gradient, clean_gradient) / norm_sq * clean_gradient
    return (sam_gradient - parallel).detach()


def recompose_looksam_direction(
    clean_gradient: Tensor,
    cached_orthogonal: Tensor,
    *,
    alpha: float = 0.7,
    epsilon: float = 1e-12,
) -> Tensor:
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    cached_norm = torch.linalg.vector_norm(cached_orthogonal)
    if float(cached_norm) <= float(epsilon):
        return clean_gradient.detach().clone()
    scale = float(alpha) * torch.linalg.vector_norm(clean_gradient) / cached_norm
    direction = clean_gradient + scale * cached_orthogonal
    if not torch.isfinite(direction).all().item():
        raise RuntimeError("LookSAM recomposition is non-finite")
    return direction.detach()


def sam_k_direction(
    model: FlatTanhMLP,
    theta: Tensor,
    inputs: Tensor,
    targets: Tensor,
    rho: float,
    *,
    step: int,
    period: int,
    epsilon: float = 1e-12,
) -> E002Trace:
    if period < 1 or step < 0:
        raise ValueError("period must be positive and step nonnegative")
    if step % int(period) == 0:
        trace = sam_direction(model, theta, inputs, targets, rho, epsilon=epsilon)
        trace.method = f"sam_{period}_refresh"
        trace.metadata.update({"refresh": True, "period": int(period), "step": int(step)})
        return trace
    trace = sgd_direction(model, theta, inputs, targets)
    trace.method = f"sam_{period}_nonrefresh"
    trace.metadata.update({"refresh": False, "period": int(period), "step": int(step)})
    return trace
