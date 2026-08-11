"""Function-preserving LoRA initialization utilities."""

from __future__ import annotations

from math import sqrt
from typing import Sequence

import torch


def kaiming_matched_scale() -> float:
    """Scale for orthonormal rows matching nn.Linear Kaiming expected norm.

    PyTorch's default ``nn.Linear`` initialization has entry variance
    ``1 / (3 * fan_in)``. An ``r by d`` matrix therefore has expected squared
    Frobenius norm ``r / 3``; orthonormal rows have norm squared ``r``.
    """

    return 1.0 / sqrt(3.0)


@torch.no_grad()
def initialize_lora_factors(
    a_modules: Sequence[object],
    b_modules: Sequence[object],
    bases: Sequence[torch.Tensor],
    *,
    scale: float | None = None,
) -> None:
    """Set ``A = scale * V.T`` and ``B = 0`` for every LoRA site."""

    if not (len(a_modules) == len(b_modules) == len(bases)):
        raise ValueError(
            "A/B/basis lengths must match: "
            f"{len(a_modules)}, {len(b_modules)}, {len(bases)}"
        )
    value_scale = kaiming_matched_scale() if scale is None else float(scale)
    for index, (a_module, b_module, basis) in enumerate(zip(a_modules, b_modules, bases)):
        a_weight = getattr(a_module, "weight", a_module)
        b_weight = getattr(b_module, "weight", b_module)
        if not isinstance(a_weight, torch.Tensor) or not isinstance(b_weight, torch.Tensor):
            raise TypeError(f"Site {index} does not expose tensor weights")
        expected = (int(a_weight.shape[1]), int(a_weight.shape[0]))
        if tuple(basis.shape) != expected:
            raise ValueError(
                f"Site {index} basis shape {tuple(basis.shape)} does not match expected {expected}"
            )
        a_weight.copy_(basis.t().to(device=a_weight.device, dtype=a_weight.dtype) * value_scale)
        b_weight.zero_()


@torch.no_grad()
def maximum_function_deviation(before: torch.Tensor, after: torch.Tensor) -> float:
    if before.shape != after.shape:
        raise ValueError(f"Output shapes differ: {tuple(before.shape)} and {tuple(after.shape)}")
    return float((after - before).abs().max().item())
