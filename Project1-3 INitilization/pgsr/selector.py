"""Preview-gradient scoring and MAP selection for PGSR-LoRA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from typing import Sequence

import torch


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    is_fresh: bool
    energy_raw: float
    energy_normalized: float
    prior: float
    map_score: float
    posterior: float


@dataclass(frozen=True)
class SelectionResult:
    selected_index: int
    selected_id: str
    posterior_entropy: float
    rows: tuple[CandidateScore, ...]

    def to_dict(self) -> dict:
        return {
            "selected_index": self.selected_index,
            "selected_id": self.selected_id,
            "selected_is_fresh": self.rows[self.selected_index].is_fresh,
            "posterior_entropy": self.posterior_entropy,
            "candidates": [asdict(row) for row in self.rows],
        }


def random_orthonormal_basis(
    input_dim: int,
    rank: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if not 0 < int(rank) <= int(input_dim):
        raise ValueError(f"rank must be in [1, input_dim], got {rank} for {input_dim}")
    sample = torch.randn(int(input_dim), int(rank), generator=generator, dtype=torch.float32)
    basis, _ = torch.linalg.qr(sample, mode="reduced")
    return basis.contiguous()


def make_fresh_candidate(
    gradients: Sequence[torch.Tensor],
    rank: int,
    *,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return [
        random_orthonormal_basis(int(gradient.shape[1]), int(rank), generator=generator)
        for gradient in gradients
    ]


def make_orthogonal_complement_candidate(
    historical_candidates: Sequence[Sequence[torch.Tensor]],
    gradients: Sequence[torch.Tensor],
    rank: int,
    *,
    seed: int,
    relative_tolerance: float = 1e-6,
) -> list[torch.Tensor]:
    """Create a seeded negative-control basis orthogonal to historical spans."""

    if not historical_candidates:
        raise ValueError("At least one historical candidate is required")
    if any(len(candidate) != len(gradients) for candidate in historical_candidates):
        raise ValueError("Historical candidates and gradients must have matching sites")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    result = []
    for site_index, gradient in enumerate(gradients):
        input_dim = int(gradient.shape[1])
        if not 0 < int(rank) <= input_dim:
            raise ValueError(f"rank must be in [1, {input_dim}], got {rank}")
        stacked = torch.cat(
            [
                candidate[site_index].detach().to(device="cpu", dtype=torch.float32)
                for candidate in historical_candidates
            ],
            dim=1,
        )
        span_u, span_s, _ = torch.linalg.svd(stacked, full_matrices=False)
        largest = float(span_s[0].item()) if span_s.numel() else 0.0
        span_rank = (
            int((span_s > largest * float(relative_tolerance)).sum().item())
            if largest > 0.0
            else 0
        )
        span = span_u[:, :span_rank]
        if input_dim - span_rank < int(rank):
            raise ValueError(
                f"Historical span leaves only {input_dim - span_rank} dimensions "
                f"for a rank-{rank} orthogonal control at site {site_index}"
            )
        sample = torch.randn(input_dim, int(rank), generator=generator, dtype=torch.float32)
        if span_rank:
            sample = sample - span @ (span.t() @ sample)
        basis, _ = torch.linalg.qr(sample, mode="reduced")
        result.append(basis.contiguous())
    return result


def candidate_energy(
    gradients: Sequence[torch.Tensor], bases: Sequence[torch.Tensor], eps: float
) -> tuple[float, float]:
    """Return raw and gradient-normalized projected energy for one candidate."""

    if len(gradients) != len(bases):
        raise ValueError(f"Gradient/basis site mismatch: {len(gradients)} versus {len(bases)}")
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    for gradient, basis in zip(gradients, bases):
        gradient32 = gradient.detach().to(device="cpu", dtype=torch.float32)
        basis32 = basis.detach().to(device="cpu", dtype=torch.float32)
        if gradient32.ndim != 2 or basis32.ndim != 2:
            raise ValueError("Every gradient and basis must be a matrix")
        if gradient32.shape[1] != basis32.shape[0]:
            raise ValueError(
                f"Incompatible gradient/basis: G{tuple(gradient32.shape)}, V{tuple(basis32.shape)}"
            )
        projected = gradient32 @ basis32
        numerator += projected.double().square().sum()
        denominator += gradient32.double().square().sum()
    raw = float(numerator.item())
    normalized = float((numerator / denominator.clamp_min(float(eps))).item())
    return raw, normalized


def select_candidate(
    gradients: Sequence[torch.Tensor],
    historical_candidates: Sequence[Sequence[torch.Tensor]],
    fresh_candidate: Sequence[torch.Tensor],
    *,
    historical_ids: Sequence[str] | None = None,
    fresh_prior: float = 0.5,
    gamma: float = 12.0,
    eps: float = 1e-12,
) -> SelectionResult:
    """Score fresh + historical task-level candidates and return the MAP choice."""

    history_count = len(historical_candidates)
    if historical_ids is None:
        historical_ids = [f"history_{idx}" for idx in range(history_count)]
    if len(historical_ids) != history_count:
        raise ValueError("historical_ids must have one entry per historical candidate")
    if history_count and not 0.0 < float(fresh_prior) < 1.0:
        raise ValueError("fresh_prior must be strictly between zero and one when history exists")

    candidates = [fresh_candidate, *historical_candidates]
    candidate_ids = ["fresh", *[str(value) for value in historical_ids]]
    if history_count:
        priors = [float(fresh_prior)] + [
            (1.0 - float(fresh_prior)) / history_count for _ in range(history_count)
        ]
    else:
        priors = [1.0]

    energies = [candidate_energy(gradients, candidate, eps) for candidate in candidates]
    logits = torch.tensor(
        [float(gamma) * normalized + log(prior) for (_, normalized), prior in zip(energies, priors)],
        dtype=torch.float64,
    )
    posterior = torch.softmax(logits, dim=0)
    selected_index = int(torch.argmax(logits).item())
    entropy = float((-(posterior * posterior.clamp_min(eps).log()).sum()).item())
    rows = tuple(
        CandidateScore(
            candidate_id=candidate_id,
            is_fresh=index == 0,
            energy_raw=energies[index][0],
            energy_normalized=energies[index][1],
            prior=priors[index],
            map_score=float(logits[index].item()),
            posterior=float(posterior[index].item()),
        )
        for index, candidate_id in enumerate(candidate_ids)
    )
    return SelectionResult(selected_index, candidate_ids[selected_index], entropy, rows)
