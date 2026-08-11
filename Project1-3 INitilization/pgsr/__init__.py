"""Core components for Preview-Guided Subspace Reuse (PGSR) LoRA."""

from .bank import SubspaceBank, right_subspace_from_factors
from .initialization import initialize_lora_factors, kaiming_matched_scale
from .selector import (
    SelectionResult,
    candidate_energy,
    make_fresh_candidate,
    make_orthogonal_complement_candidate,
    select_candidate,
)

__all__ = [
    "SelectionResult",
    "SubspaceBank",
    "candidate_energy",
    "initialize_lora_factors",
    "kaiming_matched_scale",
    "make_fresh_candidate",
    "make_orthogonal_complement_candidate",
    "right_subspace_from_factors",
    "select_candidate",
]
