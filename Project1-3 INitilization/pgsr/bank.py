"""Historical effective-update subspace bank used by PGSR-LoRA.

The bank stores the right singular subspace of ``B @ A`` rather than the raw
rows of ``A``.  Extraction uses two reduced QR decompositions and one rank by
rank SVD, so a dense ``d_out by d_in`` update never has to be materialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch


TensorPair = tuple[torch.Tensor, torch.Tensor]


def _weight(value: object) -> torch.Tensor:
    tensor = getattr(value, "weight", value)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor or module with .weight, got {type(value)!r}")
    return tensor.detach()


def right_subspace_from_factors(
    b: torch.Tensor,
    a: torch.Tensor,
    *,
    rank: int | None = None,
    supplement_seed: int = 0,
    relative_rank_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a fixed-width right-subspace basis and spectrum of ``b @ a``.

    Args:
        b: LoRA output factor with shape ``[d_out, r]``.
        a: LoRA input factor with shape ``[r, d_in]``.
        rank: Number of basis columns to return. Defaults to the factor rank.
        supplement_seed: Seed used only when the effective update is rank
            deficient and needs fresh orthogonal supplement columns.
        relative_rank_tolerance: Singular values no larger than this fraction
            of the largest value are treated as numerically zero.

    Returns:
        ``(V, singular_values)`` where ``V`` has shape ``[d_in, rank]`` and
        orthonormal columns. Its first ``numerical_rank`` columns span the
        effective update; missing columns are deterministic fresh directions
        orthogonal to that span. Computation is carried out in FP32.
    """

    if b.ndim != 2 or a.ndim != 2:
        raise ValueError(f"LoRA factors must be matrices, got B{tuple(b.shape)}, A{tuple(a.shape)}")
    if b.shape[1] != a.shape[0]:
        raise ValueError(f"Incompatible LoRA factors: B{tuple(b.shape)}, A{tuple(a.shape)}")

    factor_rank = int(a.shape[0])
    requested_rank = factor_rank if rank is None else int(rank)
    max_rank = min(factor_rank, int(b.shape[0]), int(a.shape[1]))
    if not 0 < requested_rank <= max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}], got {requested_rank}")
    if relative_rank_tolerance < 0:
        raise ValueError("relative_rank_tolerance must be non-negative")

    b32 = b.detach().to(device="cpu", dtype=torch.float32)
    at32 = a.detach().t().to(device="cpu", dtype=torch.float32)
    q_b, r_b = torch.linalg.qr(b32, mode="reduced")
    q_a, r_a = torch.linalg.qr(at32, mode="reduced")
    del q_b  # Only the right singular subspace is needed.
    middle = r_b @ r_a.t()
    _, singular_values, vh = torch.linalg.svd(middle, full_matrices=False)
    raw_basis = q_a @ vh.t()
    spectrum = singular_values[:requested_rank].contiguous()

    largest = float(spectrum[0].item()) if spectrum.numel() else 0.0
    threshold = largest * float(relative_rank_tolerance)
    effective_rank = int((spectrum > threshold).sum().item()) if largest > 0.0 else 0
    effective_rank = min(effective_rank, requested_rank)
    effective_basis = raw_basis[:, :effective_rank]

    if effective_rank == requested_rank:
        basis = effective_basis
    else:
        # Never reuse arbitrary SVD null-space directions. Supplement the
        # genuine update span with seeded fresh directions so all candidates
        # retain the fixed rank required by the LoRA modules.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(supplement_seed))
        fresh = torch.randn(
            int(a.shape[1]),
            requested_rank - effective_rank,
            generator=generator,
            dtype=torch.float32,
        )
        combined = torch.cat((effective_basis, fresh), dim=1)
        basis, _ = torch.linalg.qr(combined, mode="reduced")

    return basis[:, :requested_rank].contiguous(), spectrum


def _torch_load(path: Path):
    """Load legacy module-list checkpoints across PyTorch 2.5/2.6 defaults."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions without weights_only.
        return torch.load(path, map_location="cpu")


@dataclass
class TaskSubspace:
    task_id: int
    bases: list[torch.Tensor]
    singular_values: list[torch.Tensor]


@dataclass
class SubspaceBank:
    """A task-level bank: one candidate contains a basis for every LoRA site."""

    tasks: list[TaskSubspace] = field(default_factory=list)
    storage_dtype: torch.dtype = torch.float16

    def __len__(self) -> int:
        return len(self.tasks)

    @property
    def task_ids(self) -> list[int]:
        return [entry.task_id for entry in self.tasks]

    @property
    def candidates(self) -> list[list[torch.Tensor]]:
        return [[basis.float() for basis in entry.bases] for entry in self.tasks]

    def add_from_factors(
        self,
        task_id: int,
        factors: Iterable[TensorPair],
        *,
        rank: int | None = None,
    ) -> TaskSubspace:
        bases: list[torch.Tensor] = []
        spectra: list[torch.Tensor] = []
        for site_index, (a_value, b_value) in enumerate(factors):
            basis, singular_values = right_subspace_from_factors(
                _weight(b_value),
                _weight(a_value),
                rank=rank,
                supplement_seed=104729 * (int(task_id) + 1) + 1009 * (site_index + 1),
            )
            bases.append(basis.to(dtype=self.storage_dtype))
            spectra.append(singular_values.to(dtype=self.storage_dtype))
        if not bases:
            raise ValueError(f"Task {task_id} contains no LoRA factors")
        entry = TaskSubspace(int(task_id), bases, spectra)
        self.tasks.append(entry)
        return entry

    def add_checkpoint_task(self, checkpoint_dir: str | Path, task_id: int) -> TaskSubspace:
        checkpoint_dir = Path(checkpoint_dir)
        path_a = checkpoint_dir / f"lora_w_a_{int(task_id)}.pt"
        path_b = checkpoint_dir / f"lora_w_b_{int(task_id)}.pt"
        if not path_a.exists() or not path_b.exists():
            raise FileNotFoundError(f"Missing LoRA checkpoint pair: {path_a}, {path_b}")
        saved_as = _torch_load(path_a)
        saved_bs = _torch_load(path_b)
        if len(saved_as) != len(saved_bs):
            raise ValueError(
                f"Task {task_id} A/B list mismatch: {len(saved_as)} versus {len(saved_bs)}"
            )
        return self.add_from_factors(task_id, zip(saved_as, saved_bs))

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: str | Path,
        *,
        upto_task: int | None = None,
        max_tasks: int | None = None,
        storage_dtype: torch.dtype = torch.float16,
    ) -> "SubspaceBank":
        checkpoint_dir = Path(checkpoint_dir)
        a_ids = {
            int(path.stem.rsplit("_", 1)[-1])
            for path in checkpoint_dir.glob("lora_w_a_*.pt")
            if path.stem.rsplit("_", 1)[-1].isdigit()
        }
        b_ids = {
            int(path.stem.rsplit("_", 1)[-1])
            for path in checkpoint_dir.glob("lora_w_b_*.pt")
            if path.stem.rsplit("_", 1)[-1].isdigit()
        }
        task_ids = sorted(a_ids & b_ids)
        if upto_task is not None:
            task_ids = [task_id for task_id in task_ids if task_id < int(upto_task)]
        if max_tasks is not None:
            task_ids = task_ids[-int(max_tasks) :]

        bank = cls(storage_dtype=storage_dtype)
        for task_id in task_ids:
            bank.add_checkpoint_task(checkpoint_dir, task_id)
        return bank

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "task_ids": self.task_ids,
            "bases": [[value.cpu() for value in task.bases] for task in self.tasks],
            "singular_values": [
                [value.cpu() for value in task.singular_values] for task in self.tasks
            ],
        }
        torch.save(payload, path)

    def summary(self) -> dict:
        task_rows = []
        for task in self.tasks:
            orthogonality_errors = []
            numerical_ranks = []
            for basis, spectrum in zip(task.bases, task.singular_values):
                basis32 = basis.float()
                eye = torch.eye(basis32.shape[1])
                orthogonality_errors.append(
                    float(torch.linalg.matrix_norm(basis32.t() @ basis32 - eye).item())
                )
                threshold = float(spectrum.max().item()) * 1e-6
                numerical_ranks.append(int((spectrum.float() > threshold).sum().item()))
            task_rows.append(
                {
                    "task_id": task.task_id,
                    "num_lora_sites": len(task.bases),
                    "basis_shapes": [list(value.shape) for value in task.bases],
                    "max_orthogonality_error_after_fp16_storage": max(orthogonality_errors),
                    "min_numerical_rank": min(numerical_ranks),
                    "max_numerical_rank": max(numerical_ranks),
                }
            )
        return {"num_tasks": len(self.tasks), "task_ids": self.task_ids, "tasks": task_rows}


def factors_from_module_lists(
    a_modules: Sequence[object], b_modules: Sequence[object]
) -> list[TensorPair]:
    if len(a_modules) != len(b_modules):
        raise ValueError(f"A/B list mismatch: {len(a_modules)} versus {len(b_modules)}")
    return [(_weight(a), _weight(b)) for a, b in zip(a_modules, b_modules)]
