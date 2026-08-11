from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def make_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "gelu":
        return nn.GELU()
    if key == "softplus":
        return nn.Softplus()
    raise ValueError(f"Unsupported activation: {name}")


def _matrix_basis(matrix: torch.Tensor, complete: bool = False) -> tuple[torch.Tensor, int]:
    rank = int(torch.linalg.matrix_rank(matrix).item())
    if rank == 0:
        identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        return identity if complete else identity[:, :0], 0
    q, _ = torch.linalg.qr(matrix, mode="complete" if complete else "reduced")
    return q, rank


def lora_tangent_basis(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal effective-weight basis for {B dA + dB A}."""
    output_dim, _ = b.shape
    input_dim = a.shape[1]
    left_complete, left_rank = _matrix_basis(b, complete=True)
    right, right_rank = _matrix_basis(a.T, complete=False)
    directions: list[torch.Tensor] = []
    # Matrices whose columns lie in col(B).
    for component in range(left_rank):
        for column in range(input_dim):
            unit = torch.zeros(input_dim, device=a.device, dtype=a.dtype)
            unit[column] = 1
            directions.append(torch.outer(left_complete[:, component], unit).flatten())
    # Remaining matrices whose rows lie in row(A); using col(B)^perp avoids duplicates.
    for component in range(left_rank, output_dim):
        for row_component in range(right_rank):
            directions.append(
                torch.outer(left_complete[:, component], right[:, row_component]).flatten()
            )
    if not directions:
        return torch.empty(output_dim * input_dim, 0, device=a.device, dtype=a.dtype)
    return torch.stack(directions, dim=1)


class FullMLP(nn.Module):
    """MLP used only to create the shared base checkpoint."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 32,
        num_classes: int = 10,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.middle = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.activation_name = activation
        self.activation = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.middle(x))
        return self.fc3(x)


class AdaptiveMLP(nn.Module):
    """Frozen MLP base with a controlled middle-matrix parameterization.

    Supported values are ``dense``, ``random_subspace``,
    ``fixed_lora_tangent``, ``fixed_mature_tangent``, ``factor_lora``,
    ``balanced_lora`` and ``projected_rank``.  The legacy name ``lora`` is an
    alias for ``factor_lora``.
    """

    PARAMETERIZATIONS = {
        "dense",
        "random_subspace",
        "fixed_lora_tangent",
        "fixed_mature_tangent",
        "factor_lora",
        "balanced_lora",
        "projected_rank",
    }

    def __init__(
        self,
        base: FullMLP,
        parameterization: str = "factor_lora",
        rank: int = 4,
        lora_alpha: float | None = None,
        factor_gauge_scale: float = 1.0,
        subspace_seed: int = 0,
        subspace_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(base.fc1.in_features, base.fc1.out_features)
        self.fc3 = nn.Linear(base.fc3.in_features, base.fc3.out_features)
        self.fc1.load_state_dict(base.fc1.state_dict())
        self.fc3.load_state_dict(base.fc3.state_dict())
        for parameter in self.fc1.parameters():
            parameter.requires_grad_(False)
        for parameter in self.fc3.parameters():
            parameter.requires_grad_(False)

        self.base_weight = nn.Parameter(base.middle.weight.detach().clone(), requires_grad=False)
        self.middle_bias = nn.Parameter(base.middle.bias.detach().clone(), requires_grad=False)
        self.activation_name = base.activation_name
        self.activation = make_activation(base.activation_name)

        requested = parameterization.lower()
        self.parameterization = "factor_lora" if requested == "lora" else requested
        if self.parameterization not in self.PARAMETERIZATIONS:
            raise ValueError(f"Unsupported parameterization: {parameterization}")
        self.rank = int(rank)
        self.lora_alpha = float(lora_alpha if lora_alpha is not None else rank)
        self.factor_gauge_scale = float(factor_gauge_scale)
        if self.factor_gauge_scale <= 0:
            raise ValueError("factor_gauge_scale must be positive")
        output_dim, input_dim = self.base_weight.shape
        if not 1 <= self.rank <= min(output_dim, input_dim):
            raise ValueError(f"rank must be in [1, {min(output_dim, input_dim)}], got {self.rank}")

        self.register_parameter("dense_delta", None)
        self.register_parameter("subspace_coordinates", None)
        self.register_parameter("lora_a", None)
        self.register_parameter("lora_b", None)
        self.register_buffer("subspace_basis", None)

        if self.parameterization in {"dense", "projected_rank"}:
            self.dense_delta = nn.Parameter(torch.zeros_like(self.base_weight))
        elif self.parameterization in {
            "random_subspace", "fixed_lora_tangent", "fixed_mature_tangent"
        }:
            # Match standard LoRA's true initialization tangent (A random, B=0).
            default_dim = output_dim * self.rank
            tangent_dim = int(subspace_dim) if subspace_dim is not None else default_dim
            if tangent_dim < 1 or tangent_dim > output_dim * input_dim:
                raise ValueError(
                    f"subspace_dim must be in [1, {output_dim * input_dim}], got {tangent_dim}"
                )
            generator = torch.Generator(device="cpu").manual_seed(int(subspace_seed))
            if self.parameterization == "random_subspace":
                raw = torch.randn(output_dim * input_dim, tangent_dim, generator=generator)
                basis, _ = torch.linalg.qr(raw, mode="reduced")
            elif self.parameterization == "fixed_lora_tangent":
                a = torch.randn(self.rank, input_dim, generator=generator)
                b = torch.zeros(output_dim, self.rank)
                basis = lora_tangent_basis(a, b)
                if subspace_dim is not None and tangent_dim != basis.shape[1]:
                    raise ValueError(
                        "fixed_lora_tangent has dimension output_dim * rank; "
                        "use fixed_mature_tangent for a mature tangent plane"
                    )
            else:
                a = torch.randn(self.rank, input_dim, generator=generator)
                b = torch.randn(output_dim, self.rank, generator=generator)
                basis = lora_tangent_basis(a, b)
                if subspace_dim is not None and tangent_dim != basis.shape[1]:
                    raise ValueError(
                        f"fixed_mature_tangent dimension is {basis.shape[1]}, got {tangent_dim}"
                    )
            self.subspace_basis = basis.to(dtype=self.base_weight.dtype)
            self.subspace_coordinates = nn.Parameter(torch.zeros(basis.shape[1]))
        else:
            self.lora_a = nn.Parameter(torch.empty(self.rank, input_dim))
            self.lora_b = nn.Parameter(torch.zeros(output_dim, self.rank))
            self._reset_lora()

    @property
    def lora_scale(self) -> float:
        return self.lora_alpha / self.rank

    @property
    def is_factorized(self) -> bool:
        return self.parameterization in {"factor_lora", "balanced_lora"}

    def _reset_lora(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        with torch.no_grad():
            self.lora_a.mul_(self.factor_gauge_scale)
            self.lora_b.div_(self.factor_gauge_scale)

    def effective_delta(self) -> torch.Tensor:
        if self.parameterization in {"dense", "projected_rank"}:
            return self.dense_delta
        if self.parameterization in {
            "random_subspace", "fixed_lora_tangent", "fixed_mature_tangent"
        }:
            return (self.subspace_basis @ self.subspace_coordinates).view_as(self.base_weight)
        return self.lora_scale * (self.lora_b @ self.lora_a)

    def effective_weight(self) -> torch.Tensor:
        return self.base_weight + self.effective_delta()

    def effective_subspace_basis(self) -> torch.Tensor:
        """Current local reachable space as an orthonormal basis in vec(W)."""
        if self.parameterization == "dense":
            return torch.eye(
                self.base_weight.numel(), device=self.base_weight.device, dtype=self.base_weight.dtype
            )
        if self.parameterization in {
            "random_subspace", "fixed_lora_tangent", "fixed_mature_tangent"
        }:
            return self.subspace_basis
        if self.parameterization == "projected_rank":
            # The rank<=r variety is singular at zero and has no unique regular
            # tangent there.  Report its ambient tangent-cone span at zero; once
            # rank r is reached, report the regular r(m+n-r) tangent basis.
            if float(self.dense_delta.detach().norm()) <= 1e-12:
                return torch.eye(
                    self.base_weight.numel(), device=self.base_weight.device,
                    dtype=self.base_weight.dtype,
                )
            u, _, vh = torch.linalg.svd(self.dense_delta.detach(), full_matrices=False)
            return lora_tangent_basis(vh[: self.rank], u[:, : self.rank])
        return lora_tangent_basis(self.lora_a.detach(), self.lora_b.detach())

    def map_trainable_direction_to_weight(
        self, directions: list[torch.Tensor]
    ) -> torch.Tensor:
        """Jacobian push-forward from current trainable coordinates to effective W."""
        if self.parameterization in {"dense", "projected_rank"}:
            return directions[0]
        if self.parameterization in {
            "random_subspace", "fixed_lora_tangent", "fixed_mature_tangent"
        }:
            return (self.subspace_basis @ directions[0].flatten()).view_as(self.base_weight)
        direction_a, direction_b = directions
        return self.lora_scale * (
            self.lora_b.detach() @ direction_a + direction_b @ self.lora_a.detach()
        )

    def project_trainable_directions(
        self, directions: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Project effective-W directions for the direct rank-r control."""
        if self.parameterization != "projected_rank":
            return directions
        direction = directions[0]
        update = self.dense_delta.detach()
        if float(update.norm()) <= 1e-12:
            u, singular_values, vh = torch.linalg.svd(direction, full_matrices=False)
            projected = (u[:, : self.rank] * singular_values[: self.rank]) @ vh[: self.rank]
        else:
            u, _, vh = torch.linalg.svd(update, full_matrices=False)
            u = u[:, : self.rank]
            v = vh[: self.rank].T
            projected = u @ (u.T @ direction) + (direction @ v) @ v.T
            projected -= u @ (u.T @ direction @ v) @ v.T
        return [projected]

    def retract_trainable_update(self) -> None:
        """Truncated-SVD retraction for the direct rank-r control."""
        if self.parameterization != "projected_rank":
            return
        with torch.no_grad():
            u, singular_values, vh = torch.linalg.svd(self.dense_delta, full_matrices=False)
            self.dense_delta.copy_(
                (u[:, : self.rank] * singular_values[: self.rank]) @ vh[: self.rank]
            )

    def factor_geometry(self) -> dict[str, float] | None:
        if not self.is_factorized:
            return None
        with torch.no_grad():
            a_norm = self.lora_a.norm()
            b_norm = self.lora_b.norm()
            balance_gap = (
                self.lora_b.T @ self.lora_b - self.lora_a @ self.lora_a.T
            ).norm()
            return {
                "factor_a_norm": float(a_norm),
                "factor_b_norm": float(b_norm),
                "factor_norm_ratio": float(a_norm / b_norm.clamp_min(1e-20)),
                "factor_balance_gap": float(balance_gap),
            }

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def rebalance_factors(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Put BA in its canonical balanced SVD gauge without changing the model."""
        if self.parameterization != "balanced_lora":
            return
        with torch.no_grad():
            product = self.lora_b @ self.lora_a
            u, singular_values, vh = torch.linalg.svd(product, full_matrices=False)
            root = singular_values[: self.rank].clamp_min(0).sqrt()
            self.lora_b.copy_(u[:, : self.rank] * root.unsqueeze(0))
            self.lora_a.copy_(root.unsqueeze(1) * vh[: self.rank])
            if optimizer is not None:
                # Momentum vectors do not have a unique gauge transform; clearing only
                # factor state prevents a hidden coordinate-dependent carry-over.
                optimizer.state[self.lora_a].clear()
                optimizer.state[self.lora_b].clear()

    def merge_and_reset(self) -> None:
        """Fold the current update into the base without changing the function."""
        with torch.no_grad():
            self.base_weight.add_(self.effective_delta())
            if self.parameterization in {"dense", "projected_rank"}:
                self.dense_delta.zero_()
            elif self.parameterization in {
                "random_subspace", "fixed_lora_tangent", "fixed_mature_tangent"
            }:
                self.subspace_coordinates.zero_()
            else:
                self._reset_lora()

    def hidden_before_middle(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self.activation(F.linear(x, self.fc1.weight, self.fc1.bias))

    def forward_with_middle_weight(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x = self.hidden_before_middle(x)
        x = self.activation(F.linear(x, weight, self.middle_bias))
        return F.linear(x, self.fc3.weight, self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_middle_weight(x, self.effective_weight())
