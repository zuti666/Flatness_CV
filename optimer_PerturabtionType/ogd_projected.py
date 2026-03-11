from collections.abc import Iterable
from typing import Any, List, Optional

import torch


class OGDProjectedOptimizer(torch.optim.Optimizer):
    """Wrap a base optimizer and project gradients before the parameter update.

    This optimizer does not own continual-learning history by itself. Instead,
    history is injected at each `step(history=...)` call, which decouples:
    - history collection / update (outside)
    - gradient projection + parameter update (inside optimizer)
    """

    def __init__(
        self,
        params,
        base_optimizer,
        projection_eps: float = 1e-12,
        **kwargs: Any,
    ):
        if projection_eps <= 0.0:
            raise ValueError(f"Invalid projection_eps: {projection_eps}")

        defaults = dict(projection_eps=projection_eps, **kwargs)
        super().__init__(params, defaults)

        if isinstance(base_optimizer, torch.optim.Optimizer):
            self.base_optimizer = base_optimizer
        else:
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(getattr(self.base_optimizer, "defaults", {}))
        self.projection_eps = projection_eps

    def _flatten_gradients(self) -> tuple[Optional[torch.Tensor], List[torch.nn.Parameter]]:
        grads = []
        params: List[torch.nn.Parameter] = []

        for group in self.param_groups:
            for p in group["params"]:
                params.append(p)
                if p.grad is None:
                    grads.append(torch.zeros_like(p, memory_format=torch.preserve_format).reshape(-1))
                else:
                    grads.append(p.grad.detach().reshape(-1))

        if not grads:
            return None, params
        return torch.cat(grads), params

    def _set_flattened_gradients(self, grad_vector: torch.Tensor, params: List[torch.nn.Parameter]) -> None:
        offset = 0
        for p in params:
            numel = p.numel()
            grad_view = grad_vector[offset : offset + numel].view_as(p)
            if p.grad is None:
                p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)
            p.grad.copy_(grad_view)
            offset += numel

    def _extract_directions(
        self,
        history: Any,
        expected_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        if history is None:
            return []

        if hasattr(history, "vectors"):
            history = history.vectors

        if isinstance(history, torch.Tensor):
            if history.ndim == 1:
                history = [history]
            elif history.ndim == 2:
                if history.shape[0] == expected_dim:
                    history = [history[:, i] for i in range(history.shape[1])]
                elif history.shape[1] == expected_dim:
                    history = [history[i, :] for i in range(history.shape[0])]
                else:
                    raise ValueError(
                        f"History matrix shape {tuple(history.shape)} mismatches gradient dim {expected_dim}."
                    )
            else:
                raise ValueError("History tensor must be 1D or 2D.")
        elif not isinstance(history, Iterable):
            raise TypeError("history must be None, a tensor, a sequence of tensors, or an object with .vectors.")

        directions: List[torch.Tensor] = []
        for direction in history:
            if not torch.is_tensor(direction):
                raise TypeError("All history directions must be torch.Tensor.")
            d = direction.detach().reshape(-1).to(device=device, dtype=dtype)
            if d.numel() != expected_dim:
                raise ValueError(
                    f"Direction dim {d.numel()} mismatches current gradient dim {expected_dim}."
                )
            if torch.dot(d, d).item() > self.projection_eps:
                directions.append(d)

        return directions

    @torch.no_grad()
    def project_gradients(self, history: Any = None) -> dict[str, float]:
        flat_grad, params = self._flatten_gradients()
        if flat_grad is None:
            return {
                "num_directions": 0.0,
                "raw_grad_norm": 0.0,
                "projected_grad_norm": 0.0,
                "projected_to_raw_ratio": 1.0,
                "projection_relative_change": 0.0,
            }

        raw_norm = flat_grad.norm().item()
        directions = self._extract_directions(
            history=history,
            expected_dim=flat_grad.numel(),
            device=flat_grad.device,
            dtype=flat_grad.dtype,
        )

        if not directions:
            return {
                "num_directions": 0.0,
                "raw_grad_norm": raw_norm,
                "projected_grad_norm": raw_norm,
                "projected_to_raw_ratio": 1.0,
                "projection_relative_change": 0.0,
            }

        projected = flat_grad.clone()
        for direction in directions:
            denom = torch.dot(direction, direction).clamp_min(self.projection_eps)
            coeff = torch.dot(projected, direction) / denom
            projected = projected - coeff * direction

        self._set_flattened_gradients(projected, params)

        proj_norm = projected.norm().item()
        diff_norm = (flat_grad - projected).norm().item()

        return {
            "num_directions": float(len(directions)),
            "raw_grad_norm": raw_norm,
            "projected_grad_norm": proj_norm,
            "projected_to_raw_ratio": proj_norm / (raw_norm + 1e-8),
            "projection_relative_change": diff_norm / (raw_norm + 1e-10),
        }

    @torch.no_grad()
    def step(self, closure=None, history: Any = None, return_projection_stats: bool = False):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        projection_stats = self.project_gradients(history=history)
        self.base_optimizer.step()

        if return_projection_stats:
            return loss, projection_stats
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups

    def __repr__(self) -> str:
        return f"OGDProjectedOptimizer({self.base_optimizer.__class__.__name__})"
