from __future__ import annotations

from dataclasses import dataclass
import logging
import sys
from typing import Any, Optional, Sequence

import torch
from torch.nn.modules.batchnorm import _BatchNorm


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)


InformationType = Optional[Any]


@dataclass
class _FlatReferenceStore:
    refs: list[torch.Tensor]


class SAM_OGD(torch.optim.Optimizer):
    """
    SAM with explicit OGD-style projection logic:
    1) sigma is projected away from historical first-order directions;
    2) gradient at perturbed weights is projected away from first-order directions.
    """

    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        first_order_information: InformationType = None,
        second_order_information: InformationType = None,
        first_order_filter_strength: float = 1.0,
        second_order_filter_strength: float = 1.0,
        perturb_filter_strength: Optional[float] = None,
        result_filter_strength: Optional[float] = None,
        projection_eps: float = 1e-12,
        **kwargs,
    ):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        if projection_eps <= 0.0:
            raise ValueError(f"Invalid projection_eps, should be positive: {projection_eps}")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        if isinstance(base_optimizer, torch.optim.Optimizer):
            self.base_optimizer = base_optimizer
        else:
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self._set_param_group_defaults(rho=float(rho), adaptive=bool(adaptive))

        self.projection_eps = float(projection_eps)
        self.first_order_filter_strength = float(first_order_filter_strength)
        self.second_order_filter_strength = float(second_order_filter_strength)
        self.perturb_filter_strength = float(
            self.first_order_filter_strength if perturb_filter_strength is None else perturb_filter_strength
        )
        self.result_filter_strength = float(
            self.first_order_filter_strength if result_filter_strength is None else result_filter_strength
        )

        self._refresh_parameter_cache()
        self._first_order_information = None
        self._second_order_information = None
        self.set_first_order_information(first_order_information)
        self.set_second_order_information(second_order_information)

    def _set_param_group_defaults(self, rho: float, adaptive: bool) -> None:
        for group in self.param_groups:
            group.setdefault("rho", rho)
            group.setdefault("adaptive", adaptive)

    def _refresh_parameter_cache(self):
        ordered_params = []
        for group in self.param_groups:
            for p in group["params"]:
                ordered_params.append(p)

        self._ordered_params = ordered_params
        self._param_id_to_index = {id(p): idx for idx, p in enumerate(self._ordered_params)}

        self._flat_slices = []
        offset = 0
        for p in self._ordered_params:
            next_offset = offset + p.numel()
            self._flat_slices.append((offset, next_offset))
            offset = next_offset
        self._total_numel = offset

    def _clone_as_param_shape(self, tensor: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if tensor.numel() != param.numel():
            raise ValueError(
                f"Information tensor numel mismatch: got {tensor.numel()}, expected {param.numel()}."
            )
        return tensor.detach().clone().reshape_as(param)

    def _canonicalize_param_entry(
        self,
        entry: Any,
        param: torch.Tensor,
    ) -> Optional[list[torch.Tensor]]:
        if entry is None:
            return None

        if torch.is_tensor(entry):
            return [self._clone_as_param_shape(entry, param)]

        if isinstance(entry, (list, tuple)):
            refs = []
            for ref in entry:
                if ref is None:
                    continue
                if not torch.is_tensor(ref):
                    raise TypeError("Each reference entry must be a torch.Tensor or None.")
                refs.append(self._clone_as_param_shape(ref, param))
            return refs or None

        raise TypeError("Unsupported information entry type.")

    def _split_flat_tensor(self, flat_tensor: torch.Tensor) -> list[list[torch.Tensor]]:
        if flat_tensor.numel() != self._total_numel:
            raise ValueError(
                f"Flattened information size mismatch: got {flat_tensor.numel()}, expected {self._total_numel}."
            )

        vec = flat_tensor.reshape(-1).detach().clone()
        result = []
        for idx, p in enumerate(self._ordered_params):
            start, end = self._flat_slices[idx]
            result.append([vec[start:end].reshape_as(p)])
        return result

    def _split_flat_tensor_matrix(
        self,
        flat_tensor_matrix: torch.Tensor,
    ) -> list[Optional[list[torch.Tensor]]]:
        if flat_tensor_matrix.dim() != 2 or flat_tensor_matrix.shape[1] != self._total_numel:
            raise ValueError("Tensor matrix information must have shape [k, total_parameter_numel].")
        result: list[list[torch.Tensor]] = [[] for _ in self._ordered_params]
        for row in flat_tensor_matrix:
            row_detached = row.detach().clone()
            for idx, p in enumerate(self._ordered_params):
                start, end = self._flat_slices[idx]
                result[idx].append(row_detached[start:end].reshape_as(p))
        return [refs or None for refs in result]

    def _canonicalize_information(self, information: InformationType) -> Optional[Any]:
        if information is None:
            return None

        n_params = len(self._ordered_params)

        if isinstance(information, dict):
            result: list[Optional[list[torch.Tensor]]] = [None for _ in range(n_params)]
            for idx, p in enumerate(self._ordered_params):
                if id(p) in information:
                    result[idx] = self._canonicalize_param_entry(information[id(p)], p)
                    continue
                if idx in information:
                    result[idx] = self._canonicalize_param_entry(information[idx], p)
                    continue
                if str(idx) in information:
                    result[idx] = self._canonicalize_param_entry(information[str(idx)], p)
                    continue
            return result

        if torch.is_tensor(information):
            if information.dim() == 1:
                if information.numel() != self._total_numel:
                    if n_params == 1:
                        return [[self._clone_as_param_shape(information, self._ordered_params[0])]]
                    raise ValueError(
                        f"Flattened information size mismatch: got {information.numel()}, expected {self._total_numel}."
                    )
                return _FlatReferenceStore(refs=[information.detach().clone().reshape(-1)])
            if information.dim() == 2:
                if information.shape[1] != self._total_numel:
                    raise ValueError("Tensor matrix information must have shape [k, total_parameter_numel].")
                refs = [row.detach().clone().reshape(-1) for row in information]
                return _FlatReferenceStore(refs=refs)
            if n_params == 1:
                return [[self._clone_as_param_shape(information, self._ordered_params[0])]]
            raise ValueError("Tensor information must be [total_numel] or [k, total_numel].")

        if isinstance(information, (list, tuple)):
            if len(information) > 0 and all(
                torch.is_tensor(item) and item.numel() == self._total_numel for item in information
            ):
                refs = [item.detach().clone().reshape(-1) for item in information]
                return _FlatReferenceStore(refs=refs)

            if len(information) == n_params:
                result: list[Optional[list[torch.Tensor]]] = []
                for idx, p in enumerate(self._ordered_params):
                    result.append(self._canonicalize_param_entry(information[idx], p))
                return result

            if n_params == 1 and all(torch.is_tensor(item) for item in information):
                single_param = self._ordered_params[0]
                refs = [self._clone_as_param_shape(item, single_param) for item in information]
                return [refs or None]

            raise ValueError(
                "List/Tuple information is ambiguous. Use per-parameter list (len == #params) "
                "or flattened vectors (numel == total_numel)."
            )

        raise TypeError("Unsupported information container type.")

    def set_first_order_information(self, first_order_information: InformationType):
        self._first_order_information = self._canonicalize_information(first_order_information)

    def set_second_order_information(self, second_order_information: InformationType):
        self._second_order_information = self._canonicalize_information(second_order_information)

    def set_history_information(
        self,
        first_order_information: InformationType = None,
        second_order_information: InformationType = None,
    ):
        if first_order_information is not None:
            self.set_first_order_information(first_order_information)
        if second_order_information is not None:
            self.set_second_order_information(second_order_information)

    def _has_any_gradient(self) -> bool:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    return True
        return False

    def _get_flat_grad_vector(self) -> tuple[Optional[torch.Tensor], list[bool]]:
        grads = []
        has_grad = []
        for p in self._ordered_params:
            if p.grad is None:
                grads.append(torch.zeros_like(p, memory_format=torch.preserve_format).reshape(-1))
                has_grad.append(False)
            else:
                grads.append(p.grad.detach().reshape(-1))
                has_grad.append(True)

        if len(grads) == 0:
            return None, has_grad
        return torch.cat(grads), has_grad

    def _set_flat_grad_vector(self, grad_vector: torch.Tensor, has_grad: list[bool]) -> None:
        offset = 0
        for idx, p in enumerate(self._ordered_params):
            numel = p.numel()
            if has_grad[idx]:
                grad_view = grad_vector[offset : offset + numel].reshape_as(p)
                if p.grad is None:
                    p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)
                p.grad.copy_(grad_view)
            offset += numel

    def _flatten_references(
        self,
        info_store: Optional[Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        if info_store is None:
            return []

        if isinstance(info_store, _FlatReferenceStore):
            refs: list[torch.Tensor] = []
            for ref in info_store.refs:
                flat_ref = ref.reshape(-1)
                if flat_ref.numel() != self._total_numel:
                    logger.warning(
                        "Skipping mismatched flat reference: ref numel=%d, expected=%d.",
                        flat_ref.numel(),
                        self._total_numel,
                    )
                    continue
                refs.append(flat_ref.to(device=device, dtype=dtype))
            return refs

        refs: list[torch.Tensor] = []
        total_numel = self._total_numel

        for idx, param_refs in enumerate(info_store):
            if not param_refs:
                continue

            start, end = self._flat_slices[idx]
            p = self._ordered_params[idx]

            for ref in param_refs:
                if ref is None:
                    continue
                if ref.numel() != p.numel():
                    logger.warning(
                        "Skipping mismatched reference: ref numel=%d, param numel=%d.",
                        ref.numel(),
                        p.numel(),
                    )
                    continue

                flat_ref = torch.zeros(total_numel, device=device, dtype=dtype)
                flat_ref[start:end] = ref.to(device=device, dtype=dtype).reshape(-1)
                refs.append(flat_ref)

        return refs

    @staticmethod
    def _project_orthogonal_flat(
        gradient: torch.Tensor,
        references: Sequence[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        projected = gradient.clone()
        for ref in references:
            flat_ref = ref.reshape(-1)
            ref_norm = torch.linalg.vector_norm(flat_ref, ord=2)
            if ref_norm <= eps:
                continue
            unit_ref = flat_ref / (ref_norm + eps)
            projected = projected - torch.dot(projected, unit_ref) * unit_ref
        return projected

    @torch.no_grad()
    def first_step(
        self,
        zero_grad: bool = False,
        first_order_information: InformationType = None,
        second_order_information: InformationType = None,
    ):
        if first_order_information is not None:
            self.set_first_order_information(first_order_information)
        if second_order_information is not None:
            self.set_second_order_information(second_order_information)

        grad_norm = self._grad_norm()

        sigma_chunks = []
        has_grad = []
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + self.projection_eps)
            for p in group["params"]:
                if p.grad is None:
                    sigma_chunks.append(torch.zeros_like(p, memory_format=torch.preserve_format).reshape(-1))
                    has_grad.append(False)
                    continue

                self.state[p]["old_p"] = p.data.clone()
                rescale = torch.pow(p, 2) if group["adaptive"] else 1.0
                sigma_raw = rescale * p.grad * scale.to(p)
                sigma_chunks.append(sigma_raw.reshape(-1))
                has_grad.append(True)
                self.state[p]["sigma_raw"] = sigma_raw

        sigma_flat = torch.cat(sigma_chunks) if len(sigma_chunks) > 0 else None
        sigma_filtered_flat = sigma_flat

        if sigma_flat is not None and self.perturb_filter_strength > 0:
            refs = self._flatten_references(
                info_store=self._first_order_information,
                device=sigma_flat.device,
                dtype=sigma_flat.dtype,
            )
            if len(refs) > 0:
                sigma_orth = self._project_orthogonal_flat(
                    gradient=sigma_flat,
                    references=refs,
                    eps=self.projection_eps,
                )
                sigma_projection = sigma_flat - sigma_orth
                sigma_filtered_flat = sigma_flat - self.perturb_filter_strength * sigma_projection

        if sigma_filtered_flat is not None:
            offset = 0
            idx = 0
            for group in self.param_groups:
                for p in group["params"]:
                    numel = p.numel()
                    if has_grad[idx]:
                        sigma = sigma_filtered_flat[offset : offset + numel].reshape_as(p)
                        p.add_(sigma)
                        self.state[p]["sigma"] = sigma
                    offset += numel
                    idx += 1

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(
        self,
        zero_grad: bool = False,
        first_order_information: InformationType = None,
    ):
        if first_order_information is not None:
            self.set_first_order_information(first_order_information)

        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        flat_grad, has_grad = self._get_flat_grad_vector()
        if flat_grad is not None and self.result_filter_strength > 0:
            refs = self._flatten_references(
                info_store=self._first_order_information,
                device=flat_grad.device,
                dtype=flat_grad.dtype,
            )
            if len(refs) > 0:
                g_orth = self._project_orthogonal_flat(
                    gradient=flat_grad,
                    references=refs,
                    eps=self.projection_eps,
                )
                projection = flat_grad - g_orth
                filtered_grad = flat_grad - self.result_filter_strength * projection
                self._set_flat_grad_vector(filtered_grad, has_grad)

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(
        self,
        closure=None,
        first_order_information: InformationType = None,
        second_order_information: InformationType = None,
    ):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        if not self._has_any_gradient():
            closure()

        self.first_step(
            zero_grad=True,
            first_order_information=first_order_information,
            second_order_information=second_order_information,
        )
        loss = closure()  # g = grad(L_s(W + sigma))
        self.second_step(
            zero_grad=False,
            first_order_information=first_order_information,
        )
        return loss

    def _grad_norm(self) -> torch.Tensor:
        terms = []
        shared_device = None
        shared_dtype = None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if shared_device is None:
                    shared_device = p.device
                    shared_dtype = p.dtype
                factor = torch.abs(p) if group["adaptive"] else 1.0
                terms.append((factor * p.grad).norm(p=2).to(shared_device))

        if len(terms) == 0:
            return torch.zeros((), device=shared_device or "cpu", dtype=shared_dtype or torch.float32)
        return torch.norm(torch.stack(terms), p=2)

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
        self._set_param_group_defaults(
            rho=float(self.defaults.get("rho", 0.05)),
            adaptive=bool(self.defaults.get("adaptive", False)),
        )
        self._refresh_parameter_cache()

    def __repr__(self) -> str:
        return f"SAM_OGD({self.base_optimizer.__class__.__name__})"


SAM = SAM_OGD


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
