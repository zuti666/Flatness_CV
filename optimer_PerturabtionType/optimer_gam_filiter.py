from __future__ import annotations

import contextlib
import logging
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import torch
from torch.distributed import ReduceOp
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


class GAM_Filiter(torch.optim.Optimizer):
    """
    Accelerated GAM with explicit OGD-style historical projection.

    Projection points:
    1) perturb_weights(perturb_idx=0): project first perturbation by first-order history;
    2) perturb_weights(perturb_idx=1): project second perturbation by second-order history;
    3) gradient_decompose(...): project final update gradient by first-order history.
    """

    def __init__(
        self,
        params,
        base_optimizer,
        model,
        grad_rho=None,
        grad_norm_rho=None,
        grad_rho_scheduler=None,
        grad_norm_rho_scheduler=None,
        adaptive=False,
        perturb_eps: float = 1e-12,
        args=None,
        grad_reduce: str = "mean",
        first_order_information: InformationType = None,
        second_order_information: InformationType = None,
        first_order_filter_strength: float = 1.0,
        second_order_filter_strength: float = 1.0,
        perturb1_filter_strength: Optional[float] = None,
        perturb2_filter_strength: Optional[float] = None,
        final_gradient_filter_strength: Optional[float] = None,
        **kwargs,
    ):
        defaults = dict(adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        if perturb_eps <= 0.0:
            raise ValueError(f"Invalid perturb_eps, should be positive: {perturb_eps}")

        self.perturb_eps = float(perturb_eps)
        self.model = model

        if isinstance(base_optimizer, torch.optim.Optimizer):
            self.base_optimizer = base_optimizer
        else:
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

        self.grad_rho = 0.05 if grad_rho is None else grad_rho
        self.grad_norm_rho = 0.05 if grad_norm_rho is None else grad_norm_rho
        self.grad_rho_scheduler = grad_rho_scheduler
        self.grad_norm_rho_scheduler = grad_norm_rho_scheduler
        self.adaptive = adaptive
        self.args = args or SimpleNamespace(
            grad_beta_1=0.5,
            grad_beta_2=0.5,
            grad_beta_3=0.5,
            grad_gamma=0.5,
        )

        self._direction_projector = None
        self.get_grad_reduce(grad_reduce)
        self.update_rho_t()

        self.first_order_filter_strength = float(first_order_filter_strength)
        self.second_order_filter_strength = float(second_order_filter_strength)
        self.perturb1_filter_strength = float(
            self.first_order_filter_strength if perturb1_filter_strength is None else perturb1_filter_strength
        )
        self.perturb2_filter_strength = float(
            self.second_order_filter_strength if perturb2_filter_strength is None else perturb2_filter_strength
        )
        self.final_gradient_filter_strength = float(
            self.first_order_filter_strength
            if final_gradient_filter_strength is None
            else final_gradient_filter_strength
        )

        self._refresh_parameter_cache()
        self._first_order_information = None
        self._second_order_information = None
        self.set_first_order_information(first_order_information)
        self.set_second_order_information(second_order_information)

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

    def get_grad_reduce(self, grad_reduce: str):
        if grad_reduce.lower() == "mean":
            if hasattr(ReduceOp, "AVG"):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == "sum":
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):
        if self.grad_rho_scheduler is not None:
            self.grad_rho = self.grad_rho_scheduler.step()

        if self.grad_norm_rho_scheduler is not None:
            self.grad_norm_rho = self.grad_norm_rho_scheduler.step()

    def set_direction_projector(self, projector):
        self._direction_projector = projector

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

    def _filter_flat_direction(
        self,
        direction_flat: Optional[torch.Tensor],
        info_store: Optional[list[Optional[list[torch.Tensor]]]],
        strength: float,
    ) -> Optional[torch.Tensor]:
        if direction_flat is None:
            return None
        if strength <= 0:
            return direction_flat

        refs = self._flatten_references(
            info_store=info_store,
            device=direction_flat.device,
            dtype=direction_flat.dtype,
        )
        if len(refs) == 0:
            return direction_flat

        direction_orth = self._project_orthogonal_flat(
            gradient=direction_flat,
            references=refs,
            eps=self.perturb_eps,
        )
        projection = direction_flat - direction_orth
        return direction_flat - strength * projection

    @torch.no_grad()
    def perturb_weights(self, perturb_idx: int):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.grad_rho / (grad_norm + self.perturb_eps)

        direction_chunks = []
        has_grad = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    direction_chunks.append(torch.zeros_like(p, memory_format=torch.preserve_format).reshape(-1))
                    has_grad.append(False)
                    continue

                if perturb_idx == 0:
                    self.state[p]["g_0"] = p.grad.data.clone()
                elif perturb_idx == 1:
                    self.state[p]["g_2"] = p.grad.data.clone()
                else:
                    raise ValueError('"perturb_idx should be one of [0, 1].')

                e_w_raw = p.grad * scale.to(p)
                if self.adaptive:
                    e_w_raw *= torch.pow(p, 2)

                if perturb_idx == 0:
                    self.state[p]["e_w_0_raw"] = e_w_raw
                else:
                    self.state[p]["e_w_1_raw"] = e_w_raw

                direction_chunks.append(e_w_raw.reshape(-1))
                has_grad.append(True)

        direction_flat = torch.cat(direction_chunks) if len(direction_chunks) > 0 else None

        if perturb_idx == 0:
            direction_filtered = self._filter_flat_direction(
                direction_flat=direction_flat,
                info_store=self._first_order_information,
                strength=self.perturb1_filter_strength,
            )
        else:
            direction_filtered = self._filter_flat_direction(
                direction_flat=direction_flat,
                info_store=self._second_order_information,
                strength=self.perturb2_filter_strength,
            )

        if direction_filtered is None:
            return

        offset = 0
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                if has_grad[idx]:
                    e_w = direction_filtered[offset : offset + numel].reshape_as(p)
                    p.add_(e_w)
                    if perturb_idx == 0:
                        self.state[p]["e_w_0"] = e_w
                    else:
                        if "e_w_1_2" in self.state[p]:
                            self.state[p]["e_w_1_2"] = self.state[p]["e_w_1_2"] + e_w
                        else:
                            self.state[p]["e_w_1_2"] = e_w
                        self.state[p]["e_w_1"] = e_w
                offset += numel
                idx += 1

    @torch.no_grad()
    def grad_norm_ascent(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["g_1"] = p.grad.data.clone()
                p.grad.data -= self.state[p]["g_0"]

        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.grad_norm_rho / (grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
                self.state[p]["e_w_1_2"] = e_w

    @torch.no_grad()
    def unperturb(self, perturb_key: str):
        for group in self.param_groups:
            for p in group["params"]:
                if perturb_key in self.state[p]:
                    p.data.sub_(self.state[p][perturb_key])

    @torch.no_grad()
    def gradient_decompose(self, args=None):
        args = args or self.args
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["pro_m"] = self.state[p]["g_0"] + abs(args.grad_beta_2) * self.state[p]["g_2"]
                p.grad.data = args.grad_beta_1 * self.state[p]["g_1"] + args.grad_beta_3 * p.grad.data.detach().clone()
                inner_prod += torch.sum(self.state[p]["pro_m"] * p.grad.data)

        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by="pro_m")
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                vertical = self.state[p]["pro_m"] - cosine * old_grad_norm * p.grad.data / (
                    new_grad_norm + self.perturb_eps
                )
                p.grad.data.add_(vertical, alpha=-args.grad_gamma)

        # Final OGD-style projection to remove overlap with historical first-order directions.
        flat_grad, has_grad = self._get_flat_grad_vector()
        flat_grad_filtered = self._filter_flat_direction(
            direction_flat=flat_grad,
            info_store=self._first_order_information,
            strength=self.final_gradient_filter_strength,
        )
        if flat_grad_filtered is not None:
            self._set_flat_grad_vector(flat_grad_filtered, has_grad)

    @torch.no_grad()
    def _grad_norm(self, weight_adaptive: bool = False, by: str = "grad"):
        device = None
        dtype = None
        norm = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None and by == "grad":
                    continue
                if device is None:
                    device = p.device
                    dtype = p.dtype

                if by == "grad":
                    g = p.grad
                elif by == "pro_m":
                    st = self.state[p]
                    if "pro_m" not in st or st["pro_m"] is None:
                        continue
                    g = st["pro_m"]
                elif by == "p":
                    g = p
                else:
                    raise ValueError("Invalid 'by' argument in _grad_norm")

                if weight_adaptive:
                    term = torch.sum((g * torch.abs(p)) ** 2)
                else:
                    term = torch.sum(g ** 2)

                norm = term if norm is None else norm + term

        if norm is None:
            norm = torch.zeros((), device=device or "cpu", dtype=dtype or torch.float32)
        return torch.sqrt(norm)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        def get_grad():
            self.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    def step(
        self,
        closure=None,
        first_order_information: InformationType = None,
        second_order_information: InformationType = None,
    ):
        if first_order_information is not None:
            self.set_first_order_information(first_order_information)
        if second_order_information is not None:
            self.set_second_order_information(second_order_information)

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            outputs, loss_value = get_grad()

            self.perturb_weights(perturb_idx=0)
            disable_running_stats(self.model)
            get_grad()

            self.unperturb(perturb_key="e_w_0")
            self.grad_norm_ascent()
            get_grad()

            self.perturb_weights(perturb_idx=1)
            get_grad()

            self.gradient_decompose(args=self.args)

            if callable(self._direction_projector):
                self._direction_projector(self.param_groups)

            self.unperturb(perturb_key="e_w_1_2")

        self._sync_grad()
        self.base_optimizer.step()
        enable_running_stats(self.model)

        return outputs, loss_value

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
        self._refresh_parameter_cache()

    def __repr__(self):
        return f"GAM({self.base_optimizer.__class__.__name__})"


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
