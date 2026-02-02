import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


def _get_fc_params(module: nn.Module, include_bias: bool = False) -> List[torch.nn.Parameter]:
    picked: List[torch.nn.Parameter] = []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("fc.weight") or name == "fc.weight":
            picked.append(p)
        elif include_bias and (name.endswith("fc.bias") or name == "fc.bias"):
            picked.append(p)
        elif name.endswith("classifier.weight") or name == "classifier.weight":
            picked.append(p)
        elif include_bias and (name.endswith("classifier.bias") or name == "classifier.bias"):
            picked.append(p)
    return picked


def _weight_norm_for_params(params: List[torch.nn.Parameter], mode: str = "fro") -> Tuple[float, str]:
    mode = (mode or "fro").lower()
    if not params:
        return 0.0, mode
    if mode == "fro":
        s = 0.0
        for p in params:
            t = p.detach().float()
            s += float(torch.sum(t * t).item())
        return float(math.sqrt(max(s, 0.0))), "fro"
    spectral_vals: List[float] = []
    fro2 = 0.0
    for p in params:
        t = p.detach().float()
        if t.ndim == 2 and t.numel() > 0:
            try:
                svals = torch.linalg.svdvals(t)
                spectral_vals.append(float(svals.max().item()))
            except Exception:
                pass
        fro2 += float(torch.sum(t * t).item())
    if spectral_vals:
        return float(max(spectral_vals)), "spectral"
    return float(math.sqrt(max(fro2, 0.0))), "fro"


def _clone_params(params: Iterable[torch.nn.Parameter]) -> List[torch.Tensor]:
    """Detach and clone parameter tensors for later restoration."""
    return [p.detach().to("cpu").clone() for p in params]


def _restore_params(params: Iterable[torch.nn.Parameter], copies: List[torch.Tensor]):
    for p, saved in zip(params, copies):
        if saved.device == p.device and saved.dtype == p.dtype:
            p.data.copy_(saved)
        else:
            p.data.copy_(saved.to(device=p.device, dtype=p.dtype))


def _add_vector_to_params(params: List[torch.nn.Parameter], vec: torch.Tensor):
    """Add a flattened vector ``vec`` onto a list of parameters in-place."""
    pointer = 0
    for p in params:
        numel = p.numel()
        slice_vec = vec[pointer : pointer + numel].view_as(p)
        p.data.add_(slice_vec)
        pointer += numel


def _param_names_and_shapes(module: nn.Module, params: List[torch.nn.Parameter]):
    """Return names, shapes, and splits for the given params in order."""
    id2info = {}
    for name, p in module.named_parameters():
        id2info[id(p)] = (name, p.shape, p.numel())

    names: List[str] = []
    shapes: List[torch.Size] = []
    splits: List[int] = []
    for p in params:
        key = id(p)
        if key not in id2info:
            names.append("")
            shapes.append(p.shape)
            splits.append(p.numel())
        else:
            n, s, k = id2info[key]
            names.append(n)
            shapes.append(s)
            splits.append(k)
    return names, shapes, splits


def _unflatten_to_param_like(vec: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    pointer = 0
    for p in params:
        k = p.numel()
        out.append(vec[pointer:pointer + k].view_as(p))
        pointer += k
    return out


def _select_params_by_name(
    module: torch.nn.Module,
    substrs: Optional[List[str]],
    include_frozen: bool = False,
):
    """Filter parameters by name substrings; None falls back to requires_grad=True."""
    picked = []
    for n, p in module.named_parameters():
        if (not include_frozen) and (not p.requires_grad):
            continue
        if substrs is None or any(s in n for s in substrs):
            picked.append(p)
    return picked
