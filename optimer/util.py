import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import math
import numpy as np

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


# ------------------------------------------------------------------
# RWP noise generator
# ------------------------------------------------------------------
def generate_pertubation(param: torch.Tensor,
                   pertubation_mode: str,
                   std: float,
                   fisher_param: torch.Tensor | None = None,
                   fisher_scaler: float = 2.0) -> torch.Tensor:
    """
    Generate noise to inject into a parameter tensor using specified mode.

    Modes supported (reference-compatible names):
    - "Gauss_standard": element-wise iid N(0, std^2)
    - "Gauss_element": element-wise N(0, std^2 * |W_ij|^2)
    - "Gauss_matrix": matrix-wise N(0, std^2 * ||W||_F^2)
    - "lpf_sgd_Gauss": row-norm scaled Gaussian (per-row L2 norms)
    - "RAW_fisher_1": iid Gaussian scaled by sqrt(1 + a*F) factor
    - "RAW_fisher_2": iid Gaussian with exp(-a * log(F)_norm) scaling
    - "mARWP_fisher": row-norm Gaussian divided by sqrt(1 + a * fisher_row)
    - "flatLoRA_Gauss": row-norm Gaussian with sqrt(n) normalization

    Args:
        param: parameter tensor to match noise shape
        mode: string mode selector
        std: base standard deviation scalar
        fisher_param: optional Fisher stats tensor aligned with param
        fisher_scaler: scaling factor for Fisher-based adjustments

    Returns:
        torch.Tensor: noise tensor with same shape and device as `param`.
    """

    if param.numel() == 0:
        return torch.zeros_like(param)

    device = param.device
    dtype = param.dtype

    if pertubation_mode == "Gauss_standard":
        return torch.randn_like(param, device=device, dtype=dtype) * float(std)

    elif pertubation_mode == "Gauss_element":
        scaler = float(std) * (param.abs() + 1e-16)
        return torch.randn_like(param, device=device, dtype=dtype) * scaler

    elif pertubation_mode == "Gauss_matrix":
        fro_norm = torch.norm(param, p='fro')
        scaler = float(std) * fro_norm
        return torch.randn_like(param, device=device, dtype=dtype) * scaler

    elif pertubation_mode == "lpf_sgd_Gauss":
        if param.dim() > 1:
            sh = param.shape
            sh_mul = int(np.prod(sh[1:]))
            row_norms = param.view(sh[0], -1).norm(dim=1, keepdim=True)
            row_norms_param = row_norms.repeat(1, sh_mul).view(sh)
            scaler = float(std) * row_norms_param
            return torch.randn_like(param, device=device, dtype=dtype) * scaler
        else:
            scale = float(std) * (param.view(-1).norm().item() + 1e-16)
            return torch.randn_like(param, device=device, dtype=dtype) * scale

    # if pertubation_mode == "RAW_fisher_1":
    #     noise = torch.randn_like(param, device=device, dtype=dtype) * float(std)
    #     if fisher_param is not None:
    #         scaler_ = 1.0 / torch.sqrt(1.0 + float(fisher_scaler) * fisher_param)
    #         scaler_ = torch.clamp(scaler_, 0.3, 3.0)
    #         noise = noise * scaler_
    #     return noise

    # if pertubation_mode == "RAW_fisher_2":
    #     noise = torch.randn_like(param, device=device, dtype=dtype) * float(std)
    #     if fisher_param is not None:
    #         eps = 1e-8
    #         fisher_log = torch.log(fisher_param + eps)
    #         fmin, fmax = fisher_log.min(), fisher_log.max()
    #         frange = fmax - fmin
    #         if float(frange) < 1e-6:
    #             fisher_norm = torch.zeros_like(fisher_log)
    #         else:
    #             fisher_norm = (fisher_log - fmin) / (frange + 1e-8)
    #         weight = torch.exp(-2.0 * fisher_norm)
    #         scaler_ = torch.clamp(0.5 + 1.5 * weight, 0.2, 3.0)
    #         noise = noise * scaler_
    #     return noise

    elif pertubation_mode == "mARWP_fisher":
        sh = param.shape
        if param.dim() > 1:
            sh_mul = int(np.prod(sh[1:]))
            row_norms = param.view(sh[0], -1).norm(dim=1, keepdim=True)
            std_mat = row_norms.repeat(1, sh_mul).view(sh)
            scaler = float(std) * std_mat
            noise = torch.randn_like(param, device=device, dtype=dtype) * scaler
        else:
            scale = float(std) * (param.view(-1).norm().item() + 1e-16)
            noise = torch.randn_like(param, device=device, dtype=dtype) * scale
        if fisher_param is not None:
            if param.dim() > 1:
                sh_mul = int(np.prod(sh[1:]))
                fisher_row = fisher_param.view(sh[0], -1).sum(dim=1, keepdim=True).repeat(1, sh_mul).view(sh)
            else:
                fisher_row = fisher_param
            noise = noise / torch.sqrt(1.0 + float(fisher_scaler) * fisher_row)
        return noise

    elif pertubation_mode == "flatLoRA_Gauss":
        if param.dim() > 1:
            sh = param.shape
            sh_mul = int(np.prod(sh[1:]))
            row_norms = param.view(sh[0], -1).norm(dim=1, keepdim=True)
            std_matrix = row_norms.repeat(1, sh_mul).view(sh)
            scaler = float(std) / math.sqrt(sh[1]) * std_matrix
            noise = torch.randn_like(param, device=device, dtype=dtype) * scaler
        else:
            n = param.shape[0]
            scale = float(std) / math.sqrt(max(n, 1)) * (param.view(-1).norm().item() + 1e-16)
            noise = torch.randn_like(param, device=device, dtype=dtype) * scale
        return noise

    # default fallback (standard Gaussian)
    return torch.randn_like(param, device=device, dtype=dtype) * float(std)
