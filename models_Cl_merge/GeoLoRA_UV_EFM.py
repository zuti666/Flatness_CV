"""GeoLoRA-UV-EFM: orthogonal low-rank adaptation with shared subspace merge.

Design goals:
    - Start from the training skeleton of GeoLoRA_SlowMerge, but switch the core
      representation to a shared orthogonal subspace (U, V) plus a task-private
      orthogonal LoRA branch.
    - Enforce orthogonality on both low-rank factors during fast training:
          U_priv^T U_priv = I,  V_priv^T V_priv = I
      via a differentiable QR retraction.
    - Support weak-prior initialization for the task-private LoRA branch:
          * PiSSA-style SVD init from pretrained weights
          * LoRA-GA-style gradient-SVD init from a few task samples
    - Use a soft projection regularizer during fast training:
          L_proj = ||ΔW_t - U_sh (U_sh^T ΔW_t V_sh) V_sh^T||_F^2
    - Use feature-space Fisher-style local metrics (EFM proxy) for slow merge:
          M* = argmin_M Σ_t || U(M-M_t)V^T ||_{E_out_t, C_in_t}^2
      solved in vectorized Kronecker form.

Notes:
    - This version intentionally does not keep all historical private branches in
      the forward path. The persistent cross-task memory lives in the shared
      subspace state {(U, V, M_shared)}. The current task uses one private branch
      that is reset every task.
    - For PiSSA init we decompose the pretrained q/v weights as W = W_res + W_pri,
      replace the frozen base slice by W_res, and initialize the private branch by
      W_pri so that the initial model output is preserved.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


BranchState = Dict[str, torch.Tensor]
LayerState = Dict[str, BranchState]


def _inverse_softplus(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = torch.clamp(x, min=eps)
    return torch.log(torch.expm1(x) + eps)


def _canonical_qr(mat: torch.Tensor) -> torch.Tensor:
    if mat.numel() == 0:
        return mat
    q, r = torch.linalg.qr(mat, mode="reduced")
    diag = torch.diagonal(r)
    sign = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
    return q * sign.unsqueeze(0)


def _top_eig_basis(cov: torch.Tensor, k: int) -> torch.Tensor:
    dim = int(cov.shape[0])
    k = int(min(max(0, k), dim))
    if k <= 0:
        return cov.new_zeros(dim, 0)
    cov = 0.5 * (cov + cov.t())
    vals, vecs = torch.linalg.eigh(cov)
    idx = torch.argsort(vals, descending=True)[:k]
    return vecs[:, idx]


def _svd_rank_k(weight: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    r = int(min(rank, s.shape[0]))
    return u[:, :r].contiguous(), s[:r].contiguous(), vh[:r, :].contiguous()


def _split_qkv_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_dim = int(weight.shape[0] // 3)
    return (
        weight[:out_dim],
        weight[out_dim : 2 * out_dim],
        weight[2 * out_dim : 3 * out_dim],
    )


class StiefelFactor(nn.Module):
    """Column-orthonormal factor using QR retraction."""

    def __init__(self, out_dim: int, rank: int, init_std: float = 0.02):
        super().__init__()
        if rank > out_dim:
            raise ValueError(f"rank ({rank}) must be <= out_dim ({out_dim})")
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.theta = nn.Parameter(torch.randn(out_dim, rank) * float(init_std))

    def weight(self) -> torch.Tensor:
        return _canonical_qr(self.theta)

    @torch.no_grad()
    def set_from_matrix(self, mat: torch.Tensor, noise_std: float = 0.0) -> None:
        if tuple(mat.shape) != tuple(self.theta.shape):
            raise ValueError(f"Shape mismatch: expected {tuple(self.theta.shape)}, got {tuple(mat.shape)}")
        data = mat.detach().to(device=self.theta.device, dtype=self.theta.dtype)
        if noise_std > 0.0:
            data = data + noise_std * torch.randn_like(data)
        self.theta.copy_(data)

    @torch.no_grad()
    def reset_random(self, init_std: float = 0.02) -> None:
        self.theta.normal_(mean=0.0, std=float(init_std))


class OrthogonalLoRABranch(nn.Module):
    """ΔW = U diag(sigma) V^T with orthonormal U, V."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        sigma_init: float = 1e-4,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.u_factor = StiefelFactor(out_dim, rank, init_std=init_std)
        self.v_factor = StiefelFactor(in_dim, rank, init_std=init_std)
        self.s_logit = nn.Parameter(
            _inverse_softplus(torch.full((rank,), float(sigma_init), dtype=torch.float32))
        )
        self.register_buffer("_init_dense", torch.zeros(out_dim, in_dim))

    def sigma(self) -> torch.Tensor:
        return F.softplus(self.s_logit)

    def U(self) -> torch.Tensor:
        return self.u_factor.weight()

    def V(self) -> torch.Tensor:
        return self.v_factor.weight()

    def dense_update(self) -> torch.Tensor:
        u = self.U()
        v = self.V()
        sigma = self.sigma()
        return (u * sigma.unsqueeze(0)).matmul(v.t())

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        v = self.V()
        u = self.U()
        sigma = self.sigma()
        latent = torch.matmul(x, v) * sigma
        return torch.matmul(latent, u.t())

    @torch.no_grad()
    def snapshot_init(self) -> None:
        self._init_dense.copy_(self.dense_update().detach())

    def task_update(self) -> torch.Tensor:
        return self.dense_update() - self._init_dense

    @torch.no_grad()
    def reset_random(self, sigma_init: float = 1e-4, init_std: float = 0.02) -> None:
        self.u_factor.reset_random(init_std=init_std)
        self.v_factor.reset_random(init_std=init_std)
        self.s_logit.copy_(
            _inverse_softplus(torch.full_like(self.s_logit, float(sigma_init)))
        )

    @torch.no_grad()
    def reset_zero(self) -> None:
        self.u_factor.reset_random(init_std=0.02)
        self.v_factor.reset_random(init_std=0.02)
        self.s_logit.copy_(_inverse_softplus(torch.zeros_like(self.s_logit) + 1e-8))

    @torch.no_grad()
    def set_from_svd(self, u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor, noise_std: float = 0.0) -> None:
        r = min(self.rank, u.shape[1], vh.shape[0], s.shape[0])
        if r <= 0:
            self.reset_zero()
            return
        u_r = u[:, :r]
        v_r = vh[:r, :].t()
        s_r = s[:r]

        if r < self.rank:
            pad_u = torch.randn(self.out_dim, self.rank - r, device=u.device, dtype=u.dtype)
            pad_v = torch.randn(self.in_dim, self.rank - r, device=vh.device, dtype=vh.dtype)
            u_r = torch.cat([u_r, _canonical_qr(pad_u)], dim=1)
            v_r = torch.cat([v_r, _canonical_qr(pad_v)], dim=1)
            s_r = torch.cat([s_r, torch.zeros(self.rank - r, device=s.device, dtype=s.dtype)], dim=0)

        self.u_factor.set_from_matrix(u_r[:, : self.rank], noise_std=noise_std)
        self.v_factor.set_from_matrix(v_r[:, : self.rank], noise_std=noise_std)
        self.s_logit.copy_(_inverse_softplus(s_r[: self.rank].to(device=self.s_logit.device, dtype=self.s_logit.dtype)))


class GeoLoRAUVQKV(nn.Module):
    """QKV adapter with shared orthogonal subspace and one task-private branch."""

    def __init__(
        self,
        qkv: nn.Linear,
        rank_private: int,
        rank_shared: int,
        init_std: float = 0.02,
        sigma_init: float = 1e-4,
        add_private_branch: bool = True,
    ):
        super().__init__()
        if not isinstance(qkv, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(qkv)}")
        if qkv.out_features % 3 != 0:
            raise ValueError(f"qkv.out_features must be divisible by 3, got {qkv.out_features}")

        self.qkv = qkv
        self.in_dim = int(qkv.in_features)
        self.out_dim = int(qkv.out_features // 3)
        self.rank_private = int(rank_private)
        self.rank_shared = int(rank_shared)

        self.cur_q: Optional[OrthogonalLoRABranch] = None
        self.cur_v: Optional[OrthogonalLoRABranch] = None
        if add_private_branch:
            self.cur_q = OrthogonalLoRABranch(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                rank=self.rank_private,
                sigma_init=sigma_init,
                init_std=init_std,
            )
            self.cur_v = OrthogonalLoRABranch(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                rank=self.rank_private,
                sigma_init=sigma_init,
                init_std=init_std,
            )

        self.register_buffer("U_sh_q", torch.empty(self.out_dim, 0))
        self.register_buffer("V_sh_q", torch.empty(self.in_dim, 0))
        self.register_buffer("M_sh_q", torch.empty(0, 0))
        self.register_buffer("U_sh_v", torch.empty(self.out_dim, 0))
        self.register_buffer("V_sh_v", torch.empty(self.in_dim, 0))
        self.register_buffer("M_sh_v", torch.empty(0, 0))

        q0, _, v0 = _split_qkv_weight(self.qkv.weight.detach())
        self.register_buffer("_w0_q", q0.clone())
        self.register_buffer("_w0_v", v0.clone())

        for p in self.qkv.parameters():
            p.requires_grad = False

        self._collect_efm = False
        self._efm_in: Optional[torch.Tensor] = None
        self._efm_out_q: Optional[torch.Tensor] = None
        self._efm_out_v: Optional[torch.Tensor] = None
        self._efm_count = 0

        self._collect_init_grad = False
        self._grad_q: Optional[torch.Tensor] = None
        self._grad_v: Optional[torch.Tensor] = None
        self._grad_count = 0

    def _shared_delta(
        self,
        x: torch.Tensor,
        U_sh: torch.Tensor,
        V_sh: torch.Tensor,
        M_sh: torch.Tensor,
    ) -> torch.Tensor:
        if U_sh.numel() == 0 or V_sh.numel() == 0 or M_sh.numel() == 0:
            return torch.zeros(x.shape[0], x.shape[1], self.out_dim, device=x.device, dtype=x.dtype)
        return torch.matmul(torch.matmul(torch.matmul(x, V_sh), M_sh.t()), U_sh.t())

    def _observe_backward(self, x_det: torch.Tensor, grad_out: torch.Tensor) -> None:
        x2d = x_det.reshape(-1, x_det.shape[-1]).to("cpu", dtype=torch.float32)
        g2d = grad_out.reshape(-1, grad_out.shape[-1]).to("cpu", dtype=torch.float32)
        n = max(1, x2d.shape[0])
        gq = g2d[:, : self.out_dim]
        gv = g2d[:, 2 * self.out_dim : 3 * self.out_dim]

        if self._collect_efm:
            if self._efm_in is None:
                self._efm_in = torch.zeros(self.in_dim, self.in_dim, dtype=torch.float32)
                self._efm_out_q = torch.zeros(self.out_dim, self.out_dim, dtype=torch.float32)
                self._efm_out_v = torch.zeros(self.out_dim, self.out_dim, dtype=torch.float32)
            self._efm_in += (x2d.t() @ x2d) / float(n)
            self._efm_out_q += (gq.t() @ gq) / float(n)
            self._efm_out_v += (gv.t() @ gv) / float(n)
            self._efm_count += 1

        if self._collect_init_grad:
            if self._grad_q is None:
                self._grad_q = torch.zeros(self.out_dim, self.in_dim, dtype=torch.float32)
                self._grad_v = torch.zeros(self.out_dim, self.in_dim, dtype=torch.float32)
            self._grad_q += (gq.t() @ x2d) / float(n)
            self._grad_v += (gv.t() @ x2d) / float(n)
            self._grad_count += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        dq_sh = self._shared_delta(x, self.U_sh_q, self.V_sh_q, self.M_sh_q)
        dv_sh = self._shared_delta(x, self.U_sh_v, self.V_sh_v, self.M_sh_v)
        dq_pr = self.cur_q.forward_delta(x) if self.cur_q is not None else torch.zeros_like(q)
        dv_pr = self.cur_v.forward_delta(x) if self.cur_v is not None else torch.zeros_like(v)

        out = torch.cat([q + dq_sh + dq_pr, k, v + dv_sh + dv_pr], dim=-1)

        if (self._collect_efm or self._collect_init_grad) and out.requires_grad:
            x_det = x.detach()

            def _hook(grad: torch.Tensor) -> None:
                self._observe_backward(x_det, grad.detach())

            out.register_hook(_hook)

        return out

    @torch.no_grad()
    def pretrained_shared_state(self, rank: int) -> LayerState:
        rank = int(min(rank, self.out_dim, self.in_dim))
        uq, _, vhq = _svd_rank_k(self._w0_q.float(), rank)
        uv, _, vhv = _svd_rank_k(self._w0_v.float(), rank)
        return {
            "q": {
                "U": uq.detach().cpu(),
                "V": vhq.t().detach().cpu(),
                "M": torch.zeros(rank, rank, dtype=torch.float32),
            },
            "v": {
                "U": uv.detach().cpu(),
                "V": vhv.t().detach().cpu(),
                "M": torch.zeros(rank, rank, dtype=torch.float32),
            },
        }

    @torch.no_grad()
    def set_shared_state(self, state_q: Optional[BranchState], state_v: Optional[BranchState]) -> None:
        device = self.qkv.weight.device
        dtype = self.qkv.weight.dtype

        def _assign(prefix: str, state: Optional[BranchState], out_dim: int, in_dim: int) -> None:
            if state is None or not state:
                setattr(self, f"U_sh_{prefix}", torch.empty(out_dim, 0, device=device, dtype=dtype))
                setattr(self, f"V_sh_{prefix}", torch.empty(in_dim, 0, device=device, dtype=dtype))
                setattr(self, f"M_sh_{prefix}", torch.empty(0, 0, device=device, dtype=dtype))
                return
            U = state["U"].to(device=device, dtype=dtype)
            V = state["V"].to(device=device, dtype=dtype)
            M = state["M"].to(device=device, dtype=dtype)
            setattr(self, f"U_sh_{prefix}", U)
            setattr(self, f"V_sh_{prefix}", V)
            setattr(self, f"M_sh_{prefix}", M)

        _assign("q", state_q, self.out_dim, self.in_dim)
        _assign("v", state_v, self.out_dim, self.in_dim)

    @torch.no_grad()
    def export_shared_state(self) -> LayerState:
        return {
            "q": {
                "U": self.U_sh_q.detach().cpu(),
                "V": self.V_sh_q.detach().cpu(),
                "M": self.M_sh_q.detach().cpu(),
            },
            "v": {
                "U": self.U_sh_v.detach().cpu(),
                "V": self.V_sh_v.detach().cpu(),
                "M": self.M_sh_v.detach().cpu(),
            },
        }

    @torch.no_grad()
    def reset_private_random(self, sigma_init: float = 1e-4, init_std: float = 0.02) -> None:
        if self.cur_q is not None:
            self.cur_q.reset_random(sigma_init=sigma_init, init_std=init_std)
            self.cur_v.reset_random(sigma_init=sigma_init, init_std=init_std)

    @torch.no_grad()
    def reset_private_zero(self) -> None:
        if self.cur_q is not None:
            self.cur_q.reset_zero()
            self.cur_v.reset_zero()

    @torch.no_grad()
    def snapshot_private_init(self) -> None:
        if self.cur_q is not None:
            self.cur_q.snapshot_init()
            self.cur_v.snapshot_init()

    @torch.no_grad()
    def pissa_initialize_private(self) -> None:
        if self.cur_q is None or self.cur_v is None:
            return
        rank = int(self.rank_private)
        with torch.no_grad():
            q_w, k_w, v_w = _split_qkv_weight(self.qkv.weight.data)
            uq, sq, vhq = _svd_rank_k(self._w0_q.to(device=q_w.device, dtype=q_w.dtype), rank)
            uv, sv, vhv = _svd_rank_k(self._w0_v.to(device=v_w.device, dtype=v_w.dtype), rank)

            q_pri = (uq * sq.unsqueeze(0)).matmul(vhq)
            v_pri = (uv * sv.unsqueeze(0)).matmul(vhv)
            q_res = self._w0_q.to(device=q_w.device, dtype=q_w.dtype) - q_pri
            v_res = self._w0_v.to(device=v_w.device, dtype=v_w.dtype) - v_pri

            self.qkv.weight.data[: self.out_dim].copy_(q_res)
            self.qkv.weight.data[self.out_dim : 2 * self.out_dim].copy_(k_w)
            self.qkv.weight.data[2 * self.out_dim : 3 * self.out_dim].copy_(v_res)

            self.cur_q.set_from_svd(uq, sq, vhq)
            self.cur_v.set_from_svd(uv, sv, vhv)

    @torch.no_grad()
    def shared_prior_initialize_private(self, noise_std: float = 1e-3, sigma_scale: float = 1e-3) -> None:
        if self.cur_q is None or self.cur_v is None:
            return

        def _init_from_shared(branch: OrthogonalLoRABranch, U_sh: torch.Tensor, V_sh: torch.Tensor) -> None:
            r = branch.rank
            if U_sh.numel() == 0 or V_sh.numel() == 0:
                branch.reset_random(sigma_init=max(1e-8, sigma_scale), init_std=0.02)
                return
            u = U_sh[:, : min(r, U_sh.shape[1])]
            v = V_sh[:, : min(r, V_sh.shape[1])]
            s = torch.full((u.shape[1],), float(sigma_scale), device=u.device, dtype=u.dtype)
            branch.set_from_svd(u, s, v.t(), noise_std=noise_std)

        _init_from_shared(self.cur_q, self.U_sh_q, self.V_sh_q)
        _init_from_shared(self.cur_v, self.U_sh_v, self.V_sh_v)

    @torch.no_grad()
    def lora_ga_initialize_private(
        self,
        grad_q: torch.Tensor,
        grad_v: torch.Tensor,
        scale: float = 1e-3,
    ) -> None:
        if self.cur_q is None or self.cur_v is None:
            return
        uq, sq, vhq = _svd_rank_k(grad_q.float(), self.rank_private)
        uv, sv, vhv = _svd_rank_k(grad_v.float(), self.rank_private)
        self.cur_q.set_from_svd(uq, scale * sq, vhq)
        self.cur_v.set_from_svd(uv, scale * sv, vhv)

    def projection_penalty(self) -> torch.Tensor:
        if self.cur_q is None or self.cur_v is None:
            return self.qkv.weight.new_zeros(())

        def _penalty(branch: OrthogonalLoRABranch, U_sh: torch.Tensor, V_sh: torch.Tensor) -> torch.Tensor:
            delta = branch.task_update()
            if U_sh.numel() == 0 or V_sh.numel() == 0:
                return torch.sum(delta * delta)
            proj = torch.matmul(torch.matmul(torch.matmul(U_sh, U_sh.t()), delta), torch.matmul(V_sh, V_sh.t()))
            res = delta - proj
            return torch.sum(res * res)

        return _penalty(self.cur_q, self.U_sh_q, self.V_sh_q) + _penalty(self.cur_v, self.U_sh_v, self.V_sh_v)

    @torch.no_grad()
    def extract_task_updates(self) -> Dict[str, torch.Tensor]:
        if self.cur_q is None or self.cur_v is None:
            return {
                "q": torch.zeros(self.out_dim, self.in_dim, dtype=torch.float32),
                "v": torch.zeros(self.out_dim, self.in_dim, dtype=torch.float32),
            }
        return {
            "q": self.cur_q.task_update().detach().cpu().float(),
            "v": self.cur_v.task_update().detach().cpu().float(),
        }

    def efm_begin(self) -> None:
        self._collect_efm = True
        self._efm_in = torch.zeros(self.in_dim, self.in_dim, dtype=torch.float32)
        self._efm_out_q = torch.zeros(self.out_dim, self.out_dim, dtype=torch.float32)
        self._efm_out_v = torch.zeros(self.out_dim, self.out_dim, dtype=torch.float32)
        self._efm_count = 0

    def efm_end(self) -> None:
        self._collect_efm = False

    def efm_stats(self) -> Dict[str, torch.Tensor]:
        c = max(1, int(self._efm_count))
        return {
            "in": self._efm_in / float(c) if self._efm_in is not None else torch.zeros(self.in_dim, self.in_dim),
            "out_q": self._efm_out_q / float(c) if self._efm_out_q is not None else torch.zeros(self.out_dim, self.out_dim),
            "out_v": self._efm_out_v / float(c) if self._efm_out_v is not None else torch.zeros(self.out_dim, self.out_dim),
        }

    def grad_init_begin(self) -> None:
        self._collect_init_grad = True
        self._grad_q = torch.zeros(self.out_dim, self.in_dim, dtype=torch.float32)
        self._grad_v = torch.zeros(self.out_dim, self.in_dim, dtype=torch.float32)
        self._grad_count = 0

    def grad_init_end(self) -> None:
        self._collect_init_grad = False

    def grad_init_stats(self) -> Dict[str, torch.Tensor]:
        c = max(1, int(self._grad_count))
        return {
            "q": self._grad_q / float(c) if self._grad_q is not None else torch.zeros(self.out_dim, self.in_dim),
            "v": self._grad_v / float(c) if self._grad_v is not None else torch.zeros(self.out_dim, self.in_dim),
        }


class GeoLoRAUVViT(nn.Module):
    def __init__(
        self,
        vit_model: nn.Module,
        rank_private: int,
        rank_shared: int,
        shared_states: Optional[List[LayerState]] = None,
        add_private_branch: bool = True,
    ):
        super().__init__()
        self.vit = vit_model
        self.rank_private = int(rank_private)
        self.rank_shared = int(rank_shared)
        self._adapters: List[GeoLoRAUVQKV] = []

        for p in self.vit.parameters():
            p.requires_grad = False

        self._inject(add_private_branch=add_private_branch)
        if shared_states:
            self.apply_shared_states(shared_states)
        self.out_dim = getattr(self.vit, "num_features", 768)

    def _inject(self, add_private_branch: bool) -> None:
        for blk in self.vit.blocks:
            qkv = blk.attn.qkv
            adapter = GeoLoRAUVQKV(
                qkv=qkv,
                rank_private=self.rank_private,
                rank_shared=self.rank_shared,
                add_private_branch=add_private_branch,
            )
            blk.attn.qkv = adapter
            self._adapters.append(adapter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

    def adapters(self) -> List[GeoLoRAUVQKV]:
        return self._adapters

    @torch.no_grad()
    def apply_shared_states(self, shared_states: List[LayerState]) -> None:
        for idx, adapter in enumerate(self._adapters):
            state = shared_states[idx] if idx < len(shared_states) else {}
            adapter.set_shared_state(state.get("q"), state.get("v"))

    @torch.no_grad()
    def export_shared_states(self) -> List[LayerState]:
        return [adapter.export_shared_state() for adapter in self._adapters]


class Learner(LoraBaseLearner):
    """Orthogonal GeoLoRA with UV subspace memory and EFM-based slow merge."""

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = str(args.get("optimizer_type", "adamw")).lower()

        self._backbone_type = str(args.get("backbone_type", "vit_base_patch16_224"))
        self._rank_private = int(args.get("geouv_rank_private", args.get("geolora_r_private", args.get("lora_rank", 8))))
        self._rank_shared = int(args.get("geouv_rank_shared", args.get("geolora_r_shared", 4)))

        self._init_mode = str(args.get("geouv_init_mode", "pissa")).lower()
        self._init_sigma = float(args.get("geouv_init_sigma", 1e-4))
        self._init_std = float(args.get("geouv_init_std", 0.02))
        self._shared_prior_noise = float(args.get("geouv_shared_prior_noise", 1e-3))
        self._lorga_batches = int(args.get("geouv_lorga_batches", 2))
        self._lorga_scale = float(args.get("geouv_lorga_scale", 1e-3))

        self._proj_lambda = float(args.get("geouv_proj_lambda", 0.0))
        self._efm_batches = int(args.get("geouv_efm_batches", 10))
        self._efm_beta = float(args.get("geouv_efm_beta", 0.9))
        self._merge_eps = float(args.get("geouv_merge_eps", 1e-4))
        self._slow_beta = float(args.get("geouv_slow_beta", 0.9))
        self._expand_thresh = float(args.get("geouv_expand_thresh", 0.5))
        self._expand_step = int(args.get("geouv_expand_step", 1))
        self._rank_shared_max = int(
            args.get("geouv_rank_shared_max", max(self._rank_shared, self._rank_shared + 4))
        )

        self._shared_states: List[LayerState] = []
        self._task_deltas: List[Dict[str, Dict[int, torch.Tensor]]] = []
        self._task_metrics: List[Dict[str, Dict[int, Dict[str, torch.Tensor]]]] = []
        self._slow_stats: List[Dict[str, Dict[str, torch.Tensor]]] = []
        self._efm_ema: List[Dict[str, Dict[str, torch.Tensor]]] = []

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_backbone_name(self) -> str:
        name = self._backbone_type.lower()
        if name in {"pretrained_vit_b16_224", "vit_base_patch16_224"}:
            return "vit_base_patch16_224"
        if name in {"pretrained_vit_b16_224_in21k", "vit_base_patch16_224_in21k"}:
            return "vit_base_patch16_224_in21k"
        return self._backbone_type

    def _shared_ckpt(self, task_id: int) -> str:
        return os.path.join(self.args.get("filepath", "./"), f"geouv_shared_task_{task_id}.pt")

    def build_lora_backbone(
        self,
        *,
        shared_states: Optional[List[LayerState]] = None,
        add_private_branch: bool = True,
    ) -> GeoLoRAUVViT:
        vit = timm.create_model(self._resolve_backbone_name(), pretrained=True, num_classes=0)
        model = GeoLoRAUVViT(
            vit_model=vit.eval(),
            rank_private=self._rank_private,
            rank_shared=self._rank_shared,
            shared_states=shared_states,
            add_private_branch=add_private_branch,
        )
        model.out_dim = getattr(model, "out_dim", 768)
        return model

    def _build_eval_backbone(self, task_idx: int) -> nn.Module:
        shared_states = self._load_shared_snapshot(task_idx)
        return self.build_lora_backbone(shared_states=shared_states, add_private_branch=False)

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network

    def _ensure_storage(self, adapters: List[GeoLoRAUVQKV]) -> None:
        if self._shared_states:
            return
        self._shared_states = [adapter.pretrained_shared_state(self._rank_shared) for adapter in adapters]
        self._task_deltas = [{"q": {}, "v": {}} for _ in adapters]
        self._task_metrics = [{"q": {}, "v": {}} for _ in adapters]
        self._slow_stats = []
        self._efm_ema = []
        for adapter in adapters:
            self._slow_stats.append(
                {
                    "q": {
                        "left": torch.zeros(adapter.out_dim, adapter.out_dim, dtype=torch.float32),
                        "right": torch.zeros(adapter.in_dim, adapter.in_dim, dtype=torch.float32),
                    },
                    "v": {
                        "left": torch.zeros(adapter.out_dim, adapter.out_dim, dtype=torch.float32),
                        "right": torch.zeros(adapter.in_dim, adapter.in_dim, dtype=torch.float32),
                    },
                }
            )
            self._efm_ema.append(
                {
                    "q": {
                        "in": torch.zeros(adapter.in_dim, adapter.in_dim, dtype=torch.float32),
                        "out": torch.zeros(adapter.out_dim, adapter.out_dim, dtype=torch.float32),
                    },
                    "v": {
                        "in": torch.zeros(adapter.in_dim, adapter.in_dim, dtype=torch.float32),
                        "out": torch.zeros(adapter.out_dim, adapter.out_dim, dtype=torch.float32),
                    },
                }
            )

    @staticmethod
    def _normalize_metric(metric: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        metric = 0.5 * (metric + metric.t())
        tr = float(torch.trace(metric))
        if abs(tr) <= eps:
            return metric
        return metric / tr

    @staticmethod
    def _projection_ratio(delta: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> float:
        if U.numel() == 0 or V.numel() == 0:
            return 0.0
        core = U.t().matmul(delta).matmul(V)
        proj = U.matmul(core).matmul(V.t())
        return float(torch.sum(proj * proj) / (torch.sum(delta * delta) + 1e-12))

    def _smooth_feature_metric(self, layer_idx: int, stats: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        beta = float(max(0.0, min(0.9999, self._efm_beta)))
        ema = self._efm_ema[layer_idx]
        batch_in = stats["in"].float().cpu()
        batch_out_q = stats["out_q"].float().cpu()
        batch_out_v = stats["out_v"].float().cpu()

        ema["q"]["in"] = beta * ema["q"]["in"] + (1.0 - beta) * batch_in
        ema["q"]["out"] = beta * ema["q"]["out"] + (1.0 - beta) * batch_out_q
        ema["v"]["in"] = beta * ema["v"]["in"] + (1.0 - beta) * batch_in
        ema["v"]["out"] = beta * ema["v"]["out"] + (1.0 - beta) * batch_out_v

        return {
            "q": {
                "in": ema["q"]["in"].clone(),
                "out": ema["q"]["out"].clone(),
            },
            "v": {
                "in": ema["v"]["in"].clone(),
                "out": ema["v"]["out"].clone(),
            },
        }

    def _save_shared_snapshot(self, task_id: int) -> None:
        os.makedirs(self.args.get("filepath", "./"), exist_ok=True)
        payload = {"shared_states": self._shared_states}
        torch.save(payload, self._shared_ckpt(task_id))

    def _load_shared_snapshot(self, task_id: int) -> List[LayerState]:
        ckpt = self._shared_ckpt(task_id)
        if os.path.exists(ckpt):
            payload = torch.load(ckpt, map_location="cpu")
            return payload.get("shared_states", [])
        return self._shared_states

    def incremental_train(self, data_manager):
        self._refresh_distributed_context()
        self.data_manager = data_manager

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._log(f"Learning on {self._known_classes}-{self._total_classes}")

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args.get("train_num_workers", 8),
        )

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args.get("train_num_workers", 8),
        )

        self._train(self.train_loader, self.test_loader)
        efm_stats = self._estimate_feature_metric(self.train_loader)
        self._slow_merge(self._cur_task, efm_stats)
        self._save_shared_snapshot(self._cur_task)

        base_net = self._unwrap_network()
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(self.args.get("filepath", "./"), self._cur_task)

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._unwrap_network()

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            self._log(f"[GeoLoRA_UV_EFM][NME] Failed to compute class means: {exc}")

    def _configure_current_backbone(self) -> List[GeoLoRAUVQKV]:
        network = self._unwrap_network()
        network.backbone = self.build_lora_backbone(
            shared_states=self._shared_states if self._shared_states else None,
            add_private_branch=True,
        )
        network.backbone.to(self._device)
        self._network = network
        self._prepare_network()

        adapters = self._unwrap_network().backbone.adapters()
        self._ensure_storage(adapters)
        self._unwrap_network().backbone.apply_shared_states(self._shared_states)
        return adapters

    def _task_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cur_task == 0:
            return F.cross_entropy(logits, targets), logits, targets
        fake_targets = targets - self._known_classes
        eval_logits = logits[:, self._known_classes :]
        return F.cross_entropy(eval_logits, fake_targets), eval_logits, fake_targets

    def _initialize_private_branches(self, train_loader: DataLoader, adapters: List[GeoLoRAUVQKV]) -> None:
        mode = self._init_mode
        if mode in {"pissa", "pissa_svd"}:
            for adapter in adapters:
                adapter.pissa_initialize_private()
                adapter.snapshot_private_init()
            return

        if mode in {"lora_ga", "lorga", "ga"}:
            for adapter in adapters:
                adapter.reset_private_zero()
            grad_stats = self._estimate_initial_gradients(train_loader)
            for idx, adapter in enumerate(adapters):
                adapter.lora_ga_initialize_private(
                    grad_q=grad_stats[idx]["q"],
                    grad_v=grad_stats[idx]["v"],
                    scale=self._lorga_scale,
                )
                adapter.snapshot_private_init()
            return

        if mode in {"shared", "shared_prior"}:
            for adapter in adapters:
                adapter.shared_prior_initialize_private(
                    noise_std=self._shared_prior_noise,
                    sigma_scale=max(self._init_sigma, 1e-8),
                )
                adapter.snapshot_private_init()
            return

        for adapter in adapters:
            adapter.reset_private_random(sigma_init=self._init_sigma, init_std=self._init_std)
            adapter.snapshot_private_init()

    def _estimate_initial_gradients(self, loader: DataLoader) -> List[Dict[str, torch.Tensor]]:
        net = self._unwrap_network()
        adapters: List[GeoLoRAUVQKV] = net.backbone.adapters()
        for adapter in adapters:
            adapter.grad_init_begin()

        net.eval()
        max_batches = max(1, int(self._lorga_batches))
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            _, inputs, targets = batch if len(batch) == 3 else (None, *batch)
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            net.zero_grad(set_to_none=True)
            logits = net(inputs)["logits"]
            loss, _, _ = self._task_ce_loss(logits, targets)
            loss.backward()

        stats = [adapter.grad_init_stats() for adapter in adapters]
        for adapter in adapters:
            adapter.grad_init_end()
        return stats

    def _estimate_feature_metric(self, loader: DataLoader) -> List[Dict[str, torch.Tensor]]:
        net = self._unwrap_network()
        adapters: List[GeoLoRAUVQKV] = net.backbone.adapters()
        for adapter in adapters:
            adapter.efm_begin()

        net.eval()
        max_batches = max(1, int(self._efm_batches))
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            _, inputs, targets = batch if len(batch) == 3 else (None, *batch)
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            net.zero_grad(set_to_none=True)
            logits = net(inputs)["logits"]
            loss, _, _ = self._task_ce_loss(logits, targets)
            loss.backward()

        stats = [adapter.efm_stats() for adapter in adapters]
        for adapter in adapters:
            adapter.efm_end()
        return stats

    def _train(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        adapters = self._configure_current_backbone()
        self._initialize_private_branches(train_loader, adapters)

        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage="init" if self._cur_task == 0 else "update")
        lr = optimizer.param_groups[0]["lr"]
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("milestones", []),
            gamma=float(self.args.get("lrate_decay", 1.0)),
            T_max=self.args.get("epochs", None),
            eta_min=self.args.get("min_lr", 0.1 * lr),
        )
        epochs = int(self.args.get("init_epoch" if self._cur_task == 0 else "epochs", 1))

        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        for epoch in prog_bar:
            self._network.train()
            total_loss = 0.0
            correct, total = 0, 0
            for batch in train_loader:
                _, inputs, targets = batch if len(batch) == 3 else (None, *batch)
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                optimizer.zero_grad()
                logits = self._network(inputs)["logits"]
                ce_loss, eval_logits, eval_targets = self._task_ce_loss(logits, targets)
                proj_loss = logits.new_zeros(())
                if self._proj_lambda > 0.0:
                    base_net = self._unwrap_network()
                    for adapter in base_net.backbone.adapters():
                        proj_loss = proj_loss + adapter.projection_penalty()
                loss = ce_loss + float(self._proj_lambda) * proj_loss
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().item())
                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / max(1, total), decimals=2)
            info = (
                f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                f"Loss {total_loss / max(1, len(train_loader)):.3f}, Train_accy {train_acc:.2f}"
            )
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            if self._is_main_process:
                prog_bar.set_description(info)

        if self._is_main_process:
            self._log(info)

    @staticmethod
    def _riemannian_merge(
        task_deltas: Dict[int, torch.Tensor],
        task_metrics: Dict[int, Dict[str, torch.Tensor]],
        U: torch.Tensor,
        V: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        k = int(U.shape[1])
        if k <= 0:
            return torch.empty(0, 0, dtype=torch.float32)
        eye = torch.eye(k * k, dtype=torch.float32)
        normal = float(eps) * eye
        rhs = torch.zeros(k * k, dtype=torch.float32)
        used = 0

        for tid, delta in task_deltas.items():
            if delta.numel() == 0:
                continue
            M_t = U.t().matmul(delta.float()).matmul(V)
            metric = task_metrics.get(int(tid), None)
            if metric is not None:
                G_out = U.t().matmul(metric["out"].float()).matmul(U)
                G_in = V.t().matmul(metric["in"].float()).matmul(V)
                G_out = Learner._normalize_metric(G_out)
                G_in = Learner._normalize_metric(G_in)
                if float(torch.trace(G_out)) <= 1e-12 or float(torch.trace(G_in)) <= 1e-12:
                    K = torch.eye(k * k, dtype=torch.float32)
                else:
                    K = torch.kron(G_in.t().contiguous(), G_out.contiguous())
            else:
                K = torch.eye(k * k, dtype=torch.float32)
            normal = normal + K
            rhs = rhs + K.matmul(M_t.reshape(-1))
            used += 1

        if used <= 0:
            return torch.zeros(k, k, dtype=torch.float32)
        try:
            vec = torch.linalg.solve(normal, rhs.unsqueeze(1)).squeeze(1)
        except torch.linalg.LinAlgError:
            vec = torch.linalg.pinv(normal).matmul(rhs)
        return vec.view(k, k)

    def _update_branch_shared(
        self,
        layer_idx: int,
        branch: str,
        task_id: int,
    ) -> None:
        state = self._shared_states[layer_idx][branch]
        delta_t = self._task_deltas[layer_idx][branch][task_id].float()

        U_cur = state["U"].float()
        V_cur = state["V"].float()
        proj_ratio = self._projection_ratio(delta_t, U_cur, V_cur)
        if U_cur.numel() > 0 and V_cur.numel() > 0:
            M_cur = U_cur.t().matmul(delta_t).matmul(V_cur)
            residual = delta_t - U_cur.matmul(M_cur).matmul(V_cur.t())
        else:
            residual = delta_t

        stats = self._slow_stats[layer_idx][branch]
        stats["left"] = float(self._slow_beta) * stats["left"] + residual.matmul(residual.t())
        stats["right"] = float(self._slow_beta) * stats["right"] + residual.t().matmul(residual)

        cur_k = int(U_cur.shape[1]) if U_cur.numel() > 0 else int(self._rank_shared)
        max_k = int(min(self._rank_shared_max, delta_t.shape[0], delta_t.shape[1]))
        if proj_ratio < float(self._expand_thresh) and cur_k < max_k:
            k = min(max_k, cur_k + max(1, int(self._expand_step)))
        else:
            k = min(cur_k, max_k)
        if k <= 0:
            self._shared_states[layer_idx][branch] = {
                "U": torch.empty(delta_t.shape[0], 0),
                "V": torch.empty(delta_t.shape[1], 0),
                "M": torch.empty(0, 0),
            }
            return

        if float(torch.trace(stats["left"])) <= 1e-12 or float(torch.trace(stats["right"])) <= 1e-12:
            U_new = U_cur
            V_new = V_cur
        else:
            U_new = _top_eig_basis(stats["left"], k)
            V_new = _top_eig_basis(stats["right"], k)
            if U_new.numel() == 0 or V_new.numel() == 0:
                U_new = U_cur
                V_new = V_cur

        M_new = self._riemannian_merge(
            task_deltas=self._task_deltas[layer_idx][branch],
            task_metrics=self._task_metrics[layer_idx][branch],
            U=U_new,
            V=V_new,
            eps=self._merge_eps,
        )
        self._shared_states[layer_idx][branch] = {
            "U": U_new.detach().cpu(),
            "V": V_new.detach().cpu(),
            "M": M_new.detach().cpu(),
        }
        if self._is_main_process and k > cur_k:
            self._log(
                f"[GeoLoRA_UV_EFM] Expand layer={layer_idx} branch={branch}: "
                f"k {cur_k} -> {k} (proj_ratio={proj_ratio:.4f}, tau={self._expand_thresh:.4f})"
            )

    def _slow_merge(self, task_id: int, efm_stats: List[Dict[str, torch.Tensor]]) -> None:
        adapters: List[GeoLoRAUVQKV] = self._unwrap_network().backbone.adapters()
        for lidx, adapter in enumerate(adapters):
            updates = adapter.extract_task_updates()
            stats = self._smooth_feature_metric(lidx, efm_stats[lidx])

            self._task_deltas[lidx]["q"][task_id] = updates["q"].float()
            self._task_deltas[lidx]["v"][task_id] = updates["v"].float()
            self._task_metrics[lidx]["q"][task_id] = {
                "in": stats["q"]["in"].float().cpu(),
                "out": stats["q"]["out"].float().cpu(),
            }
            self._task_metrics[lidx]["v"][task_id] = {
                "in": stats["v"]["in"].float().cpu(),
                "out": stats["v"]["out"].float().cpu(),
            }

            self._update_branch_shared(lidx, "q", task_id)
            self._update_branch_shared(lidx, "v", task_id)

            q_state = self._shared_states[lidx]["q"]
            v_state = self._shared_states[lidx]["v"]
            q_delta = q_state["U"].float().matmul(q_state["M"].float()).matmul(q_state["V"].float().t())
            v_delta = v_state["U"].float().matmul(v_state["M"].float()).matmul(v_state["V"].float().t())
            q_ratio = float(torch.sum(q_delta * q_delta) / (torch.sum(updates["q"] * updates["q"]) + 1e-12))
            v_ratio = float(torch.sum(v_delta * v_delta) / (torch.sum(updates["v"] * updates["v"]) + 1e-12))
            if self._is_main_process:
                self._log(
                    f"[GeoLoRA_UV_EFM] Layer {lidx}: "
                    f"k_q={q_state['U'].shape[1]}, k_v={v_state['U'].shape[1]}, "
                    f"proj_q={q_ratio:.4f}, proj_v={v_ratio:.4f}"
                )
