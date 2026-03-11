"""Geo-IncLoRA: incremental GeoLoRA with per-task adapter snapshots.

Design goals (aligned with user request):
1) Keep IncLoRA training protocol (task-wise incremental training settings).
2) Replace vanilla LoRA factors with GeoLoRA factors:
      ΔW = B diag(softplus(s)+eps) A^T, A/B orthonormal-column factors.
3) Save one adapter snapshot per task.
4) Train task t on top of merged adapters from tasks [0..t-1] plus a new task-t adapter.
5) Evaluation uses all adapters learned so far.
"""

from __future__ import annotations

import os
from typing import Dict, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations

from models_LoRAbasedCL.inclora import Learner as IncLoRALearner


def _build_householder_orth_linear(out_dim: int, rank: int) -> nn.Linear:
    if rank > out_dim:
        raise ValueError(f"rank ({rank}) must be <= out_dim ({out_dim})")

    layer = nn.Linear(rank, out_dim, bias=False)
    parametrizations.orthogonal(
        layer,
        name="weight",
        orthogonal_map="householder",
        use_trivialization=False,
    )
    nn.init.normal_(layer.parametrizations.weight.original, mean=0.0, std=1.0)
    return layer


class _GeoTaskAdapter(nn.Module):
    """Single-branch Geo adapter (for either Q or V)."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, eps: float, s_init: float):
        super().__init__()
        self.theta_a = _build_householder_orth_linear(in_dim, rank)   # A: [in_dim, r]
        self.theta_b = _build_householder_orth_linear(out_dim, rank)  # B: [out_dim, r]
        self.s = nn.Parameter(torch.full((rank,), float(s_init)))
        self.eps = float(eps)

    def _factors(self):
        A = self.theta_a.weight
        B = self.theta_b.weight
        sigma = F.softplus(self.s) + self.eps
        return A, B, sigma

    def delta(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        A, B, sigma = self._factors()
        u = torch.matmul(x, A) * sigma
        return torch.matmul(u, B.t()) * scale

    def delta_normalized(self, x: torch.Tensor, scale: float, norm_eps: float = 1e-12) -> torch.Tensor:
        """
        LoRA-style factor normalization for historical branches:
        normalize by ||B*diag(sigma)||_F * ||A||_F.
        """
        A, B, sigma = self._factors()
        u = torch.matmul(x, A) * sigma
        out = torch.matmul(u, B.t())
        b_scaled = B * sigma.unsqueeze(0)
        denom = (A.norm(p="fro") * b_scaled.norm(p="fro")).clamp_min(float(norm_eps))
        return out * (scale / denom)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def export_state(self) -> Dict[str, torch.Tensor]:
        return {
            "theta_a_original": self.theta_a.parametrizations.weight.original.detach().cpu(),
            "theta_b_original": self.theta_b.parametrizations.weight.original.detach().cpu(),
            "s": self.s.detach().cpu(),
        }

    @torch.no_grad()
    def load_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.theta_a.parametrizations.weight.original.copy_(state["theta_a_original"])
        self.theta_b.parametrizations.weight.original.copy_(state["theta_b_original"])
        self.s.copy_(state["s"])


class _GeoTaskQVAdapter(nn.Module):
    """Per-task adapter containing both Q and V Geo branches."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, eps: float, s_init: float):
        super().__init__()
        self.q = _GeoTaskAdapter(in_dim, out_dim, rank, eps, s_init=s_init)
        self.v = _GeoTaskAdapter(in_dim, out_dim, rank, eps, s_init=s_init)

    def freeze(self):
        self.q.freeze()
        self.v.freeze()

    @torch.no_grad()
    def export_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {"q": self.q.export_state(), "v": self.v.export_state()}

    @torch.no_grad()
    def load_state(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        self.q.load_state(state["q"])
        self.v.load_state(state["v"])


class _GeoIncQKV(nn.Module):
    """QKV wrapper: sum Geo adapters from old tasks + current task (if training)."""

    def __init__(
        self,
        qkv: nn.Linear,
        rank: int,
        alpha: float,
        eps: float,
        s_init: float,
        layer_idx: int,
        old_task_states: List[Dict[str, Dict[str, Dict[str, torch.Tensor]]]],
        add_new_adapter: bool,
    ):
        super().__init__()
        if qkv.out_features % 3 != 0:
            raise ValueError(f"qkv.out_features must be divisible by 3, got {qkv.out_features}")

        self.qkv = qkv
        self.in_dim = int(qkv.in_features)
        self.out_dim = int(qkv.out_features // 3)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.s_init = float(s_init)
        self.layer_idx = int(layer_idx)

        if self.rank > min(self.in_dim, self.out_dim):
            raise ValueError(
                f"rank={self.rank} must satisfy rank <= min(in_dim={self.in_dim}, out_dim={self.out_dim})"
            )

        for p in self.qkv.parameters():
            p.requires_grad = False

        self.task_adapters = nn.ModuleList()

        # Load frozen adapters from previous tasks.
        for state in old_task_states:
            layer_state = state.get(str(self.layer_idx), None)
            if layer_state is None:
                continue
            adapter = _GeoTaskQVAdapter(self.in_dim, self.out_dim, self.rank, self.eps, s_init=self.s_init)
            adapter.load_state(layer_state)
            adapter.freeze()
            self.task_adapters.append(adapter)

        # Create a new trainable adapter for current task (training mode).
        self.current_adapter_idx = None
        if add_new_adapter:
            cur = _GeoTaskQVAdapter(self.in_dim, self.out_dim, self.rank, self.eps, s_init=self.s_init)
            self.task_adapters.append(cur)
            self.current_adapter_idx = len(self.task_adapters) - 1

    def freeze_all_adapters(self) -> None:
        for adp in self.task_adapters:
            adp.freeze()

    @torch.no_grad()
    def export_current_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self.current_adapter_idx is None:
            raise RuntimeError("No current adapter available to export.")
        return self.task_adapters[self.current_adapter_idx].export_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if len(self.task_adapters) == 0:
            return torch.cat([q, k, v], dim=-1)

        scale = self.alpha / float(self.rank)
        # Match IncLoRA behavior: historical branches normalized, latest branch unnormalized.
        latest_idx = self.current_adapter_idx
        if latest_idx is None and len(self.task_adapters) > 0:
            latest_idx = len(self.task_adapters) - 1

        dq, dv = None, None
        for idx, adp in enumerate(self.task_adapters):
            use_hist_norm = (latest_idx is not None) and (idx < latest_idx)
            if use_hist_norm:
                q_i = adp.q.delta_normalized(x, scale)
                v_i = adp.v.delta_normalized(x, scale)
            else:
                q_i = adp.q.delta(x, scale)
                v_i = adp.v.delta(x, scale)
            dq = q_i if dq is None else (dq + q_i)
            dv = v_i if dv is None else (dv + v_i)

        q = q + dq
        v = v + dv
        return torch.cat([q, k, v], dim=-1)


class GeoIncLoRA_ViT_timm(nn.Module):
    """Incremental GeoLoRA backbone with per-task save/load."""

    def __init__(
        self,
        vit_model: nn.Module,
        r: int,
        alpha: float = 1.0,
        eps: float = 1e-6,
        s_init: float = -1.0,
        filepath: str = "./",
        cur_task_index: int = 0,
        eval: bool = False,
        lora_layer=None,
    ):
        super().__init__()
        self.rank = int(r)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.s_init = float(s_init)
        self.save_file = filepath
        self.task_id = int(cur_task_index)
        self.eval_mode = bool(eval)

        self.vit = vit_model
        for p in self.vit.parameters():
            p.requires_grad = False

        if lora_layer is None:
            self.lora_layer = list(range(len(self.vit.blocks)))
        else:
            self.lora_layer = list(lora_layer)

        # Training task t: load states [0..t-1], add a new adapter.
        # Eval for task t: load states [0..t], do not add new adapter.
        if self.eval_mode:
            num_saved_to_load = self.task_id + 1
            add_new_adapter = False
        else:
            num_saved_to_load = self.task_id
            add_new_adapter = True

        old_task_states = self._load_task_states(num_saved_to_load)
        self._wrappers: Dict[int, _GeoIncQKV] = {}
        self._inject_adapters(old_task_states, add_new_adapter)

        if self.eval_mode:
            for w in self._wrappers.values():
                w.freeze_all_adapters()

        self.out_dim = getattr(self.vit, "num_features", 768)

    def _task_file(self, task_idx: int) -> str:
        return os.path.join(self.save_file, f"geolora_task_{task_idx}.pt")

    def _load_task_states(self, num_saved_to_load: int):
        states = []
        for t in range(max(0, int(num_saved_to_load))):
            f = self._task_file(t)
            if os.path.exists(f):
                payload = torch.load(f, map_location="cpu")
                states.append(payload.get("layers", {}))
            else:
                states.append({})
        return states

    def _inject_adapters(self, old_task_states, add_new_adapter: bool):
        for layer_idx, blk in enumerate(self.vit.blocks):
            if layer_idx not in self.lora_layer:
                continue
            qkv = blk.attn.qkv
            wrapper = _GeoIncQKV(
                qkv=qkv,
                rank=self.rank,
                alpha=self.alpha,
                eps=self.eps,
                s_init=self.s_init,
                layer_idx=layer_idx,
                old_task_states=old_task_states,
                add_new_adapter=add_new_adapter,
            )
            blk.attn.qkv = wrapper
            self._wrappers[layer_idx] = wrapper

    @torch.no_grad()
    def save_lora_parameters(self, filename: str, task_id: int) -> None:
        os.makedirs(filename, exist_ok=True)
        payload = {
            "rank": int(self.rank),
            "alpha": float(self.alpha),
            "eps": float(self.eps),
            "layers": {},
        }
        for layer_idx, wrapper in self._wrappers.items():
            payload["layers"][str(layer_idx)] = wrapper.export_current_state()
        torch.save(payload, os.path.join(filename, f"geolora_task_{task_id}.pt"))

    def forward(self, x):
        return self.vit(x)


class Learner(IncLoRALearner):
    """IncLoRA training protocol + GeoLoRA adapters."""

    def __init__(self, args):
        super().__init__(args)
        self._geolora_rank = int(args.get("geolora_rank", args.get("lora_rank", 8)))
        self._geolora_alpha = float(args.get("geolora_alpha", float(self._geolora_rank)))
        self._geolora_eps = float(args.get("geolora_eps", 1e-6))
        self._geolora_s_init = float(args.get("geolora_s_init", -1.0))
        self._backbone_type = str(args.get("backbone_type", "vit_base_patch16_224"))

    def _timm_backbone_name(self) -> str:
        name = self._backbone_type.lower()
        if name in {"pretrained_vit_b16_224", "vit_base_patch16_224"}:
            return "vit_base_patch16_224"
        if name in {"pretrained_vit_b16_224_in21k", "vit_base_patch16_224_in21k"}:
            return "vit_base_patch16_224_in21k"
        return self._backbone_type

    def _build_incremental_lora(self, eval_mode: bool = False, task_idx: int | None = None):
        # Align with IncLoRA behavior: current task index drives which historical
        # adapters are loaded and whether a new one is appended.
        cur_idx = self._cur_task if task_idx is None else int(task_idx)
        vit = timm.create_model(self._timm_backbone_name(), pretrained=True, num_classes=0)
        model = GeoIncLoRA_ViT_timm(
            vit_model=vit.eval(),
            r=self._geolora_rank,
            alpha=self._geolora_alpha,
            eps=self._geolora_eps,
            s_init=self._geolora_s_init,
            filepath=self.args.get("filepath", "./"),
            cur_task_index=cur_idx,
            eval=eval_mode,
        )
        model.out_dim = 768
        return model

    def _build_eval_backbone(self, task_idx):
        # Restore snapshot after finishing task_idx => load adapters [0..task_idx].
        return self._build_incremental_lora(eval_mode=True, task_idx=task_idx)
