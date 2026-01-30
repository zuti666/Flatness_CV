from types import MethodType
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn


def enable_last_attn(
    vit: nn.Module,
    detach: bool = True,
) -> Tuple[Dict[str, Optional[torch.Tensor]], Callable[[], None]]:
    """
    Patch the last attention block of a timm VisionTransformer so the softmaxed
    attention weights are cached during forward. Returns a cache dict and a
    restore callable that must be invoked once you are done.
    """
    if not hasattr(vit, "blocks") or len(getattr(vit, "blocks")) == 0:
        raise AttributeError("Expected a VisionTransformer with non-empty blocks.")

    attn_module = vit.blocks[-1].attn
    original_forward = attn_module.forward
    cache: Dict[str, Optional[torch.Tensor]] = {"last_attn": None}

    def patched_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        cache["last_attn"] = attn.detach() if detach else attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(self.proj_drop(x))
        return x

    attn_module.forward = MethodType(patched_forward, attn_module)

    def restore() -> None:
        attn_module.forward = original_forward
        cache["last_attn"] = None

    return cache, restore
