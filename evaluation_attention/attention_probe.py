"""Attention probe export utilities (DINO-style + optional Grad-CAM)."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def run_attention_probe(
    *,
    model,
    net: torch.nn.Module,
    data_manager,
    args: dict,
    log_dir: str,
    save_prefix: str,
    device_override: Optional[torch.device] = None,
) -> Optional[str]:
    """Export attention probe payload for a small test subset."""
    probe_dir = os.path.join(log_dir, "attention_probe")
    os.makedirs(probe_dir, exist_ok=True)

    K = max(1, int(args.get("attention_probe_samples", 64)))
    seed = int(args.get("attention_probe_seed", 42))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    probe_path = os.path.join(probe_dir, f"{save_prefix}_attention_probe_{stamp}.pt")

    probe_dataset = None
    probe_loader = None
    attn_handle = None
    attn_bwd_handle = None
    last_attn_cache = {}
    last_attn_grad = {}

    try:
        total_classes = getattr(model, "_total_classes", data_manager.nb_classes)
        probe_dataset = data_manager.get_dataset(
            np.arange(0, total_classes),
            source="test",
            mode="test",
        )
        N = len(probe_dataset)
        if N == 0:
            raise StopIteration

        rng = np.random.RandomState(seed)
        sel_idx = rng.choice(N, size=min(K, N), replace=False)
        probe_subset = Subset(probe_dataset, sel_idx)

        probe_loader = DataLoader(
            probe_subset,
            batch_size=len(probe_subset),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        probe_idx, probe_inputs, probe_labels = next(iter(probe_loader))
    except StopIteration:
        logging.warning("[AttentionProbe] No samples available for probe export.")
        return None
    except Exception as exc:
        logging.exception("[AttentionProbe] Failed to prepare probe data: %s", exc)
        return None

    # Forward and attention capture
    target_device = device_override or getattr(model, "_device", None)
    if isinstance(target_device, str):
        target_device = torch.device(target_device)
    if target_device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe_inputs = probe_inputs.to(target_device, non_blocking=True)
    net.eval()

    backbone = getattr(net, "backbone", net)
    blocks = getattr(backbone, "blocks", None)

    def _hook_last_attn(module, inp, out):
        try:
            last_attn_cache["attn"] = inp[0].detach()
        except Exception:
            pass

    def _hook_last_attn_bwd(module, grad_input, grad_output):
        try:
            if grad_input and grad_input[0] is not None:
                last_attn_grad["grad"] = grad_input[0].detach()
        except Exception:
            pass

    try:
        if blocks is not None and hasattr(blocks[-1], "attn") and hasattr(blocks[-1].attn, "attn_drop"):
            attn_mod = blocks[-1].attn.attn_drop
            attn_handle = attn_mod.register_forward_hook(_hook_last_attn)
            if bool(args.get("attention_probe_gradcam", False)):
                attn_bwd_handle = attn_mod.register_full_backward_hook(_hook_last_attn_bwd)

        outputs = net(probe_inputs)

        features = outputs.get("features") if isinstance(outputs, dict) else None
        if features is None and backbone is not None:
            with torch.no_grad():
                features = backbone(probe_inputs)

        def _find_logits(out):
            if isinstance(out, torch.Tensor) and out.dim() == 2:
                return out
            if isinstance(out, dict):
                for k in ["logits", "logit", "cls_logits", "output", "outputs"]:
                    v = out.get(k, None)
                    if isinstance(v, torch.Tensor) and v.dim() == 2:
                        return v
                for v in out.values():
                    if isinstance(v, torch.Tensor) and v.dim() == 2:
                        return v
            return None

        logits = _find_logits(outputs)
    except Exception as exc:
        logging.exception("[AttentionProbe] Failed to compute probe outputs: %s", exc)
        logits = None
        features = None

    # Build attention maps (DINO-style)
    attn_heads_up = None
    attn_avg_up = None
    ps = None
    try:
        attn = last_attn_cache.get("attn", None)
        if attn is not None and attn.ndim == 4:
            def _resolve_patch_size(m):
                pe = getattr(m, "patch_embed", None)
                ps_ = getattr(pe, "patch_size", None)
                return ps_[0] if isinstance(ps_, tuple) else (int(ps_) if ps_ is not None else None)

            ps = _resolve_patch_size(backbone)
            B, C, H, W = probe_inputs.shape
            if ps is None or ps <= 0:
                Ntok = attn.shape[-1]
                p = int(round((H * W / (Ntok - 1)) ** 0.5)) if Ntok > 1 else 16
                ps = max(1, p)

            H2, W2 = H - (H % ps), W - (W % ps)
            h, w = max(1, H2 // ps), max(1, W2 // ps)

            A = attn[:, :, 0, 1:]
            A = A.reshape(A.shape[0], A.shape[1], h, w)

            A_up = F.interpolate(A, size=(H2, W2), mode="bilinear", align_corners=False)
            A_avg = A.mean(dim=1, keepdim=True)
            A_avg_up = F.interpolate(A_avg, size=(H2, W2), mode="bilinear", align_corners=False)

            def _norm(x):
                x_min = x.amin(dim=(-2, -1), keepdim=True)
                x_max = x.amax(dim=(-2, -1), keepdim=True)
                return (x - x_min) / (x_max - x_min + 1e-8)

            attn_heads_up = _norm(A_up).cpu()
            attn_avg_up = _norm(A_avg_up).cpu()
    except Exception as exc:
        logging.exception("[AttentionProbe] Failed to build attention maps: %s", exc)

    # Optional: class-discriminative Grad-CAM on attention heads
    gradcam_up = None
    if bool(args.get("attention_probe_gradcam", False)) and logits is not None and last_attn_cache.get("attn") is not None:
        try:
            if hasattr(net, "zero_grad"):
                net.zero_grad(set_to_none=True)
            with torch.no_grad():
                top1 = logits.argmax(dim=1)
            target_y = probe_labels.to(logits.device) if (probe_labels is not None and probe_labels.numel() == logits.size(0)) else top1
            target_score = logits.gather(1, target_y.view(-1, 1)).sum()
            target_score.backward(retain_graph=True)

            G = last_attn_grad.get("grad", None)
            A = last_attn_cache.get("attn", None)
            if G is not None and A is not None and G.shape == A.shape:
                B, Hh, Nq, Nk = G.shape
                g = G[:, :, 0, 1:]
                if ps is None:
                    Himg, Wimg = probe_inputs.shape[-2], probe_inputs.shape[-1]
                    Ntok = Nk
                    p = int(round((Himg * Wimg / (Ntok - 1)) ** 0.5)) if Ntok > 1 else 16
                    ps_ = max(1, p)
                    H2, W2 = Himg - (Himg % ps_), Wimg - (Wimg % ps_)
                    h, w = max(1, H2 // ps_), max(1, W2 // ps_)
                else:
                    Himg, Wimg = probe_inputs.shape[-2], probe_inputs.shape[-1]
                    H2, W2 = Himg - (Himg % ps), Wimg - (Wimg % ps)
                    h, w = max(1, H2 // ps), max(1, W2 // ps)

                g = g.reshape(B, Hh, h, w)
                w_k = g.mean(dim=(-2, -1), keepdim=True)

                A_cls = A[:, :, 0, 1:].reshape(B, Hh, h, w)
                cam = (w_k * A_cls).sum(dim=1, keepdim=True)
                cam = F.relu(cam)
                cam_up = F.interpolate(cam, size=(H2, W2), mode="bilinear", align_corners=False)
                cam_min = cam_up.amin(dim=(-2, -1), keepdim=True)
                cam_max = cam_up.amax(dim=(-2, -1), keepdim=True)
                gradcam_up = ((cam_up - cam_min) / (cam_max - cam_min + 1e-8)).cpu()
        except Exception as exc:
            logging.exception("[AttentionProbe] Grad-CAM computation failed: %s", exc)

    # Prepare payload and save
    meta = {
        "model_tag": getattr(net, "name", type(net).__name__),
        "input_size": (int(probe_inputs.shape[-2]), int(probe_inputs.shape[-1])),
        "patch_size": int(ps) if ps is not None else None,
        "seed": int(seed),
        "num_samples": int(len(probe_subset)) if "probe_subset" in locals() else int(probe_labels.shape[0]),
    }

    payload = {
        "indices": probe_idx.cpu(),
        "labels": probe_labels.cpu(),
        "features": features.detach().cpu() if (features is not None and torch.is_tensor(features)) else None,
        "meta": meta,
    }
    if args.get("attention_probe_save_inputs", False):
        payload["inputs"] = probe_inputs.detach().cpu().to(dtype=torch.float16)
    if attn_heads_up is not None:
        payload["attn_heads"] = attn_heads_up
        payload["attn_avg"] = attn_avg_up
    if gradcam_up is not None:
        payload["gradcam_cls"] = gradcam_up

    try:
        torch.save(payload, probe_path)
        logging.info(
            "[AttentionProbe] Saved %d samples (features%s%s%s) to %s",
            int(probe_labels.shape[0]),
            ", attn_heads" if "attn_heads" in payload else "",
            ", attn_avg" if "attn_avg" in payload else "",
            ", gradcam_cls" if "gradcam_cls" in payload else "",
            probe_path,
        )
    except Exception as exc:
        logging.exception("[AttentionProbe] Saving probe payload failed: %s", exc)
        return None
    finally:
        if attn_handle is not None:
            try:
                attn_handle.remove()
            except Exception:
                pass
        if attn_bwd_handle is not None:
            try:
                attn_bwd_handle.remove()
            except Exception:
                pass

    return probe_path
