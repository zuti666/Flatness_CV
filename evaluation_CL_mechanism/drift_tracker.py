"""
drift_tracker.py
----------------
Compute the drift interference term from Theorem 1:

    drift = v_t^T q_t,  where
        v_t = ∇L₁(θ)   — old-task gradient (averaged over a few batches)
        q_t = P_OGD(∇L₂(θ)) — OGD-projected new-task gradient

After OGD, drift ≈ 0 (orthogonal complement projection).
This module verifies that empirically at every task boundary.

Also reports raw alignment cos(∇L₁, ∇L₂) before any projection,
which measures how much the tasks naturally interfere.

Public API
----------
compute_drift_interference(
    network, old_loader, new_loader, device,
    ogd_directions, max_batches, known_classes, total_classes
) -> Dict[str, float]
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Gradient helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flat_avg_grad(
    network: nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
    known_classes: int,
    total_classes: int,
    task_role: str,          # "old" | "new" | "all"
) -> Optional[torch.Tensor]:
    """
    Compute average flat gradient over at most max_batches batches.

    task_role:
        "old"  → cross_entropy on full logits, old-class targets (0..known_classes-1)
        "new"  → cross_entropy on new-class logits only, shifted targets
        "all"  → cross_entropy on full logits, all targets
    """
    if loader is None:
        return None

    params = [p for p in network.parameters() if p.requires_grad]
    total_flat: Optional[torch.Tensor] = None
    n = 0

    network.eval()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        if len(batch) == 3:
            _, inputs, targets = batch
        else:
            inputs, targets = batch
        inputs  = inputs.to(device)
        targets = targets.to(device)

        network.zero_grad()
        with torch.enable_grad():
            out    = network(inputs)
            logits = out["logits"]

            if task_role == "new" and known_classes > 0 and total_classes > known_classes:
                fake_targets = targets - known_classes
                loss = F.cross_entropy(logits[:, known_classes:], fake_targets)
            else:
                # "old" and "all": evaluate with global class indices
                loss = F.cross_entropy(logits, targets)

            loss.backward()

        flat = torch.cat(
            [p.grad.reshape(-1) if p.grad is not None
             else torch.zeros(p.numel(), device=device, dtype=p.dtype)
             for p in params]
        ).detach()

        if total_flat is None:
            total_flat = flat
        else:
            total_flat = total_flat + flat
        n += 1

    network.zero_grad()

    if total_flat is None or n == 0:
        return None
    return total_flat / n


# ─────────────────────────────────────────────────────────────────────────────
#  OGD projection (standalone, mirrors OGD_Fisher3._project)
# ─────────────────────────────────────────────────────────────────────────────

def _project_flat(
    g: torch.Tensor,
    directions: List[torch.Tensor],
) -> torch.Tensor:
    if not directions:
        return g.clone()
    g_proj = g.clone()
    for s in directions:
        n = min(g_proj.numel(), s.numel())
        if n == 0:
            continue
        s_part = s[:n].to(device=g_proj.device, dtype=g_proj.dtype)
        dot = torch.dot(g_proj[:n], s_part)
        g_proj[:n] = g_proj[:n] - dot * s_part
    return g_proj


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_drift_interference(
    network: nn.Module,
    old_loader,
    new_loader,
    device: torch.device,
    ogd_directions: Optional[List[torch.Tensor]] = None,
    max_batches: int = 3,
    known_classes: int = 0,
    total_classes: int = 0,
) -> Dict[str, float]:
    """
    Compute drift interference v_t^T q_t and related statistics.

    Returns (all floats, nan if unavailable):
        drift_dot_vq    : v^T q  (after OGD projection; should be ≈ 0)
        drift_cos_vq    : cos(v, q)  (angle between old grad and projected update)
        drift_dot_vg    : v^T g_new  (raw, before projection)
        drift_cos_vg    : cos(v, g_new)  (raw task gradient alignment)
        drift_v_norm    : |v|   = |∇L₁|
        drift_q_norm    : |q|   = |P_OGD(∇L₂)|
        drift_g_norm    : |g_new| = |∇L₂|
        drift_projection_ratio : |q|/|g_new|  (how much OGD reduces the update)
    """
    nan = float("nan")
    out: Dict[str, float] = {
        "drift_dot_vq":           nan,
        "drift_cos_vq":           nan,
        "drift_dot_vg":           nan,
        "drift_cos_vg":           nan,
        "drift_v_norm":           nan,
        "drift_q_norm":           nan,
        "drift_g_norm":           nan,
        "drift_projection_ratio": nan,
    }

    # ── old-task gradient v = ∇L₁ ──────────────────────────────────────────
    v = _flat_avg_grad(
        network, old_loader, device, max_batches,
        known_classes=0, total_classes=total_classes,
        task_role="old",
    )
    if v is None:
        return out

    # ── new-task gradient g = ∇L₂ ──────────────────────────────────────────
    g = _flat_avg_grad(
        network, new_loader, device, max_batches,
        known_classes=known_classes, total_classes=total_classes,
        task_role="new",
    )
    if g is None:
        return out

    v = v.float()
    g = g.float()

    v_norm = float(v.norm().item())
    g_norm = float(g.norm().item())

    # ── OGD projection → q ─────────────────────────────────────────────────
    q = _project_flat(g, ogd_directions or [])
    q_norm = float(q.norm().item())

    # ── dot products ───────────────────────────────────────────────────────
    # Restrict to the overlapping dimension (v and g are same dim by construction)
    n = min(v.numel(), g.numel(), q.numel())
    dot_vq = float(torch.dot(v[:n], q[:n]).item())
    dot_vg = float(torch.dot(v[:n], g[:n]).item())

    denom_vq = max(v_norm * q_norm, 1e-12)
    denom_vg = max(v_norm * g_norm, 1e-12)

    out["drift_dot_vq"]           = dot_vq
    out["drift_cos_vq"]           = dot_vq / denom_vq
    out["drift_dot_vg"]           = dot_vg
    out["drift_cos_vg"]           = dot_vg / denom_vg
    out["drift_v_norm"]           = v_norm
    out["drift_q_norm"]           = q_norm
    out["drift_g_norm"]           = g_norm
    out["drift_projection_ratio"] = q_norm / max(g_norm, 1e-12)

    return out
