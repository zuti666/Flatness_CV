---
id: exp_G
name: Raw-Factor Rescaling Control
dataset: ImageNet-R r=16 T=20
method: SeqLoRA, ViT-B/16
status: pending — next priority
---

## Purpose

Prove Sh_AB is parameterization-dependent while Sh_Delta/W_tangent is effective-update invariant.

## Setup

- Checkpoint: task-9 or task-19 from Exp E or Exp F
- Transform: for each layer, apply A' = A/c, B' = cB → ΔW = B'A' = BA unchanged
- Scales c ∈ {0.25, 0.5, 1, 2, 4}

## Metrics

| Metric | Expected behavior |
|--------|------------------|
| Model accuracy | Unchanged (ΔW = BA, same effective update) |
| ΔW (adapter effective update) | Unchanged |
| Sh_AB (sharpness in raw A,B factor space) | Varies with c |
| Sh_Delta / Sh_W_tangent (sharpness in effective update space) | Stable across c |

## Why This Matters

- Supports **C6** (raw factor-space sharpness not the correct theoretical object)
- Validates Theorem B.2 (support reduction): the bound-relevant object is W_{Δ,t}, which is basis-invariant
- Closes the key theoretical gap: current results show sam_factor is strong but do not prove factor-space is wrong

## Implementation Notes

Transform A, B in-place at each layer (no retraining). Measure sharpness using the post-hoc evaluation framework (flat_eval_loader_shuffle: false, fixed deterministic subset, same batch for base/perturbed loss). Sharpness must be non-negative.
