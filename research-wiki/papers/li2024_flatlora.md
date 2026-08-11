---
type: paper
node_id: paper:li2024_flatlora
title: "Flat-LoRA: Low-Rank Adaption over a Flat Loss Landscape"
authors: ["Tao Li", "Zhengbao He", "Yujun Li", "Yasheng Wang", "Lifeng Shang", "Xiaolin Huang"]
year: 2024
venue: NeurIPS 2024 (appears also as ICML 2025 in rebuttal docs)
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: [LoRA, flatness, SAM, fine-tuning, full-parameter-perturbation]
relevance: core
origin_skill: research-wiki
created_at: 2026-04-24T00:00:00Z
updated_at: 2026-04-24T00:00:00Z
---

# One-line thesis

Flat-LoRA argues that adapter-subspace flatness alone is insufficient and advocates perturbing ALL model parameters (including frozen backbone) to achieve meaningful full-space flatness.

## Problem / Gap

LoRA fine-tuning produces sharper landscapes in the full parameter space; SAM restricted to adapter parameters does not flatten the overall loss surface, allegedly leading to suboptimal generalization.

## Method

Constructs a mechanism to compute perturbations in the full parameter space even when the backbone is frozen, then updates only the LoRA adapter parameters using the full-space gradient. Explicitly contradicts adapter-only perturbation as insufficient.

## Key Results

- Reports improvements in standard fine-tuning benchmarks by flattening the full-space loss landscape.
- Claims adapter-only perturbation leaves the full model's loss landscape sharp.

## Assumptions

- Full-parameter perturbation is feasible and beneficial even when backbone is nominally "frozen."
- Full-space flatness is the relevant notion for generalization in LoRA fine-tuning.

## Limitations / Failure Modes

- **Directly contradicted by our paper**: In PECL with truly frozen backbone, full-parameter perturbation causes systematic performance degradation (−2.00 to −6.84 ΔAcc^cls across all three LoRA variants).
- Full-space perturbation introduces variations outside the admissible update subspace W_{Δ,t}, creating an "irreducible scope mismatch."
- The frozen-backbone constraint means full-space perturbation perturbs parameters that do not affect the task objective — these perturbations are wasteful or harmful.

## Reusable Ingredients

- Framing of flatness scope as a design choice (even if their conclusion is wrong for PECL).
- Motivating observation that LoRA produces sharper full-space landscapes is useful context.

## Open Questions

- Does Flat-LoRA's benefit appear only in non-frozen fine-tuning regimes?

## Claims

[Contested by claim:C1 — our theory shows adapter-subspace flatness IS sufficient under frozen backbone]

## Connections

<!-- AUTO-GENERATED -->
- **extends →** paper:foret2021_sam
- **←addresses_gap** idea:001 (our idea directly contests Flat-LoRA's recommendation)

## Relevance to This Project

This is the primary foil paper. Our entire project is motivated by Flat-LoRA's explicit claim that full-parameter perturbation is needed — we show this is wrong in PECL (both theoretically and empirically). The rebuttal heavily relies on differentiating from this paper.
