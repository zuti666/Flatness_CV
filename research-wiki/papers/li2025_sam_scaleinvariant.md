---
type: paper
node_id: paper:li2025_sam_scaleinvariant
title: "Implicit Regularization of Sharpness-Aware Minimization for Scale-Invariant Problems"
authors: ["Tao Li", "Qinghua Tao", "Yingwen Wu", "Xiaolin Huang"]
year: 2025
venue: arXiv 2025
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: [SAM, implicit-regularization, scale-invariant, sharpness, theory]
relevance: related
origin_skill: research-wiki
created_at: 2026-04-24T00:00:00Z
updated_at: 2026-04-24T00:00:00Z
---

# One-line thesis

Analyzes SAM's implicit regularization properties for scale-invariant problems, showing that SAM implicitly minimizes gradient norm in addition to loss.

## Problem / Gap

SAM's implicit bias and regularization effects are not fully understood theoretically, especially for scale-invariant architectures.

## Method

Theoretical analysis of SAM dynamics for scale-invariant problems. Shows connection between SAM's perturbation step and gradient norm minimization (first-order flatness).

## Key Results

- SAM implicitly regularizes gradient norm for scale-invariant problems.
- Provides theoretical grounding for why SAM finds first-order flat solutions.

## Assumptions

- Scale-invariant parameterization (e.g., batch-normalized networks).

## Limitations / Failure Modes

[TBD]

## Reusable Ingredients

- Theoretical connection between SAM and gradient norm minimization — supports the use of AS^(1) (first-order adapter-subspace perturbation) in our method.

## Open Questions

## Claims

## Connections

<!-- AUTO-GENERATED -->
- **extends →** paper:foret2021_sam

## Relevance to This Project

Provides theoretical backing for the first-order (AS^(1)) variant of adapter-subspace sharpness minimization used in our experiments. From RefMotivationPaper folder.
