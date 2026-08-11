---
type: paper
node_id: paper:zhang2023_gam
title: "Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization"
authors: ["Xingxuan Zhang", "Renzhe Xu", "Han Yu", "Hao Zou", "Peng Cui"]
year: 2023
venue: CVPR 2023
external_ids:
  arxiv: "2303.03108"
  doi: null
  s2: null
tags: [flatness, first-order, gradient-norm, generalization, GAM]
relevance: related
origin_skill: research-wiki
created_at: 2026-04-24T00:00:00Z
updated_at: 2026-04-24T00:00:00Z
---

# One-line thesis

GAM seeks first-order flatness by minimizing both the loss and the gradient norm simultaneously, avoiding the expensive worst-case perturbation step of SAM.

## Problem / Gap

SAM minimizes zeroth-order sharpness (max loss in ε-ball); first-order flatness (gradient norm) is cheaper to optimize and may be a better surrogate for generalization.

## Method

Add a gradient-norm regularization term to the standard training objective. AS^(1) in our paper is inspired by this first-order view — it approximates sharpness via gradient norm in the adapter subspace rather than performing a full perturbation step.

## Key Results

- Competitive or superior to SAM on image classification benchmarks.
- First-order flatness correlates well with generalization performance.

## Assumptions

- Gradient norm is a good proxy for local sharpness.

## Limitations / Failure Modes

- As with SAM, scope of "first-order flatness" matters — full-parameter gradient norm is not the relevant quantity in PECL.

## Reusable Ingredients

- First-order sharpness as gradient norm: the key ingredient for AS^(1) in our PECL setting.
- GAM comparison is one of the planned additional experiments (LLM+GAM results in rebuttal).

## Open Questions

## Claims

## Connections

<!-- AUTO-GENERATED -->
- **←extends** paper:foret2021_sam
- **addresses_gap →** gap:G3

## Relevance to This Project

The GAM / first-order flatness idea underpins AS^(1). Our paper shows AS^(1) outperforms AS^(0) and RS^(0) in PECL — consistent with the first-order flatness being a stronger signal than zeroth-order when restricted to the adapter subspace.
