---
type: paper
node_id: paper:bisla2022_rwp
title: "Towards Efficient and Effective Deep Learning: Sharpness-Aware Minimization (RWP variant)"
authors: ["Devansh Bisla", "Jing Wang", "Anna Choromanska"]
year: 2022
venue: AISTATS 2022
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: [sharpness, SAM, flatness, optimizer, random-perturbation]
relevance: related
origin_skill: research-wiki
created_at: 2026-04-24T00:00:00Z
updated_at: 2026-04-24T00:00:00Z
---

# One-line thesis

RWP (Random Weight Perturbation) is a stochastic variant of SAM that perturbs parameters randomly rather than using gradient-based worst-case perturbation, reducing computational overhead.

## Problem / Gap

SAM's worst-case gradient ascent step doubles training cost; a cheaper perturbation strategy is needed.

## Method

Inject random noise into parameters at each step instead of computing the gradient-ascent worst-case perturbation. Zeroth-order approach — does not require an additional forward-backward pass.

## Key Results

- Competitive with SAM at lower compute.
- Serves as a strong baseline for flatness-seeking methods.

## Assumptions

- Random perturbations are a reasonable proxy for worst-case perturbations in expectation.

## Limitations / Failure Modes

- Less principled than SAM's worst-case approach; may miss the sharpest directions.
- In PECL: same scope-misalignment issue as full-parameter SAM when applied to all parameters.

## Reusable Ingredients

- RS^(0) (Random Subspace zeroth-order perturbation) is used as a baseline in our experiments.

## Open Questions

## Claims

## Connections

<!-- AUTO-GENERATED -->
- **←extends** paper:foret2021_sam

## Relevance to This Project

RWP / RS^(0) is used as one of the experimental baselines. Our ranking: AS^(1) > AS^(0) > RS^(0) > SGD.
