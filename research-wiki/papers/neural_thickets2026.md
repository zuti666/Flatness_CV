---
type: paper
node_id: paper:neural_thickets2026
title: "Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights"
authors: ["[Authors TBD — 2026]"]
year: 2026
venue: arXiv 2026
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: [pretrained-models, LoRA, task-experts, continual-learning, adapter, landscape]
relevance: related
origin_skill: research-wiki
created_at: 2026-04-24T00:00:00Z
updated_at: 2026-04-24T00:00:00Z
---

# One-line thesis

Diverse task-specific experts (adapter solutions) are densely clustered around pretrained weights in parameter space, supporting the view that the pretrained weight neighborhood is geometrically rich for adaptation.

## Problem / Gap

How are task-specific LoRA solutions distributed in parameter space relative to the pretrained initialization?

## Method

[Details TBD — from RefInsight folder, categorized as providing geometric insight about the adapter landscape.]

## Key Results

- Task-specific solutions concentrate near pretrained weights, validating the PECL assumption that the adapter subspace captures task-relevant geometry.

## Assumptions

[TBD]

## Limitations / Failure Modes

[TBD]

## Reusable Ingredients

- Geometric intuition supporting the claim that adapter-subspace perturbations are meaningful and sufficient.
- Provides empirical grounding for why W_{Δ,t} captures the relevant geometry.

## Open Questions

## Claims

## Connections

<!-- AUTO-GENERATED -->
- **addresses_gap →** gap:G4

## Relevance to This Project

Provides supporting evidence that the adapter subspace is geometrically meaningful — consistent with our theoretical finding that flatness on W_{Δ,t} is the relevant notion for PECL generalization.
