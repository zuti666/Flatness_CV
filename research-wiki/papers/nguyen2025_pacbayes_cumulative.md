---
type: paper
node_id: paper:nguyen2025_pacbayes_cumulative
title: "PAC-Bayes Bounds for Cumulative Loss in Continual Learning"
authors: ["[Authors TBD — ICLR 2026 paper, details not yet confirmed]"]
year: 2025
venue: ICLR 2026
external_ids:
  arxiv: "2854 (internal ref: 12854_PAC_Bayes_bounds_for_cum)"
  doi: null
  s2: null
tags: [PAC-Bayes, continual-learning, theory, cumulative-loss]
relevance: related
origin_skill: research-wiki
created_at: 2026-04-24T00:00:00Z
updated_at: 2026-04-24T00:00:00Z
---

# One-line thesis

Derives PAC-Bayes bounds on cumulative loss in continual learning — a very recent (ICLR 2026) extension of classical hierarchical PAC-Bayes to the CL setting.

## Problem / Gap

Classical PAC-Bayes bounds do not account for cumulative loss across sequential tasks in CL.

## Method

[Details TBD — paper recently identified as a key differentiator in rebuttal (Hwkz-W2). Full structural analysis not yet complete.]

## Key Results

[TBD]

## Assumptions

[TBD]

## Limitations / Failure Modes

[TBD — key question: does it handle frozen backbone + LoRA subspace reduction?]

## Reusable Ingredients

[TBD]

## Open Questions

- EG-4 (partially resolved): How does our work differ structurally? Three identified deltas: (1) no M(M(H)) hierarchy, (2) plasticity-only bound vs. stability+plasticity, (3) no subspace reduction to adapter space.
- Full relationship needs user confirmation before citing in rebuttal.

## Claims

## Connections

<!-- AUTO-GENERATED -->
- **extends →** paper:pentina2014_pacbayes_cl

## Relevance to This Project

Identified by reviewer Hwkz as a missing citation. Rebuttal must differentiate from this paper on three structural dimensions identified in THEORY_COMPARISON.md. Currently flagged as partially resolved (EG-4 in ISSUE_BOARD.md).
