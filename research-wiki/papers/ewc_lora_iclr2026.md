---
type: paper
node_id: paper:ewc_lora_iclr2026
title: "Revisiting Weight Regularization for Low-Rank Continual Learning"
authors: ["[TBD — ICLR 2026]"]
year: 2026
venue: ICLR 2026
external_ids:
  arxiv: "2602.17559"
  doi: null
  s2: null
tags: [EWC, Fisher, LoRA, continual-learning, weight-regularization, stability-plasticity, SeqLoRA]
relevance: core
origin_skill: research-wiki
created_at: 2026-04-24T00:00:00Z
updated_at: 2026-04-24T00:00:00Z
---

# One-line thesis

EWC-LoRA applies EWC-style Fisher regularization to a shared LoRA module using the full-dimensional FIM, achieving a better stability-plasticity tradeoff than structural LoRA CL methods with constant memory footprint.

## Problem / Gap

Naïve integration of EWC with LoRA (applying Fisher to raw A,B factors separately) is suboptimal. How to correctly estimate parameter importance in the low-rank space?

## Method

- Single shared LoRA pair (A,B) — same SeqLoRA setting as idea:002.
- Fisher estimated in full-dimensional weight space (ΔW = BA), then projected to LoRA parameter space φ for regularization.
- EWC penalty: (φ - φ*_{t-1})^T · F^φ_{1:t-1} · (φ - φ*_{t-1})
- NO flatness optimization component.

## Key Results

- EWC-LoRA outperforms vanilla LoRA by +8.92% average.
- Matches or exceeds InfLoRA, SD-LoRA on most benchmarks.
- Better stability-plasticity tradeoff than structural isolation methods.

## Assumptions

- Fisher can be reliably estimated from full-dimensional FIM projected to LoRA space.
- Shared single LoRA pair across all tasks.

## Limitations / Failure Modes

- **No flatness component**: does not optimize current-task generalization geometry.
- Fisher penalty provides direction control but not local curvature control → may not maximize plasticity.
- Fisher accumulation over many tasks may become noisy or saturated.

## Reusable Ingredients

- Full-dimensional FIM estimation for LoRA parameters — correct way to compute Fisher for LoRA factors.
- Shared SeqLoRA + EWC as a strong baseline for idea:002.

## Open Questions

- What happens when AS^(1) flatness is added on top of EWC-LoRA? (= idea:002)

## Claims

## Connections

<!-- AUTO-GENERATED -->
- **←competing_with** idea:002

## Relevance to This Project

**Primary competitor for idea:002**. Any new method must clearly outperform EWC-LoRA on all metrics. The delta of idea:002 is AS^(1) flatness as an explicit plasticity component — EWC-LoRA has stability but no flatness-controlled plasticity.
