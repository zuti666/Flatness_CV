---
type: paper
node_id: paper:foret2021_sam
title: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
authors: ["Pierre Foret", "Ariel Kleiner", "Hossein Mobahi", "Behnam Neyshabur"]
year: 2021
venue: ICLR 2021
external_ids:
  arxiv: "2010.01412"
  doi: null
  s2: null
tags: [sharpness, flatness, optimizer, generalization, SAM]
relevance: core
origin_skill: research-wiki
created_at: 2026-04-24T00:00:00Z
updated_at: 2026-04-24T00:00:00Z
---

# One-line thesis

SAM finds parameters lying in flat loss neighborhoods via a minimax perturbation step, improving generalization over SGD.

## Problem / Gap

SGD can converge to sharp minima that generalize poorly; no efficient optimizer explicitly targets flat loss basins during training.

## Method

At each step, perturb parameters by the worst-case ε-ball perturbation (gradient ascent step), then compute gradient at the perturbed point and update original parameters. Default perturbs ALL trainable parameters. Computational cost: 2× forward-backward passes per step.

## Key Results

- Consistent generalization improvement on CIFAR-10/100 and ImageNet over SGD and Adam baselines.
- Achieves SOTA on several benchmarks when combined with data augmentation.

## Assumptions

- Full parameter space is available for perturbation.
- Loss landscape is smooth enough for gradient-based perturbation to find meaningful worst-case neighbors.

## Limitations / Failure Modes

- 2× compute cost per step.
- Default implementation perturbs ALL parameters — including frozen ones when applied naively to LoRA models, causing "perturbation misalignment" in PECL settings.
- In PECL with frozen backbone, full-parameter perturbation contradicts the frozen constraint and empirically degrades continual performance.

## Reusable Ingredients

- The ε-ball worst-case perturbation framework can be restricted to any subspace (e.g., adapter parameters only).
- First-order variant (m-SAM / AS^(1)) uses gradient norm to proxy curvature without the full ascent step.

## Open Questions

- What is the right perturbation scope when only a subspace of parameters is trainable?

## Claims

[claim:C2 — adapter-only perturbation (AS^1) consistently outperforms full-parameter SAM in frozen-backbone PECL]

## Connections

<!-- AUTO-GENERATED -->
- **extends →** paper:li2024_flatlora (Flat-LoRA adapts SAM to LoRA fine-tuning)
- **extends →** paper:bisla2022_rwp
- **extends →** paper:zhang2023_gam
- **←inspired_by** idea:001

## Relevance to This Project

SAM is the foundational method. Our project shows that naively applying SAM's full-parameter perturbation to PECL is harmful and theoretically unjustified — the relevant perturbation scope is the adapter subspace W_{Δ,t}.
