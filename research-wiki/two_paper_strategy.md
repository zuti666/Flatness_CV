# Two-Paper Strategy: Sharpness × Forgetting × LoRA

**Date**: 2026-04-27  
**Status**: Active planning — ICML 2026 paper (#2173) under review; two follow-on papers planned

---

## Overview

The existing ICML 2026 paper spans both theory and method. Splitting into two clean papers allows each to go deeper, and avoids the "two-term addition" novelty objection on the method side.

| | Paper A (Method) | Paper B (Theory) |
|---|---|---|
| **Title direction** | Orthogonal Gradient Subspaces for Stable-and-Flat Continual LoRA | PAC-Bayesian Sharpness Under Constrained Update Geometry in CL |
| **Target venue** | NeurIPS 2026 / ICLR 2027 | NeurIPS 2026 |
| **Template** | `restructured_method/` | `restructured_theory/Nips2026/` |
| **Core claim** | AS^(1) and Fisher in ΔW space address near-orthogonal gradient subspaces → principled complementary design | Pathwise hierarchical PAC-Bayes: forgetting = past-task generalization degradation under later posteriors |
| **Main contribution** | Algorithm + gradient geometry analysis | Forgetting-sharpness connection theorem |
| **Experiments** | Ablation 2×2 table + mechanism eval (cos_flat_fisher) | Validation of theoretical predictions (scope, rank, architecture) |
| **Key differentiator from ICML #2173** | Method + orthogonality evidence (new) | Deeper forgetting-sharpness theory + pathwise decomposition (new) |

---

## Paper A: Method Paper

### Core Thesis
In PECL, AS^(1) and the past-Fisher penalty address **nearly orthogonal subspaces** of the gradient space — confirmed experimentally by `cos_flat_fisher ≈ −0.02`. This is not incidental: it is the geometric consequence of using adapter-subspace perturbation (AS^(1)) alongside ΔW-space Fisher regularization. The near-orthogonality is the mechanism claim that justifies the joint design as principled complementarity, not heuristic addition.

### Key Novelty vs Prior Work

| Prior | Gap they miss |
|-------|--------------|
| EWC-LoRA (ICLR 2026) | No flatness — ignores current-task geometry; Fisher gradient direction uncontrolled |
| C-Flat / FLAD | Flatness only — no Fisher; may drift along old-task sensitive directions |
| EWC-LoRA + C-Flat (naive) | C-Flat perturbs full space, not adapter subspace → Fisher and flatness gradients NOT orthogonal; overlapping signals |
| **Ours** | AS^(1) in W_{Δ,t} + Fisher in ΔW space → near-orthogonal by geometry → genuine division of labor |

### Story Arc

1. **Observation**: SeqLoRA forgets because φ drifts; existing solutions either fix direction (Fisher) or fix curvature (flatness) but not both.
2. **Key insight**: In adapter-subspace training, the flat gradient component (g_gam − g_clean) and the Fisher gradient (g_fisher) are nearly orthogonal (cos ≈ 0, empirically −0.02).
3. **Why**: AS^(1) lives in the directions of current-task loss curvature; Fisher lives in directions of old-task parameter sensitivity → geometrically separated.
4. **Method**: The near-orthogonality is a design principle: choose perturbation scope (adapter) and Fisher space (ΔW) to maximize directional separation.
5. **Lagrangian framing**: Rewrite as constrained optimization `min L_task + λ_flat S^(1)_{Δ}  s.t. D^Fisher ≤ δ` → λ_ewc is the dual variable estimate. EWC-LoRA + C-Flat is an approximate, non-basis-invariant version of this Lagrangian.
6. **Result**: NME AvgAcc = 84.66%, Forgetting = 1.14% on CUB200 10-task.

### Required Experiments

- [ ] **Ablation 2×2**: Base / Flat-only / Fisher-only / Full — for AvgAcc, Forgetting, BWT
- [ ] **Mechanism table**: cos_flat_fisher, fisher_norm_ratio across tasks (done for CUB200)
- [ ] **Mechanism across datasets**: Same stats on ImageNet-R, CIFAR-based, Flowers
- [ ] **Pareto frontier**: Must beat EWC-LoRA + C-Flat (naive baseline), not just EWC-LoRA alone
- [ ] **Basis-invariance check**: Compare Fisher on raw (A,B) factors vs. ΔW = BA space

### What Paper A does NOT claim
- It does not re-derive the PAC-Bayes bound (cites Paper B / ICML #2173)
- It does not claim cos_flat_fisher > 0 ("synergy") — the near-orthogonality IS the claim
- It does not extend to IncLoRA/OLoRA (SeqLoRA is the natural testbed; structural isolation in others reduces the need)

---

## Paper B: Theory Paper

### Core Thesis
SAM-style objectives are motivated by PAC-Bayes, which controls single-task generalization. But forgetting is **past-task generalization degradation under later posteriors** — a different quantity. We provide the first pathwise hierarchical PAC-Bayes decomposition that unifies both:

```
R_i(Q_t) ≤ R_i(Q_i)  +  Drift(Q_i → Q_t)
         ≤ [sharpness-smoothed empirical risk at task i]  +  [KL drift from i to t]
```

Specializing to frozen-backbone LoRA: both the sharpness term and KL reduce to W_{Δ,i} (support reduction theorem). This identifies **basis-invariant effective adapter-update sharpness** as the theoretically relevant object.

### Key Novelty vs Prior Work (vs ICML #2173)

| ICML #2173 | Paper B (deeper) |
|---|---|
| Theorem 4.4: bound reduces to W_{Δ,t} | Same, but: explains WHY sharpness controls forgetting (pathwise decomposition) |
| Time-varying hyperposterior (Theorem 4.1) | Full pathwise analysis: R_i(Q_t) decomposed into task-i sharpness + drift terms |
| No forgetting-sharpness theorem | **Explicit forgetting bound**: Forget(i→t) ≤ S_i^Δ · Γ(i,t) + KL_drift(i,t) |
| Experimental validation of predictions | Deeper: prediction derivation from first principles, not just empirical |

### Three-Theorem Structure

1. **Theorem B.1 (Pathwise decomposition)**: For any task sequence, R_i(Q_t) bounds in terms of sharpness-smoothed risk at task i plus cumulative KL drift Q_i → Q_t.
2. **Theorem B.2 (Support reduction)**: Under frozen-backbone LoRA, the sharpness and KL terms in Theorem B.1 reduce to W_{Δ,i}.
3. **Theorem B.3 (Forgetting bound)**: Forget(i→t) ≤ f(S_i^Δ, Γ(i,t), η_{i+1:t}) — connects task-i adapter-subspace sharpness, task-to-task Fisher overlap, and step sizes.

### What Paper B does NOT claim
- No new method (Paper A)
- Does not experimentally validate the Flat-and-Stable method (that's Paper A)
- Does not need to beat EWC-LoRA numerically — it IS the theory foundation

---

## Separation Principle: Core Points Must NOT Overlap

| Point | Paper A | Paper B |
|-------|---------|---------|
| PAC-Bayes bound reduction to W_{Δ,t} | Cite ICML #2173 / Paper B | ✓ Prove + deepen |
| Forgetting-sharpness connection theorem | Cite Paper B | ✓ Prove |
| AS^(1) + Fisher gradient orthogonality | ✓ New claim + evidence | Not in scope |
| Lagrangian reformulation | ✓ Algorithm framing | Not in scope |
| Ablation 2×2 + mechanism stats | ✓ Core experiments | Not in scope |
| Pathwise hierarchical decomposition | Cite Paper B | ✓ Core theory |
| Adapter-subspace support reduction | Cite Paper B | ✓ Prove |

**Rule**: If a claim appears as a theorem in Paper B, Paper A cites it and builds on it. If a claim is algorithmic + empirical, it belongs to Paper A.

---

## Timeline

| Milestone | Target |
|-----------|--------|
| ICML 2026 #2173 decision | ~May 2026 |
| Ablation 2×2 experiments (Paper A) | ~May 2026 |
| Paper B theory draft (NeurIPS 2026) | ~May 2026 (deadline ~May 30) |
| Paper A method draft (NeurIPS 2026 / ICLR 2027) | ~June 2026 |
