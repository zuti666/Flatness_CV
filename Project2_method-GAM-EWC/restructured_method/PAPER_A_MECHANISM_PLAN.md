# Paper A: Method Paper — Revised Plan
# "Orthogonal Gradient Subspaces for Stable-and-Flat Continual LoRA"
# Target: NeurIPS 2026 / ICLR 2027

Updated: 2026-04-27 (after mechanism_eval experiment)

---

## Core Story (3 sentences)

SeqLoRA forgets because the single shared adapter drifts away from old-task optima with no structural protection. We show that first-order adapter-subspace flatness (AS^1) and past-Fisher regularization in ΔW-space address **nearly orthogonal subspaces** of the gradient (cos ≈ −0.02, confirmed experimentally). This orthogonality is the geometric consequence of their definitions — not a coincidence — making the joint design a principled division of labor rather than a heuristic combination.

---

## Why This Beats "EWC-LoRA + C-Flat"

The killer differentiator is in the GEOMETRY, not the formula:

```
EWC-LoRA + C-Flat (naive):
  C-Flat perturbs in FULL PARAMETER SPACE → Fisher and flatness gradients PARTIALLY OVERLAP
  → Not orthogonal → signals interfere → suboptimal

Our method:
  AS^(1): perturbs in W_{Δ,t} (adapter admissible subspace)
  Fisher: regularizes in ΔW = BA space (basis-invariant, same subspace)
  → Both live in adapter geometry → BUT target different aspects:
     AS^(1) = current-task curvature (gradient norm directions)
     Fisher = old-task sensitivity directions
  → Near-orthogonal by the spectral structure of the loss Hessian vs. past Fisher
```

This is the missing proof that "our specific combination" is not just EWC-LoRA + C-Flat.

---

## Section Structure

### §1 Introduction (1.5p)
- Hook: SeqLoRA catastrophically forgets; two families of fixes (stability: Fisher; plasticity: flatness) have never been jointly principled
- Gap: Naive combinations (EWC-LoRA + C-Flat) ignore gradient geometry; we ask: do the two objectives conflict, complement, or are they orthogonal?
- Answer: Near-orthogonal (cos_flat_fisher ≈ 0) — quantified experimentally
- Contributions (3 bullets):
  1. First joint AS^1 + past-Fisher method designed from gradient geometry principles
  2. Empirical discovery of near-orthogonality; theoretical explanation via spectral separation
  3. NeurIPS-caliber experiments: ablation 2×2, mechanism stats, multi-dataset

### §2 Background (0.75p)
- SeqLoRA setting: single (A,B) pair, φ drifts across tasks
- AS^(1): GAM applied to adapter subspace (cite Paper B / ICML #2173 for theory)
- EWC in LoRA: correct formulation uses ΔW = BA space (cite EWC-LoRA ICLR 2026)
- The naive combination gap: C-Flat ≠ AS^(1)

### §3 Method: Flat-and-Stable SeqLoRA (1p)
- Objective: min L_task + λ_flat S^(1)_{Δ} + (λ_ewc/2)(φ − φ*)^T F^φ (φ − φ*)
- Lagrangian framing: constrained form, λ_ewc as dual variable
- Why ΔW = BA for Fisher (basis-invariant, not raw (A,B) factors)
- Implementation: gradient combination, Fisher accumulation with decay γ

### §4 The Orthogonality Principle (1p) ← CORE NEW SECTION
- Define cos_flat_fisher = cos(g_flat − g_clean, g_fisher)
- **Theorem 4.1** (informal): Under adapter-subspace perturbation and ΔW-space Fisher, the expected cosine between the flatness gradient component and Fisher gradient component is bounded near zero by the spectral gap between the current-task Hessian and past-task Fisher matrix.
- Consequence: the two terms are NOT fighting each other, even approximately
- Contrast with naive EWC + C-Flat: full-space perturbation means cos ≠ 0 in general
- Experimental confirmation: Table of cos_flat_fisher across tasks/datasets

### §5 Experiments (3p)
- 5.1 Ablation 2×2 (AvgAcc, Forgetting, BWT, Plasticity, Stability)
- 5.2 Mechanism stats: cos_flat_fisher, fisher_norm_ratio across 3+ datasets
- 5.3 Pareto comparison vs. EWC-LoRA, C-Flat, EWC-LoRA + C-Flat (baseline)
- 5.4 Robustness: ImageNet-C/P under joint method

### §6 Discussion (0.5p)
- When does orthogonality break? (large rank, unfreezing backbone)
- Extension to IncLoRA (structural isolation reduces Fisher need)
- Connection to Paper B theory (forgetting bound)

---

## Key Experiments Still Needed

| Experiment | Priority | Est. time |
|-----------|----------|-----------|
| Ablation 2×2 (SeqLoRA base / Flat-only / Fisher-only / Full) on CUB200 | HIGH | 1 day |
| Same ablation on ImageNet-R | HIGH | 1 day |
| Mechanism stats (cos_flat_fisher) on ImageNet-R, Flowers | MEDIUM | 0.5 day |
| EWC-LoRA + C-Flat naive baseline | HIGH | 0.5 day |
| Basis-invariance: Fisher on (A,B) vs. ΔW comparison | LOW | 0.5 day |

---

## Claim-Evidence Matrix (Paper A specific)

| Claim | Evidence | Status |
|-------|----------|--------|
| cos_flat_fisher ≈ 0 (near-orthogonal) | E003: −0.02 to −0.04 across 10 tasks | DONE |
| Fisher gradient ≪ flatness gradient | E003: norm ratio 0.014–0.286% | DONE |
| Full method > Flat-only | Ablation 2×2 | PENDING |
| Full method > Fisher-only | Ablation 2×2 | PENDING |
| Full method > EWC-LoRA + C-Flat naive | Pareto comparison | PENDING |
| cos_flat_fisher ≈ 0 across datasets | Multi-dataset mechanism | PENDING |

---

## Differentiation from ICML #2173 (Paper B overlap)

| Content | ICML #2173 (Paper B) | Paper A |
|---------|---------------------|---------|
| PAC-Bayes bound → W_{Δ,t} | ✓ (core) | Cite only |
| Forgetting-sharpness theorem | ✓ (Paper B deepens) | Cite only |
| AS^(1) definition | ✓ | Cite + use |
| Fisher on ΔW (basis-invariant) | ✓ (EWC-LoRA ICLR 2026) | Cite + use |
| **cos_flat_fisher ≈ 0 finding** | ✗ | ✓ NEW |
| **Orthogonality theorem** | ✗ | ✓ NEW |
| **Ablation 2×2** | ✗ | ✓ NEW |
| **Lagrangian framing as constrained opt** | ✗ | ✓ NEW |
| Robustness experiments | ✓ (scope/rank ablation) | Extend to joint method |
