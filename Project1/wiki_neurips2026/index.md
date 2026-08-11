# Research Wiki — NeurIPS 2026 Theory Paper (Paper B)

**Paper**: Understanding Sharpness and Forgetting under Constrained Update Geometry in Continual Learning  
**Venue**: NeurIPS 2026 (target, deadline ~May 30 2026)  
**Template**: `Project1/NewPpaerVerisonWrite/restructured_theory/Nips2026/neurips_2026_4 (1).tex`  
**Relation to ICML #2173**: This is the deepened theory version. ICML #2173 contains Theorem 4.4 (support reduction); this paper adds the full pathwise forgetting decomposition and three-theorem structure.  
**Last updated**: 2026-04-30

---

## Paper Summary

Two research questions drive the paper:

**Q1**: How can sharpness-based generalization be connected to *trajectory-level* CL performance (forgetting) within a unified PAC-Bayesian view?

**Q2**: Under LoRA-constrained adaptation, what is the correct geometric support of the perturbation-smoothed risk that enters the trajectory-level PAC-Bayesian bound?

**Answer to Q1**: Forgetting = past-task generalization degradation under later posteriors. A sequential hierarchical PAC-Bayes decomposition shows perturbation-smoothed empirical risk (the SAM objective) accumulates into trajectory-level risk bounds.

**Answer to Q2**: Under frozen-backbone LoRA, both the sharpness term and KL complexity reduce to the admissible effective adapter-update geometry W_{Δ,t} (basis-invariant). Not full-space sharpness, not raw factor-space sharpness.

---

## Core Contributions

1. **Pathwise hierarchical PAC-Bayesian decomposition**: Trajectory risk decomposed into perturbation-smoothed empirical risk + within-task adaptation complexity + hyperposterior drift complexity.  
2. **PECL support-reduction theorem**: Under frozen-backbone LoRA, the bound-relevant sharpness and KL complexity are supported on the admissible adapter-update geometry.

---

## Three-Theorem Structure (Paper B)

| Theorem | Name | Content |
|---------|------|---------|
| B.1 | Pathwise decomposition | `R_i(Q_t) ≤ [task-i sharpness-smoothed risk] + [cumulative KL drift Q_i → Q_t]` |
| B.2 | Support reduction | Under frozen-backbone LoRA, sharpness and KL in B.1 reduce to W_{Δ,i} |
| B.3 | Forgetting bound | `Forget(i→t) ≤ f(S_i^Δ, Γ(i,t), η_{i+1:t})` connecting adapter-subspace sharpness, Fisher overlap, step sizes |

---

## Claims

| ID | Claim | Status |
|----|-------|--------|
| [C1](claims/C1.md) | Forgetting = past-task generalization degradation under later posteriors (framing) | theory |
| [C2](claims/C2.md) | Perturbation-smoothed risk at task i bounds R_i(Q_t) for t > i (Thm B.1) | theory |
| [C3](claims/C3.md) | Under frozen-backbone LoRA, bound reduces to W_{Δ,t} (Thm B.2 = ICML Thm 4.4 deepened) | theory |
| [C4](claims/C4.md) | SAM direction matters; Gaussian random direction does not reproduce the effect | empirical — Exp E supported |
| [C5](claims/C5.md) | SAM effect is trajectory-level, not merely task-local | empirical — Exp F supported |
| [C6](claims/C6.md) | Raw factor-space sharpness is parameterization-dependent; Sh_Delta is invariant | empirical — rescaling control pending |

---

## Experiments

| ID | Name | Dataset | Status | Key result |
|----|------|---------|--------|------------|
| [Exp C](experiments/exp_C.md) | Task-conditioned support pilot | CIFAR10 2-task | complete (pilot) | sam_factor best BWT; sam_random best FAA |
| [Exp D](experiments/exp_D.md) | Two-task trajectory factorial | CIFAR10 2-task | complete (pilot) | SAM at Task 2 improves old-task retention more than SAM at Task 1 |
| [Exp E](experiments/exp_E.md) | Support × Direction main | ImageNet-R r16 T20 | **complete** | sam_factor: FAA 67.94, BWT -6.81 vs SGD: 58.29, -18.99. SAM > random for every support |
| [Exp F](experiments/exp_F.md) | Forked trajectory main | ImageNet-R r16 T20 | **complete** | Same checkpoint fork: suffix SAM-factor >> suffix SGD; random-factor ≈ SGD |
| Exp G | Raw-factor rescaling control (post-hoc) | ImageNet-R | **pending** | Sh_AB should vary with scale c; Sh_Delta stable |
| [Exp Q1](experiments/exp_Q1_sharpness_forgetting_corr.md) | Per-task sharpness × forgetting correlation | ImageNet-R T20 (Exp E checkpoints) | **pending — no new training** | Sh_delta(θ_i) vs F_i scatter; directly answers Q1 |
| [Exp Q2](experiments/exp_Q2_reparameterization_invariance.md) | Reparameterization invariance training test | ImageNet-R T20 (Exp F task-9 as start) | **pending — 12 runs** | sam_delta FAA stable across c; sam_factor varies; directly answers Q2 |
| Exp H | Multi-seed robustness | ImageNet-R r16 T20 | pending | Seeds: 1993, 42, 1234 |
| Exp I | Post-hoc sharpness metrics | ImageNet-R | pending | Connect performance to Sh_param, Sh_AB, Sh_Delta |
| Exp J | Cross-LoRA structure | ImageNet-R r16 T20 | pending (after SeqLoRA story stable) | SeqLoRA/IncLoRA/OLoRA comparison |

**Strongest current evidence**: Exp F forked trajectory — same-checkpoint fork, suffix SAM-factor reduces forgetting while random-factor does not.

**Strongest negative control**: Exp E — SAM direction beats Gaussian random for every support.

---

## Open Gaps

| ID | Description | Status |
|----|-------------|--------|
| [G1](gaps/G1.md) | Sh_AB is parameterization-dependent; Sh_Delta/W_tangent is invariant — requires rescaling control | open |
| [G2](gaps/G2.md) | Multi-seed robustness for Exp E/F results | open |
| [G3](gaps/G3.md) | Post-hoc sharpness metrics connecting performance to measured curvature | open |
| [G4](gaps/G4.md) | Cross-LoRA structure validation (IncLoRA, OLoRA) | deferred |
| [G5](gaps/G5.md) | Non-vacuous (tight) PAC-Bayes bounds for PECL | acknowledged limitation |
| [G6](gaps/G6.md) | Extension to LLMs / billion-scale models | acknowledged limitation |

---

## Suggested Empirical Section Structure

```
6. Empirical Analysis of Bound-Relevant Flatness

  6.1 Setup
      ImageNet-R r=16 T=20, SeqLoRA, ViT-B/16, no replay, seed 1993.

  6.2 RQ1: Does sharpness-aware optimization affect posterior trajectories?
      Exp F forked trajectory.

  6.3 RQ2: Is the effect due to adversarial direction rather than random noise?
      Exp E SAM-vs-random support-direction comparison.

  6.4 RQ3: Which perturbation support is empirically most effective?
      Exp E support comparison: factor / full / delta / all / frozen.

  6.5 RQ4: Is raw factor-space sharpness parameterization-dependent?
      Exp G rescaling control. [pending]

  Appendix: CIFAR10 two-task diagnostics (Exp C, Exp D).
```

---

## Next Steps (priority order)

1. **Exp Q1** (Q1 direct, no new training): Post-hoc Sh_delta(θ_i) measurement on Exp E checkpoints at tasks {0,4,9,14,19}. Scatter Sh_delta vs F_i. Answers Q1 directly (Theorem B.1).
2. **Exp Q2** (Q2 direct, 12 runs): Reparameterization (A/c, cB) from Exp F task-9 checkpoint. Compare sam_factor vs sam_delta across c ∈ {0.5,1,2,4}. sam_delta should be stable, sam_factor should vary. Answers Q2 directly (Theorem B.2) and resolves sam_factor > sam_delta tension.
3. **Exp G**: Post-hoc Sh_AB vs Sh_delta at different c values (simpler than Exp Q2; measurement only, no new training). Required to prove measurement invariance.
4. **Exp H**: Multi-seed (1993, 42, 1234) for Exp E key variants.  
5. **Theorem writing**: Complete three-theorem proof structure (B.1 pathwise, B.2 support reduction, B.3 forgetting bound).

---

## Relation to Other Papers

| Paper | Relation |
|-------|---------|
| ICML 2026 #2173 | Parent paper; Theorem 4.4 (support reduction) already proved. This paper deepens with pathwise decomposition + forgetting theorem. |
| Paper A (Method) | Method paper using Flat-and-Stable SeqLoRA + AS^(1) + Fisher orthogonality. Paper B is the theory foundation it cites. |
| Pentina & Lampert 2014 | Prior lifelong PAC-Bayes; order-invariant (i.i.d. tasks), does not capture forgetting as trajectory effect. |
| Friedman et al. 2026 | Concurrent sequential PAC-Bayes CL; cumulative risk bounds with data-dependent priors but does not address flatness geometry or PECL support. |
| Flat-LoRA (Li et al. 2024) | Primary foil on the LoRA side; addresses raw-factor vs merged-weight mismatch but not CL forgetting. |

---

## Quick Navigation

- [Claims](claims/) — C1–C6
- [Experiments](experiments/) — Exp C–J
- [Gaps](gaps/) — G1–G6
- [Ideas](ideas/) — open directions
- [log.md](log.md) — append-only mutation log
- [Experiment detail source](../experiment_summary_flatness_pecl_2026-04-30.md)
- [Paper template](../NewPpaerVerisonWrite/restructured_theory/Nips2026/)
