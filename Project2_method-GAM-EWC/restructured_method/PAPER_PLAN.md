# PAPER PLAN: Revisiting Sharpness in Low-rank Subspaces for Continual Learning

## One-Sentence Contribution
Under frozen-backbone PECL, the theoretically relevant notion of flatness is task-conditioned and supported on the admissible LoRA update geometry—not on the ambient parameter space.

## Venue & Format
- ICML 2026, 9-page main body limit
- Anonymous submission

## Section Plan (9 sections + appendix)

### §1 Introduction (1.5 pages)
- Hook: PECL confines adaptation to low-rank subspace; inherited full-space flatness is not automatically relevant
- Gap: No principled theory identifies which sharpness governs continual generalization under frozen backbone
- Key result teased early: both KL and sharpness terms reduce to W_{Δ,t} (Theorem 4.4)
- Figure 1: 3D loss landscape (adapter-only SAM vs SGD)
- Contributions: 3 bullet points

### §2 PECL as Constrained Adaptation Geometry (1 page)
- 2.1 Frozen-backbone decomposition: W_t = W_froz + ΔW_t, ΔW_t ∈ W_{Δ,t}
- 2.2 LoRA organizations as geometry evolution rules: SeqLoRA, IncLoRA, OLoRA
- 2.3 From LoRA coordinates to admissible subspace (front-load Appendix C bridge)

### §3 Which Flatness Matters in PECL? (0.5 pages)
- 3.1 Three flatness objects: ambient, trainable-coordinate, admissible-update
- 3.2 Why inherited full-space notion is not automatically relevant
- Central claim: relevant flatness is task-conditioned, solution-local, on W_{Δ,t}

### §4 Sequential Hierarchical PAC-Bayes for Continual Learning (1 page)
- 4.1 Why classical hierarchical PAC-Bayes insufficient (i.i.d. task assumption)
- 4.2 Theorem 4.1 (time-varying hyperposterior) + interpretation
- Figure: multitask vs CL hyperposterior drift

### §5 PECL Specialization: Support Reduction onto W_{Δ,t} (1 page)
- 5.1 Lemma 4.2: KL reduction (KL on full space = KL on W_{Δ,t})
- 5.2 Lemma 4.3: Subspace sharpness decomposition
- 5.3 Theorem 4.4 (PECL bound) + interpretation: both terms supported on W_{Δ,t}
- Figure: KL decomposition illustration

### §6 Testable Predictions (0.5 pages)
- P1: Admissible-subspace perturbations should outperform full-space
- P2: Scope-aligned perturbations improve when curvature concentrates in adapter subspace
- P3: First-order objective (AS^(1)) > zeroth-order (AS^(0)) in fixed adapter scope
- P4: Distinction weakens as rank increases or backbone partially unfreezes

### §7 Empirical Validation (3 pages)
- 7.1 Q1: Why perturbation scope matters (Table: scope effect, Figure: c_t and α_t evolution)
- 7.2 Q2: Perturbation type comparison (Table: main results CL+flatness metrics, Figure: 1D loss probe)
- 7.3 Q3: LoRA organization as structural prior (Figure: robustness curves across SeqLoRA/IncLoRA/OLoRA)
- 7.4 Boundary conditions: rank ablation, task length ablation
- 7.5 External validity: T5 language benchmarks, Llama-3.2

### §8 Discussion and Limitations (0.25 pages)
- What this paper does NOT claim
- Conditions the analysis depends on
- Implications for future method design

### §9 Conclusion (0.25 pages)

## Claims-Evidence Matrix

| Claim | Evidence | Section |
|-------|----------|---------|
| PECL adaptation confined to W_{Δ,t} | Definition + LoRA Jacobian bridge | §2 |
| Full-space flatness not automatically relevant | Conceptual argument | §3 |
| CL requires time-varying hyperposterior | Theorem 4.1 | §4 |
| KL reduces to W_{Δ,t} | Lemma 4.2 + proof | §5 |
| Sharpness term reduces to W_{Δ,t} | Lemma 4.3 + Theorem 4.4 | §5 |
| Adapter-only perturbation consistently helps | Table (scope effect) | §7.1 |
| Full-space perturbation hurts | Table (scope effect) | §7.1 |
| c_t and α_t explain scope effect | Figure (evolution) | §7.1 |
| AS^(1) > AS^(0) > RS^(0) > SGD | Table (main results) | §7.2 |
| Subspace sharpness predicts CL performance | Table (curvature metrics) | §7.2 |
| Effect holds across LoRA organizations | Robustness figure | §7.3 |
| Effect weakens at larger rank | Ablation figure | §7.4 |
| Consistent across T5/Llama | Additional tables | §7.5 |

## Figure Plan
- Fig 1: 3D loss landscape (adapter-only SAM vs SGD) [existing: 2D_lossLandscape...]
- Fig 2: Multi-task vs CL hyperposterior [existing: multitask5.pdf]
- Fig 3: KL decomposition illustration [existing: KLshiyi1.pdf]
- Fig 4: c_t, α_t evolution + perturbation-aware loss [existing: weight_overlap_5fig.pdf]
- Fig 5: 1D loss probe [existing: lossland_curv1d_combine_loss_title2.pdf]
- Fig 6: Robustness curves ImageNet-C/P [existing: ACC_dataset_pair...]
- Table 1: Main results (CL + flatness metrics)
- Table 2: Scope effect
- Fig 7: Ablation (rank, task length) [existing: ablation_rank_length...]
