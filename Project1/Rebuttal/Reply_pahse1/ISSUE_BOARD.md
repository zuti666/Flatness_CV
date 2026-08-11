# Issue Board

**Paper**: Revisiting Sharpness in Low-rank Subspaces for Continual Learning (ICML 2026 #2173)

---

## TSrA Issues

### TSrA-C1
- **issue_id**: TSrA-C1
- **reviewer**: TSrA
- **raw_anchor**: "I am not convinced that practitioners would use full-parameter perturbation when applying SAM to LoRA… frozen parameters typically do not produce gradients."
- **issue_type**: practical_significance
- **severity**: critical (main reason for rejection)
- **reviewer_stance**: negative
- **response_mode**: direct_clarification + grounded_evidence
- **status**: answerable_from_paper
- **answer**: TSrA confuses the paper's argument direction. Standard SAM applied to a LoRA model naturally restricts to trainable parameters (no gradient → no perturbation). But **Flat-LoRA (ICML 2025)** explicitly CONSTRUCTS a mechanism to perturb frozen backbone parameters, arguing this is necessary for meaningful flatness. This is the practice our paper evaluates. The two scopes in Table 2 are: (i) adapter-only perturbation [what TSrA calls "natural"] vs. (ii) full-parameter perturbation as proposed by Flat-LoRA. Our theory + experiments show (i) is both sufficient AND better. This is a non-obvious finding that directly contradicts Flat-LoRA's published recommendation.
- **evidence**: Table 2 (Scope effect): LoRA-scope consistently improves (+0.05 to +0.86 ΔAcc^cls), full-scope consistently degrades (−2.00 to −6.84 ΔAcc^cls) across all three LoRA variants. Paper text p.7: "full-scope perturbations introduce variations in the frozen model outside the permissible update subspace, creating an irreducible scope mismatch that accelerates performance degradation."

### TSrA-C2
- **issue_id**: TSrA-C2
- **reviewer**: TSrA
- **raw_anchor**: "The PAC-Bayes bound largely builds on prior work, and I did not see major technical challenges in the derivation."
- **issue_type**: novelty / theorem_rigor
- **severity**: major
- **reviewer_stance**: negative
- **response_mode**: grounded_evidence + assumption_hierarchy
- **status**: open
- **notes**: Need to articulate specific novel technical contributions: sequentially evolving hyperposterior, explicit drift penalty, and the frozen-backbone reduction that makes both KL and sharpness terms collapse to adapter subspace.

### TSrA-C3
- **issue_id**: TSrA-C3
- **reviewer**: TSrA
- **raw_anchor**: "The paper does not introduce a new method, but mainly validates the effectiveness of combining LoRA with different variants of SAM."
- **issue_type**: novelty
- **severity**: major
- **reviewer_stance**: negative
- **response_mode**: direct_clarification
- **status**: open
- **notes**: Frame paper as theoretical + diagnostic contribution (not a new method paper). The value is resolving a contested question with principled theory + experiments.

### TSrA-C4
- **issue_id**: TSrA-C4
- **reviewer**: TSrA
- **raw_anchor**: "Experiments are conducted only in a single setting (ViT on ImageNet), and the scope and scale of evaluation are relatively limited."
- **issue_type**: empirical_support
- **severity**: major
- **reviewer_stance**: negative
- **response_mode**: grounded_evidence / future_work_boundary
- **status**: open (needs user input — can we add experiments?)

---

## ZRjU Issues

### ZRjU-Q1
- **issue_id**: ZRjU-Q1
- **reviewer**: ZRjU
- **raw_anchor**: "The classical bound in Theorem 4.1 contains a logm·logm term in the numerator inside the square root, which is absent in Theorem 4.4."
- **issue_type**: theorem_rigor
- **severity**: major
- **reviewer_stance**: positive
- **response_mode**: assumption_hierarchy + direct_clarification
- **status**: answerable_from_paper
- **answer**: The reviewer's "Theorem 4.1" = paper's Theorem 3.1 (classical PAC-Bayes) and "Theorem 4.4" = paper's Theorem 4.1 (sequential). Theorem 3.1 has `log(m/δ)` which contains `log m`. Theorem 4.1 has `log(1/δ)` only.
  - **Why the term disappears**: The two theorems use fundamentally different proof techniques. Theorem 3.1 uses McAllester's classical bound derivation, which inverts the KL-inequality and produces `log(m/δ)` as a consequence of the specific inversion technique. Theorem 4.1 uses a **supermartingale tail bound** (Markov's inequality applied to a nonneg supermartingale) combined with Donsker-Varadhan's variational KL representation at both the model level (Step 1) and hyper level (Step 2) of the proof. The Hoeffding-MGF bound in Step 1 gives `(λω_t)² / (8(m_t-1))` per task; optimizing over λ then gives `log(1/δ)` in the numerator without any `log m` factor. The supermartingale approach doesn't require union-bounding over individual samples, so `log m` never enters.
  - The `log m` is a proof artifact of the classical technique, not a fundamental bound property. The sequential martingale proof is actually tighter in this regard.
  - Proof steps: see lines 2205-2374 in the LaTeX source (Appendix B.1).

### ZRjU-Q2
- **issue_id**: ZRjU-Q2
- **reviewer**: ZRjU
- **raw_anchor**: "Lemma 4.2 assumes Qt△ ≪ Pt△ and implicitly requires prior supported on same subspace W△,t. For IncLoRA and OLoRA, where subspaces grow or are constrained, is this assumption satisfied?"
- **issue_type**: assumptions / theorem_rigor
- **severity**: major
- **reviewer_stance**: positive
- **response_mode**: assumption_hierarchy + direct_clarification
- **status**: answerable_from_paper
- **answer**: The key clarification is the scope of W_{Δ,t}. In IncLoRA, at each task t, ONLY the current task's new adapter pair (A_t, B_t) is trainable — all previous adapter pairs (A_1,B_1,...,A_{t-1},B_{t-1}) are frozen along with the backbone. Therefore, W_{Δ,t} at task t is the fixed-dimensional subspace induced by the **current task's adapter pair only**, not the entire accumulated adapter space (which would grow). Both P_t^Δ and Q_t^Δ are Gaussian distributions on this same fixed-dimensional W_{Δ,t}, so Q_t^Δ ≪ P_t^Δ is satisfied. The "growing" nature concerns how TOTAL adapter space grows across tasks (an optimizer-level phenomenon), but within each task the PAC-Bayes analysis is applied to a well-defined, fixed-dimensional W_{Δ,t}.
  - For OLoRA: same argument applies. The orthogonality constraint is an optimization penalty applied during task t training; it shapes the geometry of W_{Δ,t} relative to previous tasks but does not change the fact that both P_t^Δ and Q_t^Δ are supported on the same current W_{Δ,t}.
  - The KL decomposition KL(Q_t||P_t) = KL(Q_t^Δ||P_t^Δ) holds as long as the frozen parameters (backbone + previous adapters) contribute a degenerate (zero-variance) factor to both Q_t and P_t — which is exactly what "frozen" means. Lemma 4.2 formalizes this via translation invariance of KL under the fixed W_t^{froz} shift.

### ZRjU-Q3
- **issue_id**: ZRjU-Q3
- **reviewer**: ZRjU
- **raw_anchor**: "PAC-Bayes bounds are often too loose. Are the authors able to evaluate how tight Theorem 4.4 is?"
- **issue_type**: empirical_support
- **severity**: minor
- **reviewer_stance**: positive
- **response_mode**: narrow_concession + future_work_boundary
- **status**: open
- **notes**: Standard answer: bound is intended as conceptual motivation / qualitative direction indicator, not a numerically tight bound.

### ZRjU-Q4
- **issue_id**: ZRjU-Q4
- **reviewer**: ZRjU
- **raw_anchor**: "Figure 4 arrives late and is too technical to serve as intuitive illustration of RS(0), AS(0), AS(1)."
- **issue_type**: clarity
- **severity**: minor
- **reviewer_stance**: positive
- **response_mode**: direct_clarification
- **status**: open
- **notes**: Can commit to adding an early schematic figure in revision. Check if user can provide one.

---

## mBoF Issues

### mBoF-W1
- **issue_id**: mBoF-W1
- **reviewer**: mBoF
- **raw_anchor**: "Only ViT-B/16 pretrained on ImageNet-21K; continual learning datasets limited to ImageNet-C/R/P."
- **issue_type**: empirical_support
- **severity**: major
- **reviewer_stance**: swing
- **response_mode**: grounded_evidence / future_work_boundary
- **status**: open (needs user input — any additional architecture or dataset results available?)

### mBoF-W2
- **issue_id**: mBoF-W2
- **reviewer**: mBoF
- **raw_anchor**: "Does this analysis hold regardless of model complexity or model size?"
- **issue_type**: practical_significance
- **severity**: minor
- **reviewer_stance**: swing
- **response_mode**: future_work_boundary + direct_clarification
- **status**: open

### mBoF-W3
- **issue_id**: mBoF-W3
- **reviewer**: mBoF
- **raw_anchor**: "Observations appear somewhat incremental (just very similar to full fine-tuning)."
- **issue_type**: novelty
- **severity**: minor
- **reviewer_stance**: swing
- **response_mode**: direct_clarification
- **status**: open
- **notes**: PECL with frozen backbone is fundamentally different from full fine-tuning. The theoretical machinery must be rebuilt from scratch.

### mBoF-Q1
- **issue_id**: mBoF-Q1
- **reviewer**: mBoF
- **raw_anchor**: "FlatLoRA highlights that LoRA fine-tuning leads to sharper landscapes in the full parameter space. Do the authors think their approach would influence the generalization bound derived in this work?"
- **issue_type**: baseline_comparison
- **severity**: minor
- **reviewer_stance**: swing
- **response_mode**: nearest_work_delta
- **status**: open

---

## Hwkz Issues

### Hwkz-W1
- **issue_id**: Hwkz-W1
- **reviewer**: Hwkz
- **raw_anchor**: "Applying AS(1) perturbation only to the LoRA adapter appears to be the standard way of using SAM… this practice does not seem to introduce the claimed perturbation misalignment issue."
- **issue_type**: practical_significance / novelty
- **severity**: critical
- **reviewer_stance**: swing
- **response_mode**: direct_clarification + grounded_evidence
- **status**: answerable_from_paper
- **notes**: Same core misunderstanding as TSrA-C1. Additional clarification: the reviewer says adapter-only SAM "is already the most effective" — this is exactly what the paper shows, but it's not obvious, because Flat-LoRA (ICML 2025), a peer-reviewed ICML paper, explicitly argues the OPPOSITE. Our contribution is resolving this active debate with theory and systematic experiments.
- **answer**: (See TSrA-C1 answer.) Additionally: Hwkz says empirical results show this default strategy is already the most effective — agreed, and this IS the paper's finding. But the point is that this finding (a) was not known before; (b) is non-obvious given Flat-LoRA's explicit argument to the contrary; and (c) now has theoretical backing from Theorems 4.1 and 4.4 which prove WHY adapter-only is sufficient and full-parameter is misaligned. The contribution is theory + systematic evidence resolving an open question, not proposing a new method.

### Hwkz-W2
- **issue_id**: Hwkz-W2
- **reviewer**: Hwkz
- **raw_anchor**: "The paper should discuss prior work that applies PAC-Bayesian analysis to continual learning, such as [1] A PAC-Bayesian bound for Lifelong Learning (ICML 2014) and [2] PAC-Bayes bounds for cumulative loss in CL (ICLR 2026)."
- **issue_type**: clarity / novelty
- **severity**: major
- **reviewer_stance**: swing
- **response_mode**: nearest_work_delta
- **status**: open
- **notes**: Need to differentiate from these two works. ICLR 2026 paper is very recent — need user to confirm relationship.

### Hwkz-W3
- **issue_id**: Hwkz-W3
- **reviewer**: Hwkz
- **raw_anchor**: "Additional PEFT approaches should be included to better validate the effectiveness of the proposed subspace-restricted updates."
- **issue_type**: empirical_support
- **severity**: major
- **reviewer_stance**: swing
- **response_mode**: grounded_evidence / future_work_boundary
- **status**: open (needs user input — any other PEFT methods tested?)

### Hwkz-W4
- **issue_id**: Hwkz-W4
- **reviewer**: Hwkz
- **raw_anchor**: "Some notations are used before being defined. For example, H in Line 101 (Page 2) is not formally introduced."
- **issue_type**: clarity
- **severity**: minor
- **reviewer_stance**: swing
- **response_mode**: direct_clarification
- **status**: open
- **notes**: Easy fix — confirm what H denotes and commit to fix in revision.

---

## Issue Count Summary

| Reviewer | Critical | Major | Minor | Total |
|----------|----------|-------|-------|-------|
| TSrA     | 1        | 3     | 0     | 4     |
| ZRjU     | 0        | 2     | 2     | 4     |
| mBoF     | 0        | 1     | 3     | 4     |
| Hwkz     | 1        | 2     | 1     | 4     |
| **Total**| **2**    | **8** | **6** | **16**|
