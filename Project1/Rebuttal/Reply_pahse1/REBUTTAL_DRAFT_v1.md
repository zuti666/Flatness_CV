# Rebuttal Draft v1

**Paper**: Revisiting Sharpness in Low-rank Subspaces for Continual Learning (ICML 2026 #2173)
**Character count (PASTE_READY.txt)**: see separate file
**Status**: Awaiting EG-4 (ICLR 2026 paper differentiation) and EG-6 (character limit) before finalizing

---

## Structured Response (5 Groups)

---

### Group 1: Motivation — Perturbation Scope in PECL
*Addresses: TSrA W1, Hwkz W1, mBoF Q1*

**Core argument**: Both reviewers view adapter-only SAM as the natural default. Our point is that the literature still leaves room for a real disagreement once the goal is full-model flatness under LoRA.

**Response text**:

TSrA raises that "frozen parameters typically do not produce gradients," and Hwkz states that "adapter-only SAM appears to be the standard way." We agree with this observation: when SAM is directly applied to a LoRA model with a frozen backbone, perturbing only the trainable LoRA parameters is indeed the most immediate implementation choice. However, this does not fully settle the question of how SAM should be used with LoRA.

The reviewer-cited **Flat-LoRA (ICML 2025, Li et al.)** takes a different position. It argues that perturbing only the LoRA parameters may be insufficient to guarantee flatness of the full model, and therefore motivates perturbations over the **full parameter space including the frozen backbone**. This creates a concrete difference in the literature:

- **LoRA-SAM (NeurIPS 2024)**: adapter-only perturbation is effective
- **Flat-LoRA (ICML 2025)**: broader perturbation is needed to ensure full-model flatness

This distinction becomes especially important in PECL. Flat-LoRA is developed for general LoRA fine-tuning, whereas PECL is built on the principle that the backbone remains fixed across tasks. Perturbing the full model therefore conflicts with the design of PECL, because it introduces perturbations in frozen directions that are outside the permissible update subspace. For this reason, the question "how should SAM be applied in LoRA-based continual learning?" is not trivial; it is precisely the starting point of our paper. We further study not only perturbation scope, but also different perturbation mechanisms within the subspace and their behavior across SeqLoRA, IncLoRA, and OLoRA.

For the PECL setting, our results support a clear conclusion. Figure 1 and Table 2 directly compare these two scopes. Table 2 shows: full-scope consistently **degrades** accuracy (ΔAcc^cls = −2.00 to −6.84 across all LoRA variants), while adapter-scope consistently **improves** it (ΔAcc^cls = +0.05 to +0.86). Theorem 4.4 explains why: under frozen backbone, full-parameter perturbation introduces noise in frozen directions that LoRA updates cannot counteract — an "irreducible scope mismatch" (Section 5.1 of the paper).

**Re mBoF Q1 (FlatLoRA's effect on our bound)**: FlatLoRA's perturbation is defined over the full parameter space W, not W_{Δ,t}. Our Theorem 4.4 shows that for frozen-backbone PECL, the bound-relevant sharpness is exclusively within W_{Δ,t}. Extending perturbations beyond W_{Δ,t} adds terms that the bound does not control — which is exactly why Table 2 shows consistent degradation from full-scope perturbation.

**Why "adapter-only is better" is non-obvious**: The finding directly contradicts Flat-LoRA's published recommendation. Practitioners who follow Flat-LoRA would make the wrong choice in PECL. Our paper provides both the theory (Theorem 4.4) and empirical evidence (Table 2, Figures 5–7) to guide the correct design decision.

---

### Group 2: Theoretical Novelty
*Addresses: TSrA W2/W3, ZRjU Q1/Q2, Hwkz W2*

**Core argument**: Three specific novelties not present in any prior work. The paper is a theory + analysis contribution, not a method paper — both are valid at ICML.

**Response text** (vs. PAC-Bayes prior work):

**Pentina & Lampert (ICML 2014)** [Hwkz Ref 1, already cited in our paper]: we thank the reviewer for pointing out this relevant work. We respectfully note that both its formulation and presentation are closer to multi-task/lifelong learning than to sequential continual learning. It assumes a **single, time-invariant hyperdistribution** π over task priors, so task priors are i.i.d. samples from π and task ordering does not affect the bound. In contrast, continual learning requires the prior at task t to depend on tasks 1,...,t-1 (e.g., EWC changes the prior geometry through Fisher information; IncLoRA changes which parameters remain trainable). Our framework models this via a **filtration-adapted hyperposterior process** {P_t}_{t=0}^T with a drift penalty KL(P_t ‖ P_{t-1}) that quantifies how inductive bias evolves across task transitions. P&L is a **special case of our Theorem 4.1** when the hyperposterior does not change across tasks, as stated in Section 4.1.

**Friedman & Meir (ICLR 2026)** [Hwkz Ref 2]: we also thank the reviewer for pointing out this concurrent work. Their formulation uses a flat posterior chain, whereas our framework introduces a genuine two-level hierarchical process with a time-varying hyperposterior and an explicit drift term. This distinction matters because it more directly reflects the sequential accumulation of knowledge and interference across tasks in continual learning. In addition, our paper further specializes the analysis to frozen-backbone LoRA and shows that the bound reduces to adapter-subspace quantities (Lemma 4.2 + Lemma 4.3 → Theorem 4.4), which is absent from prior PAC-Bayes CL analyses.

**Novel technical elements** (TSrA W2, ZRjU Q1):
1. **Time-varying hyperposterior process**: classical hierarchical PAC-Bayes (P&L 2014, Rothfuss et al. 2023) assumes fixed hyperdistribution. Our construction requires carefully showing that {P_t} is a valid adapted stochastic process and that the bound remains valid pathwise — this is non-trivial.
2. **Explicit drift penalty**: the KL(P_t ‖ P_{t-1}) term is new and separates within-task adaptation cost from cross-task interference cost. This decomposition enables direct analysis of forgetting.
3. **Frozen-backbone subspace reduction** (Lemma 4.2 + Lemma 4.3 → Theorem 4.4): showing that KL(Q_t ‖ P_t) = KL(Q_t^Δ ‖ P_t^Δ) under frozen backbone requires a careful translation-invariance argument for the KL under the affine structure of W_t^{froz} + W_{Δ,t}. Similarly, the sharpness term reduces to TS_t^Δ via the Gaussian posterior support argument. These reductions are absent from all prior hierarchical PAC-Bayes work.

**Re TSrA W3 ("no new method")**: Analysis and understanding papers are a recognized contribution at ICML. The paper establishes WHEN and WHY adapter-only flatness control suffices — a criterion that guides the design of any LoRA-based method. The curvature diagnostic framework (c_t, α_t) and ordered perturbation comparison (RS(0) < AS(0) < AS(1)) also go beyond mere validation.

**ZRjU Q1 — Where does the log m term go?**

Theorem 3.1 (classical, McAllester) has `log(m/δ) = log m + log(1/δ)` in the numerator. Theorem 4.1 (sequential) has `log(1/δ)` only. The term disappears because the two theorems use different proof techniques:

- Theorem 3.1 uses McAllester's bound, which inverts a KL inequality and produces `log(m/δ)` as a proof artifact of that inversion technique.
- Theorem 4.1 uses a **supermartingale tail bound**: we construct a nonneg supermartingale {M_t} via Donsker-Varadhan's variational KL inequality at both the model level (Step 1) and hyper level (Step 2). Markov's inequality then gives P(M_T ≥ 1/δ) ≤ δ, producing `log(1/δ)` only. The per-task Hoeffding-MGF bound (Step 1 of Appendix B.1) yields (λω_t)²/(8(m_t-1)) per task; optimizing λ gives the `sum(m_t-1)` denominator without any `log(m_t)` factor. The martingale approach never needs to union-bound over individual samples.

This is a standard and beneficial property of the variational-martingale PAC-Bayes technique — the sequential bound is tighter in this regard. We will add a clarifying remark to Appendix B.1.

**ZRjU Q2 — IncLoRA/OLoRA subspace support assumption**

The key clarification: W_{Δ,t} in our framework refers to the **current task's adapter subspace only**, not the entire accumulated adapter space. In IncLoRA, at task t, only (A_t, B_t) is trainable — all previous adapter pairs (A_1,B_1,...,A_{t-1},B_{t-1}) are frozen. Therefore W_{Δ,t} = span of current (A_t, B_t): a fixed-dimensional subspace. Both P_t^Δ and Q_t^Δ are Gaussians on this same W_{Δ,t}, satisfying Q_t^Δ ≪ P_t^Δ. The "growing" is a global phenomenon across tasks; within each task t, the PAC-Bayes analysis applies to a well-defined, fixed-dimensional W_{Δ,t}. For OLoRA: the orthogonality constraint is an optimization penalty that shapes the geometry of W_{Δ,t} relative to previous tasks but does not change the fact that both P_t^Δ and Q_t^Δ are supported on the same current W_{Δ,t}. We will add a clarifying remark to the proof of Lemma 4.2 in Appendix B.2.

---

### Group 3: Experiments
*Addresses: TSrA W4, mBoF W1/W2, Hwkz W3*

**Response text**:

We are conducting additional experiments to address the scope concerns:

**New vision benchmarks**: ViT-B/16 on **CUB200** (200 classes, 20-task incremental), **Cars196** (196 classes), and **CIFAR100** (100 classes, 10-task incremental). These cover different fine-grained recognition domains and span a broader range of task difficulty.

**LLM experiments + GAM**: We have existing LLM results with SAM and will add GAM comparisons to complete the evaluation. This directly addresses the "different scale and architecture" concern — LLMs are orders of magnitude larger than ViT-B/16 and have a fundamentally different architecture (decoder-only transformer vs. ViT encoder), covering a different modality (language vs. vision).

**Re mBoF W2 (model complexity/size scalability)**: Theorem 4.4 holds for any dimension of W_{Δ,t} — the key structural requirement is the frozen backbone condition, not model size. The LLM results provide empirical evidence that the ordering (adapter-only SAM > full-parameter perturbation) generalizes beyond ViT scale.

**Re Hwkz W3 (additional PEFT approaches)**: The framework is designed for LoRA because of its low-rank subspace structure. In principle, any PEFT where updates are confined to a structured subspace (prefix tuning, adapter modules) admits a similar analysis — the key is the existence of a well-defined W_{Δ,t}. We will discuss this generalization as future work in the revision.

---

### Group 4: Minor Technical Points
*Addresses: ZRjU Q3/Q4, Hwkz W4*

**ZRjU Q3 (Bound tightness)**: PAC-Bayes bounds are typically qualitative guides rather than numerically tight predictors. Theorem 4.4 identifies which quantities govern stability-plasticity — the curvature diagnostics (Table 1: λ_max(H), tr(H), λ_max(F), tr(F)) serve as the empirical validation. Computing tight numerical bound values requires estimating the partition function of the Gaussian posterior, which is challenging at this parameter scale. We will add a remark clarifying the bound's intended role.

**ZRjU Q4 (Figure 4 placement)**: We agree an early schematic would help. In the revision, we will add an intuitive figure to Section 3.3 illustrating the geometric differences between RS(0), AS(0), and AS(1) — showing how they concentrate perturbations differently within W_{Δ,t} relative to the curvature.

**Hwkz W4 (Undefined H notation)**: H denotes the hypothesis class (space of predictors f_W: X → Y). It first appears in the Related Work paragraph on "PAC-Bayesian Theory" (page 2, line 101 in the submission) without a formal definition. We will add "where H denotes the hypothesis class" at first use and consolidate the formal definition in Section 3.1.

---

## Closing Paragraph

The central finding of this paper — that adapter-only sharpness control is both theoretically justified (Theorem 4.4) and empirically superior (Table 2, Figures 6–7) to full-parameter perturbation in PECL — helps clarify how the LoRA-SAM (NeurIPS 2024) and Flat-LoRA (ICML 2025) perspectives should be interpreted in the frozen-backbone continual learning setting, while providing actionable guidance for practitioners. The theoretical framework introduces three genuinely novel elements (time-varying hyperposterior, drift penalty, frozen-backbone subspace reduction) that extend beyond prior hierarchical PAC-Bayes work. The supplementary experiments on CUB200, Cars196, CIFAR100, and LLM settings will substantially broaden the empirical validation. We believe these clarifications address the reviewers' main concerns and strengthen the case for acceptance.

---

## CHARACTER COUNT ESTIMATE

Group 1 (motivation): ~700 chars
Group 2 (theory): ~1400 chars
Group 3 (experiments): ~500 chars
Group 4 (minor): ~400 chars
Closer: ~300 chars
**Total: ~3300 chars** (well within 5000 chars; can expand if limit allows more)

---

## NOTES FOR FINALIZATION
- [ ] Get character limit (EG-6) to calibrate
- [ ] Get ICLR 2026 paper description (EG-4) to fill placeholder in Group 2
- [ ] Confirm LLM setup (model name, task type) for Group 3
- [ ] Confirm GAM will be added to LLM results
- [ ] Per-reviewer grouping: consider whether to split by reviewer or keep by theme
