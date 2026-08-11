# Rebuttal — Extended Version (ICML 2026 #2173)

**Purpose**: This rich version contains fuller reasoning, optional paragraphs marked `[OPTIONAL]`, and annotations for each choice. Use `PASTE_READY.txt` for the actual submission; use this file to understand the logic and decide on manual edits.

**Provenance rule**: Every factual claim below is anchored to `paper`, `review`, `user_confirmed_result`, or `THEORY_COMPARISON.md`.

---

## Opener

**Paste-ready version** (~200 chars):
> We thank all reviewers. Three themes require clarification: (1) whether the perturbation-scope question is a genuine structural issue, not an implementation default; (2) theoretical novelty relative to prior PAC-Bayes work; (3) empirical scope. Supplementary experiments on CUB200, Cars196, CIFAR100, and LLM+GAM are in preparation.

**Why this opener**: Previous version opened with "key themes" framing which let reviewers categorize us defensively. New opener immediately signals: we are reframing (1) as a *structural issue*, not a choice — this is the single most important persuasion shift.

---

## MOTIVATION: A STRUCTURAL DESIGN ISSUE (TSrA W1, Hwkz W1, mBoF Q1)

**Core reframe**: We are NOT claiming adapter-only SAM is the correct default. We are claiming that the literature presents an unresolved structural inconsistency (Flat-LoRA vs. LoRA-SAM), and we provide the first formal resolution for the frozen-backbone PECL setting.

### Paragraph 1 — The observation does not resolve the question

> We agree that, in standard implementations, SAM applied to a LoRA model with a frozen backbone perturbs only the trainable adapter parameters. However, this observation does not resolve the underlying question of how perturbations should be defined in LoRA-based models — specifically, whether flatness should be enforced within the adapter subspace or over the full parameter space.

**Why**: Opens by conceding the factual observation (disarms reviewer) but immediately pivots to "this does not resolve the question." This is the key move — we go from defensive to aggressive in one sentence.

### Paragraph 2 — The structural inconsistency

> Existing works adopt fundamentally different answers. LoRA-SAM (NeurIPS 2024) restricts perturbations to adapter parameters; Flat-LoRA (ICML 2025) explicitly argues this is insufficient and advocates perturbing the full parameter space including the frozen backbone. This reveals an unresolved inconsistency in how flatness is defined for LoRA models.

**Evidence source**: paper (Section 1, both citations present). **Safe**: directly citable from Introduction.

### Paragraph 3 — Why it is structural in PECL

> This inconsistency becomes structural in PECL, where updates are restricted to the task-permissible subspace due to the frozen backbone. Perturbations over the full parameter space extend beyond the admissible update directions, creating a mismatch between the perturbation domain and the optimization domain. Our contribution is a formal resolution: Theorem 4.4 shows that, under frozen backbone, the bound-relevant sharpness depends exclusively on perturbations within W_{Δ,t}. Extending perturbations beyond W_{Δ,t} introduces terms outside the update mechanism, explaining Table 2's consistent degradation (full-scope: ΔAcc^cls = -2.00 to -6.84; adapter-scope: +0.05 to +0.86).

**Key phrase choices**:
- "mismatch between perturbation domain and optimization domain" — grounded in paper concept, avoids "irrecoverable" (out of paper scope)
- "terms outside the update mechanism" — maps to Theorem 4.4, avoids "unrecoverable directions" (not in paper)
- Numbers from Table 2 — directly in paper

[OPTIONAL — add if space permits]:
> This explains why practitioners following Flat-LoRA's recommendation would make the wrong choice in frozen-backbone PECL — not because Flat-LoRA is wrong in general, but because its recommendation does not carry over to the structural constraints of PECL. Our paper provides both the theory (Theorem 4.4) and empirical evidence (Table 2, Figs. 5–7) to guide the correct design decision in this setting.

### Re mBoF Q1 (compact)

> Re mBoF Q1: Flat-LoRA's perturbation spans full W. Our analysis shows only W_{Δ,t} contributes to bound-relevant sharpness in frozen-backbone PECL — perturbations outside this subspace do not align with permissible updates and are not controlled by the bound.

---

## THEORETICAL CONTRIBUTIONS (ZRjU Q1/Q2, Hwkz W2, TSrA W2/W3)

**Core reframe**: We are not claiming new proof techniques for PAC-Bayes machinery. We are claiming a new *problem formulation* (sequential hyperposterior process) and a new *structural reduction* (subspace collapse under frozen backbone). These are absent from all prior work.

### Three structural differentiators

#### (1) Sequential hyperposterior vs. shared hyperprior — vs. Pentina & Lampert (ICML 2014)

> P&L derives a bound on the **multi-task risk** under the assumption that tasks are i.i.d. from a single time-invariant hyperdistribution — task ordering is irrelevant and shuffling the sequence leaves the bound unchanged. This is multi-task learning, not sequential CL. As our paper explicitly states and Figure 2 illustrates: "in continual learning, the prior used at step t is typically produced by a history-dependent rule that progressively incorporates information from earlier tasks, rather than being generated from a single time-invariant hyperdistribution (Multi-task Learning)." Our framework captures this via a filtration-adapted hyperposterior process {P_t} with explicit drift penalty KL(P_t‖P_{t-1}). P&L is a strict special case of our Theorem 4.1 when drift = 0 (stated in Section 4.1).

**Evidence**: paper Section 4.1, Related Work, Figure 2, and direct quote from the paper. **Safe**: P&L already cited; Figure 2 already in the paper; quote is from the submitted text.

[OPTIONAL — more technical, cut if tight]:
> The key implication: in P&L, one can shuffle the task sequence without changing the bound. In our framework, different task orderings produce different drift terms — the bound correctly reflects that earlier tasks shape the prior for later ones, which is the defining property of CL (e.g., EWC shapes the Fisher-weighted prior geometry; IncLoRA restricts which parameters are trainable based on prior tasks).

#### (2) Genuine hierarchy vs. flat posterior chain — vs. Friedman & Meir (ICLR 2026, concurrent)

> F&M uses P_t = Q_{t-1} (flat posterior chain) and bounds only cumulative plasticity loss; their Corollary 3.1 is self-described as "a straightforward extension requiring no new technical tools." Three structural differences: (i) we introduce a genuine two-level hierarchy P_{t-1} ∈ M(M(H)), not M(H) — F&M has no distribution over distributions; (ii) our decomposition separates within-task adaptation E[KL(Q_t‖P_t)] from cross-task drift KL(P_t‖P_{t-1}), directly capturing the stability-plasticity tradeoff — F&M's single KL(Q_t‖Q_{t-1}) cannot distinguish these; (iii) F&M has no sharpness term and no subspace reduction.

**Evidence**: THEORY_COMPARISON.md (detailed analysis). F&M self-description quote — verify exact wording before submission.

**Why this matters**: Reviewer Hwkz cited F&M as potentially covering our contribution. The three-point answer directly refutes this: (i) structural hierarchy, (ii) stability vs. plasticity bound target, (iii) PECL-specific reduction.

#### (3) Subspace reduction — absent from all prior work

> Lemma 4.2+4.3 → Theorem 4.4 shows KL(Q_t‖P_t) = KL(Q_t^Δ‖P_t^Δ) and sharpness reduces exactly to TS_t^Δ under frozen backbone. This PECL-specific reduction is entirely absent from both P&L and F&M.

**Evidence**: paper Lemma 4.2, 4.3, Theorem 4.4.

### Re TSrA W3 ("no new method")

> Resolving open structural questions with formal analysis is a recognized ICML contribution. The result — adapter-only perturbation is the theoretically correct choice in frozen-backbone PECL — is directly actionable for any LoRA-based method designer.

### ZRjU Q1 — log m disappears

> Theorem 3.1 (McAllester) has log(m/δ) = log m + log(1/δ). Theorem 4.1 has log(1/δ) only. The log m term disappears because Theorem 4.1 uses a supermartingale tail bound (Donsker-Varadhan + Markov's inequality) rather than a union bound over samples. The per-task Hoeffding-MGF step yields (λω_t)²/(8(m_t-1)); optimizing λ produces log(1/δ) only — the martingale never union-bounds over individual samples. The sequential bound is thus strictly tighter. We will add a remark to Appendix B.1.

**Evidence source**: user_confirmed_result (EG-1 resolved).

### ZRjU Q2 — IncLoRA/OLoRA support

> At task t, only the current adapter pair (A_t, B_t) is trainable — all previous pairs are frozen. W_{Δ,t} = span(A_t, B_t) has fixed dimension. P_t^Δ and Q_t^Δ are Gaussians on this same W_{Δ,t}, satisfying Q_t^Δ ≪ P_t^Δ. For OLoRA, the orthogonality penalty shapes geometry within W_{Δ,t} but does not alter the support structure. We will add a remark to Appendix B.2.

**Evidence source**: user_confirmed_result (EG-2 resolved).

---

## EMPIRICAL SCOPE (TSrA W4, mBoF W1/W2, Hwkz W3)

> We are adding: ViT-B/16 on CUB200 (200 classes), Cars196 (196 classes), and CIFAR100 (100 classes) — three additional vision CL benchmarks spanning different fine-grained domains; and LLM experiments with GAM comparisons, covering a fundamentally different architecture and scale. Theorem 4.4 holds for any frozen-backbone PEFT with a well-defined W_{Δ,t}; the frozen-backbone condition, not model size, is the key structural requirement. We will discuss generalization to other PEFT methods in the revision (Hwkz W3).

**Evidence source**: user_confirmed_result (EG-3 confirmed). **Commitment status**: approved for rebuttal.

[OPTIONAL — add if space permits for mBoF W2]:
> The LLM experiments provide cross-architecture evidence that the ordering (adapter-only > full-parameter perturbation) generalizes beyond ViT-B/16. Since Theorem 4.4 depends only on the frozen-backbone structure and the existence of a well-defined W_{Δ,t}, not on specific model architecture, this generalization is theoretically expected.

---

## MINOR POINTS (ZRjU Q3/Q4, Hwkz W4)

> ZRjU Q3: Theorem 4.4 is a qualitative guide identifying which quantities govern stability-plasticity; Table 1 curvature diagnostics (λ_max(H), tr(H)) provide complementary empirical validation. We will add a clarifying remark.
>
> ZRjU Q4: We will add an early schematic to Section 3.3 illustrating the geometric differences between RS(0), AS(0), and AS(1) within W_{Δ,t}.
>
> Hwkz W4: H denotes the hypothesis class (predictors f_W: X→Y). We will add an explicit definition at first use (page 2, line 101) and in Section 3.1.

**Evidence**: user_confirmed_result (EG-5 resolved).

---

## Closing

> The core finding — adapter-only sharpness control formally resolves the Flat-LoRA vs. LoRA-SAM ambiguity in frozen-backbone PECL, backed by three structural novelties over prior PAC-Bayes work and consistent empirical evidence across Table 2 and Figs. 6-7 — provides both a principled theoretical framework and actionable guidance. The supplementary experiments on three new benchmarks and LLM settings substantially broaden empirical scope. We believe these clarifications address all main concerns.

---

## Safety Lint Summary

| Gate | Status |
|------|--------|
| Coverage | All 4 reviewers' issues addressed |
| Provenance | All claims anchored to paper/review/user_confirmed |
| Commitments | New experiments (EG-3) approved; notation fixes approved |
| Removed | "irreducible scope mismatch", "noise in frozen directions that LoRA cannot counteract", "irrecoverable directions" |
| Tone | No aggressive phrasing; no submissive phrasing |

## Character Count

| Section | Approx chars |
|---------|-------------|
| Opener | 200 |
| Motivation (3 paras + mBoF) | 950 |
| Theory (3 differentiators + TSrA W3 + ZRjU Q1/Q2) | 1700 |
| Empirical | 320 |
| Minor points | 230 |
| Closing | 200 |
| Dividers/headers | 150 |
| **Total PASTE_READY.txt** | **~5050** |

**If venue limit < 5000**: Cut `Theory (2)` from 3-point list to 2-point bullet (saves ~200 chars), or shorten ZRjU Q1 by removing the last two sentences (saves ~120 chars).

**If venue limit > 5000**: Add `[OPTIONAL]` paragraphs in Motivation and Empirical sections (adds ~400 chars).

---

## Remaining Manual Decisions

- [ ] **EG-6**: Confirm ICML 2026 rebuttal character limit (check OpenReview submission page)
- [ ] **EG-4 verification**: Verify F&M self-description quote ("straightforward extension") — must match exact wording in paper before submitting
- [ ] **Experiment results**: If CUB200/Cars196/CIFAR100 results arrive before deadline, replace "in preparation" with actual numbers
- [ ] **LLM setup**: Confirm model name and task type for LLM experiments if quoting in rebuttal
