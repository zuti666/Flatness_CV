# Strategy Plan

**Paper**: Revisiting Sharpness in Low-rank Subspaces for Continual Learning (ICML 2026 #2173)
**Date**: 2026-03-24

---

## Reviewer Posture Summary

| Reviewer | Score | Confidence | Priority | Strategy |
|----------|-------|------------|----------|----------|
| ZRjU     | 5     | 4          | Maintain | Answer proof questions precisely; reinforce positive framing |
| mBoF     | 3     | 2          | Flip     | Low confidence = persuadable; address empirical breadth with available evidence + FlatLoRA relationship |
| Hwkz     | 3     | 4          | Flip     | High confidence = needs principled argument; fix framing on motivation + cite missing PAC-Bayes CL works |
| TSrA     | 2     | 4          | Partial  | Hard reject; high confidence. Address misunderstanding of paper direction; unlikely to flip but must not alienate |

**Priority order**: Hwkz → mBoF → ZRjU → TSrA

---

## Global Themes (Opener — 4 resolutions)

**GT-1: The perturbation misalignment problem is real, not a straw man.**
Standard SAM perturbs ALL parameters by default; Flat-LoRA (ICML 2025) explicitly advocates full-parameter perturbation even when the backbone is frozen. Our paper shows this hurts in PECL (empirically) and explains WHY (theoretically). Applying SAM only to adapter parameters requires a deliberate design choice — our paper justifies and recommends it.

**GT-2: The theoretical contribution is the frozen-backbone reduction.**
The novel contribution is not classical PAC-Bayes machinery per se, but: (a) extending hierarchical bounds to sequentially evolving hyperposteriors with an explicit drift penalty; (b) showing that under frozen backbone the bound reduces EXACTLY to adapter-subspace quantities — a non-trivial reduction that depends critically on the support structure of the posterior.

**GT-3: The empirical finding is non-obvious and practically important.**
The consistent superiority of AS(1) across three LoRA strategies and three benchmarks, and the systematic harm of full-parameter perturbation, is a robust finding that practitioners and LoRA-SAM designers can immediately act on.

**GT-4: Scope is deliberately focused; generalization claims are appropriately bounded.**
We study the PECL with frozen backbone setting. The findings apply to this well-defined, increasingly deployed setting. Extension to other architectures/datasets is important future work, not a flaw.

---

## Per-Issue Response Plan

### CRITICAL Issues

**TSrA-C1 / Hwkz-W1 (SHARED — perturbation misalignment motivation)**
- *Response mode*: direct_clarification + grounded_evidence
- *Core argument*: SAM's default implementation perturbs all parameters; Flat-LoRA argues for this even in PECL. The paper's setup correctly reflects a genuine design choice faced by practitioners. Our theory + experiments resolve this choice.
- *Evidence source*: paper (Section 1 intro + Flat-LoRA citation); experimental results showing full-perturbation harm
- *Key line*: "Standard SAM and Flat-LoRA (ICML 2025) both operate over the full parameter space. Our paper asks and answers whether this is appropriate when the backbone is frozen — a question not previously resolved theoretically."

---

### MAJOR Issues

**TSrA-C2 (PAC-Bayes novelty)**
- *Response mode*: assumption_hierarchy
- *Core argument*: Three specific novel elements: (1) sequentially evolving hyperposterior — classical hierarchical PAC-Bayes assumes fixed hyperdistribution; (2) explicit drift penalty for cross-task interference; (3) frozen-backbone reduction proving KL + sharpness collapse to adapter subspace.
- *Evidence source*: paper (Theorem 3.1 vs classical; Lemma 4.2)

**TSrA-C3 (no new method)**
- *Response mode*: direct_clarification
- *Core argument*: The paper is explicitly a theoretical + empirical investigation, not a method paper. The value is resolving the open question of whether adapter-only perturbation suffices — which guides practitioners in how to use existing methods.

**TSrA-C4 / mBoF-W1 / Hwkz-W3 (SHARED — limited experiments)**
- *Response mode*: grounded_evidence + future_work_boundary
- *Status*: NEEDS USER INPUT — ask if any additional architecture/dataset results exist
- *Fallback*: Acknowledge the limitation honestly; explain choice of ViT-B/16 + ImageNet-C/R/P as principled (standard PECL benchmark); commit to expanding in camera-ready

**ZRjU-Q1 (logm·logm term missing from Theorem 4.4)**
- *Response mode*: assumption_hierarchy
- *Status*: NEEDS USER INPUT — requires exact proof accounting from authors
- *Ask user*: "Where does the logm·logm term from Theorem 4.1 go in the sequential bound of Theorem 4.4?"

**ZRjU-Q2 (IncLoRA/OLoRA subspace support assumption)**
- *Response mode*: assumption_hierarchy + direct_clarification
- *Status*: NEEDS USER INPUT
- *Ask user*: "For IncLoRA and OLoRA, is the prior supported on the same subspace as the posterior for each task? How is this maintained?"

**Hwkz-W2 (missing PAC-Bayes CL citations)**
- *Response mode*: nearest_work_delta
- *Core argument*: Cite and differentiate from ICML 2014 paper and ICLR 2026 paper. Key deltas: (a) we handle sequentially evolving hyperposterior; (b) we operate in LoRA subspace with frozen backbone; (c) we add drift penalty.
- *Status*: NEEDS USER INPUT — ask user to confirm relationship to ICLR 2026 paper

---

### MINOR Issues

**ZRjU-Q3 (bound tightness)**
- *Response mode*: narrow_concession + future_work_boundary
- *Script*: "We agree PAC-Bayes bounds are typically loose numerically. Theorem 4.4 is intended as a qualitative guide identifying which quantities govern stability and plasticity, not a numerically tight predictor. The curvature diagnostics in Section 5 provide complementary empirical support for the theoretical prediction."

**ZRjU-Q4 (Figure 4 too late/technical)**
- *Response mode*: direct_clarification
- *Script*: "We agree an early conceptual figure would help. We will add a schematic to Section 2 illustrating the geometric differences between RS(0), AS(0), and AS(1) in revision."
- *Status*: NEEDS USER INPUT — can user provide or describe such a figure?

**mBoF-W2 (model size scalability)**
- *Response mode*: future_work_boundary
- *Script*: "The frozen-backbone PECL regime is most relevant for large pre-trained models where full fine-tuning is prohibitive. We believe the subspace reduction argument extends to larger models since the frozen-backbone condition is the key structural requirement, not model size. Empirical validation at larger scale is important future work."

**mBoF-W3 (incremental vs full fine-tuning)**
- *Response mode*: direct_clarification
- *Script*: "The PECL setting fundamentally differs from full fine-tuning: the backbone is frozen, optimization is confined to a low-dimensional adapter subspace, and the sharpness of the full model is not directly controlled. Standard SAM theory does not apply. Our analysis must be rebuilt from scratch for this setting."

**mBoF-Q1 (FlatLoRA and the bound)**
- *Response mode*: nearest_work_delta
- *Script*: "FlatLoRA perturbs all parameters to flatten the full-space landscape, which contradicts the frozen backbone assumption in our setting. If the backbone is truly frozen, full-parameter perturbation is either infeasible or introduces perturbations to parameters that do not affect the task objective — precisely the misalignment our paper identifies. Our bound formalizes why restricting perturbation to the adapter subspace is both necessary and sufficient."

**Hwkz-W4 (undefined H notation)**
- *Response mode*: direct_clarification
- *Script*: "We will formally define H at its first use in Line 101 in the revision. [USER: please confirm what H denotes in your paper]"

---

## Evidence Gaps — Status

### Resolved (from reading paper + user input)

**[EG-1] RESOLVED**: `log m` in Theorem 3.1 vs. absent in Theorem 4.1:
- Theorem 3.1 uses McAllester's bound → `log(m/δ)` form
- Theorem 4.1 uses supermartingale + Donsker-Varadhan → produces `log(1/δ)` only (no `log m` needed because martingale approach doesn't union-bound over samples)
- This is a standard and beneficial aspect of the martingale-based PAC-Bayes technique

**[EG-2] RESOLVED**: IncLoRA/OLoRA subspace support:
- At each task t, W_{Δ,t} = span of CURRENT task's (A_t, B_t) only — fixed dimension
- All previous adapters are frozen, contributing zero perturbation to P_t and Q_t
- Absolute continuity Q_t^Δ ≪ P_t^Δ satisfied because both are Gaussians on same W_{Δ,t}
- The "growing" subspace is across tasks, not within a task

**[EG-5] RESOLVED**: H = hypothesis class (space of predictors). Appears informally in Related Work before formal definition in Section 3. Fix: add "where H denotes the hypothesis class" at first use in the Related Work paragraph on PAC-Bayesian Theory (page 2).

**[EG-3] CONFIRMED by user**: New experiments planned:
- Vision: ViT-B/16 on CUB200 (200 classes), Cars196 (196 classes), CIFAR100 (100 classes)
- LLM: Already have LLM results; will add GAM comparison

### Still Needed from User

**[EG-4]** How does your work differ from "PAC-Bayes bounds for cumulative loss in CL" (ICLR 2026)?
**[EG-6]** What is the ICML 2026 rebuttal character limit?

---

## Character Budget (assuming 5000-char limit — TBC)

| Section | Chars | % |
|---------|-------|---|
| Global opener (GT-1 through GT-4) | 600 | 12% |
| TSrA responses | 700 | 14% |
| ZRjU responses | 800 | 16% |
| mBoF responses | 600 | 12% |
| Hwkz responses | 900 | 18% |
| Meta-reviewer closing | 400 | 8% |
| **Total** | **4000** | **80%** |
| Buffer | 1000 | 20% |

---

## Blocked Claims (do NOT include without user confirmation)

- Any specific numbers from experiments not in the paper
- Any proof fix for ZRjU-Q1 without author confirmation
- Any claim about subspace support for IncLoRA/OLoRA without author confirmation
- Any commitment to camera-ready additions without user approval
