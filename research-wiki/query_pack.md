# Query Pack — Project1 PECL Flatness Wiki
# Generated: 2026-04-24 | Budget: ~8000 chars

## Project Direction (300 chars)

Theoretical + empirical investigation of which notion of sharpness governs generalization in Pre-trained Encoder Continual Learning (PECL) with frozen backbone and LoRA adapters. Core finding: adapter-subspace (W_{Δ,t}) flatness is both sufficient and superior to full-parameter perturbation. ICML 2026 submission #2173.

---

## Top Gaps (1200 chars)

**G1** [ADDRESSING] — No principled theory for frozen-backbone PECL flatness. What sharpness notion governs generalization when backbone is frozen and optimization is confined to W_{Δ,t}? Our PAC-Bayes theory resolves this (Theorem 4.4).

**G2** [ADDRESSING] — Unresolved LoRA-SAM vs Flat-LoRA debate. LoRA-SAM: adapter-only sufficient. Flat-LoRA (ICML 2025): full-param perturbation needed. Our work: adapter-only is sufficient AND full-param is harmful in PECL. Resolved with theory + experiments.

**G3** [ADDRESSING] — First-order vs zeroth-order flatness comparison not systematic in CL. AS^(1) (gradient-norm proxy) vs AS^(0) (random perturbation) in LoRA CL. Our finding: AS^(1) > AS^(0).

**G5** [OPEN] — Scalability to LLMs. Current theory and experiments focused on ViT-B/16. LLM extension planned but pending. Key reviewer concern (mBoF-W2).

**G6** [OPEN] — Non-vacuous PAC-Bayes bounds for PECL. Current Theorem 4.4 is qualitative; numerically tight bounds are open. Acknowledged limitation (ZRjU-Q3).

---

## Paper Clusters (1600 chars)

**Cluster A: Sharpness-Aware Optimization**
foret2021_sam (SAM, core), bisla2022_rwp (RWP/RS^0), zhang2023_gam (GAM/first-order), li2024_flatlora (Flat-LoRA, primary foil), li2025_sam_scaleinvariant (implicit regularization).
These papers establish the flatness-seeking optimizer landscape. The project's key contribution is determining which of these applies in the PECL setting — the answer (AS^1 wins, RS^0 hurts) directly contradicts Flat-LoRA.

**Cluster B: PAC-Bayes Theory for CL**
pentina2014_pacbayes_cl (P&L 2014 — classical hierarchical, i.i.d. tasks, fixed hyperposterior), nguyen2025_pacbayes_cumulative (ICLR 2026 — cumulative loss bound, details TBD).
Our Theorem 3.1 / 4.1 extends P&L by: (1) time-varying hyperposterior, (2) explicit drift penalty, (3) supermartingale/DV technique (removes log m). Differentiation from ICLR 2026 paper is a pending rebuttal concern (Hwkz-W2 / EG-4 partially resolved: 3 structural differences identified).

**Cluster C: Geometric Insights**
neural_thickets2026 — Task-expert solutions cluster near pretrained weights, supporting the geometric intuition that W_{Δ,t} captures the relevant geometry for PECL.

---

## Failed Ideas (1400 chars)

No fully failed ideas yet — idea:001 is **active/partial** (under review).

**Failed approaches within idea:001 (sub-attempts)**:
- **Full-scope perturbation in PECL** [CONFIRMED FAILED]: Applying SAM / Flat-LoRA's full-parameter perturbation to PECL gives −2.00 to −6.84 ΔAcc^cls degradation. Do NOT investigate full-scope perturbation as a PECL strategy.
- **Framing as "implementation default"** [ABANDONED]: Early drafts framed the problem as practitioners mistakenly using the default SAM implementation. Flat-LoRA's explicit advocacy for full-perturbation is the real foil — reframed to "unresolved structural inconsistency between Flat-LoRA and LoRA-SAM."
- **Classical hierarchical PAC-Bayes (P&L style)** [INSUFFICIENT]: Fixed hyperposterior assumption violated in sequential CL. Must use time-varying hyperposterior + drift penalty.
- **Zeroth-order adapter perturbation as primary contribution**: AS^(0) < AS^(1) — first-order (gradient-norm-based) is consistently better. First-order is the right recommendation.

---

## Top Papers (1800 chars)

| Paper | Relevance | Key Role |
|-------|-----------|----------|
| paper:li2024_flatlora | **core** | Primary foil — advocates full-param perturbation, directly contradicted by our empirical + theoretical results |
| paper:foret2021_sam | **core** | Foundation method — full-param SAM by default; we show this is wrong in PECL |
| paper:pentina2014_pacbayes_cl | **core** | Theory baseline — classical hierarchical PAC-Bayes we extend |
| paper:zhang2023_gam | **related** | First-order flatness insight — underpins AS^(1); GAM comparison planned in rebuttal |
| paper:bisla2022_rwp | **related** | RS^(0) baseline — random perturbation, consistently worse than AS^(1) in experiments |
| paper:nguyen2025_pacbayes_cumulative | **related** | Must-differentiate — ICLR 2026 PAC-Bayes CL paper; 3 structural differences identified but details TBD |
| paper:neural_thickets2026 | **related** | Geometric support — task solutions cluster near pretrained weights, validates adapter-subspace focus |
| paper:li2025_sam_scaleinvariant | **related** | SAM implicit regularization theory — theoretical support for first-order flatness approach |

---

## Active Chains (900 chars)

**Chain 1: Scope Misalignment → Performance Degradation**
Flat-LoRA advocates full-param perturbation → full-scope perturbation in PECL → violations of frozen-backbone constraint → scope mismatch → curvature injected into W_froz → cascading task interference → −2.00 to −6.84 ΔAcc^cls.

**Chain 2: Adapter Subspace → Generalization**
Frozen backbone → optimization confined to W_{Δ,t} → KL(Q_t||P_t) = KL(Q_t^Δ||P_t^Δ) (Lemma 4.2) → sharpness term reduces to W_{Δ,t} curvature (Lemma 4.3) → Theorem 4.4 (adapter-only PAC bound) → AS^(1) theoretically justified → AS^(1) empirically best.

---

## Open Unknowns (500 chars)

- **EG-4 (partially open)**: Full structural comparison with ICLR 2026 PAC-Bayes CL paper (nguyen2025_pacbayes_cumulative) — 3 differences identified but details not confirmed.
- **G5**: Does frozen-backbone flatness theory hold for billion-parameter LLMs?
- **G6**: Can PAC-Bayes bounds for PECL be made non-vacuous (numerically tight)?
- **Broader PEFT**: Does the adapter-subspace principle extend beyond LoRA (prefix-tuning, adapters, etc.)?
