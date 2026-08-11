# Research Wiki Index — Project1: PECL Flatness

**Paper**: Revisiting Sharpness in Low-rank Subspaces for Continual Learning
**Venue**: ICML 2026 (submission #2173, under review)
**Last updated**: 2026-04-27

---

## Papers (8)

### Core
- [foret2021_sam](papers/foret2021_sam.md) — SAM: sharpness-aware minimization, perturbs all params; foundation method
- [li2024_flatlora](papers/li2024_flatlora.md) — Flat-LoRA: argues full-param perturbation needed even in LoRA fine-tuning; **primary foil**
- [pentina2014_pacbayes_cl](papers/pentina2014_pacbayes_cl.md) — P&L 2014: classical hierarchical PAC-Bayes for lifelong learning; theory baseline

### Related
- [bisla2022_rwp](papers/bisla2022_rwp.md) — RWP: random weight perturbation variant of SAM; RS^(0) baseline in experiments
- [ewc_lora_iclr2026](papers/ewc_lora_iclr2026.md) — EWC-LoRA (ICLR 2026): Fisher regularization for shared LoRA CL; **primary competitor for idea:002** (no flatness)
- [fr_lora_cikm2025](papers/fr_lora_cikm2025.md) — FR-LoRA (CIKM 2025): Fisher-regularized LoRA for multilingual CL
- [zhang2023_gam](papers/zhang2023_gam.md) — GAM: first-order flatness via gradient norm; underpins AS^(1)
- [nguyen2025_pacbayes_cumulative](papers/nguyen2025_pacbayes_cumulative.md) — ICLR 2026 PAC-Bayes CL paper; must differentiate in rebuttal (Hwkz-W2)
- [neural_thickets2026](papers/neural_thickets2026.md) — Task experts cluster near pretrained weights; geometric support for adapter-subspace focus
- [li2025_sam_scaleinvariant](papers/li2025_sam_scaleinvariant.md) — SAM implicit regularization theory; supports first-order flatness view

---

## Ideas (2)

- [idea:001](ideas/001.md) — **Adapter-Subspace Flatness Sufficient for PECL** [active/partial — ICML 2026 review]
- [idea:002](ideas/002.md) — **Flat-and-Stable SeqLoRA: AS^(1) + Past-Fisher** [active — mechanism confirmed 2026-04-27; orthogonality story adopted]

---

## Claims (5)

| ID | Title | Status |
|----|-------|--------|
| [C1](claims/C1.md) | PAC-Bayes bound reduces to adapter-subspace quantities | supported |
| [C2](claims/C2.md) | AS^(1) consistently outperforms full-param perturbation | supported |
| [C3](claims/C3.md) | Full-param perturbation systematically harms PECL | supported |
| [C4](claims/C4.md) | Benefit holds across SeqLoRA / IncLoRA / OLoRA | supported |
| [C5](claims/C5.md) | Benefit weakens at larger LoRA rank | reported |

---

## Gaps (6)

| ID | Description | Status |
|----|-------------|--------|
| G1 | No PECL-specific flatness theory (frozen backbone) | addressing |
| G2 | LoRA-SAM vs Flat-LoRA debate unresolved | addressing |
| G3 | First-order vs zeroth-order in CL not systematic | addressing |
| G4 | Geometric distribution of adapter solutions | open |
| G5 | Theory extension to LLMs / large models | open |
| G6 | Non-vacuous PAC-Bayes bounds for PECL | open |

---

## Experiments (3)

- [exp:E001](experiments/E001.md) — Perturbation scope comparison (adapter-only vs full-space) [complete]
- [exp:E002](experiments/E002.md) — Robustness + ablation (LoRA variants, rank, task length) [complete]
- [exp:E003](experiments/E003.md) — Mechanism eval: cos_flat_fisher, gradient orthogonality (CUB200 t10) [complete — 2026-04-27]

## Two-Paper Strategy

- [two_paper_strategy.md](two_paper_strategy.md) — Plan for splitting into Paper A (method) + Paper B (theory)
  - **Paper A** (NeurIPS 2026/ICLR 2027): Flat-and-Stable SeqLoRA — orthogonal gradient subspaces
  - **Paper B** (NeurIPS 2026): PAC-Bayes pathwise decomposition — sharpness→forgetting theory

## Quick Navigation

- [gap_map.md](gap_map.md) — All field gaps with stable IDs
- [graph/edges.jsonl](graph/edges.jsonl) — Materialized relationship graph
- [query_pack.md](query_pack.md) — Compressed context for idea generation (~8000 chars)
- [log.md](log.md) — Append-only mutation log
