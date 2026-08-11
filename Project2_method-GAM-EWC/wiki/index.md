# Project 2 Wiki — Flat-and-Stable SeqLoRA

**Paper A**: AS^1 + Past-Fisher in Near-Orthogonal Gradient Subspaces  
**Target venue**: NeurIPS 2026 / ICLR 2027  
**Last updated**: 2026-05-04

---

## Quick Navigation

- [Setting](setting.md) — PECL problem setup, datasets, training protocol
- [Method Core Design](method_core_design.md) — Chinese method summary: motivation, update rule, normalized Fisher, and implementation boundary
- [Methods](methods.md) — Three-layer method hierarchy and key differences
- [Experiments](experiments.md) — Config directories, scripts, result file locations
- [Results](results.md) — All experiment results with JSON-verified numbers (updated 2026-05-01)
- [Method Result Selection](method_result_selection_2026-05-04.md) — latest `sgd_tuned`, invalid `as1_normfisher_gam` audit, and FitArchitecture baseline selection (2026-05-04)
- [Exp Plan: Paper4 Benchmarks](exp_plan_paper4_benchmarks.md) — CIFAR-100/DomainNet/ImageNet-R/ImageNet-A run plan (2026-05-02)
- [Paper4 Method3 Config Summary](paper4_method3_config_summary_2026-05-03.md) — corrected config/log audit for CIFAR-100/DomainNet/ImageNet-R/ImageNet-A (2026-05-03)
- [Next Steps](next_steps.md) — Open experiments and analysis needed

---

## One-Sentence Story

SeqLoRA forgets because the single shared adapter drifts away from old-task optima. We show that AS^1 (current-task flatness) and past-Fisher regularization in ΔW-space operate in **near-orthogonal gradient subspaces** (cos ≈ −0.02 to −0.04, confirmed across datasets), enabling a principled joint design where each component handles a separate aspect of the stability-plasticity tradeoff.

---

## Core Finding

**cos_flat_fisher ≈ −0.02 to −0.04** (near-zero, across tasks and datasets).

This is not a coincidence — it reflects the spectral separation between the current-task Hessian (which flatness targets) and the past-task Fisher (which forgetting targets). The two gradients address different directions by construction.

---

## Method Summary

| Layer | Model Name | Key Difference |
|-------|-----------|----------------|
| Layer 1 | `ewclora_youyue_fitarchitecture` | GAM sees `L_task + Fisher` jointly (coupled, baseline) |
| Layer 2 | `ewclora_youyue_fitarchitecture_gam` | GAM sees only `L_task`; Fisher added as separate direction (decoupled) |
| Layer 3 | `ewclora_normfisher_gam` | Layer 2 + trace-normalized Fisher + optional dual ascent |

**Current best (Layer 3):** `lambda_flat=1.0, ewc_lambda=2000, ewc_normalize_fisher=true`

---

## Key Results — Cross-Dataset (Layer 3, ewc_lambda=2000, seed 0/1993)

| Dataset | CNN FAA | CNN Forget | fisher/clean | cos_flat_fisher |
|---------|---------|-----------|-------------|----------------|
| CUB200 t20 | 74.224 | 11.709 | 0.025 | −0.030 |
| Cars196 | 51.741 | 12.321 | 0.191 | −0.012 |
| Aircraft | 46.660 | 15.567 | 0.075 | −0.001 |
| Flowers | **86.747** | 4.584 | 0.010 | −0.025 |
| OxfordPet | 78.824 | 16.546 | 0.154 | −0.018 |
| ImageNet-R (s=1993) | 73.143 | 9.605 | 0.017 | −0.017 |

cos_flat_fisher ≈ 0 across **all** datasets (range −0.030 to −0.001).

---

## Paper Draft Location

- Draft paper (LaTeX): [restructured_method/paper/](../restructured_method/paper/)
- Section plan: [restructured_method/PAPER_A_MECHANISM_PLAN.md](../restructured_method/PAPER_A_MECHANISM_PLAN.md)
- ICML #2173 (Paper B, theory): separate — see Project1/
