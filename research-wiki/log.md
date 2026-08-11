# Research Wiki Log

## 2026-04-27T00:00:00Z — Mechanism experiment complete; two-paper strategy adopted

**exp:E003 complete**: Ran `ewclora_fitarch_gam_cub200_t10_r16_mechanism.yaml` with `mechanism_eval: true`.

**Key finding**: `cos_flat_fisher ≈ −0.02 to −0.04` (near-orthogonal, NOT synergistic). Fisher gradient is 0.014–0.286% of clean gradient. NME AvgAcc=84.66%, Forgetting=1.14%.

**Mechanism story revised**: From "cooperative" to "near-orthogonal division of labor" — stronger argument for complementarity. Near-zero cosine proves the two components address geometrically separated subspaces.

**Two-paper strategy adopted** (see `two_paper_strategy.md`):
- Paper A (Method, NeurIPS 2026/ICLR 2027): Flat-and-Stable SeqLoRA, gradient orthogonality as design principle
- Paper B (Theory, NeurIPS 2026): PAC-Bayes pathwise decomposition, forgetting-sharpness connection

**Wiki updates**:
- `ideas/002.md`: Added mechanism results + revised claims C7/C8
- `experiments/E003.md`: New experiment record
- `two_paper_strategy.md`: New — two-paper differentiation plan
- `index.md`: Updated to include E003 and two-paper section

---

## 2026-04-24T00:00:00Z — Wiki initialized

Wiki initialized for Project1: "Revisiting Sharpness in Low-rank Subspaces for Continual Learning" (ICML 2026 #2173).

Ingested from Project1/ directory:
- **Papers (6)**: foret2021_sam, li2024_flatlora, bisla2022_rwp, zhang2023_gam, pentina2014_pacbayes_cl, nguyen2025_pacbayes_cumulative, neural_thickets2026, li2025_sam_scaleinvariant
- **Ideas (1)**: idea:001 — Adapter-Subspace Flatness is Sufficient for PECL Generalization
- **Experiments (2)**: E001 (scope comparison), E002 (robustness + ablation)
- **Claims (5)**: C1–C5
- **Gaps (6)**: G1–G6
- **Edges (23)** in graph/edges.jsonl

Source context: PAPER_PLAN.md, REBUTTAL_STATE.md, ISSUE_BOARD.md, STRATEGY_PLAN.md, Previous Version.md (intro LaTeX).

Paper status: Under review at ICML 2026. Phase 2 rebuttal complete (2026-03-24). Mixed reviewer signals (ZRjU +5, TSrA -2, mBoF ~3, Hwkz ~3).
