# Gap Map

| Gap ID | Description | Status | Linked Ideas | Linked Papers |
|--------|-------------|--------|--------------|---------------|
| G1 | No principled theory identifies which notion of flatness governs generalization in frozen-backbone PECL | **addressing** | idea:001 | paper:foret2021_sam, paper:li2024_flatlora |
| G2 | Active unresolved debate: LoRA-SAM (adapter-only sufficient) vs Flat-LoRA (full-param needed) | **addressing** | idea:001 | paper:li2024_flatlora |
| G3 | First-order flatness (gradient norm) vs zeroth-order in LoRA fine-tuning not systematically compared in CL | **addressing** | idea:001 | paper:zhang2023_gam |
| G4 | Geometric understanding of adapter solution distribution around pretrained weights | **open** | — | paper:neural_thickets2026 |
| G5 | Extension of frozen-backbone flatness theory to larger models (LLMs, billion-scale) | **open** | — | — |
| G6 | Tight (non-vacuous) PAC-Bayes bounds for CL with subspace constraints | **open** | — | paper:nguyen2025_pacbayes_cumulative |

## Gap Notes

- **G1 + G2** are the primary gaps addressed by idea:001 / the paper.
- **G4** partially addressed by Neural Thickets (2026) — their findings support our geometric intuition.
- **G5** is a key limitation acknowledged in the paper and reviewers' concerns (mBoF-W2); future work.
- **G6** acknowledged (ZRjU-Q3): PAC-Bayes bounds are typically loose; our bound is qualitative, not numerically tight.
