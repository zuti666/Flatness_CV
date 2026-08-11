# Rebuttal State

**Paper**: Revisiting Sharpness in Low-rank Subspaces for Continual Learning
**Submission**: ICML 2026 #2173
**Venue**: ICML 2026
**Character Limit**: 5000 chars per reviewer (confirmed — separate response per reviewer)
**Response Mode**: TEXT_ONLY
**Current Phase**: Phase 2 — Concerns Atomized
**Last Updated**: 2026-03-24

## Reviewers
| ID   | Score | Confidence | Stance   |
|------|-------|------------|----------|
| TSrA | 2     | 4          | negative |
| ZRjU | 5     | 4          | positive |
| mBoF | 3     | 2          | swing    |
| Hwkz | 3     | 4          | swing    |

## Current Phase
Phase 7 complete — four per-reviewer responses + combined PASTE_READY produced. Manual blockers remain before submission.

## Output Files
- RESPONSE_TSrA.txt — ~3700 chars (limit: 5000) ✓
- RESPONSE_ZRjU.txt — ~3400 chars (limit: 5000) ✓
- RESPONSE_mBoF.txt — ~4400 chars (limit: 5000) ✓
- RESPONSE_Hwkz.txt — ~4700 chars (limit: 5000) ✓
- PASTE_READY.txt — combined fallback (~5050 chars)
- REBUTTAL_DRAFT_rich.md — annotated extended version

## Evidence Available & Status
- Paper PDF: 2173_Revisiting_Sharpness_in_L (8).pdf — READ (all 8 pages)
- Paper source: LIYING_ICML2026_Final (1).tex — READ (intro, theorems, proof appendix)
- Historical source: ICML_oldTemplate/liying_ICML.tex — READ (intro, motivation)
- Reviews: All 4 reviews fully parsed
- THEORY_COMPARISON.md: Full P&L vs F&M structural analysis complete

## Confirmed User Evidence
- **EG-1 RESOLVED**: log m disappears because sequential proof uses supermartingale/DV, not McAllester union bound
- **EG-2 RESOLVED**: IncLoRA/OLoRA support condition satisfied — W_{Δ,t} is CURRENT task adapter only (fixed dimension)
- **EG-5 RESOLVED**: H = hypothesis class; fix with inline definition at page 2 line 101
- **EG-3 CONFIRMED**: New experiments: CUB200, Cars196, CIFAR100 (ViT-B/16) + LLM+GAM results
- **EG-4 RESOLVED**: F&M differentiation complete (THEORY_COMPARISON.md) — three structural differences: no M(M(H)) hierarchy, plasticity-only bound, no subspace reduction

## Key Framing Changes (v1 → final)
- Motivation reframed: "implementation default" → "unresolved structural inconsistency between Flat-LoRA and LoRA-SAM"
- Removed out-of-scope phrases: "irreducible scope mismatch", "noise LoRA cannot counteract"
- Theory: P&L and F&M differentiated via 3 structural dimensions (not just "we are sequential")
- Commitment language: "in preparation" for new experiments (not "completed")

## Remaining Blockers Before Submission
- **EG-6**: ICML 2026 rebuttal character limit (check OpenReview) — PASTE_READY.txt is ~5050 chars
- **F&M quote**: Verify exact wording of F&M self-description ("straightforward extension") before citing
- **Experiment results**: Replace "in preparation" with numbers if results arrive before deadline
