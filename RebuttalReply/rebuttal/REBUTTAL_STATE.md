# Rebuttal State

**Paper**: Revisiting Sharpness in Low-rank Subspaces for Continual Learning
**Submission**: ICML 2026 #2173
**Venue**: ICML 2026
**Character Limit**: 5000 chars per reviewer (confirmed — separate response per reviewer)
**Response Mode**: TEXT_ONLY
**Current Phase**: Phase 8 — Follow-Up Round (new experiments incorporated)
**Last Updated**: 2026-03-29

## Reviewers
| ID   | Score | Confidence | Stance   |
|------|-------|------------|----------|
| TSrA | 2     | 4          | negative |
| ZRjU | 5     | 4          | positive |
| mBoF | 3     | 2          | swing    |
| Hwkz | 3     | 4          | swing    |

## Current Phase
Phase 8 — all four per-reviewer responses updated with new experimental results. PASTE_READY.txt deprecated. Remaining action: replace [ANON_LINK] and upload.

## Output Files
- RESPONSE_TSrA.txt — 4849 chars (limit: 5000, headroom: 151) ✓
- RESPONSE_ZRjU.txt — 4763 chars (limit: 5000, headroom: 237) ✓
- RESPONSE_mBoF.txt — 4867 chars (limit: 5000, headroom: 133) ✓
- RESPONSE_Hwkz.txt — 4761 chars (limit: 5000, headroom: 239) ✓
- ANON_REPO_README.md — supplementary results (Setting 1/2/3 tables, ready to upload)
- PASTE_READY.txt — combined fallback (outdated; use per-reviewer files)
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
- **EG-3 COMPLETED**: Five fine-grained vision datasets (Aircraft/Cars196/CUB200/Flowers/OxfordPet) — SAM>SGD on 5/5 per method; GAM best 3-5/5; InfLoRA and SDLoRA added. Het5 heterogeneous benchmark — SeqLoRA SAM +9.4 pp FAA, IncLoRA +1.4 pp (cross-domain evidence). T5-small/T5-large × 3 orders — T5-large shows positive average gains for all three methods; T5-small is mixed.
- **EG-4 RESOLVED**: F&M differentiation complete (THEORY_COMPARISON.md) — three structural differences: no M(M(H)) hierarchy, plasticity-only bound, no subspace reduction

## Key Framing Decisions (final)
- Motivation: "unresolved structural inconsistency between Flat-LoRA and LoRA-SAM" (not "implementation default")
- Theory: P&L and F&M differentiated via 3 structural dimensions
- New experiments: results are COMPLETE — no "in preparation" language anywhere in the 4 response files
- NLP framing: use as scale/architecture validation; emphasize T5-large and keep T5-small as supplementary context
- Het5 cross-domain numbers used for mBoF Q3/Q4: SeqLoRA +9.4 pp FAA, IncLoRA +1.4 pp

## Remaining Blockers Before Submission
- **ANON_LINK**: Replace all `[ANON_LINK]` placeholders in the 4 response files with the actual anonymous repository URL before submitting
- **Setting 2 completion**: OLoRA SAM/GAM/RWP and InfLoRA/SDLoRA results on the het5 heterogeneous benchmark are pending — update ANON_REPO_README.md when available
- **Cross-dataset results**: If cross-dataset (train-on-one, test-on-another) experiments complete before deadline, add as Setting 2b in the README
- **F&M quote**: Verify exact wording of F&M self-description ("straightforward extension") before citing
- **NLP scope note**: NLP experiments cover only Adam vs SAM (not GAM). Responses are framed as "validating the adapter-scope conclusion" — do not add GAM language unless results exist
