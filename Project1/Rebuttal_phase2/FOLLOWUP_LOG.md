# Follow-Up Log — Phase 2 (ICML 2026 #2173)

**Date**: 2026-04-06
**Stage**: Phase 2 reviewer comments — author follow-up replies

---

## Phase 2 Reviewer Comments (verbatim)

### TSrA — Phase 2 Update

> "Partially resolved. I will raise my score to 3, as I still find the motivation of the paper to be weak and insufficient to meet the acceptance standard."

- **Score change**: 2 → 3
- **Linked issue**: TSrA-C1, TSrA-C3 (motivation + no new method)
- **Stance**: Still negative, not flipped. Score improvement suggests partial rebuttal success.
- **New issue**: None — this is a continuation of TSrA-C1/C3 motivation concern.
- **Reply file**: REPLY_TSrA_phase2.md

---

### mBoF — Phase 2 Update

> "I still believe that scalability remains an issue, as the results only cover a limited set of architectures (ViT → T5) and limited scale variation."
> "I still think the insights provided in this paper also appear somewhat incremental."

- **Linked issue**: mBoF-W1 (limited architectures), mBoF-W3 (incremental)
- **Stance**: Swing — not flipped yet. Scalability remains the main block.
- **New reply handles**: Added Llama-3.2-1B and Llama-3.2-3B results (Table 1).
- **Reply file**: REPLY_mBoF_phase2.md

---

### Hwkz — Phase 2 Update

> "Partially resolved. I might concur with Reviewer TSrA that the paper's motivation requires better justification."

- **Linked issue**: Hwkz-W1 (motivation), echoing TSrA-C1/C3
- **Stance**: Swing — shifted closer to TSrA's negative view. Key to address motivation clearly and independently from TSrA reply.
- **Original concerns addressed in Phase 1**: Hwkz-W2 (PAC-Bayes CL citations), Hwkz-W4 (undefined H notation). Hwkz-W3 (PEFT coverage) partially addressed.
- **Reply file**: REPLY_Hwkz_phase2.md (v2 improved)

---

## Safety Lint — Phase 2 Drafts

### TSrA Phase 2 Reply
- [x] Coverage: Addresses TSrA's remaining concern (motivation weakness)
- [x] Provenance: Claims grounded in paper (Theorem 4.4) and the paper's Phase 1 motivation framing
- [x] Commitment: No unapproved promises
- [x] Tone: Professional, not defensive
- **Status**: SAFE TO SUBMIT

### mBoF Phase 2 Reply
- [x] Coverage: Addresses scalability (Llama results) + incrementality
- [x] Provenance: New LLM results are user-confirmed (noted in STRATEGY_PLAN.md EG-3)
- [x] Commitment: No unapproved promises
- [x] Table formatting: compact summary table with full row labels
- [x] Tone: OK
- **Status**: SAFE TO SUBMIT

### Hwkz Phase 2 Reply
- [x] Coverage: Addresses motivation directly and ties back to Hwkz-W2
- [x] Provenance: Grounded in Theorem 4.4, Lemma 4.2, and the Phase 1 PAC-Bayes comparison
- [x] Commitment: None
- [x] Tone: Professional, not defensive
- **Status**: SAFE TO SUBMIT

### Bundle
- [x] Paste-ready combined file added: `PASTE_READY_phase2.md`

---

## Phase 2 Issue Status Summary

| Issue       | Phase 1 Status     | Phase 2 Action            | Outcome     |
|-------------|-------------------|---------------------------|-------------|
| TSrA-C1     | Addressed         | Deepened motivation arg   | Score: 2→3  |
| TSrA-C2     | Addressed         | N/A                       | Partial     |
| TSrA-C3     | Addressed         | Deepened motivation arg   | Score: 2→3  |
| TSrA-C4     | Addressed w/exps  | N/A                       | Partial     |
| ZRjU-Q1     | Answered          | No Phase 2 comment        | Maintained  |
| ZRjU-Q2     | Answered          | No Phase 2 comment        | Maintained  |
| ZRjU-Q3     | Addressed         | No Phase 2 comment        | OK          |
| ZRjU-Q4     | Acknowledged      | No Phase 2 comment        | OK          |
| mBoF-W1     | Partially addr    | Added Llama 1B/3B results | Pending     |
| mBoF-W2     | Addressed         | N/A                       | OK          |
| mBoF-W3     | Addressed         | Reiterated theory contrib | Pending     |
| mBoF-Q1     | Addressed         | N/A                       | OK          |
| Hwkz-W1     | Addressed         | Deepened motivation arg   | Pending     |
| Hwkz-W2     | Addressed (cites) | Referenced in v2 reply    | Pending     |
| Hwkz-W3     | Partial           | N/A                       | Open        |
| Hwkz-W4     | Addressed (H def) | N/A                       | Resolved    |
