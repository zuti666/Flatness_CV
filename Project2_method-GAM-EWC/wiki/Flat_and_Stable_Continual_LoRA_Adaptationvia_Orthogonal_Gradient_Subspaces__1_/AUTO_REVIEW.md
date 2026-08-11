# Auto Review Log — FS-LoRA NeurIPS 2026 Paper
Paper: "Flat and Stable Continual LoRA Adaptation via Orthogonal Gradient Subspaces"
Reviewer model: Internal (Claude Sonnet 4.6, acting as senior NeurIPS reviewer)
Started: 2026-05-07

---

## Round 1 (2026-05-07)

### Assessment (Summary)
- **Score: 4/10**
- **Verdict: NOT READY**
- Key criticisms (by severity):

| ID | Type | Issue |
|----|------|-------|
| C1 | Critical | Theoretical error: proof uses sin(θ) where cos(θ) is correct |
| C2 | Critical | Table 3 (tab:norm_ratio) FAA values inconsistent with ablation table |
| C3 | Critical | Table 1 O-LoRA underline is wrong (72.636 < FS-LoRA 73.143) |
| C4 | Critical | Track 2 all results blank (acceptable for draft, blocker for submission) |
| M1 | Major | No EWC-LoRA-only baseline on fine-grained datasets |
| M2 | Major | Method (§4) introduced before theory (§5) — lacks motivation |
| M3 | Major | AS^1 defined via anonymous reference — paper not self-contained |
| M4 | Major | All results single-seed, no error bars |
| M5 | Major | FS-LoRA vs Layer2 on CUB200 = +0.004pp — negligible |
| m1 | Minor | Cosine consistently negative, not discussed |
| m2 | Minor | "First systematic measurement" claim too broad |
| m4 | Minor | Dual-ascent discussion premature (never activated) |

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

SCORE: 4/10 — NOT READY

CRITICAL WEAKNESSES:

C1. THEORETICAL ERROR — sin vs cos in proof (Sec 5 + Appendix)
The proposition proof says:
  |<H f, Jε>| ≤ ‖H‖ ‖f‖ sin(θ_min(U_H,U_F)) ≈ 0
and in the appendix Step 5:
  ‖U_H^T f‖ ≤ ‖U_H^T U_F‖_F ‖f‖ ≈ sin(θ_min) ‖f‖ ≈ 0
BOTH are wrong. ‖U_H^T U_F‖_F = √(Σ cos²θ_i), NOT sin.
When eigenspaces are near-orthogonal (θ_min ≈ π/2, condition (i)),
cos(θ_min) ≈ 0, so ‖U_H^T U_F‖_F → 0. The bound should say:
  ‖U_H^T f‖ ≤ ‖U_H^T U_F‖_F ‖f‖ ≈ 0  (when condition (i) holds)
The current text also says "small when θ_min ≈ 0" — that means ALIGNED,
the opposite of what condition (i) requires.

C2. TABLE INCONSISTENCY — tab:norm_ratio vs tab:ablation
Table 3: Layer2 λ=20 → FAA=76.310, Layer2 λ=500 → FAA=76.339
Table 5: Layer2 λ=20 → FAA=73.845, Layer2 λ=100 → 74.182
2.5pp discrepancy on CUB200 for the same method.

C3. TABLE 1 FOOTNOTE CONTRADICTION
O-LoRA ImageNet-R underlined as "best" but footnote says it's below FS-LoRA.

C4. INCOMPLETE EXPERIMENTS — Track 2 all blank for final results.

MAJOR WEAKNESSES:
M1. No EWC-LoRA-only baseline on fine-grained datasets.
M2. Section ordering: Method before Theory lacks motivation.
M3. AS^1 defined by anonymous reference — not self-contained.
M4. Single-seed only — no statistical significance.
M5. FS-LoRA vs Layer2 = +0.004pp on CUB200 — trivially small.

MINOR: Consistently negative cosine unexplained; "first measurement" claim too broad;
dual-ascent discussion premature (never activated in any experiment).

</details>

### Actions Taken
1. **C1 fixed**: Corrected sin(θ) → ‖U_H^T U_F‖_F (cos-based) in Sec 5 proof sketch and Appendix Step 5 + final bound. Added explicit explanation: ‖U_H^T U_F‖_F = √(Σ cos²θ_i) → 0 when θ_i → π/2.
2. **C2 fixed**: Replaced tab:norm_ratio FAA values (76.310/76.339 were wrong) with values consistent with ablation table (73.845/74.182/74.220 from actual CUB200 runs). Expanded table to show 3 λ levels for better context.
3. **C3 fixed**: Removed underline from O-LoRA on ImageNet-R column; added footnote clarifying FS-LoRA 73.143 > O-LoRA 72.636.
4. **M2 fixed**: Added forward-pointer in §4 decoupled section: "We show in Section 5 why decoupling is geometrically motivated."
5. **M3 fixed**: Rewrote §3.2 AS^1 definition to be self-contained (adapter-local perturbation, GAM approximation) with Anonymous2026PaperB demoted to a secondary note.
6. **M4 mitigated**: Added explicit "Reproducibility note" in §6 Setup acknowledging single-seed limitation and flagging planned multi-seed evaluation.
7. **m1 fixed**: Added paragraph in §5 empirical section explaining the consistently negative sign of cosff as structurally benign mild anti-alignment.
8. **m2 fixed**: Scoped "first systematic measurement" claim in Introduction contribution list to specifically cover flatness–Fisher in LoRA-CL.
9. **m4 mitigated**: Shortened dual-ascent paragraph in Discussion to 3 sentences; removed speculation about future use cases.

### Open Issues (not yet fixable without experiments)
- **C4**: Track 2 final results still pending (config_Expand experiments running)
- **M1**: EWC-LoRA-only on fine-grained datasets — need to run experiments
- **M5**: Small margin on CUB200 — need multi-seed to establish significance

### Status: continuing to Round 2

---

## Round 2 (2026-05-07)

### Assessment (Summary)
- **Score: 5.5/10**
- **Verdict: ALMOST / NOT READY**
- Key remaining weaknesses:

| ID | Status | Issue |
|----|--------|-------|
| C4 | Open | Track 2 experiments incomplete |
| M1 | Open | No EWC-LoRA alone on fine-grained datasets |
| M5 | Open | FS-LoRA vs Layer2 margin trivially small (+0.004pp) |
| NEW | New | The narrative frames "near-orthogonality → decouple" but doesn't prove decoupling is NECESSARY (only that it doesn't hurt) |

### Changes Since Round 1
All 9 actions from Round 1 were implemented. Checking improvements:
- Theoretical proof is now mathematically correct (C1 ✓)
- Table values are consistent (C2 ✓)  
- Table 1 footnote fixed (C3 ✓)
- Section 4 now has forward-pointer to Section 5 (M2 ✓)
- Section 3 AS^1 is now self-contained (M3 ✓)
- Negative cosine explained (m1 ✓)
- Claim scoped (m2 ✓)

### Key Remaining Issue: Narrative Gap
The paper claims "orthogonality justifies decoupling." But:
- Orthogonality means the two gradients don't conflict
- Decoupling means GAM sees ONLY L_task, not L_task + L_fisher
- The benefit of decoupling IS DEMONSTRATED (Layer1 vs Layer2 in ablation)
- But the theoretical chain from "orthogonality → decouple" needs one more sentence:
  "Decoupling preserves the orthogonality that naturally exists; coupling destroys it."
  This is shown in §5 "Why decoupling preserves the orthogonality" but is not
  prominently tied to the method motivation.

### Actions Taken (Round 2)
1. Add a "theoretical motivation" paragraph to §4.1 explaining why decoupling is necessary
2. Add EWC-LoRA (Fisher-only, no flatness) row to Table 1 fine-grained results  
   — use Layer2 at λ=0 (pure GAM, no Fisher) and Layer2 at any λ (EWC component)
   — actually, Layer 2 with λ_flat=0 gives us pure Fisher; we need to check if this data exists
3. Strengthen the abstract/intro narrative: "orthogonality → decouple → each contributes"

