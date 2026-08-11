# Wiki Log — NeurIPS 2026 Theory Paper

## 2026-04-30T14:00:00Z — Paper structure finalized; reparameterization guarantee confirmed from code

**Section 5 structure finalized**:
- Main text: Exp F (Q1, forked trajectory) + Exp E (Q2, support × direction) + one post-hoc sharpness diagnostic sentence
- Appendix: Exp C/D (CIFAR10 pilot), Reparameterization control, Multi-seed

**ΔW convention confirmed from code** (backbone/lora.py:211-212):
- `delta_w = B @ A`, so reparameterization is A' = A/c, B' = cB

**sam_delta scale-invariance guaranteed by implementation** (seqlora.py:633-634):
- `_project_grad_to_current_lora_tangent` uses orthonormal columns of B and A.T
- These column spaces are invariant under (A/c, cB) scaling → sam_delta direction is exactly invariant
- Can be stated in Appendix C without further proof

**sam_random ≠ random_delta** (seqlora.py:685):
- `sam_random` = SAM gradient projected onto `_project_grad_to_random_matched_tangent`
- `random_delta` = Gaussian noise in delta tangent support
- `sam_random` is NOT in Exp E; should not appear in the paper

**Key protective statement placement**: Must be LAST sentence of RQ2 paragraph, with forward reference to Appendix (reparameterization control). Do not bury it in the middle.

**Post-hoc sharpness framing**: "consistent with ... being a contributor" — not causal claim.

---

## 2026-04-30T12:00:00Z — Two direct experiments designed for Q1 and Q2

**Analysis**: Existing Exp D/E/F do not directly answer Q1/Q2:
- Exp D/F show "SAM training → less forgetting" but not "task-i sharpness predicts task-i forgetting"
- Exp E shows "sam_factor > sam_delta" — creates tension with theory; does not prove W_{Δ,t} invariance
- Neither experiment has per-task sharpness-forgetting correlation data

**Exp Q1 designed** ([experiments/exp_Q1_sharpness_forgetting_corr.md](experiments/exp_Q1_sharpness_forgetting_corr.md)):
- Post-hoc measurement on existing Exp E checkpoints — no new training
- Scatter Sh_delta(θ_i) vs F_i across all tasks and variants
- Direct validation of Theorem B.1

**Exp Q2 designed** ([experiments/exp_Q2_reparameterization_invariance.md](experiments/exp_Q2_reparameterization_invariance.md)):
- Reparameterize Exp F task-9 checkpoint: (A/c, cB), c ∈ {0.5, 1, 2, 4}
- Compare sam_factor vs sam_delta across c: sam_delta should be stable (invariant), sam_factor should vary
- Directly validates Theorem B.2 and resolves the sam_factor > sam_delta tension
- 12 runs total (4 c-values × 3 optimizers)

**Key insight added**: sam_factor > sam_delta at standard init (Exp E) because at standard LoRA init (B ≈ 0), T_Delta is ill-conditioned and sam_factor approximates A-space perturbation which has practical advantages. At non-standard scales (reparameterization), this advantage should change or disappear.

---

## 2026-04-30T00:00:00Z — Wiki initialized

Wiki initialized for NeurIPS 2026 Paper B: "Understanding Sharpness and Forgetting under Constrained Update Geometry in Continual Learning".

**Sources ingested**:
- `Project1/NewPpaerVerisonWrite/restructured_theory/Nips2026/neurips_2026_4 (1).tex` — current paper draft
- `Project1/experiment_summary_flatness_pecl_2026-04-30.md` — full experiment summary (Exp C/D/E/F + next steps)
- `research-wiki/` — prior wiki for ICML 2026 #2173 (theory foundation)
- `research-wiki/two_paper_strategy.md` — Paper A/B split strategy

**Paper status**: NeurIPS 2026 submission target. Template ready. Three-theorem structure planned (B.1 pathwise, B.2 support reduction, B.3 forgetting bound).

**Key experimental state as of 2026-04-30**:
- Exp E (ImageNet-R, support × direction): complete. sam_factor strongest (FAA 67.94, BWT -6.81 vs SGD 58.29, -18.99). SAM > random for every support.
- Exp F (ImageNet-R, forked trajectory): complete. Same-checkpoint fork: suffix SAM-factor >> suffix SGD, random-factor ≈ SGD.
- Exp G (rescaling control): pending — next priority.
- Multi-seed, post-hoc sharpness, cross-LoRA: pending.

**Wiki files created**:
- `index.md` — full overview
- `claims/C4.md`, `C5.md`, `C6.md`
- `experiments/exp_E.md`, `exp_F.md`, `exp_G.md`
- `gaps/G1.md`
- `log.md`
