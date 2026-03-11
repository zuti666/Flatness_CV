# Methods Overview (Chronological Log)

> Project: Flatness_CV
> Scope: LoRA-CL method variants and evaluation refactors
> Format: Each entry lists date, change purpose, and core logic.

## 2026-02-09 — InfoProbabilisticCL (Algorithm B) introduced
Purpose
- Create a second, independent method line that does **not** rely on projection/drift (Algorithm A), to support a separate paper narrative.

Core logic
- Build a head-free discriminative subspace per task using whitened cross-cov (features × labels).
- Maintain per-task references (U*, Kb*, energy ratio) at task end.
- Use differentiable **projection-energy ratio** on current batch features as proxy for:
  - Mix (budget deficit), Spec (spectral forget proxy), Align (subspace retention).
- Optional InfoNCE on U_hist and KL-like surrogate penalties.

Files
- `models_LoRAbasedCL/infoprob_gam_lora.py`
- `utils/factory.py` (model registry)
- `config_exps/exp0_infoprob_gam_lora/*.yaml`
- `scripts/exp0_infoprob_gam_lora/run_infoprob_gam_memory_parallel.sh`

Key fixes in this iteration
- Eliminated gradient detachment for mix/spec/align by moving them into `_composite_loss` as **tensor operations**.
- Removed SVD from inner loop; SVD only at task end.
- Unified data source handling for historical stats (`ipb_mem_source`).

---

## 2026-02-09 — InfoBudget-GAM diagnostics & projection validity improvements
Purpose
- Make projection effects **measurable and testable**, and ensure no hidden test leakage in stats mode.

Core logic
- `mem_source=stats` disables drift and prevents old-sample fallback to test set.
- Projection can be applied to SGD/SAM/GAM; GAM uses direction projector hook.
- Use **energy-based** protection basis for B_prot to avoid mean cancellation.
- Record **absolute drift energy** (interf_abs), per-epoch curves, and old-task accuracy.

Files
- `models_LoRAbasedCL/infobudget_gam_lora.py`

---

## 2026-02-08 — InfoBudget-GAM + Subspace InfoNCE (A+NCE)
Purpose
- Extend Algorithm A with **subspace InfoNCE**, preserving relational structure in U_hist, beyond drift magnitude control.

Core logic
- InfoNCE on normalized projections of current vs. teacher features in U_hist.
- Loss = task + drift + NCE (optional).

Files
- `models_LoRAbasedCL/infobudget_gam_lora_subnce.py`
- `config_exps/exp0_infobudget_gam_lora_subnce/*.yaml`
- `scripts/exp0_infobudget_gam_lora_subnce/run_subnce_gam_memory_parallel.sh`

---

## 2026-02-08 — Projection-on-SGD/SAM parity
Purpose
- Ensure projection mechanism is not GAM-only; enable fair comparisons for SGD/SAM.

Core logic
- Project gradients directly in SGD/SAM steps;
- GAM uses optimizer projector hook as before.

Files
- `models_LoRAbasedCL/infobudget_gam_lora.py`
- `config_exps/exp0_infobudget_gam_lora/*proj_opt_*`
- `scripts/exp0_infobudget_gam_lora/run_proj_opt_compare_parallel.sh`

---

## 2026-02-07 — Evaluation & logging refactor (clean API)
Purpose
- Decouple heavy evaluation logic from `trainer.py`, unify feature cache & drift comparisons, improve GitHub readiness.

Core logic
- Moved attention probe export to standalone module.
- Moved OOD eval to standalone module.
- Unified feature cache + first/last drift comparison logic.
- Added task cache controls (only cache selected tasks).

Files
- `evaluation_attention/attention_probe.py`
- `evaluation_ood/ood_eval.py`
- `evaluation_feature/eval_flat_feature.py`
- `evaluation_feature/feature_drift.py`
- `src/trainer.py` (orchestration only)

---

## 2026-02-07 — Metrics book consolidation
Purpose
- Centralize evaluation metric logging and JSON writing for consistency and reuse.

Core logic
- Consolidated append/final metric JSON helpers and matrix utilities.

Files
- `utils/metrics_book.py`

---

## 2026-02-06 — Config/entry structure cleanup
Purpose
- GitHub-ready layout; unify entrypoints, keep configs/logs separate.

Core logic
- Move reproducible code under `src/` and configs under `config_exps/`.
- Logs/outputs to `.outputs_logs/` or `outputs_logs/` with `.gitignore`.

---

## 2026-02-05 — Flatness/feature eval cleanup
Purpose
- Make flatness/feature eval repeatable and decoupled from training logic.

Core logic
- One shared loader built for flatness and feature metrics.
- Flatness configs read from args (no hard-coded flags).

Files
- `evaluation_sharpness/*`
- `evaluation_feature/*`

---

# Version Map (Summary)

- **Algorithm A (InfoBudget-GAM LoRA)**
  - Projection + drift (feature-space) + optional NCE.
  - Uses U_t / U_hist and B_prot (LoRA param space).
  - `models_LoRAbasedCL/infobudget_gam_lora.py`

- **Algorithm A + Subspace InfoNCE**
  - Adds InfoNCE on U_hist projections (teacher vs current).
  - `models_LoRAbasedCL/infobudget_gam_lora_subnce.py`

- **Algorithm B (InfoProbabilisticCL)**
  - No projection/drift; uses MI‑budget proxies + align/spec + NCE + KL surrogates.
  - Differentiable proxy losses from projection energy ratio.
  - `models_LoRAbasedCL/infoprob_gam_lora.py`
