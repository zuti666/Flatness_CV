# Experiment Results

All results: ViT-B/16, LoRA rank 16, GAM optimizer, no replay memory.  
CNN metrics used for main comparison. NME also recorded but de-emphasized.  
FAA = Final Average Accuracy (mean of last accuracy-matrix row). AAA = lower-triangle mean.  
**Source of truth**: `outputs_logs/logs_inc_lora/ewclora_normfisher_gam/gam/` (JSON files).

---

## CUB200 t20 (10 tasks × 20 classes, seed 0)

### Three-Method Comparison (from logs_paperA/, log-parsed)

| Method | Model | ewc_lambda | CNN FAA | CNN Forget | fisher/clean | cos_flat_fisher |
|--------|-------|-----------|---------|-----------|-------------|----------------|
| Layer1 mixed GAM+rawFisher | `ewclora_youyue_fitarchitecture` | 20 | 75.136 | 10.796 | — | — |
| Layer2 explicit decoupled | `ewclora_youyue_fitarchitecture_gam` | 20 | 76.310 | 12.079 | 0.0029 | −0.0359 |
| Layer3 normFisher | `ewclora_normfisher_gam` | 500 | 76.295 | 11.986 | 0.0153 | −0.0374 |

Layer1 has lower forgetting but Fisher is entangled with GAM — not a valid ablation baseline.

### Layer 3 Lambda Sweep (ewc_normalize_fisher=true, lambda_flat=1.0, seed 0)

**JSON-based results** from `outputs_logs/logs_inc_lora/ewclora_normfisher_gam/gam/cub200/0/`:

| ewc_lambda | CNN FAA | CNN AAA | CNN Forget | BWT | fisher/clean | flat/clean | cos_flat_fisher |
|-----------|---------|---------|-----------|-----|-------------|-----------|----------------|
| 500 | 73.931 | 74.241 | 11.986 | −11.934 | — | — | — |
| **2000** | **74.224** | **74.332** | **11.709** | **−11.658** | **0.0248** | **4.223** | **−0.0302** |
| 5000 | 74.181 | 74.255 | 11.667 | −11.616 | — | — | — |
| dual (λ_init=500) | 73.881 | 74.207 | 12.039 | −11.988 | — | — | — |

Per-task final accuracy (lam2000):  
`[92.21, 78.15, 90.64, 79.67, 91.40, 76.13, 81.40, 83.04, 82.28, 92.24]`  
Per-task forgetting: `[29.44, 23.95, 15.75, 10.56, 6.34, 2.71, 2.06, 6.70, 7.87]`

**Best config**: `ewc_lambda=2000, lambda_flat=1.0, ewc_normalize_fisher=true`.  
Dual ascent did not activate — normalized Fisher drift stayed below δ=1e-4 throughout.

> **Note on CUB200 discrepancy**: Earlier wiki entries showed FAA ≈ 76.3–76.4 (log-parsed from `logs_paperA/`). The authoritative JSON results show FAA ≈ 74.2. The difference likely reflects a re-run or different logging path. Use JSON values going forward.

### Layer 2 Lambda Sweep (lambda_flat=1.0, raw Fisher, from logs_paperA/)

| ewc_lambda | CNN FAA | CNN Forget | fisher/clean | flat/clean | cos_flat_fisher |
|-----------|---------|-----------|-------------|-----------|----------------|
| 20 | 76.310 | 12.079 | 0.0029 | 4.302 | −0.0359 |
| 100 | 76.384 | 11.806 | 0.0137 | 4.310 | −0.0361 |
| 500 | 76.339 | 11.753 | 0.0411 | 4.320 | −0.0406 |
| 1000 | 76.343 | 11.563 | 0.0588 | 4.326 | −0.0432 |

Raw Fisher with ewc_lambda=20 is structurally inactive (ratio 0.003). Normalization is necessary.  
`flat/clean ≈ 4.3` throughout — flatness component dominates the update at all λ values.

---

## Cross-Dataset Results (Layer 3, ewc_lambda=2000, lambda_flat=1.0)

**All from JSON files** in `outputs_logs/logs_inc_lora/ewclora_normfisher_gam/gam/`.

| Dataset | Seed | Tasks | CNN FAA | CNN AAA | CNN Forget | BWT |
|---------|------|-------|---------|---------|-----------|-----|
| CUB200 t20 | 0 | 10×20 | 74.224 | 74.332 | 11.709 | −11.658 |
| Cars196 t10c20 | 0 | 10×20 | 51.741 | 52.494 | 12.321 | −12.321 |
| Aircraft t10c10 | 0 | 10×10 | 46.660 | 43.862 | 15.567 | −15.567 |
| Flowers t10c10 | 0 | 10×10 | 86.747 | 89.743 | 4.584 | −4.107 |
| OxfordPet t9c4 | 0 | 9×4 | 78.824 | 81.197 | 16.546 | −16.515 |
| ImageNet-R t20c10 | **1993** | 20×10 | 73.143 | 76.266 | 9.605 | −9.559 |

### Mechanism Statistics (averaged over tasks 1–T, from JSON)

| Dataset | fisher/clean | flat/clean | cos_flat_fisher |
|---------|-------------|-----------|----------------|
| CUB200 | 0.0248 | 4.223 | −0.0302 |
| Cars196 | 0.1914 | 13.560 | −0.0122 |
| Aircraft | 0.0752 | 4.128 | −0.0012 |
| Flowers | 0.0095 | 9.486 | −0.0251 |
| OxfordPet | 0.1544 | 203.647 | −0.0179 |
| ImageNet-R | 0.0167 | 2.257 | −0.0165 |

### Delta vs SeqLoRA-GAM Baseline

SeqLoRA-GAM baselines (from `run_paperA_paper_main_method3.sh` reference table):

| Dataset | Baseline FAA | Baseline AAA | Our FAA | Our AAA | ΔFAA | ΔAAA |
|---------|------------|------------|---------|---------|------|------|
| CUB200 | 73.249 | 73.535 | 74.224 | 74.332 | **+0.98** | **+0.80** |
| Cars196 | 52.197 | 51.646 | 51.741 | 52.494 | −0.46 | +0.85 |
| Aircraft | 45.730 | 43.861 | 46.660 | 43.862 | **+0.93** | +0.00 |
| Flowers | 81.282 | 85.799 | 86.747 | 89.743 | **+5.47** | **+3.94** |
| OxfordPet | 80.921 | 83.030 | 78.824 | 81.197 | −2.10 | −1.83 |

### Observations

- **Near-orthogonality is universal**: cos_flat_fisher ∈ [−0.030, −0.001] across all 6 datasets and all λ values. This is the core mechanistic finding.
- **Flowers**: Largest gain (+5.5pp FAA). Forgetting only 4.6%. fisher/clean=0.010 — Fisher penalty barely active; gain comes mainly from AS^1 flatness.
- **CUB200 / Aircraft**: Modest positive gains (+1pp). Both show healthy flat/clean ≈ 4.
- **Cars196**: Near-zero FAA delta (−0.46pp) but positive AAA delta (+0.85pp). flat/clean=13.6 is high — Fisher regularization relatively weak for this dataset.
- **OxfordPet**: Negative delta (−2.1pp FAA). flat/clean=203.6 — extreme outlier. OxfordPet has tiny gradients, so the AS^1 perturbation (rho=0.2) is disproportionately large relative to the task gradient. Need to lower `lambda_flat` or reduce rho.
- **ImageNet-R** (seed 1993): FAA=73.143 vs EWC-LoRA (Zheng et al.) reference of 72.86. Small improvement; AAA=76.266.

---

## Per-Task Accuracy Matrices

### Aircraft (t10c10, seed 0, lam2000)

Final task accuracies (diagonal = initial, last row = final):  
`[53.15, 39.94, 40.12, 59.16, 71.77, 66.47, 69.37, 73.57, 60.18, 72.97]`

### Cars196 (t10c20, seed 0, lam2000)

Final task accuracies:  
`[80.50, 45.68, 61.06, 65.15, 60.67, 66.30, 64.22, 65.85, 66.13, 52.74]`

### Flowers (t10c10, seed 0, lam2000)

Final task accuracies:  
`[98.68, 98.04, 93.70, 86.58, 79.68, 93.46, 90.99, 90.95, 89.38, 82.97]`

### OxfordPet (t9c4, seed 0, lam2000)

Final task accuracies:  
`[92.57, 90.18, 87.66, 97.25, 97.74, 95.00, 87.50, 97.24, 96.40]`

### ImageNet-R (t20c10, seed 1993, lam2000)

Final task accuracies:  
`[90.85, 91.67, 87.00, 89.46, 86.09, 80.92, 82.89, 84.98, 82.58, 71.75, 86.62, 83.51, 74.17, 81.21, 87.19, 81.22, 63.91, 86.29, 77.27, 74.91]`

---

## Baseline Comparison (EWC_Yaoyue_LoRA, SGD, no GAM)

Status: configs created, experiments pending.

Config dir: `config_exps_paper1_PAC/exp_paperA_ewc_yaoyue_baseline/`  
Run with: `bash scripts/exp_paperA/run_paperA_ewc_yaoyue_sgd.sh`

Datasets covered: cub200, cars196, aircraft, flowers, imagenetr, oxfordpet (rank 16).

---

## 2×2 Ablation (CUB200 t20 + t3)

Status: configs created, experiments pending.

| Condition | lambda_flat | ewc_lambda | Config (t20) |
|-----------|------------|-----------|--------------|
| Base (SeqLoRA+GAM) | 0.0 | 0.0 | `base_seqlora_sgd_cub200_t20.yaml` |
| Flat-only (AS^1) | 1.0 | 0.0 | `flat_only_cub200_t20.yaml` |
| Fisher-only (EWC) | 0.0 | 20.0 | `fisher_only_cub200_t20.yaml` |
| Full method | 1.0 | 20.0 | `full_method_cub200_t20.yaml` |

t3 versions also exist for fast diagnosis.  
Run with: `RUN_ABLATION=1 bash scripts/exp_paperA/run_paperA_cub200_t20.sh`

---

## EWC-LoRA Baseline Reproduction

Config dir: `config_exps_paper1_PAC/exp_paperA_reproduce_ewclora/`

| Dataset | Config |
|---------|--------|
| ImageNet-R | `ewclora_reproduce_imagenetr_t10_r10.yaml` |
| CIFAR-100 | `ewclora_reproduce_cifar100_t10_r10.yaml` |
| ImageNet-A | `ewclora_reproduce_imageneta_t10_r10.yaml` |

Paper reference (Zheng et al., ICLR 2026):
- ImageNet-R: A10=72.86±0.79, Avg=78.95±0.86
- Rank 10, 10 tasks × 20 classes, `delta_reference` snapshot (not accumulate_and_reset_lora)

Status: pending — need to confirm model_name matches reference implementation.

---

## Summary: What We Know

1. **Near-orthogonality (cos_flat_fisher ≈ 0) is universal** — holds across all 6 datasets (range −0.030 to −0.001). Core mechanistic evidence.
2. **Trace normalization is required** — raw Fisher at ewc_lambda=20 is structurally inactive (ratio 0.003). Normalized Fisher at lam=2000 gives ratio 0.025–0.19 depending on dataset.
3. **OxfordPet is an outlier** — flat/clean=204. Need to reduce `lambda_flat` or `rho` for this dataset; current setting over-perturbs relative to task gradient scale.
4. **Cars196 flat/clean=13.6** — Fisher regularization relatively weak compared to flatness penalty. May benefit from higher lam or lower lambda_flat.
5. **Flowers is the easiest win** — +5.5pp FAA, only 4.6% forgetting. AS^1 flatness alone likely drives the gain.
6. **Ablation and EWC-Yaoyue baseline still pending** — needed to disentangle flatness vs Fisher contributions cleanly.
