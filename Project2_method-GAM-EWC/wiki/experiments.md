# Experiment Configurations and Scripts

All configs in `config_exps_paper1_PAC/`. All scripts in `scripts/exp_paperA/`.  
Results land in `outputs_logs/logs_inc_lora/<model_name>/<optimizer_type>/<dataset>/<seed>/<prefix>/`.

---

## Config Directories

### `exp_paperA_three_methods/` — Method hierarchy comparison (CUB200 t20)

| Config file | Model | ewc_lambda | Notes |
|------------|-------|-----------|-------|
| `layer1_mixed_gam_rawfisher_cub200_t20.yaml` | `ewclora_youyue_fitarchitecture` | 20 | GAM sees L_task + Fisher jointly |
| `layer2_explicit_gam_rawfisher_cub200_t20.yaml` | `ewclora_youyue_fitarchitecture_gam` | 20 | GAM sees only L_task; raw Fisher added separately |
| `layer3_as1_normfisher_lam500_cub200_t20.yaml` | `ewclora_normfisher_gam` | 500 | AS^1 + trace-normalized Fisher |

Run with: `RUN_THREE_METHODS=1 bash scripts/exp_paperA/run_paperA_cub200_t20.sh`  
Logs: `logs_paperA/cub200_t20/`

---

### `exp_paperA_ablation/` — 2×2 component ablation (CUB200 t20 + t3)

| Config file | lambda_flat | ewc_lambda | Purpose |
|------------|------------|-----------|---------|
| `base_seqlora_sgd_cub200_t20.yaml` | 0.0 | 0.0 | Baseline: SeqLoRA + GAM only |
| `flat_only_cub200_t20.yaml` | 1.0 | 0.0 | AS^1 flatness only |
| `fisher_only_cub200_t20.yaml` | 0.0 | 20.0 | EWC Fisher only (raw) |
| `full_method_cub200_t20.yaml` | 1.0 | 20.0 | Full method (flat + Fisher) |
| `base_seqlora_sgd_cub200_t3.yaml` | 0.0 | 0.0 | t3 fast diagnosis versions |
| `flat_only_cub200_t3.yaml` | 1.0 | 0.0 | |
| `fisher_only_cub200_t3.yaml` | 0.0 | 20.0 | |
| `full_method_cub200_t3.yaml` | 1.0 | 20.0 | |

Run with: `RUN_ABLATION=1 bash scripts/exp_paperA/run_paperA_cub200_t20.sh`  
Status: **pending**

---

### `exp_paperA_newmethod/` — Layer 3 lambda sweep (CUB200 t20 + t3)

Model: `ewclora_normfisher_gam`, `ewc_normalize_fisher=true`, `lambda_flat=1.0`

| Config file | ewc_lambda | ewc_dual_ascent | Seed |
|------------|-----------|----------------|------|
| `as1_normfisher_lam500_cub200_t20.yaml` | 500 | false | 0 |
| `as1_normfisher_lam2000_cub200_t20.yaml` | 2000 | false | 0 |
| `as1_normfisher_lam5000_cub200_t20.yaml` | 5000 | false | 0 |
| `as1_normfisher_dual_cub200_t20.yaml` | 500 | **true** | 0 |
| `as1_normfisher_lam500_cub200_t3.yaml` | 500 | false | 0 |
| `as1_normfisher_lam2000_cub200_t3.yaml` | 2000 | false | 0 |
| `as1_normfisher_lam5000_cub200_t3.yaml` | 5000 | false | 0 |
| `as1_normfisher_dual_cub200_t3.yaml` | 500 | true | 0 |

Common hyperparams: `lr=0.01`, `epochs=40`, `batch=128`, `gam_rho=0.2`, `scheduler=cosine`

Run with: `RUN_NORMFISHER_SWEEP=1 bash scripts/exp_paperA/run_paperA_cub200_t20.sh`  
Results: `outputs_logs/logs_inc_lora/ewclora_normfisher_gam/gam/cub200/0/` — **complete**

---

### `exp_paperA_cross_dataset/` — Cross-dataset validation (Layer 3, lam2000)

All use `ewclora_normfisher_gam`, `ewc_lambda=2000`, `ewc_normalize_fisher=true`, `lambda_flat=1.0`

| Config file | Dataset | Tasks×Classes | Seed | LR | GAM rho |
|------------|---------|--------------|------|----|---------|
| `as1_normfisher_lam2000_aircraft_t10c10_r16.yaml` | aircraft | 10×10 | 0 | 0.05 | 0.01/0.1 |
| `as1_normfisher_lam2000_cars196_t10c20_r16.yaml` | cars196 | 10×20 | 0 | 0.05 | 0.05/0.2 |
| `as1_normfisher_lam2000_flowers_t10c10_r16.yaml` | flowers | 10×10 | 0 | 0.0025 | default |
| `as1_normfisher_lam2000_oxfordpet_t9c4_r16.yaml` | pets | 9×4 | 0 | 0.0025 | default |
| `as1_normfisher_lam2000_imagenetr_t20c10_r16.yaml` | imagenetr | 20×10 | **1993** | 0.01 | 0.2/0.2 |

Note: Aircraft and Cars use `gam_grad_rho=0.01/0.05` (lower than CUB200's 0.2) due to different gradient scales.

Run with: `bash scripts/exp_paperA/run_paperA_cross_datasets_method3.sh`  
Results: `outputs_logs/logs_inc_lora/ewclora_normfisher_gam/gam/{dataset}/{seed}/` — **all complete**

---

### `exp_paperA_paper_main_method3/` — Paper main table (5 datasets, Layer 3)

Same method as cross_dataset but configs use `paper_main` prefix naming.  
Datasets: cub200, cars196, aircraft, flowers, oxfordpet.

Run with: `bash scripts/exp_paperA/run_paperA_paper_main_method3.sh`  
Generates: `logs_paperA/paper_main_method3/summary_latest.csv` with delta vs SeqLoRA-GAM baseline.

---

### `exp_paperA_layer2_sweep/` — Layer 2 raw Fisher sweep (CUB200 t20)

Grid over `lambda_flat` × `ewc_lambda`. Model: `ewclora_youyue_fitarchitecture_gam` (raw Fisher).

| lambda_flat | ewc_lambda options |
|------------|-------------------|
| 0.25 | 100, 500, 1000 |
| 0.5 | 100, 500, 1000 |
| 1.0 | 20, 100, 500, 1000 |

Total: 10 configs.  
Run with: `bash scripts/exp_paperA/run_layer2_rawfisher_sweep_cub200_t20.sh`

---

### `exp_paperA_ewclora_paper4_method3/` — EWC-LoRA on paper 4 datasets (Layer 3)

| Config file | Dataset | Tasks |
|------------|---------|-------|
| `as1_normfisher_lam2000_cifar100_t10c10_r10.yaml` | cifar100 | 10×10 |
| `as1_normfisher_lam2000_imageneta_t10c20_r10.yaml` | imageneta | 10×20 |
| `as1_normfisher_lam2000_imagenetr_t10c20_r10.yaml` | imagenetr | 10×20 |
| `as1_normfisher_lam2000_domainnet_t5c69_r30.yaml` | domainnet | 5×69 |

Rank 10/30 (matching EWC-LoRA paper setting).  
Run with: `bash scripts/exp_paperA/run_paperA_ewclora_paper4_method3.sh`  
Status: **pending**

---

### `exp_paperA_reproduce_ewclora/` — EWC-LoRA baseline reproduction

| Config file | Dataset | Split | Notes |
|------------|---------|-------|-------|
| `ewclora_reproduce_imagenetr_t10_r10.yaml` | imagenetr | 10×20, rank 10 | delta_reference snapshot |
| `ewclora_reproduce_cifar100_t10_r10.yaml` | cifar100 | 10×10, rank 10 | |
| `ewclora_reproduce_imageneta_t10_r10.yaml` | imageneta | 10×20, rank 10 | |

Reproducing Zheng et al. (ICLR 2026) reference numbers.  
Run with: `bash scripts/exp_paperA/run_reproduce_ewclora_all.sh` or `run_reproduce_ewclora_imagenetr.sh`  
Status: **pending** — verify model_name matches paper implementation

---

### `exp_paperA_ewc_yaoyue_baseline/` — EWC-Yaoyue SGD baseline (no GAM)

Same datasets as main method but with plain SGD optimizer, no GAM, no flatness.

| Config file | Dataset |
|------------|---------|
| `ewc_yaoyue_lora_sgd_cub200_t20_r16.yaml` | cub200 |
| `ewc_yaoyue_lora_sgd_cars196_t10c20_r16.yaml` | cars196 |
| `ewc_yaoyue_lora_sgd_aircraft_t10c10_r16.yaml` | aircraft |
| `ewc_yaoyue_lora_sgd_flowers_t10c10_r16.yaml` | flowers |
| `ewc_yaoyue_lora_sgd_imagenetr_t20c10_r16.yaml` | imagenetr |
| `ewc_yaoyue_lora_sgd_oxfordpet_t9c4_r16.yaml` | oxfordpet |

Run with: `bash scripts/exp_paperA/run_paperA_ewc_yaoyue_sgd.sh`  
Status: **pending**

---

## Script Reference

| Script | Stage | Configs | Log dir |
|--------|-------|---------|---------|
| `run_paperA_cub200_t20.sh` | 3-stage: three-method + ablation + lam sweep | `exp_paperA_three_methods/`, `exp_paperA_ablation/`, `exp_paperA_newmethod/` | `logs_paperA/cub200_t20/` |
| `run_paperA_imagenetr_method3.sh` | ImageNet-R Layer 3 | `exp_paperA_cross_dataset/` (imagenetr) | `logs_paperA/imagenetr_method3/` |
| `run_paperA_cross_datasets_method3.sh` | Aircraft/Cars/Flowers/Pets Layer 3 | `exp_paperA_cross_dataset/` | `logs_paperA/cross_datasets_method3/` |
| `run_paperA_paper_main_method3.sh` | 5-dataset paper table | `exp_paperA_paper_main_method3/` | `logs_paperA/paper_main_method3/` |
| `run_paperA_cub200_t3.sh` | CUB200 t3 (fast) | `exp_paperA_newmethod/` (*t3 configs) | `logs_paperA/cub200_t3/` |
| `run_layer2_rawfisher_sweep_cub200_t20.sh` | Layer 2 λ grid | `exp_paperA_layer2_sweep/` | `logs_paperA/cub200_t20_layer2_sweep/` |
| `run_paperA_ewclora_paper4_method3.sh` | CIFAR/ImageNet-A/DomainNet | `exp_paperA_ewclora_paper4_method3/` | TBD |
| `run_paperA_ewc_yaoyue_sgd.sh` | EWC-Yaoyue baseline | `exp_paperA_ewc_yaoyue_baseline/` | `logs_paperA/ewc_yaoyue_sgd/` |
| `run_reproduce_ewclora_all.sh` | EWC-LoRA reproduction (all 3) | `exp_paperA_reproduce_ewclora/` | TBD |
| `run_reproduce_ewclora_imagenetr.sh` | EWC-LoRA reproduction (ImageNet-R only) | `exp_paperA_reproduce_ewclora/` | TBD |

### Usage notes

- Scripts use `logs_paperA/` for logging; actual model outputs (JSON metrics) go to `outputs_logs/logs_inc_lora/`.
- GPU assignment: `GPUS_THREE="0 1"`, `GPUS_ABL="0 1"` etc. can be overridden from the shell.
- Stage toggles: `RUN_THREE_METHODS=0`, `RUN_ABLATION=0`, `RUN_NORMFISHER_SWEEP=0`.
- `TMPDIR` must be set to a local path (not NFS) to avoid `.nfs*` cleanup errors — the scripts handle this via `RUN_TMP_ROOT`.
- `DRY_RUN=1` prints commands without running them (supported in `run_paperA_paper_main_method3.sh`).

---

## Result File Locations (completed experiments)

```
outputs_logs/logs_inc_lora/ewclora_normfisher_gam/gam/
├── aircraft/0/paperA_as1normfisher_lam2000_aircraft_t10c10_r16_s0/exp_run/10/  *_cl_metrics.json
├── cars196/0/paperA_as1normfisher_lam2000_cars196_t10c20_r16_s0/exp_run/20/   *_cl_metrics.json
├── cub200/0/
│   ├── paperA_as1normfisher_lam500_cub200_t20/exp_run/20/                     *_cl_metrics.json
│   ├── paperA_as1normfisher_lam2000_cub200_t20/exp_run/20/                    *_cl_metrics.json  ← BEST
│   ├── paperA_as1normfisher_lam5000_cub200_t20/exp_run/20/                    *_cl_metrics.json
│   └── paperA_as1normfisher_dual_cub200_t20/exp_run/20/                       *_cl_metrics.json
├── flowers/0/paperA_as1normfisher_lam2000_flowers_t10c10_r16_s0/exp_run/10/   *_cl_metrics.json
├── imagenetr/1993/paperA_as1normfisher_lam2000_imagenetr_t20c10_r16_s1993/exp_run/10/  *_cl_metrics.json
└── pets/0/paperA_as1normfisher_lam2000_oxfordpet_t9c4_r16_s0/exp_run/4/       *_cl_metrics.json
```

Each `*_cl_metrics.json` contains:
- `cnn.final`: FAA, AAA, Forget_avg, BWT_final_avg, CA (per-task final accuracy), full accuracy matrices
- `nme.final`: same metrics for NME classifier
- `mechanism.final.tasks`: per-task mechanism stats (grad norms, fisher/clean ratio, flat/clean ratio, cos_flat_fisher, delta drift)
