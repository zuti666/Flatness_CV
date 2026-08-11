# Experiment Plan: AS^1 + NormFisher on EWC-LoRA Paper 4 Benchmarks

**Goal**: Compare `as1_normfisher_gam` against EWC-LoRA (Zheng et al., ICLR 2026) on the four datasets from that paper's Table 2.  
**Status (2026-05-02)**: Phase 1 launched — CIFAR-100/ImageNet-R/ImageNet-A running on GPUs 0/2/3. DomainNet skipped (disk space issue, see below).  
**Created**: 2026-05-02

---

## Background

The fine-grained cross-dataset experiments (CUB200, Aircraft, Cars196, Flowers, OxfordPet) use SGD+GAM and provide our internal ablation story. The 4-dataset benchmark (CIFAR-100, DomainNet, ImageNet-R, ImageNet-A) is the standard comparison table used by EWC-LoRA, SD-LoRA, InfLoRA, SeqLoRA, etc. — we need results here to be publishable.

**Key implementation note**: `models_CL/baseLearner.py:648` was fixed — `optimizer_type: gam` now correctly uses the `optimizer:` field (adam/adamw/sgd) as the base optimizer instead of hardcoding SGD. All new configs use `optimizer: "adam"`.

---

## Method: as1_normfisher_gam

File: `models_LoRAbasedCL/as1_normfisher_gam.py`  
Factory aliases: `as1_normfisher_gam`, `as1_normfisher`, `ewclora_normfisher_gam`

What it adds over plain EWC-LoRA:
1. **AS^1 flatness** via GAM perturbation on `L_task`: `λ_flat · S^(1)(θ, ΔW)`
2. **Trace-normalized Fisher**: `F̃ = F / (trace(F) + ε)` — makes `λ_ewc` scale-invariant to the ΔW magnitude
3. Optional dual ascent for adaptive `λ_ewc` (not used in main runs)

---

## Experiment Settings

All settings follow EWC-LoRA paper Table 8, with our method additions.

| Dataset | repo key | Split | Rank | Epochs | LR | WD | Batch |
|---------|----------|-------|------|--------|----|----|-------|
| CIFAR-100 | `cifar224` | 10×10 | 10 | 20 | 5e-4 | 0.0 | 128 |
| DomainNet | `domainnet` | 5×69 | 30 | 5 | 5e-4 | 0.0 | 128 |
| ImageNet-R | `imagenetr` | 10×20 | 10 | 50 | 5e-4 | 0.005 | 128 |
| ImageNet-A | `imageneta` | 10×20 | 10 | 10 | 5e-4 | 0.0 | 128 |

Our method additions (same for all 4):
- `optimizer: adam`, `optimizer_type: gam` (Adam base + GAM perturbation)
- `gam_grad_rho: 0.2`, `gam_grad_norm_rho: 0.2`
- `lambda_flat: 1.0`, `ewc_lambda: 2000`, `ewc_normalize_fisher: true`
- `ewc_max_batches: 0` (full train loader for Fisher, matching paper)
- `ewc_gamma: 0.9` (EWC decay per task)
- `class_shuffle: false` (matching paper)
- `seed: [0]` for single-seed run, `[0, 42, 521, 1024, 1993]` for mean±std

---

## Reference Numbers (EWC-LoRA Table 2)

| Dataset | EWC-LoRA A (last) | EWC-LoRA Avg |
|---------|-------------------|-------------|
| CIFAR-100 | 87.91 ± 0.36 | 92.27 ± 0.21 |
| DomainNet | 73.46 ± 0.35 | 79.58 ± 0.45 |
| ImageNet-R | 72.86 ± 0.79 | 78.95 ± 0.86 |
| ImageNet-A | 59.89 ± 0.30 | 68.33 ± 0.30 |

---

## Config Files

Directory: `config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3/`

| Config | Dataset | Prefix |
|--------|---------|--------|
| `as1_normfisher_lam2000_cifar100_t10c10_r10.yaml` | CIFAR-100 | `paper4_as1normfisher_lam2000_cifar100_t10c10_r10_s0` |
| `as1_normfisher_lam2000_domainnet_t5c69_r30.yaml` | DomainNet | `paper4_as1normfisher_lam2000_domainnet_t5c69_r30_s0` |
| `as1_normfisher_lam2000_imagenetr_t10c20_r10.yaml` | ImageNet-R | `paper4_as1normfisher_lam2000_imagenetr_t10c20_r10_s0` |
| `as1_normfisher_lam2000_imageneta_t10c20_r10.yaml` | ImageNet-A | `paper4_as1normfisher_lam2000_imageneta_t10c20_r10_s0` |

---

## Run Commands

### Phase 1 — Single seed (seed=0), all 4 datasets

Assign one GPU per dataset (4 GPUs total, runs in parallel):

```bash
GPUS_PAPER4="0 1 2 3" bash scripts/exp_paperA/run_paperA_ewclora_paper4_method3.sh
```

Subset only time-critical datasets first:

```bash
DATASETS="imagenetr imageneta" GPUS_PAPER4="0 1" \
  bash scripts/exp_paperA/run_paperA_ewclora_paper4_method3.sh
```

### Phase 2 — Multi-seed for mean ± std

Run after Phase 1 passes (results above reference):

```bash
SEEDS_JSON='[0,42,521,1024,1993]' GPUS_PAPER4="0 1 2 3" \
  bash scripts/exp_paperA/run_paperA_ewclora_paper4_method3.sh
```

Each GPU runs one dataset, cycling seeds sequentially per dataset. Produces 5 runs per dataset.

### Dry run (verify commands without running)

```bash
DRY_RUN=1 GPUS_PAPER4="0 1 2 3" bash scripts/exp_paperA/run_paperA_ewclora_paper4_method3.sh
```

### Outputs

- Logs: `logs_paperA/ewclora_paper4_method3/`
- Metrics JSON: `outputs_logs/logs_inc_lora/as1_normfisher_gam/gam_adam/{dataset}/0/{prefix}/exp_run/{inc}/`
- Summary CSV: `logs_paperA/ewclora_paper4_method3/summary_latest.csv`

---

## Experiment Phases

### Phase 1: Baseline run (ewc_lambda=2000, single seed)
Priority: **immediate**. This gives us a first read vs. EWC-LoRA reference.

Expected time:
- CIFAR-100: ~40 min (20 epochs × 10 tasks, ~2 min/epoch on V100)
- DomainNet: ~30 min (5 epochs × 5 tasks, large per-task size)
- ImageNet-R: ~4–5 hr (50 epochs × 10 tasks)
- ImageNet-A: ~30 min (10 epochs × 10 tasks)

### Phase 2: ewc_lambda sweep (if Phase 1 result ≠ desired)

If any dataset shows:
- `fisher/clean < 0.01` → ewc_lambda too small, try 5000/10000
- `fisher/clean > 0.3` → ewc_lambda too large, try 500
- Forgetting >> EWC-LoRA ref → increase ewc_lambda or decrease lambda_flat

Suggested sweep grid per problematic dataset:

| ewc_lambda | Notes |
|-----------|-------|
| 500 | Weak Fisher constraint |
| 2000 | Default (Phase 1) |
| 5000 | Strong Fisher constraint |

Add configs in `exp_paperA_ewclora_paper4_method3/` with `_lam500_` / `_lam5000_` suffix.

### Phase 3: rho sweep (if plasticity is hurt)

If `CNN A` (current-task accuracy) drops vs baseline:
- Reduce `gam_grad_rho: 0.1` or `0.05`

DomainNet in particular may need lower rho (large per-batch gradient scale due to rank=30).

### Phase 4: Multi-seed (for paper table)

5 seeds: `[0, 42, 521, 1024, 1993]`. Required for reporting mean ± std matching EWC-LoRA format.

---

## Phase 1 Run Log (2026-05-02, seed=1993)

### Launch command used

```bash
PYTHON_BIN="/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python" \
GPUS_PAPER4="0 1 2 3" \
SEEDS_JSON='[1993]' \
bash scripts/exp_paperA/run_paperA_ewclora_paper4_method3.sh
```

### Per-dataset status

| Dataset | GPU | Status | Note |
|---------|-----|--------|------|
| CIFAR-100 | 0 | **running** | Task 0, ~2 min/epoch, est. ~40 min total |
| DomainNet | 1 | **FAILED** | FileNotFoundError: `/data/140-0/datasets/DomainNet/splits/domainnet_train.yaml` not found; disk full (22G avail, ~25G needed) |
| ImageNet-R | 2 | **running** | Task 0, ~67 s/epoch, est. ~12 hr total (50 ep × 10 tasks) |
| ImageNet-A | 3 | **running** | Task 1, ~26 s/epoch, est. ~45 min total |

### Output paths (model_name=ewclora_normfisher_gam, pre-config-update)

> Note: configs were updated to `model_name: "as1_normfisher_gam"` after launch, so this run's output lands under `ewclora_normfisher_gam/`. Future runs will use `as1_normfisher_gam/`.

> Note: `--override seed=[1993]` changes the actual training seed to 1993, but the output **directory** uses seed=0 (from config's `seed: - 0`). The training is reproducible at seed=1993.

```
outputs_logs/logs_inc_lora/ewclora_normfisher_gam/gam_adam/
├── cifar224/0/paper4_as1normfisher_lam2000_cifar100_t10c10_r10_s0/exp_run/10/   ← CIFAR-100 final JSON
├── imagenetr/0/paper4_as1normfisher_lam2000_imagenetr_t10c20_r10_s0/exp_run/20/ ← ImageNet-R final JSON
├── imageneta/0/paper4_as1normfisher_lam2000_imageneta_t10c20_r10_s0/exp_run/20/ ← ImageNet-A final JSON
└── domainnet/0/paper4_as1normfisher_lam2000_domainnet_t5c69_r30_s0/             ← empty (failed)
```

Run logs: `logs_paperA/ewclora_paper4_method3/`
Summary CSV (auto-generated after all complete): `logs_paperA/ewclora_paper4_method3/summary_latest.csv`

### DomainNet resolution options

1. **Request sudo to create** `/data/140-0/datasets/DomainNet/` and download (~25GB needed, disk currently 22G free — need to free space first).
2. **Download to user-writable path** (e.g., `/data/115-2/users/liying/datasets/DomainNet/`) and set env var `DATA_ROOT=/data/115-2/users/liying/datasets` before running.
3. **Skip DomainNet for Phase 1**, add it in Phase 2 after freeing disk.

Preferred: Option 2 — avoids needing admin, run with:
```bash
DATA_ROOT=/data/115-2/users/liying/datasets \
bash data/download_domainnet.sh --root /data/115-2/users/liying/datasets/DomainNet

DATASETS="domainnet" DATA_ROOT=/data/115-2/users/liying/datasets \
GPUS_PAPER4="1" SEEDS_JSON='[1993]' \
bash scripts/exp_paperA/run_paperA_ewclora_paper4_method3.sh
```

---

## Monitoring Checklist per Dataset

After Phase 1, check these from the summary CSV or log:

| Check | Target | Action if failed |
|-------|--------|-----------------|
| `run_status == complete` | all 4 | check log for error/OOM |
| `cnn_FAA ≥ EWC-LoRA ref A` | ≥ 87.91 / 73.46 / 72.86 / 59.89 | try ewc_lambda sweep |
| `cnn_forgetting ≤ EWC-LoRA forgetting` | compare | increase ewc_lambda |
| `fisher/clean ∈ [0.02, 0.20]` | healthy range | adjust ewc_lambda |
| `flat/clean < 50` | not OxfordPet-style outlier | reduce lambda_flat |
| `cos_flat_fisher ≈ 0` | near-zero expected | report; if large → mechanism changed |

---

## Expected Mechanism Findings

Based on fine-grained results, we expect:
- `cos_flat_fisher ≈ −0.01 to −0.05` — near-orthogonality should hold on these datasets too
- `fisher/clean` — will vary by dataset; CIFAR-100/ImageNet should behave like CUB200 range
- DomainNet: large tasks (69 classes), Fisher may be noisier due to short training (5 epochs)

---

## Reporting Target (Paper Table)

Final table format matching EWC-LoRA Table 2:

| Method | CIFAR-100 A / Avg | DomainNet A / Avg | ImageNet-R A / Avg | ImageNet-A A / Avg |
|--------|------------------|------------------|-------------------|-------------------|
| SeqLoRA | — | — | — | — |
| EWC-LoRA | 87.91 / 92.27 | 73.46 / 79.58 | 72.86 / 78.95 | 59.89 / 68.33 |
| SD-LoRA | — | — | — | — |
| InfLoRA | — | — | — | — |
| **Ours (AS^1+NormFisher)** | **?** / **?** | **?** / **?** | **?** / **?** | **?** / **?** |

Fill in after Phase 1 (single seed) and Phase 4 (mean ± std).
