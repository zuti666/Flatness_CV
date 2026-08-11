# Experimental Setting

## Problem: PECL (Parameter-Efficient Continual Learning)

Frozen ViT-B/16 backbone + single shared LoRA adapter (A, B) trained sequentially across tasks.
No replay memory. SeqLoRA catastrophically forgets old tasks.

**ΔW = B @ A**: The effective adapter update in (d_out × d_in) space. Used for both Fisher estimation and EWC penalty — this is basis-invariant (does not depend on the factorization A, B separately).

## Standard Setup (all Paper A experiments)

| Config | Value |
|--------|-------|
| Backbone | ViT-B/16 (`vit_base_patch16_224`) |
| LoRA rank | 16 |
| LoRA locations | Q and V projections per transformer block |
| Delta terms | 24 (12 blocks × Q + V) |
| Memory | 0 (no replay) |
| Batch size | 128 |
| Optimizer | SGD |
| Optimizer type | GAM (`optimizer_type: gam`) |
| GAM rho | `as1_rho=0.2`, `as1_norm_rho=0.2` |
| Scheduler | cosine |
| Seed | 0 (default) |

## Datasets

| Dataset | Split | Tasks | Classes/task | Epochs | LR | Notes |
|---------|-------|-------|-------------|--------|-----|-------|
| CUB200 | t20 | 10 | 20 | 40 | 0.01 | Standard paper split |
| ImageNet-R | t20c10 | 20 | 10 | 20 | 0.01 | `class_shuffle=true` |
| Aircraft | t10c10 | 10 | 10 | 40 | 0.05 | `gam_rho=0.01/0.1` |
| Cars196 | t10c20 | 10 | 20 | 40 | 0.05 | `gam_rho=0.05/0.2` |
| Flowers | t10c10 | 10 | 10 | 40 | 0.0025 | |
| OxfordPet | t9c4 | 9 | 4 | 40 | 0.0025 | `dataset: "pets"` |

## Config Directories

| Purpose | Directory |
|---------|-----------|
| Three-method comparison | `config_exps_paper1_PAC/exp_paperA_three_methods/` |
| 2×2 ablation | `config_exps_paper1_PAC/exp_paperA_ablation/` |
| Method 3 lambda sweep | `config_exps_paper1_PAC/exp_paperA_newmethod/` |
| Cross-dataset (Layer 3) | `config_exps_paper1_PAC/exp_paperA_cross_dataset/` |
| EWC-Yaoyue baseline | `config_exps_paper1_PAC/exp_paperA_ewc_yaoyue_baseline/` |

## Log Directories

| Purpose | Directory |
|---------|-----------|
| CUB200 t20 | `logs_paperA/cub200_t20/` |
| Layer 2 sweep | `logs_paperA/cub200_t20_layer2_sweep/` |
| ImageNet-R | `logs_paperA/imagenetr_method3/` |
| Cross-dataset | `logs_paperA/cross_datasets_method3/` |
| EWC-Yaoyue baseline | `logs_paperA/ewc_yaoyue_sgd/` |

## Run Scripts

| Script | Purpose |
|--------|---------|
| `scripts/exp_paperA/run_paperA_cub200_t20.sh` | Three stages: three-method / ablation / lambda sweep |
| `scripts/exp_paperA/run_layer2_rawfisher_sweep_cub200_t20.sh` | Layer 2 ewc_lambda + lambda_flat grid |
| `scripts/exp_paperA/run_paperA_imagenetr_method3.sh` | ImageNet-R, Layer 3 best |
| `scripts/exp_paperA/run_paperA_cross_datasets_method3.sh` | Aircraft/Cars/Flowers/OxfordPet |
| `scripts/exp_paperA/run_paperA_ewc_yaoyue_sgd.sh` | EWC-Yaoyue SGD baseline |

## NFS Warning

`TMPDIR` defaults to NFS path (`/data/115-2/users/liying/cache/tmp`), which causes `.nfs...` cleanup errors with multi-worker DataLoaders. Fixed in all run scripts by setting `TMPDIR=/tmp/...` per process. Always use the run scripts rather than direct `python3 -m src.main`.
