# Paper4 Method3 Experiment Configuration Summary

**Last checked**: 2026-05-03  
**Log directory**: `logs_paperA/ewclora_paper4_method3/`  
**Config directory**: `config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3/`  
**Run script**: `scripts/exp_paperA/run_paperA_ewclora_paper4_method3.sh`

This page records the current AS^1 + trace-normalized Fisher configuration for the four vision datasets used by the EWC-LoRA paper: CIFAR-100, DomainNet, ImageNet-R, and ImageNet-A.

---

## Key Correction

The newest real logs in `logs_paperA/ewclora_paper4_method3/` were generated before the latest config fix. They used:

```yaml
model_name: "as1_normfisher_gam"
```

This is not a valid LoRA training configuration in this codebase. The trainer and backbone builder use `"lora" in model_name` to select the LoRA backbone and output path. With `as1_normfisher_gam`, the run uses the full ViT backbone path, stores outputs under `logs_inc/`, and the Fisher/Delta-W mechanism becomes inactive (`delta_tensors=0`, `fisher/clean=0`).

The corrected configs now use:

```yaml
model_name: "ewclora_normfisher_gam"
```

Future runs should output to:

```text
outputs_logs/logs_inc_lora/ewclora_normfisher_gam/gam_adam/{dataset}/{seed}/{prefix}/exp_run/{increment}/
```

The run script now also guards against configs whose `model_name` does not contain `lora`.

---

## Current Corrected Configs

| Dataset | Config | Repo key | Split | Rank | Epochs/task | LR | Weight decay |
|---|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | `as1_normfisher_lam2000_cifar100_t10c10_r10.yaml` | `cifar224` | 10 x 10 | 10 | 20 | 0.0005 | 0.0 |
| DomainNet | `as1_normfisher_lam2000_domainnet_t5c69_r30.yaml` | `domainnet` | 5 x 69 | 30 | 5 | 0.0005 | 0.0 |
| ImageNet-R | `as1_normfisher_lam2000_imagenetr_t10c20_r10.yaml` | `imagenetr` | 10 x 20 | 10 | 50 | 0.0005 | 0.005 |
| ImageNet-A | `as1_normfisher_lam2000_imageneta_t10c20_r10.yaml` | `imageneta` | 10 x 20 | 10 | 10 | 0.0005 | 0.0 |

Common dataset/training settings:

| Key | Value |
|---|---|
| `seed` | `[0]` by default |
| `class_shuffle` | `false` |
| `backbone_type` | `vit_base_patch16_224` |
| `batch_size` | `128` |
| `scheduler` | `cosine` |
| `init_epoch` / `epochs` | same per dataset |
| `init_lr` / `lrate` | same per dataset |
| `init_weight_decay` / `weight_decay` | same per dataset |
| replay memory | disabled (`memory_size=0`) |

---

## Optimizer and Method Hyperparameters

All four corrected configs use Adam as the base optimizer inside GAM:

| Key | Value |
|---|---:|
| `optimizer` | `adam` |
| `optimizer_type` | `gam` |
| `optimizer_tag_override` | `gam_adam` |
| `adam_beta1` | `0.9` |
| `adam_beta2` | `0.99` |
| `gam_adaptive` | `false` |
| `gam_grad_rho` | `0.2` |
| `gam_grad_norm_rho` | `0.2` |
| `gam_grad_beta_1` | `1.0` |
| `gam_grad_beta_2` | `1.0` |
| `gam_grad_beta_3` | `1.0` |
| `gam_grad_gamma` | `0.1` |

Method-specific settings:

| Key | Value | Meaning |
|---|---:|---|
| `lambda_flat` | `1.0` | AS^1 flatness weight |
| `ewc_lambda` | `2000` | normalized Fisher penalty weight |
| `ewc_gamma` | `0.9` | past Fisher decay |
| `ewc_max_batches` | `0` | full current-task train loader for Fisher estimation |
| `ewc_normalize_fisher` | `true` | trace-normalize Fisher |
| `ewc_dual_ascent` | `false` | fixed multiplier, no adaptive dual update |
| `mechanism_eval` | `true` | record mechanism diagnostics |
| `mechanism_grad_stats` | `true` | record gradient cosine/norm ratios |

Implementation note: `models_CL/baseLearner.py` was changed so the `optimizer_type: gam` branch reads `optimizer: adam/adamw/sgd` instead of always constructing an SGD base optimizer. These configs therefore run GAM with Adam base optimizer.

---

## Latest Log Status

Latest log files currently present:

| Log | Dataset | Seed | Printed model | Status | Last task seen | Last CNN avg | Notes |
|---|---|---:|---|---|---|---:|---|
| `as1_normfisher_lam2000_cifar100_t10c10_r10_gpu0.log` | CIFAR-100 | 1993 | `as1_normfisher_gam` | partial | `60-70` | 23.8067 | invalid full-ViT run; stopped with multiprocessing tmp finalizer errors |
| `as1_normfisher_lam2000_domainnet_t5c69_r30_gpu1.log` | DomainNet | 1993 | `as1_normfisher_gam` | partial | `69-138` | 26.13 | invalid full-ViT run; stopped with multiprocessing tmp finalizer errors |
| `as1_normfisher_lam2000_imagenetr_t10c20_r10_gpu2.log` | ImageNet-R | 1993 | `as1_normfisher_gam` | partial | `80-100` | 10.15 | invalid full-ViT run; stopped with multiprocessing tmp finalizer errors |
| `as1_normfisher_lam2000_imageneta_t10c20_r10_gpu3.log` | ImageNet-A | 1993 | `as1_normfisher_gam` | complete | `180-200` | 1.913 | invalid full-ViT run; discard |
| `as1_normfisher_lam2000_cifar100_t10c10_r10_gpu5.log` | CIFAR-100 | 0 | `ewclora_normfisher_gam` | complete | `90-100` | 81.268 | valid Method3 run |
| `as1_normfisher_lam2000_domainnet_t5c69_r30_gpu5.log` | DomainNet | 0 | `ewclora_normfisher_gam` | failed early | none | n/a | failed because DomainNet split did not exist at run time |
| `as1_normfisher_lam2000_imagenetr_t10c20_r10_gpu5.log` | ImageNet-R | 0 | `ewclora_normfisher_gam` | partial | `20-40` | 93.24 | valid Method3 start, incomplete |

Conclusion: only the CIFAR-100 seed-0 log is a complete valid Method3 run. The seed-1993 logs should not be used for method comparison because they used `as1_normfisher_gam` instead of the LoRA alias.

The old DomainNet failure was:

```text
FileNotFoundError: DomainNet/splits/domainnet_train.yaml
```

The file now exists at:

```text
data/DomainNet/splits/domainnet_train.yaml
```

---

## Recommended Re-run Command

Use the corrected configs and keep temporary directories during the run to avoid multiprocessing finalizer noise:

```bash
KEEP_RUN_TMP=1 GPUS_PAPER4="0 1 2 3" \
  bash scripts/exp_paperA/run_paperA_ewclora_paper4_method3.sh
```

Single-dataset checks:

```bash
KEEP_RUN_TMP=1 DATASETS="cifar100" GPUS_PAPER4="0" \
  bash scripts/exp_paperA/run_paperA_ewclora_paper4_method3.sh

KEEP_RUN_TMP=1 DATASETS="domainnet" GPUS_PAPER4="1" \
  bash scripts/exp_paperA/run_paperA_ewclora_paper4_method3.sh
```

Multi-seed run for paper-style mean/std:

```bash
KEEP_RUN_TMP=1 SEEDS_JSON='[0,42,521,1024,1993]' GPUS_PAPER4="0 1 2 3" \
  bash scripts/exp_paperA/run_paperA_ewclora_paper4_method3.sh
```

Dry run:

```bash
DRY_RUN=1 DATASETS="cifar100 domainnet imagenetr imageneta" GPUS_PAPER4="0 1 2 3" \
  bash scripts/exp_paperA/run_paperA_ewclora_paper4_method3.sh
```

---

## Reference EWC-LoRA Paper Setting

The corrected configs align the dataset split, rank, epoch count, batch size, class order, and Adam learning rate with the EWC-LoRA paper's vision setting:

| Dataset | Paper split | Paper rank | Paper epochs | Paper LR |
|---|---:|---:|---:|---:|
| CIFAR-100 | 10 x 10 | 10 | 20 | 0.0005 |
| DomainNet | 5 x 69 | 30 | 5 | 0.0005 |
| ImageNet-R | 10 x 20 | 10 | 50 | 0.0005 |
| ImageNet-A | 10 x 20 | 10 | 10 | 0.0005 |

The main methodological difference is that our optimizer is GAM with Adam as the base optimizer, while EWC-LoRA itself is Adam without AS^1.
