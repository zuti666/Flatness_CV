## Usage Guide

### Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main experiment script:
   ```bash
   bash scripts/exp1_script/run_q1_seed.sh
   ```
   The corresponding settings live in `exp_1_sam-sgd_imageR_r16_t20_seed_42/`. You can change the seed, dataset, LoRA method, and optimizer there.

### Ablation Experiments
Run the length ablation script:
```bash
bash scripts/exp4_script/run_imagenet-r_length_r16_l40_cosine_1993.sh
```
The settings are in `exp4_ablation_length/`. Edit the seed, dataset, LoRA method, and optimizer in those YAML files.

### Evaluation
Evaluation utilities are in `eval_flat/eval_flatness_weight_Loss.py`. You can switch on/off specific metrics by editing the config files used in your experiment.

## Methods Overview

### LoRA-Based Continual Learning (loraCL)
The LoRA-based CL methods are implemented in `loraCL/`:
- `seqlora`: sequential LoRA adapters per task.
- `inclora`: incremental LoRA variant for class-incremental learning.
- `olora`:  LoRA-based continual learning strategy.

### Sharpness/Optimizer Methods (optimer)
Sharpness-related optimizers are implemented in `optimer/`:
- `sam`: Sharpness-Aware Minimization.
- `rwp`: robust weight perturbation style updates.
- `gam`: gradient-regularized sharpness variant.

### Flatness Evaluation (eval_flat)
`eval_flat/eval_flatness_weight_Loss.py` implements weight-space flatness metrics, including sharpness and Hessian-based statistics. Adjust the config flags to enable the metrics you want to compute.
