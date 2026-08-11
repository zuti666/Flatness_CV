---
id: exp_F
name: Forked Trajectory Main Experiment
dataset: ImageNet-R r=16 T=20
method: SeqLoRA, ViT-B/16, no replay, seed 1993
status: complete
config: config_exps_paper1_PAC/exp_F_imagenetr_r16_t20_forked_taskwise
results: outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/
---

## Design

Train prefix (Task 0–9) once per optimizer type. Copy task-9 checkpoint. Continue Task 10–19 with different suffix optimizers.

- SGD prefix group: sgd_sgd, sgd_sam_factor, sgd_random_factor — all share the same task-9 checkpoint (hash verified).
- SAM prefix group: sam_factor_sgd, sam_factor_sam_factor — share the same SAM task-9 checkpoint.

Checkpoint hash verification: lora_w_a_9, lora_w_b_9, fc_state_9 are identical within each prefix group.

## Key Results

| Variant | FAA | AAA | BWT | Forget | Prefix Forget | Suffix Acc |
|---------|-----|-----|-----|--------|---------------|------------|
| sgd_sgd | 61.72 | 67.60 | -15.32 | 15.49 | 17.19 | 64.33 |
| sgd_sam_factor | **66.38** | **69.51** | **-9.97** | **10.17** | **11.03** | 67.49 |
| sgd_random_factor | 61.72 | 67.59 | -15.32 | 15.50 | 17.19 | 64.33 |
| sam_factor_sgd | 65.29 | 69.84 | -10.38 | 10.72 | 11.87 | 66.46 |
| sam_factor_sam_factor | **67.50** | **70.42** | **-7.94** | **8.24** | **8.72** | 67.72 |

## Interpretation

**From same SGD prefix (task 0–9 identical)**:
- Suffix SAM-factor: FAA 61.72 → 66.38 (+4.66), Prefix Forget 17.19 → 11.03 (-6.16)
- Suffix random-factor: FAA 61.72 → 61.72 (no change), Prefix Forget 17.19 → 17.19 (no change)
- Conclusion: SAM direction in suffix is what drives the improvement, not noise.

**From same SAM prefix (task 0–9 SAM-trained)**:
- Continuing SAM: FAA 65.29 → 67.50 (+2.21), BWT -10.38 → -7.94
- Switching to SGD: lower FAA, higher forgetting

**Core conclusion** (supports **C5**):
> Sharpness-aware optimization is not only a task-local effect. Applied in later tasks, it changes posterior movement and reduces trajectory-level forgetting, including forgetting of prefix tasks trained with SGD.

sgd_random_factor ≡ sgd_sgd rules out noise as mechanism.

## Figures

- `figures/sgd_prefix_time_curves.png`
- `figures/sam_prefix_time_curves.png`
- `figures/all_variants_time_curves.png`
- `figures/sgd_prefix_final_forgetting.png`
- `figures/sam_prefix_final_forgetting.png`
- `figures/all_variants_final_forgetting.png`

## Supports Claims

- **C5** (trajectory-level effect) — strongest current evidence
- **C4** (direction specificity, via random_factor control)
