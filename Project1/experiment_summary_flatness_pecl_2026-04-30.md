# LoRA PECL Flatness Experiments Summary

Date: 2026-04-30  
Code root: `/data/140-0/users/liying/Flatness_CV`

## 1. Current Experimental Claim

当前实验不应表述为“提出一个新的 optimizer 并 beating baselines”，而应表述为：

> Empirical analysis of bound-relevant flatness in LoRA-based PECL.

核心问题是：

1. **trajectory effect**: sharpness-aware optimization 是否只提升当前任务，还是会通过 posterior trajectory 影响后续 retention / forgetting。
2. **support-direction decomposition**: 改善来自 perturbation support、adversarial SAM direction，还是普通 random noise。
3. **LoRA geometry**: raw factor-space sharpness 是否可靠，effective adapter-update / tangent-space sharpness 是否更接近理论对象。

目前最强证据来自 ImageNet-R r=16 T=20 的两个主实验：

- Exp E: support x direction 主实验，比较 `sam_*` 与 `random_*`。
- Exp F: forked trajectory 主实验，从同一 task-9 checkpoint 分叉，比较后续 SGD / SAM-factor / random-factor。

## 2. Key Implementation Meanings

所有 scoped SAM / random 设置最终更新对象都是 LoRA A/B + classifier head；区别只在训练时临时 perturbation 的位置和方向。

| Variant | Perturbation direction | Perturbation support |
|---|---|---|
| `sam_factor` | normalized adversarial gradient | raw LoRA factors `(A,B)` |
| `sam_full` | normalized adversarial gradient | merged effective qkv full weight |
| `sam_delta` | normalized adversarial gradient | LoRA effective update tangent `T_Delta = {B dA + dB A}` |
| `sam_all` | normalized adversarial gradient | raw all-parameter coordinates |
| `sam_frozen` | normalized adversarial gradient | frozen backbone coordinates |
| `random_factor` | normalized Gaussian random direction | raw LoRA factors `(A,B)` |
| `random_full` | normalized Gaussian random direction | merged effective qkv full weight |
| `random_delta` | normalized Gaussian random direction | LoRA effective update tangent |
| `random_all` | normalized Gaussian random direction | raw all-parameter coordinates |
| `random_frozen` | normalized Gaussian random direction | frozen backbone coordinates |

Important naming:

- `sam_random` should **not** be called `random_delta`.
- `sam_random` means SAM gradient direction projected onto a random matched tangent support.
- `random_delta` means Gaussian random direction inside Delta tangent support.

Mathematically:

```text
SAM:    epsilon_S = rho * Pi_S grad L / ||Pi_S grad L||
Random: z_S ~ N_S(0,I), epsilon_S = rho * z_S / ||z_S||
```

## 3. CIFAR10 Two-task Diagnostics

### Exp C: task-conditioned support pilot

Path:

```text
config_exps_paper1_PAC/exp_C_cifar10_task_conditioned
outputs_logs/exp_C_cifar10_task_conditioned_summary/summary_by_variant.csv
```

Main result:

| Variant | FAA | AAA | BWT / Forget |
|---|---:|---:|---:|
| `sgd` | 96.78 | 97.52 | -3.54 / 3.54 |
| `sam_factor` | 96.92 | 97.59 | -2.76 / 2.76 |
| `sam_delta` | 96.96 | 97.67 | -3.00 / 3.00 |
| `sam_random` | 97.11 | 97.78 | -3.14 / 3.14 |
| `sam_full` | 96.83 | 97.57 | -3.78 / 3.78 |
| `sam_frozen` | 96.33 | 97.25 | -4.86 / 4.86 |

Interpretation:

- `sam_factor` has the best forgetting / BWT.
- `sam_random` has the best FAA / AAA in this small two-task pilot.
- `sam_frozen` is clearly worse, so frozen-only is a useful negative control.
- This pilot is useful mechanistically, but too small for the final main claim.

### Exp D: two-task trajectory factorial diagnostic

Path:

```text
config_exps_paper1_PAC/exp_D_taskwise_sam_trajectory
outputs_logs/exp_D_taskwise_sam_trajectory_summary/summary_by_variant.csv
outputs_logs/exp_D_taskwise_sam_trajectory_summary/contrasts_by_seed.csv
```

Main result:

| Variant | A_1,1 | A_1,2 | A_2,2 | FinalAvg | Forget |
|---|---:|---:|---:|---:|---:|
| `sgd_sgd` | 98.98 | 95.64 | 97.96 | 96.80 | 3.34 |
| `sam_factor_sgd` | 98.92 | 95.70 | 98.24 | 96.97 | 3.22 |
| `sgd_sam_factor` | 98.98 | 96.00 | 97.58 | 96.79 | 2.98 |
| `sam_factor_sam_factor` | 98.92 | 96.16 | 97.72 | 96.94 | 2.76 |
| `sgd_random_factor` | 98.98 | 95.66 | 97.94 | 96.80 | 3.32 |
| `random_factor_random_factor` | 98.98 | 95.66 | 97.96 | 96.81 | 3.32 |

Key contrasts:

| Perturbation | Delta T1 flat A_1,2 | Delta T2 protect A_1,2 | Interaction A_1,2 |
|---|---:|---:|---:|
| `random_factor` | +0.02 | +0.02 | -0.02 |
| `sam_factor` | +0.06 | +0.36 | +0.10 |

Interpretation:

- Applying SAM only at Task 2 improves old-task retention more than applying SAM only at Task 1.
- This supports a posterior-trajectory interpretation: SAM affects the later update path, not only the local generalization of the task where it is applied.
- Random-factor does not reproduce the effect, so the benefit is not generic noise injection.

## 4. ImageNet-R Main Experiment 1: Support x Direction

Experiment:

```text
config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction
```

Results:

```text
outputs_logs/exp_E_imagenetr_r16_t20_support_direction_summary/summary_by_variant.csv
outputs_logs/exp_E_imagenetr_r16_t20_support_direction_summary/figures/support_direction_pairwise.csv
outputs_logs/exp_E_imagenetr_r16_t20_support_direction_summary/figures/support_direction_faa_bwt.png
outputs_logs/exp_E_imagenetr_r16_t20_support_direction_summary/figures/support_direction_faa_bwt.pdf
```

Main table:

| Variant | FAA | AAA | BWT | Forget |
|---|---:|---:|---:|---:|
| `sgd` | 58.29 | 67.05 | -18.99 | 19.09 |
| `sam_factor` | 67.94 | 70.60 | -6.81 | 7.10 |
| `sam_full` | 64.67 | 69.04 | -12.01 | 12.04 |
| `sam_delta` | 65.41 | 69.63 | -11.56 | 11.66 |
| `sam_all` | 63.19 | 67.55 | -12.28 | 12.28 |
| `sam_frozen` | 62.91 | 67.29 | -12.37 | 12.46 |
| `random_factor` | 58.79 | 67.12 | -18.47 | 18.57 |
| `random_full` | 58.15 | 67.01 | -19.24 | 19.34 |
| `random_delta` | 58.26 | 67.04 | -19.08 | 19.18 |
| `random_all` | 58.32 | 67.04 | -19.01 | 19.11 |
| `random_frozen` | 58.24 | 67.05 | -19.10 | 19.20 |

Pairwise support comparison:

| Support | SAM FAA | Random FAA | Delta FAA | SAM BWT | Random BWT | Delta BWT |
|---|---:|---:|---:|---:|---:|---:|
| factor | 67.94 | 58.79 | +9.14 | -6.81 | -18.47 | +11.67 |
| full | 64.67 | 58.15 | +6.52 | -12.01 | -19.24 | +7.23 |
| delta | 65.41 | 58.26 | +7.15 | -11.56 | -19.08 | +7.52 |
| all | 63.19 | 58.32 | +4.87 | -12.28 | -19.01 | +6.72 |
| frozen | 62.91 | 58.24 | +4.67 | -12.37 | -19.10 | +6.73 |

Interpretation:

- For every support, SAM direction is much better than Gaussian random direction.
- `random_*` is very close to SGD, so simply injecting noise does not explain the improvement.
- `sam_factor` is strongest in this current implementation, both in FAA and BWT.
- `sam_delta` improves strongly over SGD/random, but does not beat `sam_factor`.
- This result supports the direction claim: sharpness-aware adversarial perturbation matters.
- It does **not by itself** prove that raw factor-space is the correct theoretical object; raw-factor rescaling control is still required.

## 5. ImageNet-R Main Experiment 2: Forked Trajectory

Experiment:

```text
config_exps_paper1_PAC/exp_F_imagenetr_r16_t20_forked_taskwise
```

Design:

- Train prefix `Task 0-9` once.
- Copy the same task-9 checkpoint into multiple branches.
- Continue `Task 10-19` with different suffix optimizers.

This ensures:

- `sgd_sgd`, `sgd_sam_factor`, `sgd_random_factor` start from the exact same SGD task-9 checkpoint.
- `sam_factor_sgd`, `sam_factor_sam_factor` start from the exact same SAM-factor task-9 checkpoint.

Checkpoint hash verification:

- The SGD-prefix branch task-9 `lora_w_a_9`, `lora_w_b_9`, and `fc_state_9` hashes are identical across `sgd_sgd`, `sgd_sam_factor`, `sgd_random_factor`.
- The SAM-prefix branch task-9 checkpoint hashes are identical across `sam_factor_sgd`, `sam_factor_sam_factor`.

Results:

```text
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/summary_official_and_prefix.csv
```

Figures:

```text
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/figures/sgd_prefix_time_curves.png
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/figures/sam_prefix_time_curves.png
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/figures/all_variants_time_curves.png
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/figures/sgd_prefix_final_forgetting.png
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/figures/sam_prefix_final_forgetting.png
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/figures/all_variants_final_forgetting.png
```

Main table:

| Variant | FAA | AAA | BWT | Forget | Prefix Forget | Suffix Acc |
|---|---:|---:|---:|---:|---:|---:|
| `sgd_sgd` | 61.72 | 67.60 | -15.32 | 15.49 | 17.19 | 64.33 |
| `sgd_sam_factor` | 66.38 | 69.51 | -9.97 | 10.17 | 11.03 | 67.49 |
| `sgd_random_factor` | 61.72 | 67.59 | -15.32 | 15.50 | 17.19 | 64.33 |
| `sam_factor_sgd` | 65.29 | 69.84 | -10.38 | 10.72 | 11.87 | 66.46 |
| `sam_factor_sam_factor` | 67.50 | 70.42 | -7.94 | 8.24 | 8.72 | 67.72 |

Interpretation:

- From the same SGD prefix, suffix SAM-factor is much better than suffix SGD:
  - FAA: `61.72 -> 66.38`
  - BWT: `-15.32 -> -9.97`
  - prefix forgetting: `17.19 -> 11.03`
- From the same SAM prefix, continuing SAM-factor is better than switching to SGD:
  - FAA: `65.29 -> 67.50`
  - BWT: `-10.38 -> -7.94`
  - prefix forgetting: `11.87 -> 8.72`
- `sgd_random_factor` is essentially identical to `sgd_sgd`.
- This is the strongest current evidence for the trajectory-level claim:

> Sharpness-aware optimization is not only a task-local effect; when applied to later updates, it changes posterior movement and reduces trajectory-level forgetting.

## 6. What Current Experiments Support

Supported:

1. **Perturbation direction matters.**
   - Exp E: SAM direction beats Gaussian random direction for every support.
   - Exp F: random-factor does not reproduce SAM-factor retention gains.

2. **The effect is trajectory-level, not merely task-local.**
   - Exp D and Exp F show that applying SAM in later tasks improves old-task retention.

3. **Negative controls are useful.**
   - Frozen-only and random perturbations do not explain the full effect.

4. **ImageNet-R r16 T20 is a stronger main setting than CIFAR10 two-task.**
   - T=20 gives enough sequence length for BWT trajectories and final per-task forgetting curves.

Not fully supported yet:

1. **Raw factor-space is not the correct theoretical object.**
   - Current results show `sam_factor` is empirically strong.
   - To argue against raw factor sharpness, we still need reparameterization / rescaling control.

2. **Sh_Delta is invariant and better aligned with the PAC-Bayes object.**
   - This needs post-hoc rescaling:

```text
(A, B) -> (A/c, cB)
Delta W = B A unchanged
Sh_AB changes
Sh_Delta should remain stable
```

3. **Cross-method LoRA organization claim.**
   - Current main experiments are SeqLoRA only.
   - IncLoRA / OLoRA should be used later for geometry robustness, not as first-line causal proof.

## 7. Recommended Paper Mapping

Suggested empirical section:

```text
6. Empirical Analysis of Bound-Relevant Flatness

6.1 Setup
ImageNet-R r=16 T=20, SeqLoRA, ViT-B/16, no replay, seed 1993.

6.2 RQ1: Does sharpness-aware optimization affect posterior trajectories?
Use Exp F forked trajectory.

6.3 RQ2: Is the effect due to adversarial sharpness direction rather than random noise?
Use Exp E SAM-vs-random support-direction comparison.

6.4 RQ3: Which perturbation support is empirically most effective?
Use Exp E support comparison: factor/full/delta/all/frozen.

6.5 RQ4: Is raw factor-space sharpness parameterization-dependent?
Add rescaling control. This remains to be completed.

6.6 Appendix: CIFAR10 two-task diagnostics
Use Exp C and Exp D as mechanism pilots.
```

## 8. Next Experiments

### Next 1: Raw-factor rescaling control

Purpose:

```text
Prove Sh_AB is parameterization-dependent, while Sh_Delta is effective-update invariant.
```

Setting:

```text
Dataset: ImageNet-R
Method: SeqLoRA
Rank: 16
Checkpoint: task 9 or task 19 from Exp E/Exp F
Scales: c in {0.25, 0.5, 1, 2, 4}
Transform: A' = A / c, B' = c B
Metrics: Sh_AB, Sh_Delta/W_tangent, model accuracy unchanged
```

Expected:

```text
Sh_AB varies with c.
Sh_Delta remains stable.
Accuracy and Delta W remain unchanged.
```

### Next 2: Repeat key ImageNet-R results with more seeds

Minimal seed extension:

```text
Seeds: 1993, 42, 1234
Variants:
sgd
sam_factor
sam_delta
random_factor
random_delta
```

Reason:

- Current Exp E/F are single-seed but strong.
- More seeds are needed for final paper robustness.

### Next 3: Post-hoc sharpness metrics on selected checkpoints

Do not run full flatness on all variants first. Start with:

```text
Exp E variants:
sgd
sam_factor
sam_delta
random_factor
random_delta

Tasks:
0, 4, 9, 14, 19

Metrics:
Sh_param_full
Sh_AB
Sh_Delta/W_tangent
Sh_rand/W_tangent
Sh_frozen_coords
```

Purpose:

- Connect performance improvements to measured sharpness.
- Avoid the earlier noisy issue from shuffled loader / train transform.

Implementation requirements:

```text
flat_eval_loader_shuffle: false
flat_eval_data_mode: test
fixed deterministic subset
same batch for base loss and perturbed loss
sharpness should be non-negative
```

### Next 4: Cross-LoRA structure robustness

Only after SeqLoRA story is stable:

```text
Methods: SeqLoRA, IncLoRA, OLoRA
Dataset: ImageNet-R r=16 T=20
Variants: sgd, sam_factor, sam_delta
```

Purpose:

- Show the relevant support changes with LoRA organization.
- Do not use this as the first causal proof.

## 9. Most Important Takeaway

The current strongest evidence is:

```text
ImageNet-R Exp F:
same checkpoint fork + suffix SAM-factor improves retention,
while suffix random-factor equals SGD.
```

This directly supports:

```text
sharpness-aware optimization affects posterior trajectories,
not merely current-task local generalization.
```

The current strongest negative-control evidence is:

```text
ImageNet-R Exp E:
SAM direction beats Gaussian random direction under the same support.
```

This supports:

```text
the gain is not from generic noise injection.
```

The remaining theoretical gap is:

```text
show Sh_AB is not invariant under equivalent LoRA factorization,
while Sh_Delta/W_tangent is stable.
```

