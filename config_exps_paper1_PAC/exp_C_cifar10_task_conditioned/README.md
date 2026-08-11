# exp_C_cifar10_task_conditioned

SeqLoRA CIFAR10 two-task diagnostic for task-conditioned, support-conditioned flatness.

Task split:

```text
T1 = classes 0..4
T2 = classes 5..9
```

The dataset key is `cifar10_224`: it uses CIFAR10 data with 224-sized ViT transforms so it is compatible with `vit_base_patch16_224`.

Main hypothesis:

```text
Sharpness is indexed by evaluated task loss and perturbation support.
The forgetting-relevant quantity is old-task loss sharpness along the future/current effective LoRA update support.
```

Enabled post-hoc support metrics:

- `Sh_param_full`
- `Sh_AB`
- `Sh_Delta/W_tangent`
- `Sh_rand/W_tangent`
- `Sh_frozen_coords` with `flat_eval_frozen_sharpness_match_delta_dim: true`

Enabled task-conditioned outputs:

```text
flatness_task_conditioned/*_theta_t00_loss_task00_metrics.json
flatness_task_conditioned/*_theta_t01_loss_task00_metrics.json
flatness_task_conditioned/*_theta_t01_loss_task01_metrics.json
```

Run the causal support variants:

```bash
bash config_exps_paper1_PAC/exp_C_cifar10_task_conditioned/run_cifar10_task_conditioned.sh
```

Run the support-matched random perturbation variants:

```bash
bash config_exps_paper1_PAC/exp_C_cifar10_task_conditioned/run_cifar10_random_scopes.sh
```

Naming note: `sam_random` is SAM in a random matched tangent support. It is not the same as `random_delta`, which uses the real Delta support but replaces the SAM ascent direction with a normalized Gaussian direction.

For a cheaper first pass:

```bash
EXP_C_VARIANTS="sgd sam_delta sam_random sam_frozen" EXP_C_GPUS="0 1 2 3" \
bash config_exps_paper1_PAC/exp_C_cifar10_task_conditioned/run_cifar10_task_conditioned.sh
```

After runs finish:

```bash
python config_exps_paper1_PAC/exp_C_cifar10_task_conditioned/summarize_cifar10_task_conditioned.py
```

This writes:

- `outputs_logs/exp_C_cifar10_task_conditioned_summary/summary_by_variant.csv`
- `outputs_logs/exp_C_cifar10_task_conditioned_summary/task_conditioned_flatness_long.csv`

Key columns:

- `loss_forget_1_2`: `L_{T1}(theta_2) - L_{T1}(theta_1)`.
- `acc_forget_1_2`: `Acc_{T1}(theta_1) - Acc_{T1}(theta_2)`.
- `task_cond_gap_delta_theta2`: relative gap between `Sh_{T1}^{Delta}(theta_2)` and `Sh_{T2}^{Delta}(theta_2)`.
