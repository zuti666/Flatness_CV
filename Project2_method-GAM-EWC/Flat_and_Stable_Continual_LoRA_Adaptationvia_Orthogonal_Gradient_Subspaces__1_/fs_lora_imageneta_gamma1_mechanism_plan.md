# ImageNet-A Gamma-1 Mechanism Plan

## Motivation

The current best ImageNet-A complete result is:

```text
ewc_gamma=0.99
ewc_lambda=2000
FAA=54.124
AAA=61.852
Task0_final=69.71
Forget=6.603
```

It improves over `ewc_gamma=0.9, ewc_lambda=2000`, where `FAA=53.813` and `Forget=6.914`.
This suggests that extending the Fisher memory horizon helps old-task retention.

In the current implementation, Fisher accumulation is:

```text
F_accum <- gamma * F_old + F_new
```

This is not a normalized EMA. Therefore `gamma=1.0` means old Fisher evidence is never decayed.
At task 9, task-0 Fisher retention changes from `0.99^9 = 0.914` to `1.0`.
The total old-Fisher mass changes from `sum_{k=0}^8 0.99^k = 8.648` to `9.0`.

## Configs

### G1-Control

```text
ewc_gamma=1.0
ewc_lambda=2000
init_lr=0.04
lrate=0.04
optimizer_type=gam
rho=0.2
norm_rho=0.2
gam_grad_gamma=0.1
```

Purpose: strict test of replacing `0.99` with `1.0`.

### G1-Balanced, Recommended

```text
ewc_gamma=1.0
ewc_lambda=1900
init_lr=0.04
lrate=0.04
optimizer_type=gam
rho=0.2
norm_rho=0.2
gam_grad_gamma=0.1
```

Purpose: preserve the no-decay memory horizon while keeping the total Fisher constraint close to
the current best `gamma=0.99, lambda=2000` setting.

## Files

```text
config_kl_mechanism_imageneta_g1/as2_normfisher_kl_lam2000_ewcg100_imageneta_t10c20_r10_sgd_lr004_e20_rho02_nr02_g01.yaml
config_kl_mechanism_imageneta_g1/as2_normfisher_kl_lam1900_ewcg100_imageneta_t10c20_r10_sgd_lr004_e20_rho02_nr02_g01.yaml
config_kl_mechanism_imageneta_g1/run_g1_kl_mechanism_imageneta.sh
config_kl_mechanism_imageneta_g1/summarize_g1_kl_mechanism_imageneta.py
```

Run:

```bash
cd /data/140-0/users/liying/Flatness_CV
GPUS="3 4" bash config_kl_mechanism_imageneta_g1/run_g1_kl_mechanism_imageneta.sh
```

Summarize:

```bash
cd /data/140-0/users/liying/Flatness_CV
/home-local/liying/.conda/envs/Pilot_new_local/bin/python \
  config_kl_mechanism_imageneta_g1/summarize_g1_kl_mechanism_imageneta.py
```

Outputs:

```text
outputs_logs/config_kl_mechanism_imageneta_g1
logs_kl_mechanism_imageneta_g1
```

## Mechanism Check

The mechanism is supported if:

```text
Task0_final improves or Forget decreases
FAA does not drop materially
cos(flat,fisher) remains close to 0
kl_fisher_delta and kl_fisher_to_iso_ratio are controlled relative to gamma=0.99
fisher/clean increases only moderately
```

Interpretation:

```text
If gamma=1 improves retention while flat/fisher cosine stays near zero,
the result supports the claim that performance changes come from the Fisher-KL
memory channel, not from coupling flatness and Fisher inside one gradient.
```

The mechanism is not supported by this experiment if:

```text
Forget does not improve
kl_fisher_to_iso_ratio does not decrease or becomes unstable
fisher/clean grows sharply and FAA drops
```

That would indicate that the remaining forgetting is not solved by preserving cumulative Fisher
mass alone, and that the next mechanism to test should be per-task Fisher snapshots or adaptive
Fisher-KL budgets rather than another global gamma sweep.
