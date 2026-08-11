# exp_B_imager_r16_t20_seed1993

Controlled SeqLoRA/ImageNet-R/rank16/20-task/seed1993 perturbation-scope experiment.

Goal: this is a causal probe for bound-relevant flatness, not a new-method benchmark. All variants keep the same dataset, task order, rank, epochs, learning rate, optimizer base, and trainable parameters. The controlled variable is only the perturbation support used for the sharpness-aware second loss.

Primary RQ:

```text
Does the support of the training-time perturbation causally affect old-task forgetting?
```

Primary comparison:

```text
sgd
vs sam_factor
vs sam_full
vs sam_delta
vs sam_random
vs sam_frozen
vs sam_all
```

Main outcomes:

- FAA/LAA, AAA, BWT from the standard CL accuracy matrix.
- Per-task forgetting, when available from the downstream summary script.
- Post-hoc deterministic sharpness metrics saved by flat eval:
  - `Sh_param_full`
  - `Sh_AB`
  - `Sh_Delta/W_tangent` and alias `Sh_Delta_tangent`
  - `Sh_rand/W_tangent`
  - `Sh_frozen_coords`

Interpretation:

- `sam_delta` is the theory-aligned intervention: adversarial perturbation in the current effective LoRA update tangent.
- `sam_random` controls for dimension/rank: same rank structure, wrong random support.
- `sam_frozen` controls for inaccessible directions: perturbing frozen coordinates should not reliably reduce LoRA PECL forgetting if the theory is correct.
- `sam_factor` tests raw factor-space perturbation, which is parameterization-dependent.
- `sam_full` and `sam_all` test broader ambient-space flatness.

All scoped SAM variants update the ordinary trainable parameters through the base SGD optimizer, i.e. LoRA `A/B` and the classifier head as defined by the existing SeqLoRA model. The intervention only changes where the first SAM ascent perturbation is applied.

Variants:

- `sgd`: no sharpness-aware perturbation.
- `sam_factor`: raw LoRA factor-space perturbation. This corresponds to evaluating the second loss at `W + (B+eps_B)(A+eps_A)`.
- `sam_full`: full effective qkv weight perturbation. This corresponds to `W_eff + eps`, where `W_eff = W_qkv + Delta W`; the perturbation is applied to the underlying qkv weight so the forward function sees the merged effective weight plus `eps`.
- `sam_delta`: effective adapter-update tangent perturbation. This uses the q/v rows and projects the qkv-weight gradient onto the current LoRA tangent support `{B dA + dB A}` before the second loss.
- `sam_random`: random matched tangent control. This uses fixed random row/column bases with the same LoRA rank as the current q/v adapters, then projects the qkv-weight gradient into that random tangent-like support before the second loss.
- `sam_frozen`: frozen-coordinate negative control. Frozen backbone parameters are temporarily enabled only to compute and apply the ascent perturbation, then restored before the second loss; the optimizer still updates only LoRA `A/B` and the classifier head.
- `sam_all`: raw all-parameter perturbation. Frozen parameters are temporarily enabled for the ascent perturbation, then restored; the optimizer still updates only the ordinary trainable parameters.

Relation to FlatLoRA:

- The existing `flatlora_full` branch perturbs the same effective qkv weight location as `sam_full` by adding random/generated noise to `module.qkv.weight`.
- `sam_full` differs because the perturbation direction is SAM-style gradient ascent rather than random noise.
- `sam_delta` is more restricted than `sam_full`: it perturbs only the effective LoRA update tangent support in q/v rows.

Run:

```bash
bash config_exps_paper1_PAC/exp_B_imager_r16_t20_seed1993/run_exp_B_sam_scopes.sh
```

Environment overrides:

```bash
EXP_B_VARIANTS="sgd sam_factor sam_full sam_delta sam_random sam_frozen" EXP_B_GPUS="0 1 2 3 4 5" bash config_exps_paper1_PAC/exp_B_imager_r16_t20_seed1993/run_exp_B_sam_scopes.sh
```

The launcher runs at most one job per listed GPU at a time. `sam_all` and `sam_frozen` are substantially heavier because they compute ascent gradients for frozen backbone coordinates.

After runs finish:

```bash
python config_exps_paper1_PAC/exp_B_imager_r16_t20_seed1993/summarize_exp_B.py
```

Support-matched random perturbation variants are also available:

```bash
bash config_exps_paper1_PAC/exp_B_imager_r16_t20_seed1993/run_exp_B_random_scopes.sh
```

These use the same supports as `sam_factor/full/delta/all/frozen`, but replace the SAM ascent direction with a normalized Gaussian direction.

This writes:

- `outputs_logs/exp_B_imager_r16_t20_seed1993_summary/summary_by_variant.csv`
- `outputs_logs/exp_B_imager_r16_t20_seed1993_summary/per_task_flatness.csv`
