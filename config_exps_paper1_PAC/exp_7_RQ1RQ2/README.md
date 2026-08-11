# exp_7_RQ1RQ2

This folder contains the SeqLoRA/ImageNet-R rank-16 configs for the paper RQ1/RQ2 checks.

Strict theory alignment:

- The current RQ1 script measures current-task `Sh_Delta/W_tangent`, not the final paper object `Sh_{i<-t}^Delta`.
- `Sh_param_full` is raw parameter-space sharpness. If the paper table needs full effective merged-weight-space sharpness, add a separate `Sh_W_full` metric rather than reusing this value.
- RQ2 is the cleanest implemented check of the paper claim: `Sh_AB` should change under `(A,B)->(cA,B/c)`, while `Sh_Delta/W_tangent` should remain stable.

## RQ1: which sharpness object predicts forgetting?

Object of study: perturbation space, not optimizer. The same trained SeqLoRA model is evaluated with:

- `Sh_param_full`: raw parameter full-space sharpness. This is a diagnostic baseline and is not the same as merged effective-weight sharpness.
- `Sh_AB`: raw LoRA factor-space sharpness (`Sh^{A,B}`).
- `Sh_Delta/W_tangent`: effective Delta-W tangent-space sharpness (`Sh_t^Delta` local tangent support). `Sh_Delta_tangent` is also written as a script-safe alias.

The legacy keys `sh0_max`, `sh_ab_max`, and `sh_delta_max` are still written for backward compatibility.

SGD and SAM are both included because they produce different trained solutions and forgetting regimes. They are not the controlled variable of the sharpness definition. Keep `rho` fixed across the three sharpness objects in the main table.

Run:

```bash
bash config_exps_paper1_PAC/exp_7_RQ1RQ2/run_rq1_seqlora_imagenetr.sh
```

## RQ2: raw-factor rescaling control

Object of study: parameterization dependence. Training is fixed to SeqLoRA/SGD/ImageNet-R/rank16/seed1993/task10, and the script sweeps:

```text
(A, B) -> (c A, B / c), c in {0.1, 0.3, 1, 3, 10}
```

The expected result is that `Sh_AB` changes with `c`, while `Sh_Delta/W_tangent` and `base_loss` stay approximately invariant.

Run:

```bash
bash config_exps_paper1_PAC/exp_7_RQ1RQ2/run_rq2_rescale_sweep.sh
```

By default the RQ2 script retrains to task 10 for each `c` because the current training entrypoint evaluates flatness in-process. The rescaling itself is evaluation-only and is restored after metric extraction.

## Summaries

After runs finish:

```bash
python config_exps_paper1_PAC/exp_7_RQ1RQ2/summarize_rq1rq2.py
```

It writes CSV files to `outputs_logs/exp_7_RQ1RQ2_summary/`.
