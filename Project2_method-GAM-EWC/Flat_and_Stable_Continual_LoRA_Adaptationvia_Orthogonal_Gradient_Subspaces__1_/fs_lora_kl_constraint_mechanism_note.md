# FS-LoRA KL Constraint Mechanism Note

## Goal

Use ImageNet-A as the main diagnostic dataset and keep the current best setting fixed:

```text
dataset=imageneta
model=ewclora_normfisher_gam
lr=0.04
optimizer_type=gam
rho=0.2
norm_rho=0.2
gam_grad_gamma=0.1
ewc_gamma=0.99
ewc_lambda=2000
ewc_normalize_fisher=true
```

The new diagnostic is not a new optimizer. It adds an explicit KL-distance view to the existing Fisher/EWC stability channel.

Strictly, this should be described as a compatibility bridge rather than an identity claim:

```text
FS-LoRA's Fisher-EWC constraint is a local second-order Laplace/Fisher proxy
for adapter-update-space KL complexity. After trace normalization, it is a
normalized Fisher-KL drift proxy, not an exact PAC-Bayesian KL certificate.
```

## What The Reference Theory Says

`4476_Sharpness_under_Constrain (2).pdf` gives two facts that are directly relevant to FS-LoRA.

First, the sequential PAC-Bayesian view decomposes continual learning into a trajectory-level perturbation-smoothed empirical term plus KL complexity terms. Sharpness-aware optimization controls the perturbation-smoothed empirical risk along the posterior path.

Second, under PECL/LoRA-constrained updates, the task-level KL is preserved by the pushforward from effective adapter updates to full model weights:

```text
KL(Q_t || P_t) = KL(Q_Delta,t || P_Delta,t)
```

Therefore both the sharpness term and the KL complexity term should be measured in the admissible effective update geometry, not in the ambient full parameter space.

## How This Maps To FS-LoRA

FS-LoRA already implements this decomposition in a practical optimizer:

```text
g_update = g_clean + lambda_flat * (g_gam - g_clean) + g_fisher
```

The AS1/GAM channel estimates the current-task perturbation-smoothed empirical term. The Fisher/EWC channel constrains old-task sensitive drift in `Delta W = BA`. The trace-normalized Fisher makes that stability channel measurable and scale-calibrated under low-rank updates.

This means Fisher-EWC is best interpreted as a local KL-drift surrogate, not as an unrelated regularizer. The claim should remain local and approximate.

## Is EWC The KL Constraint?

Not exactly. The safer statement is:

```text
EWC is the mean-shift term of a DeltaW-space Gaussian/Laplace KL under explicit
posterior/prior assumptions.
```

If the old task posterior over effective updates is approximated as a Gaussian with precision given by the old Fisher, then the mean-shift part of the KL is

```text
KL_Fisher(Delta W_t || Delta W_ref)
  ~= 1/2 * sum_k F_k * (Delta W_t,k - Delta W_ref,k)^2
```

The current code's EWC penalty is exactly the scaled version of this quantity:

```text
ewc_penalty_value = lambda_ewc * KL_Fisher_eta
                  = 0.5 * lambda_ewc * sum_k (F_k + eta) * drift_k^2
```

So the paper should say: EWC is a tractable local Fisher-Rao/KL proxy for KL-style stability complexity in the admissible `Delta W` space.

It should not say that EWC is the exact KL, because the implementation uses a diagonal empirical Fisher, ignores covariance and log-determinant terms, and uses trace normalization. Trace normalization changes the scale, so the resulting term is a calibrated KL proxy or normalized Fisher-KL budget.

One implementation detail matters. In the current FS-LoRA code, `delta_reference` is the `Delta W` snapshot at the beginning of the current task, while `fisher_past_delta` is the accumulated Fisher from previous tasks. Therefore the online training penalty is best read as a consecutive posterior-step drift proxy weighted by cumulative old-task Fisher precision. A literal per-task sum over all old snapshots would require storing each past task's Fisher and `Delta W_tau` snapshot separately.

## Added Metrics

The mechanism logger now reports explicit KL diagnostics:

```text
kl_fisher_proxy
kl_fisher_eta_proxy
kl_fisher_proxy_per_dim
kl_fisher_eta_proxy_per_dim
kl_fisher_eta_proxy_scaled_by_lambda
kl_iso_delta
kl_step_iso_proxy
kl_fisher_delta
kl_fisher_delta_weighted
kl_fisher_to_iso_ratio
kl_fisher_per_tensor_mean
kl_fisher_per_layer_mean
kl_fisher_per_task
kl_step_fisher_proxy
kl_gaussian_l2_sigma2
kl_gaussian_l2_per_dim
delta_numel
```

Interpretation:

```text
kl_fisher_proxy = 0.5 * sum_k F_k * drift_k^2
```

This is the raw Fisher-KL proxy.

```text
kl_fisher_delta = 0.5 * sum_k (F_k + eta) * drift_k^2
kl_fisher_delta_weighted = lambda_ewc * kl_fisher_delta
```

These are the main diagnostics aligned with the active EWC penalty.

```text
kl_iso_delta = kl_gaussian_l2_sigma2 = ||drift||^2 / (2 sigma^2)
```

This is an unweighted Gaussian mean-shift KL diagnostic. It is useful only as a scale comparison because it does not know which directions are old-task sensitive.

```text
kl_fisher_to_iso_ratio = kl_fisher_delta / (kl_iso_delta + eps)
```

This ratio asks whether the adapter drift is concentrated in old-task Fisher-sensitive directions or is mostly harmless movement in insensitive directions.

`kl_fisher_per_task` is written once per mechanism step, so the task index in `cl_metrics.json` provides the task-wise trajectory. `kl_fisher_per_layer_mean` is currently an alias of the mean over logged `Delta W` tensors, because the implementation stores q/v delta terms separately.

## ImageNet-A Diagnostic Config

Config:

```text
config_kl_mechanism_imageneta/as2_normfisher_kl_lam2000_ewcg099_imageneta_t10c20_r10_sgd_lr004_e20_rho02_nr02_g01.yaml
```

Run script:

```text
config_kl_mechanism_imageneta/run_kl_mechanism_imageneta.sh
```

Command:

```bash
cd /data/140-0/users/liying/Flatness_CV
GPU=3 bash config_kl_mechanism_imageneta/run_kl_mechanism_imageneta.sh
```

Outputs:

```text
outputs_logs/config_kl_mechanism_imageneta
logs_kl_mechanism_imageneta
```

## Expected Mechanism Claims To Test

1. `kl_fisher_delta` should reflect old-task-sensitive drift better than `kl_iso_delta`.
2. `gamma=0.99` should reduce final forgetting because it preserves old Fisher mass longer, increasing the Fisher-KL budget pressure on old-task sensitive directions.
3. `kl_iso_delta` can grow without necessarily predicting forgetting; the Fisher-weighted KL should be the more meaningful stability indicator.
4. `cos_flat_fisher` should remain near zero, showing that the flatness and Fisher-KL channels are geometrically separable.
5. Better retention should correlate with controlled `kl_fisher_delta`, `kl_fisher_to_iso_ratio`, and `high_fisher_weighted_energy_ratio`, not merely with smaller total `delta_norm`.

## Paper Implication

The core story should become:

```text
FS-LoRA decouples the two PAC-Bayesian ingredients under LoRA-constrained update geometry:
AS1 controls the perturbation-smoothed empirical/sharpness term, while trace-normalized Fisher-EWC controls a normalized local KL-drift proxy in Delta W space.
```

This is stronger and cleaner than saying FS-LoRA is just GAM plus EWC. It explains the mechanism: current-task flatness and old-task KL stability are separate gradient channels in the same admissible update geometry.

## Suggested Method Text

```latex
\subsection{Fisher-EWC as a Local KL-Drift Proxy}

The Fisher constraint used by FS-LoRA can be interpreted as a local KL-drift
proxy in the effective adapter-update space. For a past posterior state,
consider a Laplace approximation to the adapter-update posterior centered at
the reference update $\Delta W_{\mathrm{ref}}$, with diagonal precision given by
the trace-normalized Fisher $\widetilde F+\eta I$. If the current adapter state
is represented by a Gaussian posterior with the same covariance and mean
$\Delta W_t$, the mean-shift term of the KL reduces to the Fisher-weighted
quadratic drift:
\begin{equation}
D_{\mathrm{FisherKL}}^t
=
\frac{1}{2}
\sum_l
\left\langle
\widetilde F_l+\eta,
\left(
\Delta W_{t,l}
-
\Delta W_{\mathrm{ref},l}
\right)^{\odot 2}
\right\rangle .
\end{equation}
Thus, the EWC term used by FS-LoRA is a tractable local proxy for KL-style
stability complexity in $\Delta W$-space. We emphasize that this is not a
numerical PAC-Bayesian certificate: after trace normalization, the quantity is
a scale-calibrated relative KL-drift proxy rather than an exact predictive KL.
```
