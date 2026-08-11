# Revised Section 6 Experiment Plan for LoRA PECL Flatness

Date: 2026-04-30  
Code root: `/data/140-0/users/liying/Flatness_CV`

## 1. Core Principle

The experiment section should not say "verify Q1" or "verify Q2". A theorem is
not verified directly by experiments. The safer and more publishable wording is:

```text
We test empirical implications of the theory.
```

The experimental thesis should be:

```text
Our experiments are designed to test two empirical implications of the theory:
sharpness-aware optimization should influence continual learning through the
posterior trajectory, and the sharpness most predictive of PECL performance
should be measured along the local effective adapter-update support rather than
in the ambient parameter space or raw LoRA factor space.
```

This connects directly to the two theoretical chains:

1. Section 4: perturbation-smoothed empirical risk enters a trajectory-level
   PAC-Bayesian bound, so flatness should be interpreted along posterior
   trajectories, not only at a single checkpoint.
2. Section 5: under frozen-backbone LoRA PECL, the bound-relevant support is the
   local effective adapter-update geometry, not arbitrary full-space or raw
   factor-space perturbations.

## 2. Recommended Section 6 Structure

Use four RQs in the paper, but keep the evidence hierarchy clear.

```latex
\section{Experiments}

\subsection{Experimental Setup}

\subsection{RQ1: Does Sharpness-Aware Optimization Affect the Posterior Trajectory?}

\subsection{RQ2: Which Perturbation Geometry Is Relevant in PECL?}

\subsection{RQ3: How Do Sharpness Objectives Behave within the Adapter-Update Support?}

\subsection{RQ4: Robustness across LoRA Organizations and Evaluation Protocols}
```

Important:

- RQ1 and RQ2 already have strong ImageNet-R evidence.
- RQ3 and RQ4 are next-step experiments. They should be included in the plan,
  but not written as finished unless we run them.
- CIFAR10 experiments are pilots and should be appendix evidence.

Appendix:

```latex
\appendix

\section{CIFAR10 Pilot Experiments}
\section{Reparameterization Control}
\section{Multi-seed Robustness}
\section{Implementation Details of Perturbation Supports}
\section{Additional Robustness Results}
```

## 3. Experimental Setup

The setup should explicitly include:

### Continual protocol

Main benchmark:

```text
ImageNet-R
T = 20 class-incremental tasks
10 classes per task
seed = 1993 currently; later add 42 and 1234
```

Robustness benchmarks:

```text
ImageNet-C
ImageNet-P
```

These should be used for RQ4, not for the first causal proof.

### Model and LoRA organization

Main:

```text
Backbone: ViT-B/16
LoRA method: SeqLoRA
Rank: r = 16
Frozen backbone
No replay
```

Robustness:

```text
SeqLoRA
IncLoRA
OLoRA
```

### Metrics

Main metrics:

```text
FAA / LAA
AAA
BWT
Final classifier accuracy
NCM accuracy, if available
Prefix Forget, for forked trajectory experiments
Suffix Acc, for forked trajectory experiments
```

Sharpness diagnostics:

```text
Sh_param_full
Sh_AB / Sh_factor
Sh_Delta/W_tangent
Sh_rand/W_tangent
Sh_frozen_coords
```

Use deterministic loaders for post-hoc sharpness:

```text
class_shuffle: false
flat_eval_loader_shuffle: false
flat_eval_data_mode: test
same data batch for base and perturbed losses
```

## 4. RQ1: Does Sharpness-Aware Optimization Affect the Posterior Trajectory?

### Empirical implication

The Section 4 trajectory PAC-Bayes result suggests that sharpness-aware
optimization should not only affect current-task fitting. It should affect the
posterior trajectory:

```text
Q_1 -> Q_2 -> ... -> Q_T
```

Empirical question:

```text
Given the same posterior state Q_9, does changing the suffix optimizer change
Q_10:19 and old-task retention?
```

### Main experiment: Exp F, ImageNet-R forked trajectory

Existing path:

```text
config_exps_paper1_PAC/exp_F_imagenetr_r16_t20_forked_taskwise
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/summary_official_and_prefix.csv
```

Design:

```text
Dataset: ImageNet-R
Backbone: ViT-B/16
Tasks: T = 20
LoRA: SeqLoRA
Rank: r = 16
Fork point: after Task 9
```

Branches:

| Prefix optimizer | Suffix optimizer | Purpose |
|---|---|---|
| SGD | SGD | baseline trajectory |
| SGD | SAM-factor | test whether later SAM changes retention |
| SGD | random-factor | control for generic perturbation noise |
| SAM-factor | SGD | test whether earlier SAM alone is sufficient |
| SAM-factor | SAM-factor | full sharpness-aware trajectory |

Existing result:

| Prefix | Suffix | FAA | BWT | Prefix Forget | Suffix Acc |
|---|---:|---:|---:|---:|---:|
| SGD | SGD | 61.72 | -15.32 | 17.19 | 64.33 |
| SGD | SAM-factor | **66.38** | **-9.97** | **11.03** | **67.49** |
| SGD | random-factor | 61.72 | -15.32 | 17.19 | 64.33 |
| SAM-factor | SGD | 65.29 | -10.38 | 11.87 | 66.46 |
| SAM-factor | SAM-factor | **67.50** | **-7.94** | **8.72** | **67.72** |

Figures:

```text
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/figures/sgd_prefix_time_curves.png
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/figures/sam_prefix_time_curves.png
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/figures/all_variants_time_curves.png
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/figures/sgd_prefix_final_forgetting.png
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/figures/sam_prefix_final_forgetting.png
```

### Main-text interpretation

Use this wording:

```text
From an identical SGD-prefix posterior Q_9, suffix SAM-factor improves FAA from
61.72 to 66.38 and reduces prefix forgetting from 17.19 to 11.03. Suffix
random-factor reproduces the SGD trajectory. Thus the improvement is not due to
generic perturbation noise, but to the adversarial sharpness-aware update
direction shaping the posterior trajectory.
```

Avoid:

```text
SAM directly prevents forgetting.
```

Use:

```text
SAM changes the posterior trajectory in a way that reduces later degradation of
earlier-task performance.
```

### Appendix pilot: Exp D, CIFAR10 two-task factorial

Existing path:

```text
config_exps_paper1_PAC/exp_D_taskwise_sam_trajectory
outputs_logs/exp_D_taskwise_sam_trajectory_summary/summary_by_variant.csv
outputs_logs/exp_D_taskwise_sam_trajectory_summary/contrasts_by_seed.csv
```

Use only as pilot:

| Variant | A_1,2 | A_2,2 | Final Avg | Forget |
|---|---:|---:|---:|---:|
| `sgd_sgd` | 95.64 | 97.96 | 96.80 | 3.34 |
| `sam_factor_sgd` | 95.70 | 98.24 | 96.97 | 3.22 |
| `sgd_sam_factor` | 96.00 | 97.58 | 96.79 | 2.98 |
| `sam_factor_sam_factor` | 96.16 | 97.72 | 96.94 | 2.76 |
| `sgd_random_factor` | 95.66 | 97.94 | 96.80 | 3.32 |
| `random_factor_random_factor` | 95.66 | 97.96 | 96.81 | 3.32 |

Pilot sentence:

```text
Applying SAM only at Task 2 improves old-task retention more than applying SAM
only at Task 1, while random-factor has no comparable effect.
```

## 5. RQ2: Which Perturbation Geometry Is Relevant in PECL?

### Empirical implication

The Section 5 support-reduction theorem says the bound-relevant perturbation
should be associated with the effective adapter-update geometry. But the
experiment must distinguish:

```text
support effect
direction effect
generic random-noise effect
```

Therefore RQ2 should be written as:

```text
Which perturbation geometry best explains PECL performance?
```

### Main experiment: Exp E, ImageNet-R support x direction

Existing path:

```text
config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction
outputs_logs/exp_E_imagenetr_r16_t20_support_direction_summary/summary_by_variant.csv
outputs_logs/exp_E_imagenetr_r16_t20_support_direction_summary/figures/support_direction_pairwise.csv
outputs_logs/exp_E_imagenetr_r16_t20_support_direction_summary/figures/support_direction_faa_bwt.png
```

Two axes:

| Axis | Values | Meaning |
|---|---|---|
| Direction | `sam`, `random` | adversarial SAM gradient direction vs Gaussian random direction |
| Support | `factor`, `full`, `delta`, `all`, `frozen` | perturbation coordinate/support spaces |

Naming:

```text
[direction]_[support]
```

Examples:

```text
sam_delta:
adversarial SAM direction restricted to effective adapter-update tangent support.

random_delta:
Gaussian random direction restricted to the same support.

sam_factor:
adversarial SAM direction in raw LoRA factor coordinates.

random_factor:
Gaussian random direction in raw LoRA factor coordinates.
```

Do not use `sam_random` in the main text. It means SAM gradient projected onto a
random matched support and is not the same as `random_delta`.

### Existing result

| Support | SAM FAA | Random FAA | Delta FAA | SAM BWT | Random BWT | Delta BWT |
|---|---:|---:|---:|---:|---:|---:|
| factor | 67.94 | 58.79 | **+9.14** | -6.81 | -18.47 | **+11.67** |
| delta | 65.41 | 58.26 | **+7.15** | -11.56 | -19.08 | **+7.52** |
| full | 64.67 | 58.15 | **+6.52** | -12.01 | -19.24 | **+7.23** |
| all | 63.19 | 58.32 | **+4.87** | -12.28 | -19.01 | **+6.72** |
| frozen | 62.91 | 58.24 | **+4.67** | -12.37 | -19.10 | **+6.73** |
| SGD | 58.29 | -- | -- | -18.99 | -- | -- |

### Main interpretation

The strongest supported claim from Exp E is:

```text
For every support, SAM direction substantially outperforms Gaussian random
direction. Random perturbations stay close to SGD. Therefore the improvement is
not generic perturbation noise.
```

Boundary:

```text
Exp E does not by itself prove that delta support is empirically best.
sam_factor is currently strongest under the default LoRA parameterization.
```

Protective wording:

```text
Exp E supports the necessity of adversarial sharpness-aware perturbations, but
it does not by itself establish raw factor-space sharpness as the theoretically
correct object. The factor-space advantage under the default parameterization is
interpreted as an implementation-level proxy, and the reparameterization control
distinguishes this from basis-invariant theoretical relevance.
```

## 6. RQ3: How Do Sharpness Objectives Behave within the Adapter-Update Support?

### Status

This is a planned extension. It is not fully run yet.

### Question

Once the perturbation support is fixed to the adapter-update tangent support
`T_Delta`, compare different sharpness objectives:

| Variant | Sharpness type | Meaning |
|---|---|---|
| `random_delta` | RS(0) | random perturbation in Delta support |
| `sam_delta` | AS(0) | zeroth-order adversarial sharpness in Delta support |
| `gam_delta` / `as1_delta` | AS(1) | first-order / gradient-norm aware sharpness in Delta support |

### Required implementation

Current code has:

```text
random_delta
sam_delta
```

Need to implement or verify:

```text
gam_delta or as1_delta
```

It must perturb inside the same `T_Delta` support, otherwise it is not a fair
comparison.

### Experiment design

```text
Dataset: ImageNet-R
Backbone: ViT-B/16
LoRA: SeqLoRA
Rank: r = 16
Tasks: T = 20
Variants: random_delta, sam_delta, gam_delta/as1_delta
```

Metrics:

```text
FAA / LAA
AAA
BWT
Final classifier accuracy
NCM accuracy if available
```

Sharpness curves:

```text
RS_Delta_t
AS0_Delta_t
AS1_Delta_t
```

### Expected conclusion

If `gam_delta` / `as1_delta` works best:

```text
Within the theoretically relevant adapter-update support, first-order
sharpness more directly constrains local update instability than AS(0).
```

If not:

```text
AS(0) is sufficient in the current LoRA PECL setting, and AS(1) is not needed
for the main empirical claim.
```

This RQ should not distract from RQ1/RQ2. It can be a shorter subsection or
appendix if results are not clean.

## 7. RQ4: Robustness across LoRA Organizations and Evaluation Protocols

### Status

This is a planned robustness extension. It should not be used as the first-line
evidence.

### Question

Is the adapter-update sharpness principle robust across PECL organizations and
robustness benchmarks?

### Design

LoRA organizations:

```text
SeqLoRA
IncLoRA
OLoRA
```

Datasets / protocols:

```text
ImageNet-R
ImageNet-C
ImageNet-P
```

Key variants:

```text
sgd
sam_factor
sam_delta
```

Do not run the full 11 variants for every method and dataset unless there is
enough compute. RQ4 is robustness, not the core causal proof.

### Expected conclusion

The claim should not be:

```text
The same variant is always best.
```

The claim should be:

```text
The relevant perturbation support should track the admissible effective
adapter-update geometry of the LoRA organization.
```

For IncLoRA, this may require distinguishing:

```text
new branch support
accumulated adapter support
```

## 8. Required Control: Reparameterization

This control is essential for RQ2, even if placed in appendix.

### Motivation

Current Exp E has:

```text
sam_factor FAA = 67.94
sam_delta  FAA = 65.41
```

Theory says the invariant object should be effective adapter-update geometry.
Therefore we need to explain why raw factor perturbation is practically strong
without making it the theoretical object.

### Design

Current code uses:

```text
Delta W = B A
```

Use:

```text
A' = A / c
B' = c B
Delta W' = B' A' = B A
```

Scales:

```text
c in {0.5, 1.0, 2.0, 4.0}
```

Optional stronger version:

```text
c in {0.25, 0.5, 1.0, 2.0, 4.0}
```

Start checkpoint:

```text
Exp F SGD-prefix task-9 checkpoint
outputs_logs/logs_inc_lora/seqlora/sgd/imagenetr/1993/
exp_F_imagenetr_r16_t20_prefix_sgd_t0_9/exp_run/checkpoints
```

Continue Task 10-19 with:

```text
sgd
sam_factor
sam_delta
```

Minimum compute version:

```text
suffix in {sam_factor, sam_delta}
```

Sanity checks:

```text
relative ||Delta W' - Delta W|| close to 0
max logit difference on fixed batch close to 0
Task 0-9 accuracy unchanged before suffix training
```

Expected:

| Method | Expected behavior under rescaling |
|---|---|
| SGD | nearly invariant |
| SAM-delta | nearly invariant |
| SAM-factor | scale-sensitive |

Paper interpretation:

```text
The practical advantage of factor-space SAM under the default LoRA scale is not
a basis-invariant property. When the same effective update is represented by
different factor scalings, factor-space perturbation changes, while Delta-space
perturbation remains stable. This supports the theoretical role of effective
adapter-update geometry.
```

## 9. Post-hoc Sharpness Diagnostic

This should stay compact.

### Goal

Connect performance to measured `Sh_delta`, without claiming causality.

### Design

```text
Checkpoints: t in {0, 4, 9, 14, 19}
Variants: sgd, sam_factor, sam_delta, random_factor
Metrics:
  Sh_delta
  Sh_factor / Sh_AB
  final forgetting F_t
```

Primary figure:

```text
x-axis: task index
y-axis: Sh_delta
curves: sgd, sam_factor, sam_delta, random_factor
```

Implementation requirements:

```text
flat_eval_loader_shuffle: false
flat_eval_data_mode: test
class_shuffle: false
fixed deterministic subset
same batch for base and perturbed loss
```

Acceptance:

```text
sharpness values should be non-negative, up to tiny numerical tolerance
```

Paper wording:

```text
Post-hoc measurements show that SAM-trained trajectories have lower effective
adapter-update sharpness than SGD and random perturbation trajectories, and the
reduction is aligned with lower final forgetting.
```

## 10. Multi-seed Robustness

Keep minimal.

Seeds:

```text
1993, 42, 1234
```

Exp E subset:

```text
sgd
sam_factor
sam_delta
random_factor
random_delta
```

Exp F subset:

```text
sgd_sgd
sgd_sam_factor
sgd_random_factor
```

Report:

```text
mean +/- std for FAA, BWT, Prefix Forget
```

Do not multi-seed all 11 Exp E variants unless compute is abundant.

## 11. Section 6 Opening Draft

```latex
\section{Experiments}

Our experiments are designed to test empirical implications of the theory
rather than to propose a new optimizer. First, the sequential PAC-Bayesian
decomposition predicts that sharpness-aware optimization can affect continual
learning through the posterior trajectory, not only through the local solution
of the task on which it is applied. Second, the PECL support-reduction theorem
predicts that, under frozen-backbone LoRA adaptation, the relevant flatness
object is the task-local sharpness of the effective adapter update rather than
ambient full-space sharpness or non-identifiable raw factor-space sharpness.

We therefore organize the experiments around four questions. RQ1 asks whether
changing the sharpness-aware optimizer after a shared checkpoint changes the
subsequent posterior trajectory and the forgetting of earlier tasks. RQ2 asks
which perturbation geometry best explains PECL performance by decomposing
perturbation direction and support. RQ3 compares different sharpness objectives
within the adapter-update support. RQ4 examines whether the support principle is
robust across LoRA organizations and evaluation protocols.
```

## 12. Critical Wording Boundaries

Do not write:

```text
verify Q1
verify Q2
Exp E proves delta support is best
Exp E proves factor-space sharpness is the correct object
```

Write instead:

```text
test empirical implications
posterior trajectory effect
support x direction decomposition
factor-space advantage under default parameterization
basis-invariant relevance of Delta-space tested by reparameterization control
```

Most important protection sentence for RQ2:

```text
Exp E shows that adversarial sharpness-aware direction is essential, while the
support comparison reveals an implementation-level advantage of factor-space
perturbation under the default LoRA parameterization. The reparameterization
control distinguishes practical effectiveness from basis-invariant theoretical
relevance.
```

## 13. Immediate Next Actions

1. Implement reparameterization control.
2. Smoke test:

```text
c = 2.0
suffix in {sam_factor, sam_delta}
1 epoch
verify Delta W error, logit error, resume correctness
```

3. Run full reparameterization control:

```text
c in {0.5, 1.0, 2.0, 4.0}
suffix in {sam_factor, sam_delta}
```

4. Then run minimal multi-seed robustness.

