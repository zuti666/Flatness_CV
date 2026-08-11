# FS-LoRA Core Mechanism Experiment Design

## 0. Goal

The next experiments should verify the core mechanism of FS-LoRA, not search for a better number by tuning learning rates.

The central claim to validate is:

> Continual LoRA should decouple current-task flatness control from old-task Fisher stabilization. The two signals are measurable as separate gradient channels in the effective adapter-update space `Delta W = BA`; trace-normalized Fisher makes the stability channel active, and the Fisher memory horizon controls old-task anchoring.

Therefore the experiments must answer five questions:

1. Is decoupling necessary, or is coupled GAM on `L_task + L_fisher` enough?
2. Does trace-normalized Fisher actually activate the stability channel?
3. Are `g_flat` and `g_fisher` near-orthogonal in the runs where the method works?
4. Does `ewc_gamma` behave as a Fisher memory-horizon mechanism?
5. Do the failure cases match the proposed mechanism: insufficient old-task anchoring or flatness-scale mismatch?

Primary metrics must come from the final accuracy matrix:

- `Task0_final`: final matrix last row, first task.
- `FAA`: final row mean.
- `AAA`: lower-triangular matrix mean.
- `Forget`: final forgetting average.

Secondary diagnostics:

- `cos(g_flat, g_fisher)`.
- `cos(g_clean, g_flat)`.
- `fisher/clean` gradient norm ratio.
- `flat/clean` gradient norm ratio.
- normalized Fisher drift.
- high-Fisher weighted energy ratio.

Top accuracy curves are useful for debugging but should not support final benchmark claims.

## 1. Hypotheses

### H1: Coupled Sharpness-Aware Fisher Is the Wrong Composition

Naive coupling applies GAM to:

```text
L_task + lambda_ewc * R_fisher
```

This makes the perturbation depend on both current-task loss and old-task Fisher penalty. The resulting sharpness correction is not a pure current-task flatness signal.

FS-LoRA should do better or be more interpretable because it uses:

```text
g = g_clean + lambda_flat * (g_gam - g_clean) + g_fisher
```

where the GAM closure sees only `L_task`.

Validation signal:

- Decoupled FS-LoRA should improve or match FAA while reducing forgetting or improving Task0 retention.
- Mechanism logs should show that the Fisher channel is independent and measurable.
- Coupled runs should have weaker diagnostic separation, worse final matrix, or less stable old-task retention.

### H2: Trace Normalization Is Not Cosmetic

Raw Fisher in low-rank `Delta W` can be too small, making the Fisher penalty inactive unless `lambda_ewc` is heavily retuned.

Validation signal:

- Raw-Fisher decoupled runs should show smaller `fisher/clean`.
- Trace-normalized Fisher should show measurable `fisher/clean` while preserving `cos(g_flat, g_fisher)` near zero.
- If raw Fisher requires a much larger `lambda_ewc` to match the same `fisher/clean`, trace normalization is justified as scale calibration.

### H3: Near-Orthogonality Supports Decoupling

The method does not enforce orthogonality. It uses near-orthogonality as a diagnostic.

Validation signal:

- In successful runs, `cos(g_flat, g_fisher)` should be close to zero across tasks.
- `cos(g_clean, g_flat)` should remain positive, showing the flatness channel is related to current-task learning.
- `fisher/clean` should be nonzero but not dominant.

Expected current pattern from existing logs:

```text
ImageNet-A best: cos(flat,fisher) about -0.002
ImageNet-R best: cos(flat,fisher) about  0.003
CIFAR-100 best:  cos(flat,fisher) about  0.009
DomainNet best:  cos(flat,fisher) about  0.004
```

The important point is not exactly zero. The point is that the two channels are not strongly aligned or opposed.

### H4: `ewc_gamma` Is a Memory-Horizon Mechanism

The current implementation accumulates Fisher as:

```python
F_accum = gamma * F_old + F_new
```

This is not normalized EMA. It is memory-mass accumulation with retention factor `gamma`.

Validation signal:

- Larger `gamma` should improve old-task anchoring, especially `Task0_final` and forgetting.
- Too large `gamma` may hurt later-task plasticity because accumulated Fisher mass becomes too strong.
- This should be tested with fixed learning rate, optimizer, and GAM settings.

Existing ImageNet-A evidence:

```text
gamma=0.90, lambda=2000:
  FAA=53.813, AAA=61.751, Task0_final=69.71, Forget=6.914

gamma=0.99, lambda=2000:
  FAA=54.124, AAA=61.852, Task0_final=69.71, Forget=6.603
```

This suggests the memory channel is active, but it needs a systematic table.

### H5: Failure Cases Should Be Mechanistically Predictable

FS-LoRA can fail for at least two different reasons:

1. Old-task anchoring is insufficient:
   - CIFAR-100 and DomainNet final-matrix results show lower `Task0_final` than EWC-LoRA references.
2. Flatness correction is too large:
   - OxfordPet has an extreme `flat/clean` ratio in the draft analysis.

Validation signal:

- CIFAR-100 and DomainNet should improve old-task retention when `gamma` is increased or `lambda_ewc` is adjusted without changing LR/optimizer.
- OxfordPet or Cars should improve when `rho` or `lambda_flat` is reduced.
- If these targeted changes help the predicted failure mode, negative results become diagnostic evidence.

## 2. Experiment Set A: Coupled vs Decoupled Composition

### Purpose

Directly test the central design claim:

> Flatness control and Fisher stabilization should not be mixed inside the same sharpness-aware closure.

### Datasets

Use two datasets first:

1. CUB200 or another stable fine-grained benchmark from Track 1.
2. ImageNet-A from Track 2 because it is sensitive and already has complete final-matrix results.

Add CIFAR-100 only after the first two show interpretable results.

### Runs

Keep fixed:

- backbone: ViT-B/16;
- LoRA rank: same as current benchmark setting;
- optimizer: SGD + GAM where applicable;
- learning rate: current best per dataset;
- `lambda_ewc`: 2000 unless doing the Fisher-specific sweep;
- seed: 0 for first pass, then add 1993 and 42 for paper results.

Rows:

```text
A0: SeqLoRA + SGD
A1: SeqLoRA + GAM only
A2: EWC only, no flatness
A3: Coupled GAM on L_task + Fisher
A4: Decoupled raw Fisher
A5: Decoupled trace-normalized Fisher = FS-LoRA
```

Implementation mapping:

```text
A0: model_name=seqlora, optimizer=sgd
A1: model_name=seqlora, optimizer_type=gam
A2: model_name=ewclora_normfisher_gam, lambda_flat=0, ewc_normalize_fisher=true
A3: model_name=ewclora_youyue_fitarchitecture, optimizer_type=gam
A4: model_name=ewclora_normfisher_gam, ewc_normalize_fisher=false
A5: model_name=ewclora_normfisher_gam, ewc_normalize_fisher=true
```

Note: A2 still uses the GAM optimizer wrapper if required by the implementation, but `lambda_flat=0` makes the update `g_clean + g_fisher`.

### Metrics

Main table columns:

```text
Method | FAA | AAA | Task0_final | Forget | Final top1 | Notes
```

Mechanism columns:

```text
cos(flat,fisher) | cos(clean,flat) | fisher/clean | flat/clean | normalized_fisher_drift
```

### Expected Evidence

The strongest support would be:

- A5 > A3 in FAA or Task0 retention.
- A5 shows measurable `fisher/clean`.
- A5 keeps `cos(flat,fisher)` near zero.
- A1 has good plasticity but worse forgetting.
- A2 protects old tasks but may underfit or lack flatness gains.

If A3 matches A5, the decoupling claim must be weakened. The paper can still argue that decoupling improves interpretability and diagnostics, but not that it is empirically necessary on that dataset.

## 3. Experiment Set B: Raw Fisher vs Trace-Normalized Fisher

### Purpose

Show that trace normalization solves a scale pathology, not just a hyperparameter preference.

### Datasets

Use:

- CUB200 for fine-grained stable evidence.
- ImageNet-A for PECL diagnostic evidence.

Optional:

- CIFAR-100 because old-task retention is currently weak.

### Runs

Use decoupled update for all rows:

```text
B0: raw Fisher, lambda=200
B1: raw Fisher, lambda=2000
B2: raw Fisher, lambda=20000
B3: normalized Fisher, lambda=1000
B4: normalized Fisher, lambda=1500
B5: normalized Fisher, lambda=2000
```

Keep:

- same LR;
- same GAM rho;
- same gamma;
- same epochs.

### Metrics

Primary:

```text
FAA | AAA | Task0_final | Forget
```

Mechanism:

```text
fisher/clean | train_fisher_loss_mean | ewc_penalty_value | normalized_fisher_drift
```

### Expected Evidence

Trace normalization is supported if:

- raw Fisher has very small `fisher/clean` at ordinary lambda;
- raw Fisher needs much larger lambda to become active;
- normalized Fisher reaches meaningful `fisher/clean` with stable lambda values;
- normalized Fisher improves old-task retention without destroying current-task CA.

This table should support the claim:

> Trace normalization makes `lambda_ewc` a relative stability coefficient instead of a compensation factor for arbitrary Fisher scale.

## 4. Experiment Set C: Fisher Memory Horizon (`ewc_gamma`)

### Purpose

Test whether `ewc_gamma` controls old-task anchoring through Fisher memory retention.

### Datasets

Primary:

- ImageNet-A, because `gamma=0.99` already improved final-matrix forgetting.

Secondary:

- CIFAR-100, because Task0 final is weak.
- DomainNet, because Task0 final is weak.
- ImageNet-R once current gamma=0.99 runs are complete.

### Runs

Keep fixed:

- LR and optimizer from the current best config.
- GAM rho and norm rho.
- `lambda_flat=1.0`.
- `ewc_normalize_fisher=true`.

Rows:

```text
C0: gamma=0.90, lambda=2000
C1: gamma=0.95, lambda=2000
C2: gamma=0.99, lambda=2000
C3: gamma=1.00, lambda=2000
C4: gamma=0.99, lambda=1500
C5: gamma=1.00, lambda=1500
```

For paper space, the minimal table can keep only:

```text
gamma=0.90, lambda=2000
gamma=0.99, lambda=2000
gamma=0.99, lambda=1500
```

### Metrics

Main:

```text
FAA | AAA | Task0_final | Forget | Last-task CA
```

Mechanism:

```text
fisher/clean | ewc_penalty_value | normalized_fisher_drift | high_fisher_weighted_energy_ratio
```

### Expected Evidence

If the memory-horizon mechanism is correct:

- increasing gamma should improve Task0 retention and reduce forgetting;
- too high gamma may reduce last-task CA or FAA if the Fisher anchor becomes too strong;
- reducing lambda at high gamma may recover plasticity.

The key plot:

```text
x-axis: gamma
y-axis 1: Task0_final
y-axis 2: last-task CA or FAA
```

This directly shows the stability-plasticity tradeoff induced by Fisher memory.

## 5. Experiment Set D: Flatness Scale and Failure Modes

### Purpose

Show that failure cases can be explained by flatness-scale mismatch rather than random instability.

### Datasets

Primary:

- OxfordPet, because prior diagnostics show extreme `flat/clean`.

Secondary:

- Cars196 or Aircraft if they show mixed behavior.
- ImageNet-A if high rho affects current-task plasticity.

### Runs

Keep Fisher settings fixed:

```text
ewc_normalize_fisher=true
lambda_ewc=2000
gamma=current best
```

Sweep flatness:

```text
D0: lambda_flat=0.0
D1: lambda_flat=0.25
D2: lambda_flat=0.5
D3: lambda_flat=1.0
```

or sweep GAM radius:

```text
D4: rho=0.02, norm_rho=0.10
D5: rho=0.05, norm_rho=0.10
D6: rho=0.10, norm_rho=0.10
D7: rho=0.20, norm_rho=0.20
```

Do not sweep both lambda_flat and rho at full grid size unless the first pass is inconclusive.

### Metrics

Main:

```text
FAA | AAA | Forget | last-task CA
```

Mechanism:

```text
flat/clean | cos(clean,flat) | cos(flat,fisher) | fisher/clean
```

### Expected Evidence

If failure is caused by flatness-scale mismatch:

- reducing `lambda_flat` or `rho` should reduce `flat/clean`;
- FAA should recover if flatness was over-dominant;
- Fisher metrics should remain similar if only flatness scale changes.

This supports the discussion claim:

> FS-LoRA fails when the flatness correction dominates the clean gradient or when the Fisher memory horizon is too short.

## 6. Experiment Set E: `Delta W` Fisher vs Factor-Space Regularization

### Purpose

Validate that the stability object should be the effective adapter update `Delta W = BA`.

### Runs

If implementation supports it:

```text
E0: raw LoRA factor EWC
E1: Delta-W Fisher, raw
E2: Delta-W Fisher, trace-normalized
```

Datasets:

- CUB200.
- ImageNet-A.

Metrics:

```text
FAA | AAA | Task0_final | Forget | fisher/clean
```

### Expected Evidence

The expected result is not necessarily that `Delta W` always wins. The needed evidence is:

- `Delta W` Fisher is the interpretable functional constraint;
- raw factor regularization can be sensitive to LoRA factorization scale;
- trace-normalized `Delta W` Fisher gives a more stable gradient scale.

If factor-space EWC is unavailable, this can be moved to appendix or discussed as conceptual motivation supported by prior EWC-LoRA comparison.

## 7. Experiment Set F: Metric Sanity Check

### Purpose

Show why final matrix must be used instead of top accuracy curves.

### Datasets

Use already completed runs:

- ImageNet-A.
- CIFAR-100.
- DomainNet.

### Table

Columns:

```text
Dataset | Config | Top1 final | Final-matrix FAA | AAA | Task0_final | Interpretation
```

### Expected Evidence

The point is:

- top accuracy is sample-weighted and can hide per-task retention issues;
- final matrix exposes task-block forgetting;
- FS-LoRA should be evaluated by final matrix because the method is about continual retention.

This is especially important for CIFAR-100 and DomainNet, where current-task learning can look good but Task0 retention is weaker than EWC-LoRA.

## 8. Priority Run Plan

### Tier 1: Minimum Paper-Critical Runs

These are the most important if compute is limited.

```text
T1-A: Coupled vs decoupled on CUB200
T1-B: Coupled vs decoupled on ImageNet-A
T1-C: Raw vs normalized Fisher on CUB200
T1-D: Gamma memory horizon on ImageNet-A
```

Minimum rows:

```text
SeqLoRA + GAM only
EWC only
Coupled GAM + Fisher
Decoupled raw Fisher
Decoupled normalized Fisher
```

Minimum gamma rows:

```text
gamma=0.90, lambda=2000
gamma=0.99, lambda=2000
gamma=0.99, lambda=1500
```

### Tier 2: Failure-Mode Diagnostics

```text
T2-A: CIFAR-100 gamma=0.95/0.99 with fixed LR
T2-B: DomainNet gamma=0.95/0.99 with fixed LR
T2-C: OxfordPet lambda_flat/rho reduction
```

### Tier 3: Paper-Strengthening Runs

```text
T3-A: 3-seed confirmation for CUB200 mechanism table
T3-B: 3-seed confirmation for ImageNet-A gamma table
T3-C: Delta-W vs factor-space Fisher, if available
```

## 9. Recommended Tables for the Paper

### Table 1: Main Results

Purpose:

Show useful performance without overclaiming universal superiority.

Columns:

```text
Dataset | SeqLoRA+GAM FAA | EWC-LoRA FAA | FS-LoRA FAA | FS-LoRA AAA | Forget | Note
```

Notes should honestly mark mixed behavior.

### Table 2: Core Mechanism Ablation

Purpose:

Validate the method design.

Columns:

```text
Method
FAA
AAA
Task0_final
Forget
cos(flat,fisher)
fisher/clean
flat/clean
```

Rows:

```text
SeqLoRA + SGD
SeqLoRA + GAM
EWC only
Coupled GAM + Fisher
Decoupled raw Fisher
FS-LoRA
```

This is the most important table for the paper.

### Table 3: Fisher Memory Horizon

Purpose:

Show gamma is a mechanism.

Columns:

```text
Dataset | gamma | lambda_ewc | FAA | AAA | Task0_final | Forget | last-task CA | fisher/clean
```

### Table 4: Standard PECL Diagnostic Extension

Purpose:

Show broader behavior without claiming final SOTA.

Columns:

```text
Dataset | EWC-LoRA reference | FS-LoRA best complete | Task0_final gap | FAA gap | Mechanism interpretation
```

Example interpretations:

```text
ImageNet-A: memory horizon active; gamma=0.99 improves forgetting.
CIFAR-100: current-task learning is strong, old-task anchoring remains insufficient.
DomainNet: old-task anchoring remains insufficient under current settings.
ImageNet-R: strong complete result; gamma=0.99 pending final matrix.
```

## 10. Recommended Figures

### Figure 1: Coupled vs Decoupled Mechanism

Left:

```text
GAM closure over L_task + R_fisher
```

Right:

```text
task-only GAM closure -> g_flat
Delta-W Fisher -> g_fisher
final update = g_clean + lambda_flat g_flat + g_fisher
```

### Figure 2: Gradient Channel Diagnostics

Plot across tasks:

```text
cos(flat,fisher)
fisher/clean
flat/clean
```

Expected:

- `cos(flat,fisher)` stays near zero;
- `flat/clean` explains flatness pressure;
- `fisher/clean` explains stability pressure.

### Figure 3: Gamma Memory Horizon

Plot:

```text
gamma vs Task0_final
gamma vs last-task CA
```

This directly visualizes the stability-plasticity tradeoff.

### Figure 4: Failure Mode Map

Scatter:

```text
x-axis: flat/clean
y-axis: fisher/clean
marker size: Forget
color: FAA
```

Interpretation:

- too high flat/clean -> flatness over-dominance;
- too low fisher/clean -> weak old-task anchoring;
- balanced region -> best FS-LoRA behavior.

## 11. Acceptance Criteria

### Strong Evidence for the Paper

The mechanism claim is well supported if the experiments show:

1. Decoupled FS-LoRA beats or matches coupled GAM+Fisher on final matrix.
2. Normalized Fisher has more meaningful `fisher/clean` than raw Fisher.
3. `cos(flat,fisher)` is consistently near zero in successful runs.
4. Gamma changes old-task retention in the predicted direction.
5. Failure cases can be explained by flatness-scale or memory-horizon diagnostics.

### Evidence That Requires Rewriting

If:

- coupled and decoupled are indistinguishable;
- normalized Fisher does not change Fisher activity;
- `cos(flat,fisher)` is large and unstable;
- gamma does not affect old-task retention;

then the paper should be rewritten as a narrower empirical optimizer study rather than a mechanism paper.

### Evidence That Is Still Useful

If FS-LoRA does not beat EWC-LoRA on CIFAR-100 or DomainNet but diagnostics show weak old-task anchoring, that still supports the paper's mechanism framing.

The claim should become:

> FS-LoRA exposes and partially addresses the flatness-stability coupling, but standard PECL datasets require adaptive Fisher memory and flatness scaling for robust gains.

## 12. Concrete First Batch

If we run only one batch of new experiments, run this:

### CUB200 Mechanism Batch

```text
M1_cub_gam_only
M2_cub_ewc_only_norm
M3_cub_coupled_gam_fisher
M4_cub_decoupled_raw
M5_cub_fslora_norm
```

Goal:

- verify decoupling and trace normalization in a stable fine-grained setting.

### ImageNet-A Mechanism Batch

```text
M6_ina_coupled_gam_fisher
M7_ina_decoupled_raw
M8_ina_fslora_gamma090_lam2000
M9_ina_fslora_gamma099_lam2000
M10_ina_fslora_gamma099_lam1500
```

Goal:

- verify memory horizon and final-matrix retention.

### CIFAR-100 Retention Diagnostic Batch

```text
M11_cifar_gamma095_lam1600
M12_cifar_gamma099_lam1500
M13_cifar_gamma099_lam2000
```

Goal:

- test whether weak Task0 final is a Fisher-memory issue.

### DomainNet Retention Diagnostic Batch

```text
M14_domainnet_gamma095_lam1600
M15_domainnet_gamma099_lam1500
M16_domainnet_gamma099_lam2000
```

Goal:

- test whether DomainNet's old-task anchoring improves with longer Fisher memory.

Use independent output roots:

```text
outputs_logs/config_mechanism_validation
logs_mechanism_validation
```

Use config folder:

```text
config_mechanism_validation
```

Do not mix these logs with previous learning-rate sweeps.

## 13. Paper Interpretation After These Runs

If the expected pattern holds, the paper can say:

```text
The main empirical contribution is not that a single fixed hyperparameter setting dominates all benchmarks.
Rather, the experiments show that FS-LoRA exposes the separate roles of current-task flatness and old-task Fisher stability.
Decoupling improves the interpretability and often the performance of continual LoRA optimization; trace normalization activates the stability channel; and gamma controls the memory horizon of old-task Fisher anchoring.
The remaining failures are consistent with the same mechanism: either the Fisher channel is too weak for long sequences, or the flatness channel is too strong relative to the clean task gradient.
```

This is the strongest version of the experimental story because it aligns method, code, logs, and conclusion strength.
