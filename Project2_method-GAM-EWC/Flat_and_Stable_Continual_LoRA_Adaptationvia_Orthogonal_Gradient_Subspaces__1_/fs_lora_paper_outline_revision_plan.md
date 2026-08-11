# FS-LoRA Paper Outline and Revision Plan

## 0. Fixed Paper Positioning

The paper should not be framed as "GAM + EWC with better hyperparameters".
The stable positioning is:

> FS-LoRA identifies and addresses a design flaw in combining sharpness-aware optimization with Fisher regularization for continual LoRA. It separates current-task flatness and old-task stability into two measurable gradient channels, calibrates the Fisher scale in the effective adapter-update space `Delta W = BA`, and shows when this design improves performance or fails due to scale and memory-horizon mismatch.

The paper's main contribution is therefore mechanistic:

- Problem: a single shared LoRA adapter drifts across tasks, while naive sharpness-aware Fisher regularization mixes plasticity and stability inside one perturbation closure.
- Mechanism: decouple `g_flat` and `g_fisher`, compute Fisher in `Delta W = BA`, trace-normalize Fisher, and use gradient diagnostics to verify whether the two channels are separable.
- Evidence: near-zero `cos(g_flat, g_fisher)`, raw Fisher scale pathology, coupled-vs-decoupled behavior, and final-matrix task retention diagnostics.
- Scope: FS-LoRA improves several settings, but is not yet a universal SOTA result across all standard PECL benchmarks.

Avoid the claim:

> FS-LoRA beats EWC-LoRA and SeqLoRA everywhere.

Use the claim:

> FS-LoRA provides a principled and measurable way to decouple plasticity and stability in continual LoRA; its effectiveness depends on Fisher activity, flatness scale, and Fisher memory horizon.

## 1. Claim Strength

### Claims to Keep

1. Continual LoRA has a structural stability-plasticity problem because every task updates the same effective adapter `Delta W = BA`.
2. Sharpness-aware optimization and Fisher regularization are functionally distinct:
   - sharpness-aware updates improve current-task flatness/plasticity;
   - Fisher regularization constrains drift in old-task-sensitive directions.
3. Naively applying GAM/SAM to `L_task + L_fisher` couples these signals and makes the sharpness correction no longer a clean current-task flatness estimate.
4. FS-LoRA decouples the update:

   ```text
   g = g_clean + lambda_flat * (g_gam - g_clean) + g_fisher
   ```

5. Fisher must be computed and regularized in effective adapter-update space `Delta W = BA`, not raw LoRA factors.
6. Trace-normalization is needed because raw Fisher in the low-rank update space can be too small and therefore nearly inactive.
7. `cos(g_flat, g_fisher)` is close to zero in the observed runs, supporting a decoupled design.
8. Failure cases are meaningful diagnostics:
   - CIFAR-100 and DomainNet show insufficient old-task anchoring under current settings.
   - OxfordPet shows flatness-scale mismatch.
   - ImageNet-A shows sensitivity to Fisher memory horizon and learning-rate schedule.

### Claims to Remove or Weaken

Remove or rewrite:

- "four of six datasets" unless the final table exactly supports it.
- "consistent reductions in forgetting" because forgetting is not consistently reduced.
- "partial runs exceed EWC-LoRA" unless the run is complete and evaluated by final matrix.
- "FS-LoRA is competitive or superior on all standard benchmarks."
- Any conclusion based on top-accuracy curves instead of final matrix.

Use this safer conclusion:

```latex
Experiments on six fine-grained recognition datasets show that FS-LoRA improves FAA on CUB200, Flowers, and ImageNet-R relative to SeqLoRA+GAM, with the largest gain on Flowers, while showing mixed behavior on Cars196, Aircraft, and OxfordPet.
The results indicate that the decoupled design can be beneficial, but its effectiveness depends on Fisher activity and the relative scale of the flatness correction.
On standard PECL benchmarks, completed ImageNet-A runs show that LR-tuned FS-LoRA approaches the EWC-LoRA reference, while CIFAR-100, ImageNet-R, and DomainNet results should be interpreted as diagnostic extensions until all final-matrix runs are complete.
```

## 2. Recommended Paper Outline

### Title

Keep:

```text
Flat and Stable Continual LoRA Adaptation via Orthogonal Gradient Subspaces
```

Possible more mechanism-focused alternative:

```text
Decoupling Flatness and Fisher Stability in Continual LoRA Adaptation
```

The current title is acceptable if the text consistently treats orthogonality as a diagnostic, not as a guaranteed enforced property.

### Abstract

Structure:

1. Problem:
   - Sequential LoRA is memory-efficient but a single shared adapter drifts and forgets.
   - Existing methods separately address flatness/plasticity or Fisher/stability.
   - Naive combination couples current-task perturbation with old-task Fisher penalty.
2. Method:
   - FS-LoRA computes first-order adapter-subspace sharpness from current-task loss only.
   - It adds Fisher stability as an independent gradient in `Delta W = BA`.
   - It trace-normalizes Fisher to prevent low-rank scale collapse.
3. Mechanism:
   - `g_flat` and `g_fisher` are nearly orthogonal in logs.
   - This supports decoupling as a design principle.
4. Evidence and scope:
   - Improves several fine-grained settings.
   - Reveals mixed behavior when flatness scale or Fisher memory is mismatched.

Do not write the abstract as a broad SOTA claim.

### 1. Introduction

Main flow:

1. Parameter-efficient continual learning with LoRA is attractive because it freezes the backbone and updates a small adapter.
2. A single shared adapter is also the source of forgetting: later tasks move the same `Delta W = BA`.
3. Existing remedies:
   - SAM/GAM: current-task flatness and plasticity.
   - EWC/EWC-LoRA: old-task Fisher stability.
4. Design flaw:
   - Applying sharpness-aware optimization to `L_task + L_fisher` contaminates the perturbation.
   - The flatness correction no longer measures current-task sharpness alone.
5. FS-LoRA answer:
   - current-task flatness and old-task Fisher stability should be separate gradient channels.
6. Contributions:
   - decoupled flat-stable update;
   - trace-normalized Fisher in `Delta W`;
   - constrained optimization interpretation, with optional dual controller;
   - gradient diagnostics and final-matrix evaluation.

Revised contribution 3 should be weak:

```latex
\item \textbf{A constrained optimization interpretation.}
We interpret FS-LoRA as a penalty-method approximation to minimizing current-task loss and first-order adapter-subspace sharpness subject to a Fisher drift budget.
This view clarifies the role of the Fisher weight and yields an optional dual-ascent controller for regimes where the stability constraint becomes active.
```

### 2. Related Work

Organize by problem, not method list:

1. LoRA continual learning:
   - SeqLoRA, IncLoRA, O-LoRA, EWC-LoRA, FR-LoRA.
   - Emphasize that EWC-LoRA is the stability component without flatness control.
2. Sharpness-aware optimization:
   - SAM, GAM, LoRA-SAM, Flat-LoRA.
   - Emphasize that these are mostly single-task or do not study Fisher interaction.
3. Fisher regularization and stability:
   - EWC, Fisher penalties, delta-space regularization.
4. Gradient geometry:
   - GEM/A-GEM/OGD project or constrain gradients.
   - FS-LoRA does not perform gradient surgery; it uses near-orthogonality as a diagnostic for decoupled design.

### 3. Background and Problem Formulation

Recommended subsections:

#### 3.1 Sequential LoRA and Effective Update Space

Define:

```latex
\Delta W_l = B_l A_l.
```

Explain why raw factor coordinates are not the right stability object:

- `(A, B)` is non-unique.
- `Delta W` is the functional update seen by the backbone.
- Forgetting is drift in the effective update, not merely parameter drift.

#### 3.2 Sharpness-Aware Updates in Adapter Space

Define clean gradient and GAM gradient:

```latex
g_{\mathrm{flat}} = g_{\mathrm{gam}} - g_{\mathrm{clean}}.
```

State clearly:

- `g_flat` is a current-task signal.
- The GAM closure must use only `L_task`.

#### 3.3 Fisher Stability in Delta Space

Define Fisher-weighted drift:

```latex
\mathcal R_{\mathrm{Fisher}}
=
\sum_{t'<t}\sum_l
\langle \widetilde F_{t',l},
(\Delta W_l - \Delta W^{\star}_{t',l})^2
\rangle.
```

Explain old-task anchoring.

### 4. Method: FS-LoRA

This should be the central section. Recommended subsections:

#### 4.1 Why Coupled Sharpness-Aware Fisher Is Problematic

State the coupled baseline:

```latex
\mathrm{GAM}(L_t + \lambda_{\mathrm{ewc}} R_{\mathrm{Fisher}})
```

Explain:

- the perturbation depends on both current task and old-task penalty;
- the sharpness correction becomes a mixed signal;
- plasticity and stability become hard to diagnose.

This subsection is important because it defines the design problem, not just the solution.

#### 4.2 Decoupled Flat-Stable Update

Define the actual update:

```latex
g_{\mathrm{FS}}
=
g_{\mathrm{clean}}
+ \lambda_{\mathrm{flat}}(g_{\mathrm{gam}} - g_{\mathrm{clean}})
+ g_{\mathrm{fisher}}.
```

Interpret each term:

- `g_clean`: current-task fitting.
- `lambda_flat (g_gam - g_clean)`: current-task flatness correction.
- `g_fisher`: old-task Fisher stability gradient.

Add the key sentence:

> This update is not a generic sum of losses; it preserves the functional separation between plasticity and stability before the sharpness-aware perturbation is computed.

#### 4.3 Trace-Normalized Fisher in Effective Adapter Space

Keep the trace-normalization subsection, but make it central rather than auxiliary.

Formula:

```latex
\widetilde F_{t,l}
=
\frac{F_{t,l}}
{\sum_{l'} \mathrm{tr}(F_{t,l'}) + \epsilon}.
```

Interpretation:

- raw Fisher in low-rank `Delta W` can be around `1e-6`;
- without normalization, `lambda_ewc` compensates for numerical scale;
- with normalization, `lambda_ewc` acts as a relative stability coefficient.

#### 4.4 Fisher Memory Horizon

The paper and code must be made consistent.

Current code behavior is:

```python
F_accum = gamma * F_old + F_new
```

Recommended paper formula if keeping the code:

```latex
\bar F_t \leftarrow \gamma \bar F_{t-1} + \widetilde F_t.
```

Add explanation:

```latex
Here $\gamma$ controls the retention of past Fisher mass.
Unlike a normalized exponential moving average, this update allows Fisher evidence to accumulate across tasks, so larger $\gamma$ increases the effective memory horizon and strengthens old-task anchoring.
```

Do not write:

```latex
\bar F_t \leftarrow \gamma \bar F_{t-1} + (1-\gamma)\widetilde F_t
```

unless the code is changed to match it.

#### 4.5 Constrained Optimization View

Keep this as interpretation, not empirical contribution.

```latex
\min_\phi
L_t(\phi) + \lambda_{\mathrm{flat}}\mathcal S_{\Delta}^{(1)}(\phi)
\quad
\mathrm{s.t.}
\quad
\widetilde D_{\mathrm{Fisher}}(\phi) \le \delta.
```

Then:

- fixed `lambda_ewc` is the penalty-method approximation used in experiments;
- dual ascent is optional and passive in reported runs;
- do not overclaim dual ascent.

#### 4.6 Gradient-Geometric Diagnostic

Define:

```latex
\cos_{\mathrm{flat,fisher}}
=
\frac{\langle g_{\mathrm{flat}}, g_{\mathrm{fisher}}\rangle}
{\|g_{\mathrm{flat}}\|\|g_{\mathrm{fisher}}\|+\epsilon}.
```

Use this wording:

> We use near-orthogonality as a diagnostic that supports decoupling, not as an assumption enforced by FS-LoRA. FS-LoRA does not perform projection or gradient surgery.

### 5. Experiments

The experiments should be organized around mechanism validation, not just leaderboard-style results.

#### 5.1 Evaluation Protocol

Must state:

- Use final accuracy matrix for primary reporting.
- Report:
  - Task0 final accuracy;
  - FAA from final row;
  - AAA from lower-triangular matrix;
  - forgetting.
- Top accuracy curves are secondary diagnostics and must not be used as final benchmark claims.

#### 5.2 Main Fine-Grained Benchmarks

Purpose:

- show FS-LoRA is useful in several realistic fine-grained CL sequences;
- show mixed cases honestly.

Recommended statement:

```latex
FS-LoRA improves FAA on CUB200, Flowers, and ImageNet-R relative to SeqLoRA+GAM, while showing mixed behavior on Cars196, Aircraft, and OxfordPet.
```

Add a short diagnostic paragraph:

- OxfordPet: flat/clean ratio too large;
- Cars/Aircraft: Fisher activity and flatness scale may be mismatched.

#### 5.3 Mechanism Ablation

This should become a key table.

Minimum rows:

```text
SeqLoRA + SGD
SeqLoRA + GAM only
SeqLoRA + EWC only, no GAM
Coupled GAM on task + Fisher
Decoupled raw Fisher
Decoupled normalized Fisher = FS-LoRA
```

Minimum datasets:

- CUB200: stable fine-grained mechanism test.
- ImageNet-A: memory-horizon and LR-sensitive diagnostic.

Columns:

```text
FAA | AAA | Task0_final | Forget | cos(flat,fisher) | fisher/clean | flat/clean
```

Expected story:

1. Coupled version shows why shared closure is not ideal.
2. Raw Fisher shows Fisher inactivity or weak stability.
3. Normalized Fisher shows stability channel becomes measurable.
4. FS-LoRA works when flatness scale and Fisher memory are matched.

#### 5.4 Standard PECL Benchmarks as Diagnostic Extension

Do not frame this as final SOTA.

Use this framing:

```latex
Track 2 is used as a diagnostic extension rather than a final benchmark claim.
The completed ImageNet-A runs show that FS-LoRA is sensitive to the memory horizon and learning-rate schedule.
Increasing the Fisher accumulation factor improves old-task retention, suggesting that the stability channel is active.
However, current CIFAR-100 and DomainNet runs remain below the EWC-LoRA reference when evaluated by the final accuracy matrix.
These results indicate that FS-LoRA's effectiveness depends on matching the Fisher memory horizon and flatness scale to the task sequence.
```

Current final-matrix evidence to reflect:

```text
ImageNet-A:
  best complete FS-LoRA gamma=0.99, lambda=2000:
  FAA=54.124, AAA=61.852, Task0_final=69.71, Forget=6.603

ImageNet-R:
  best complete FS-LoRA gamma=0.9, lambda=2000, rho=0.2:
  FAA=79.642, AAA=81.875, Task0_final=79.09, Forget=7.777
  gamma=0.99 runs should be treated as incomplete until final matrix is available.

CIFAR-100:
  best current AS1/GAM:
  FAA=87.150, AAA=89.404, Task0_final=84.10
  EWC-LoRA reference:
  FAA=88.070, AAA=90.420, Task0_final=92.10

DomainNet:
  best current AS1/GAM:
  FAA=72.826, AAA=76.355, Task0_final=69.86
  EWC-LoRA reference:
  FAA=73.300, AAA=77.059, Task0_final=77.83
```

Interpretation:

- CIFAR/DomainNet are not current-task learning failures alone.
- They expose old-task anchoring and memory-horizon limitations.
- This supports the diagnostic value of FS-LoRA.

#### 5.5 Gamma Memory-Horizon Study

Add a compact table:

```text
gamma | lambda_ewc | FAA | AAA | Task0_final | Forget | fisher/clean
0.90  | 2000       | ...
0.99  | 2000       | ...
0.99  | 1500       | ...
```

Dataset:

- ImageNet-A first.
- ImageNet-R once complete.

Interpretation:

- `gamma` is the Fisher memory retention factor.
- Larger `gamma` preserves more past Fisher mass.
- It can improve old-task anchoring but may reduce plasticity if too strong.

### 6. Discussion

Recommended subsections:

#### 6.1 What FS-LoRA Solves

State:

- It solves a signal-coupling problem in sharpness-aware Fisher regularization.
- It makes plasticity and stability measurable separately.
- It avoids raw LoRA-factor regularization and uses the functional update `Delta W`.

#### 6.2 When FS-LoRA Works

FS-LoRA works best when:

- `g_flat` and `g_fisher` are near-orthogonal;
- Fisher gradient is active but not dominating;
- flatness correction is not too large relative to clean gradient;
- memory horizon is long enough for early tasks.

#### 6.3 When FS-LoRA Fails

Add this paragraph:

```latex
\paragraph{When does FS-LoRA fail?}
FS-LoRA can fail when the Fisher memory horizon is too short or when the flatness correction dominates the clean task gradient.
CIFAR-100 and DomainNet indicate that final-matrix performance is limited by insufficient old-task anchoring rather than by current-task plasticity alone.
OxfordPet shows a different failure mode: the flatness correction becomes disproportionately large relative to the clean gradient.
These observations suggest two future improvements: adaptive Fisher memory through $\gamma$ and adaptive perturbation scaling through $\rho$ or $\lambda_{\mathrm{flat}}$.
```

#### 6.4 Relationship to EWC-LoRA

Write:

- EWC-LoRA is the stability component.
- FS-LoRA adds a decoupled flatness channel.
- When Fisher dominates, improvements may be small.
- When Fisher is lightly active and flatness is well-scaled, FS-LoRA can improve FAA.

#### 6.5 Limitations

Be explicit:

- Results are not uniformly better across all datasets.
- Some standard PECL runs are diagnostic rather than final.
- `rho`, `lambda_flat`, and `gamma` may require dataset-adaptive calibration.
- The near-orthogonality explanation is empirical/local, not a global theorem.

### 7. Conclusion

Conclusion should match the evidence.

Use:

```latex
FS-LoRA addresses a design issue in continual LoRA optimization: current-task flatness control and old-task Fisher stabilization should not be mixed inside a shared sharpness-aware closure.
By separating these signals, applying Fisher regularization in the effective adapter-update space, and trace-normalizing the Fisher estimate, FS-LoRA provides a measurable flat-stable optimization framework.
The experiments show that this design can improve continual LoRA adaptation in several fine-grained settings and provides useful diagnostics in standard PECL benchmarks.
The mixed results further indicate that flatness scale and Fisher memory horizon are important conditions for success.
```

Remove:

- "consistent reductions in forgetting";
- "four of six" unless exactly true;
- "partial runs exceed";
- any universal superiority claim.

## 3. Appendix and Theory Revision

### Problem

Current appendix uses an overly strong isotropy assumption:

```latex
JJ^\top \approx c^2 I_{Ld^2}.
```

This is not rigorous because `J` maps LoRA factor perturbations into a reachable low-rank update subspace. `JJ^T` cannot be close to full identity in the whole ambient `Delta W` space.

### Fix

Rename:

```latex
Full Proof of Proposition
```

to:

```latex
A Local Explanation for the Near-Orthogonality Diagnostic
```

Replace full-space isotropy with restricted isotropy:

```latex
P_J JJ^\top P_J \approx c^2 P_J,
```

where `P_J` is the projector onto the reachable update subspace `Im(J)`.

Use:

```latex
JJ^\top f = JJ^\top P_J f \approx c^2 P_J f.
```

Conclusion should be:

> When the reachable current-task curvature directions and reachable past-task Fisher directions have small overlap, the flatness and Fisher gradients are expected to have small cosine.

Do not state this as a global guarantee.

## 4. Code-Paper Consistency Checklist

### Fisher Accumulation

Current implementation:

```python
out[key] = old[key] * self._ewc_gamma + new[key]
```

Paper should use:

```latex
\bar F_t \leftarrow \gamma \bar F_{t-1} + \widetilde F_t.
```

unless the code is changed.

### Decoupled Update

Implementation uses:

```text
g_clean + lambda_flat * (g_gam - g_clean) + g_fisher
```

Paper should not describe this as simply optimizing:

```latex
L_task + lambda_flat S + lambda_ewc R
```

without explaining that the GAM closure is task-only.

### Dual Ascent

If `ewc_dual_ascent=false` in all reported runs:

- do not claim dual ascent as an empirical component;
- keep it as optional constrained-view extension.

### Metrics

Primary metrics:

- final matrix FAA;
- final matrix AAA;
- Task0 final;
- forgetting.

Secondary diagnostics:

- top accuracy curve;
- `cos(flat,fisher)`;
- `flat/clean`;
- `fisher/clean`;
- normalized Fisher drift.

## 5. Immediate Editing Priorities

1. Fix conclusion overclaims.
2. Rewrite Track 2 as diagnostic extension using final-matrix metrics.
3. Align Fisher accumulation formula with code: choose `gamma old + new` unless code is changed.
4. Strengthen mechanism-ablation table:
   - Fisher only;
   - flat only;
   - coupled vs decoupled;
   - raw vs normalized Fisher;
   - gamma memory horizon.
5. Revise appendix from proof to local diagnostic explanation with restricted isotropy.
6. Weaken dual ascent contribution.
7. Add failure-mode discussion for CIFAR-100, DomainNet, and OxfordPet.

## 6. Recommended New Experimental Table Set

### Table 1: Main Fine-Grained Results

Rows:

- SeqLoRA + SGD
- SeqLoRA + GAM
- EWC-LoRA
- FS-LoRA

Columns:

- dataset;
- FAA;
- AAA;
- Forget;
- Task0 final.

### Table 2: Mechanism Ablation

Rows:

- SeqLoRA + SGD
- GAM only
- Fisher only
- Coupled GAM + Fisher
- Decoupled raw Fisher
- Decoupled normalized Fisher

Columns:

- FAA;
- AAA;
- Task0 final;
- Forget;
- `cos(flat,fisher)`;
- `fisher/clean`;
- `flat/clean`.

### Table 3: Standard PECL Diagnostic Extension

Rows:

- EWC-LoRA reference
- FS-LoRA best complete
- FS-LoRA gamma/memory variants

Datasets:

- ImageNet-A
- ImageNet-R
- CIFAR-100
- DomainNet

Text must state that this table is diagnostic unless all runs are complete and final-matrix evaluated.

### Figure 1: Mechanism Diagram

Show:

- coupled baseline: one closure over `L_task + R_fisher`;
- FS-LoRA: task-only GAM closure plus independent Fisher gradient in `Delta W`.

### Figure 2: Cosine Diagnostic

Histogram or per-task line:

- `cos(g_flat, g_fisher)` near zero;
- optional: `flat/clean`, `fisher/clean`.

### Figure 3: Failure-Mode Diagnostic

Show one or two examples:

- OxfordPet: high `flat/clean`;
- CIFAR/DomainNet: final Task0 degradation;
- ImageNet-A: gamma improves retention.

## 7. Final Narrative Template

Use this as the paper's through-line:

1. Continual LoRA forgets because a shared `Delta W=BA` must serve all tasks.
2. Flatness-aware optimization and Fisher stabilization target different failure modes.
3. Naively coupling them inside one sharpness-aware closure corrupts the flatness signal.
4. FS-LoRA decouples them into separate gradient channels.
5. Trace-normalized `Delta W` Fisher makes the stability channel active and scale-calibrated.
6. Near-orthogonality verifies that the two channels are usually separable.
7. Final-matrix experiments show where this helps and where scale/memory calibration is still needed.

This is the strongest and most defensible version of the paper.
