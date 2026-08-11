---
id: exp_Q1
name: Per-task Sharpness × Per-task Forgetting Correlation
dataset: ImageNet-R r=16 T=20 (Exp E checkpoints)
method: SeqLoRA, ViT-B/16, post-hoc measurement — NO new training
status: pending — next after Exp G
answers: Q1 (Theorem B.1 direct validation)
---

## Motivation

**Q1** requires: task-i sharpness-smoothed risk predicts R_i(Q_t) for t > i.

**Gap in current experiments**: Exp E/F show "SAM training → lower aggregate forgetting" but do not directly test "sharpness of θ_i at training time → forgetting of task i at task T". The causal direction needs to be tested per-task.

**Core prediction of Theorem B.1**: regardless of which optimizer produced θ_i, the sharpness Sh_delta(θ_i) should positively correlate with eventual per-task forgetting F_i.

---

## Design

**No new training required.** Uses existing Exp E checkpoints.

### Step 1: Load checkpoints

For each variant in Exp E: `sgd`, `sam_factor`, `sam_delta`, `random_factor`, `random_delta`  
For each task position i ∈ {0, 4, 9, 14, 19}: load checkpoint θ_i

### Step 2: Post-hoc sharpness measurement at each task i

For each checkpoint θ_i, measure two sharpness quantities:

```
Sh_delta(θ_i) = max_{||ε||≤ρ, ε ∈ T_Delta_i} [ L(θ_i + ε, D_i) - L(θ_i, D_i) ]
Sh_AB(θ_i)    = max_{||(εA,εB)||≤ρ} [ L(θ_i + ε, D_i) - L(θ_i, D_i) ]
```

Where D_i is a fixed deterministic subset of task-i data (same batch for base and perturbed loss; non-negative sharpness constraint).

**Implementation requirements** (critical — avoid previous noisy sharpness issue):
- `flat_eval_loader_shuffle: false`
- Fixed deterministic subset (same data across all checkpoints for comparability)
- Use task-specific data D_i (not mixed)
- Sharpness must be non-negative

### Step 3: Compute per-task forgetting

F_i = A_{i,i} - A_{i,T=19}

This is already available in Exp E results (full accuracy matrix). If not logged, re-run evaluation only (no training).

### Step 4: Correlation analysis

Main plot: scatter Sh_delta(θ_i) vs F_i  
- All variants × all tasks as separate points
- Color by variant (sgd=gray, sam_factor=red, sam_delta=blue, random_factor=orange)
- Fit a linear/Spearman correlation across all points
- Expected: strong positive correlation (r > 0.7), consistent across variants

Secondary plot: Sh_AB(θ_i) vs F_i  
- Expected: weaker/noisier correlation than Sh_delta
- This comparison directly supports Q2: Sh_delta is a better predictor than Sh_AB

---

## Predicted Results

| Pattern | Expected | What it proves |
|---------|----------|---------------|
| Sh_delta(θ_i) vs F_i: positive correlation | Yes, across all variants | Theorem B.1: sharpness at task i predicts forgetting |
| Sh_AB(θ_i) vs F_i: weaker correlation | Yes | Sh_AB is noisier theoretical object (Q2 hint) |
| Within sgd run: natural variation in Sh_delta across tasks correlates with variation in F_i | Yes | Effect is not only about SAM training, but about the sharpness value itself |
| sam_factor checkpoints have lower Sh_delta than sgd | Yes | SAM training produces flatter solutions in W_Δ space |

---

## Key Figure

**Figure Q1a**: 2D scatter plot
- x: Sh_delta(θ_i) measured at time of learning task i
- y: F_i = A_{i,i} - A_{i,T} (eventual forgetting of task i)
- Points: each (variant, task) pair
- Message: "Higher task-i sharpness predicts higher eventual forgetting of task i"

**Figure Q1b**: Same plot with Sh_AB(θ_i) on x-axis
- Message: "Raw factor sharpness is a noisier predictor, consistent with non-invariance (Q2)"

---

## Connection to Theory

This experiment directly validates the chain:
```
SAM training at task i  →  lower Sh_delta(θ_i)  →  lower R_i(Q_t) for t > i  →  less forgetting F_i
```

The correlation is evidence that Sh_delta(θ_i) is not merely a proxy for "SAM was used" — it is the mechanistically relevant quantity, regardless of which optimizer produced the value.
