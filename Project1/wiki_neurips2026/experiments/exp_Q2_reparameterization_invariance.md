---
id: exp_Q2
name: Reparameterization Invariance Training Test
dataset: ImageNet-R r=16 T=20 (Exp F task-9 checkpoint as starting point)
method: SeqLoRA, ViT-B/16, seed 1993
status: pending — priority 2 after Exp Q1
answers: Q2 (Theorem B.2 direct validation) + resolves sam_factor > sam_delta tension
---

## Motivation

**Q2** requires: the bound-relevant perturbation support is W_{Δ,t} (basis-invariant effective adapter update).

**Theorem B.2 predicts**:
- Sh_delta is invariant under (A,B) → (A/c, cB): same ΔW, different factorization, same sharpness
- Sh_AB is NOT invariant: same ΔW, but different factorization, different sharpness

**Training prediction**: 
- sam_delta's training effect should be STABLE across reparameterizations (it perturbs in ΔW tangent space, which is invariant)
- sam_factor's training effect should VARY with reparameterization (it perturbs in raw (A,B) coordinates, which change with c)

**Resolves the sam_factor > sam_delta tension**:
- Exp E shows sam_factor (FAA 67.94) > sam_delta (FAA 65.41) at standard initialization
- Possible explanation: at standard LoRA init (A ~ N(0,1/r), B ≈ 0), the scale is c ≈ 1 and both are similar; sam_factor has a practical advantage at this particular scale
- If reparameterization changes sam_factor's performance but not sam_delta's → proves sam_factor's advantage is scale-dependent artifact, and sam_delta is the theoretically invariant object

---

## Design

### Step 0: Starting point

Use Exp F SGD-prefix task-9 checkpoint (hash-verified: same initial state for sgd_sgd, sgd_sam_factor, sgd_random_factor). This is a well-characterized intermediate checkpoint after 10 tasks.

### Step 1: Apply reparameterization

For each LoRA layer in the checkpoint:
```python
# For scale c:
A_new = A / c   # shape (r, d_in)
B_new = B * c   # shape (d_out, r)

# Verify:
assert torch.allclose(B_new @ A_new, B @ A)   # ΔW unchanged
assert forward_pass(A_new, B_new) == forward_pass(A, B)   # model output unchanged
```

Scales c ∈ {0.5, 1.0, 2.0, 4.0}. Note c=1.0 is the original (no change).

### Step 2: Post-hoc sharpness verification (at task-9 checkpoint)

Before training, verify at c=1 checkpoint:
- Sh_AB(c) varies with c: expected ~ proportional to c² or similar
- Sh_delta(c) stable: should be ~identical across all c
- Accuracy: identical across all c

This is the "measurement" part of Q2. Record results in a table.

### Step 3: Continue training tasks 10-19

For each (c, optimizer) combination:

| c | optimizer | run name |
|---|-----------|----------|
| 0.5 | sam_factor | reparam_c05_sam_factor |
| 0.5 | sam_delta | reparam_c05_sam_delta |
| 0.5 | sgd | reparam_c05_sgd |
| 1.0 | sam_factor | reparam_c10_sam_factor (= sgd_sam_factor from Exp F) |
| 1.0 | sam_delta | reparam_c10_sam_delta |
| 1.0 | sgd | reparam_c10_sgd (= sgd_sgd from Exp F) |
| 2.0 | sam_factor | reparam_c20_sam_factor |
| 2.0 | sam_delta | reparam_c20_sam_delta |
| 2.0 | sgd | reparam_c20_sgd |
| 4.0 | sam_factor | reparam_c40_sam_factor |
| 4.0 | sam_delta | reparam_c40_sam_delta |
| 4.0 | sgd | reparam_c40_sgd |

Total: 12 runs × 10 tasks each.

Note: c=1.0 runs reuse existing Exp F data (sgd_sam_factor and sgd_sgd) for sam_factor and sgd.

### Step 4: Measure performance

For each run: FAA, AAA, BWT, Prefix Forget (tasks 0-9), Suffix Acc (tasks 10-19).

---

## Predicted Results

### Main prediction table

| optimizer | c=0.5 FAA | c=1.0 FAA | c=2.0 FAA | c=4.0 FAA | Interpretation |
|-----------|-----------|-----------|-----------|-----------|----------------|
| sgd | ~61.7 | 61.72 | ~61.7 | ~61.7 | Stable (baseline) |
| sam_factor | varies | 66.38 | varies | varies | **Scale-dependent** |
| sam_delta | ~65.x | ~65.x | ~65.x | ~65.x | **Scale-invariant** |

The specific direction of sam_factor variation with c depends on the scale of B after 10 tasks. Key prediction: variance of FAA across c values should be significantly larger for sam_factor than for sam_delta.

### Secondary prediction: at what c does sam_factor ≈ sam_delta?

This tells us the "natural scale" at which factor-space ≈ update-space. This is informative about the LoRA parameterization geometry.

---

## Key Figure

**Figure Q2a**: Line plot, x = log(c), y = FAA
- Three lines: sam_factor, sam_delta, sgd
- sam_factor: non-flat (varies with c)
- sam_delta: flat (invariant across c)
- sgd: flat (baseline)
- Message: "sam_delta's training effect is invariant to LoRA factorization; sam_factor's is not"

**Figure Q2b**: Same with Prefix Forget on y-axis
- Highlights that old-task retention under sam_factor is c-dependent

**Table**: Post-hoc Sh_AB(c) and Sh_delta(c) at task-9 checkpoint
- Sh_AB: varies; Sh_delta: stable
- Message: "Sh_AB is not a basis-invariant quantity; Sh_delta is"

---

## Connection to Theory

Theorem B.2 states: under frozen-backbone LoRA, the bound-relevant sharpness and KL both reduce to W_{Δ,t}.

This means the correct sharpness object is:
```
Sh_delta(θ) = max_{ε ∈ T_Delta, ||ε||≤ρ} [L(θ+ε) - L(θ)]
```
where T_Delta = span{B dA + dB A} is the tangent space of the effective update manifold. This is invariant under (A,B) → (A/c, cB).

The experiment directly operationalizes this: if sam_delta's training result is invariant, it confirms W_{Δ,t} is the correct support. If sam_factor's is not, it confirms raw factor space is NOT the correct theoretical object — despite being empirically competitive at standard scale.

---

## Notes on the sam_factor > sam_delta gap

At standard LoRA init (A ~ N(0,1/r), B=0 at task start):
- Initial ΔW ≈ 0 so T_Delta is ill-defined
- After a few gradient steps, B grows and T_Delta becomes well-defined
- sam_factor at this point is approximately perturbing in A-space (since B is small, the factor-space perturbation is mostly (εA, 0))

This means sam_factor at standard init is approximately a "perturbation in A-space only", which may accidentally be a better-conditioned direction than the full sam_delta computation. The reparameterization test at c=2,4 (large B') will show if enlarging B changes this behavior.
