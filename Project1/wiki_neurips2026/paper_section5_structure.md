# Section 5 Structure — NeurIPS 2026 Paper

**Finalized 2026-04-30**

---

## Main Text Structure (finalized 2026-04-30)

```latex
\section{Experiments}

\subsection{Experimental Setup}

\subsection{RQ1: Does Sharpness-Aware Optimization Affect the Posterior Trajectory?}
  → Main experiment: Exp F (ImageNet-R, trajectory factorial = forked trajectory)
  → Theorem linked: B.1 (sequential PAC-Bayesian decomposition)
  → NO new training needed: Exp F already IS the 2×2 early/later intervention

\subsection{RQ2: Which Perturbation Geometry Is Relevant in PECL?}
  → Main experiment: Exp E (ImageNet-R, support comparison: sam_full / sam_factor / sam_delta)
  → Theorem linked: B.2 (PECL support-reduction)
  → NO new training needed: use 3 rows from Exp E + post-hoc sharpness correlation

\subsection{RQ3: Which Sharpness Objective Works Best within the Adapter-Update Support?}
  → Main experiment: Exp E (random_delta vs sam_delta) + NEW gam_delta if implemented
  → AS(0) vs RS(0) already in Exp E; AS(1)/GAM-delta is new if desired

\subsection{RQ4: Robustness across LoRA Organizations}
  → New experiments: IncLoRA + OLoRA × {sgd, sam_delta, sam_factor}
  → 6 new runs needed (SeqLoRA already done)

\subsection{Post-hoc Sharpness Diagnostics}
  → One figure: Sh_delta vs task index, four curves (no new training)
  → One sentence: "consistent with TS^Delta_t being a contributor"
  → NOT a causal claim
```

---

## Appendix Structure

```latex
\appendix

\section{CIFAR10 Pilot Experiments}         % Exp C (Q2 pilot) + Exp D (Q1 pilot)

\section{Reparameterization Control}         % Exp G / Exp Q2
  Confirms sam_delta is scale-invariant, sam_factor is not.
  Implementation guarantee: seqlora.py uses orthonormal columns of B and A.T
  which are invariant under (A/c, cB) scaling.

\section{Multi-seed Robustness}              % Exp H
  Exp E subset: sgd, sam_factor, sam_delta, random_factor, random_delta
  Exp F subset: sgd_sgd, sgd_sam_factor, sgd_random_factor
  Seeds: 1993, 42, 1234

\section{Implementation Details of Perturbation Supports}
  Table: variant name / direction / support definition / code function
  Clarify: sam_random ≠ random_delta (different axes entirely)
```

---

## Key Technical Facts for Writing

### ΔW convention (confirmed from backbone/lora.py:211-212)
```python
delta_w = self.linear_b_q.weight @ self.linear_a_q.weight   # ΔW = B @ A
```
→ Reparameterization: A' = A/c, B' = cB → B'A' = BA = ΔW unchanged

### sam_delta invariance (from seqlora.py:633-634)
```python
U = orthonormal_columns(B)     # column space of B — invariant under B → cB
V = orthonormal_columns(A.T)   # column space of A.T — invariant under A → A/c
```
→ sam_delta perturbation direction is exactly invariant under (A/c, cB). Not just theoretical.

### Naming conventions (must be consistent throughout paper)
```
[direction]_[support] convention:
  sam_factor    = adversarial SAM direction, raw LoRA factor support
  random_factor = Gaussian random direction, raw LoRA factor support
  sam_delta     = adversarial SAM direction, effective adapter-update tangent support
  random_delta  = Gaussian random direction, effective adapter-update tangent support
  sam_full      = adversarial SAM direction, merged effective weight support
  random_full   = Gaussian random direction, merged effective weight support
  sam_all       = adversarial SAM direction, all raw parameter support
  random_all    = Gaussian random direction, all raw parameter support
  sam_frozen    = adversarial SAM direction, frozen backbone coordinate support
  random_frozen = Gaussian random direction, frozen backbone coordinate support

sam_random ≠ random_delta. sam_random = "SAM gradient projected onto random matched
tangent" (seqlora.py: _project_grad_to_random_matched_tangent). NOT in Exp E.
Do not use this name in the paper unless explicitly defining and using it.
```

---

## Critical Constraints on Paper Text

### Q1 (RQ1) — what NOT to write
- ❌ "SAM directly prevents forgetting"
- ❌ "SAM makes solutions flatter and therefore more resilient"
- ✓ "SAM changes the posterior trajectory in a way that reduces later degradation of earlier-task performance"
- ✓ "From the same posterior state Q_9, suffix SAM changes the subsequent posterior trajectory and reduces prefix forgetting"

### Q2 (RQ2) — protective statement (MUST be last sentence of RQ2, with forward ref)
```latex
Exp E supports the necessity of the adversarial sharpness-aware direction,
but does not by itself establish raw factor-space perturbation as the
theoretically correct object: the factor-support advantage is specific to the
default LoRA parameterization.
Appendix~\ref{app:reparam} shows that under reparameterization preserving
$\Delta W$, SAM-delta's performance is stable while SAM-factor's varies,
confirming that the bound-relevant object is the basis-invariant effective
adapter-update geometry.
```

### Post-hoc sharpness — diagnostic NOT causal
- ❌ "lower sharpness causes lower forgetting"
- ✓ "consistent with $\mathrm{TS}^\Delta_t$ being a contributor to trajectory-level risk"

---

## Perturbation Norm Fairness Issue (must address in paper)

The three support variants (sam_full, sam_factor, sam_delta) use the same ρ in their
respective coordinate spaces, but the effective ΔW perturbation magnitude differs:
- sam_factor: ||B δA + δB A||_F ≠ sam_delta: ||ε_ΔW||_F at same ρ

Option A (conservative): Add a Setup sentence explaining this; interpret comparisons
as "easier to optimize under fixed coordinate budget" not "equal effective magnitude."

Option B (stronger): Run norm-equalized variant in Appendix (normalize all on ||δΔW||_F).

Recommendation: Option A for main text, Option B as Appendix if compute allows.

---

## Main Tables

### Table 1 (RQ1, from Exp F): Forked Trajectory
Group rows by prefix optimizer using multirow. Columns: FAA, BWT, Prefix Forget, Suffix Acc.
Key: sgd|random-factor ≈ sgd|sgd → controls for noise.

### Table 2 (RQ2, from Exp E): Support × Direction Contrast
Use pairwise contrast format (NOT 11-row raw table).
Columns: Support | SAM FAA | Random FAA | ΔFAA | SAM BWT | Random BWT | ΔBWT
+ SGD baseline row at bottom.
ΔFAA column is the primary reading signal.
