# Theory Comparison: Our Paper vs. Prior PAC-Bayes CL Works

## Three Works in Context

| | Pentina & Lampert (ICML 2014) | Friedman & Meir (ICLR 2026) | Our Paper |
|---|---|---|---|
| **Task model** | i.i.d. tasks from fixed meta-distribution | Sequential tasks, arbitrary | Sequential tasks, CL-specific |
| **Paradigm** | Meta-learning (NOT CL) | Cumulative loss / plasticity | Stability-plasticity tradeoff |
| **Prior structure** | Single fixed hyperdistribution π; P_t ~i.i.d. π | Flat chain: P_t = Q_{t-1} | Time-varying hyperposterior process {P_t}: P_t ~ P_{t-1} ∈ M(M(H)) |
| **Drift term** | None (stationary) | Implicit: KL(Q_t‖Q_{t-1}) | Explicit: KL(P_t‖P_{t-1}) — cross-task interference |
| **Sharpness** | None | None | TS_t^Δ sharpness term |
| **PECL/LoRA** | None | None | Frozen-backbone subspace reduction |
| **Bound target** | Average generalization loss | Cumulative loss (CuL) | Trajectory-averaged test risk |
| **Special cases** | — | — | P&L is special case (drift=0); F&M is separate dimension |
| **Technical novelty** | Hierarchical PAC-Bayes (meta-learning) | "Straightforward extension" (self-described) | New hyperposterior process, drift penalty, subspace reduction |
| **Concurrent with us** | No (2014) | YES (ICLR 2026, submitted ~same time) | — |

---

## Detailed Analysis: Why F&M is Structurally Different

### 1. The Prior Structure: Flat Chain vs. Genuine Hierarchy

**F&M's structure (flat posterior chain):**
- P_1 = P (data-free prior, fixed)
- P_t(S_{1:t-1}) = Q_{t-1}(S_{1:t-1}) for t ≥ 2
- The prior for task t IS the posterior from task t-1
- This is a deterministic mapping: given Q_{t-1}, the prior P_t is determined
- There is NO distribution over distributions — only M(H) at each step
- Conceptually: "the model I learned yesterday is my starting point for today"

**Our structure (genuine hierarchical hyperposterior):**
- P_{t-1} ∈ M(M(H)) — a distribution OVER task-level prior distributions
- P_t is DRAWN (sampled) from P_{t-1}: the prior is stochastic
- This is a two-level hierarchy:
  - Level 1 (within task): Q_t adapts to P_t given S_t → cost: E_{P_t~P_{t-1}} KL(Q_t‖P_t)
  - Level 2 (cross task): P_{t-1} → P_t as new tasks arrive → cost: KL(P_t‖P_{t-1})
- Conceptually: "the meta-prior that generates task priors evolves with experience"

**Why this matters for CL**: In CL, what changes between tasks is not just "which hypothesis I hold" (which is what Q_{t-1} tracks) but "what kind of hypothesis space I believe is appropriate for this sequence" (which is what P_{t-1} tracks). EWC shapes the prior geometry via Fisher information — this changes the META-BELIEF, not just the belief. IncLoRA restricts which parameters are trainable — this changes what W_{Δ,t} is available as a prior support, again changing the META-BELIEF. F&M's flat chain misses this meta-level evolution entirely.

### 2. Decomposition of Cross-Task Interference

**F&M's KL term**: KL(Q_t‖P_t) where P_t = Q_{t-1}
- This measures: how much does the posterior need to change from task to task?
- This is a PLASTICITY measure: it captures how much the model must adapt
- It does NOT distinguish between adaptation cost and interference cost

**Our decomposition**:
- Within-task: E_{P_t~P_{t-1}} KL(Q_t‖P_t) — how much the learner deviates from the task prior
- Cross-task drift: KL(P_t‖P_{t-1}) — how much the inductive bias generation mechanism changes
- These two terms are conceptually and mathematically distinct:
  - Large KL(Q_t‖P_t) + small drift = the learner adapts a lot but the meta-prior is stable → high plasticity, low interference
  - Small KL(Q_t‖P_t) + large drift = the learner stays close to the prior but the meta-prior shifts dramatically → stability under a changing meta-belief

This decomposition is exactly what's needed to analyze the stability-plasticity tradeoff — something F&M's framework cannot do.

### 3. Bound Target: Cumulative Loss vs. Average Test Risk

**F&M**: Bound on CuL = sum_{t=1}^T L(Q_t, D_t) — cumulative error at CURRENT task t when learning on task t
- This measures PLASTICITY (forward transfer): how well does the model learn each new task?
- Self-described: "to the best of our knowledge, this is the first general upper bound on learning plasticity for continual learning"
- Does NOT bound forgetting / stability on previous tasks
- F&M's key result (Theorem 3.2) self-notes: "Corollary 3.1 is a straightforward extension of Theorem A.1 and requires no new technical tools"

**Our paper**: Bound on (1/T) Σ_t L_t — trajectory-averaged test risk across all tasks
- This bounds BOTH stability (performance on previous tasks) and plasticity (adaptation to new tasks)
- The drift penalty KL(P_t‖P_{t-1}) directly quantifies cross-task interference (forgetting)
- This is the correct CL objective — the full stability-plasticity tradeoff

### 4. Addressing the Sequential Requirement of CL

**P&L 2014 fails the sequential requirement** because:
- Assumes tasks are i.i.d. from a SINGLE time-invariant hyperdistribution
- No ordering of tasks matters — shuffle the task sequence and the bound is identical
- This is meta-learning, not CL: there is no notion of "tasks arrive in sequence and earlier tasks affect later priors"

**F&M partially addresses it** by chaining posteriors, but:
- Still no hyperposterior hierarchy
- Still only bounds plasticity (not forgetting)
- Their framework reduces to online learning (Haddouche & Guedj 2022) as a special case

**Our paper fully addresses it** because:
- {P_t} is F_{t-1}-measurable (adapted to task history)
- P&L 2014 is a SPECIAL CASE of our Theorem 4.1 when drift = 0
- The drift KL(P_t‖P_{t-1}) directly captures sequential ordering effects
- Different task orderings give different drift terms → the bound correctly reflects ordering

### 5. F&M is Concurrent Independent Work

**Timeline note**: Our paper was first submitted to ICML 2025 (submitted last year, reviewed as ICML 2026). F&M was published at ICLR 2026. These are genuinely concurrent, independent works targeting different aspects of PAC-Bayes for CL.

**Complementarity**: F&M provides plasticity bounds for general CL algorithms. Our paper provides stability-plasticity bounds specifically for LoRA-based PECL with sharpness analysis. The two works are complementary rather than competing.

---

## Key Sentences for Rebuttal (Draft)

**One sentence version**:
"Friedman & Meir (ICLR 2026) is a concurrent independent work that uses a flat posterior chain (P_t = Q_{t-1}) to bound only the cumulative plasticity loss, while our framework introduces a genuine time-varying hyperposterior process P_{t-1} ∈ M(M(H)) with an explicit cross-task drift penalty to bound the full stability-plasticity tradeoff, and further specializes to LoRA-based PECL via the frozen-backbone subspace reduction — three contributions absent from both Friedman & Meir and Pentina & Lampert."

**Structured version for rebuttal**:
"We are aware of Friedman & Meir (ICLR 2026) as a concurrent independent work (our initial submission predates ICLR 2026 publication). Their framework uses a flat posterior chain (P_t = Q_{t-1}) and bounds only the cumulative plasticity loss — indeed, their Corollary 3.1 is self-described as a 'straightforward extension requiring no new technical tools.' Three fundamental differences separate our work: (1) **Prior structure**: we introduce a genuine hyperposterior process {P_t} ∈ M(M(H)) adapted to task history F_{t-1}, where P_t is sampled from P_{t-1} — F&M have no distribution over distributions. (2) **Explicit drift decomposition**: our bound separates within-task adaptation cost (E KL(Q_t‖P_t)) from cross-task inductive bias drift (KL(P_t‖P_{t-1})) — F&M's KL(Q_t‖Q_{t-1}) is a single plasticity measure that cannot distinguish these. (3) **Sharpness + PECL specificity**: we integrate sharpness-dependent terms and reduce to adapter subspace under frozen backbone (Lemma 4.2+4.3, Theorem 4.4) — entirely absent from F&M. Furthermore, Pentina & Lampert (ICML 2014) assumes i.i.d. tasks from a fixed meta-distribution (meta-learning, not CL) — our Theorem 4.1 contains P&L as a special case when drift is zero. Our work is the first PAC-Bayes framework that correctly models the sequential, history-dependent evolution of inductive bias in CL."
