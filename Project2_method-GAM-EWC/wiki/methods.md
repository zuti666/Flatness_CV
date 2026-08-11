# Methods

## Inheritance Hierarchy

```
SeqLoRALearner
  └── ewclora_youyue_fitArchitechture.py     [Layer 1 — Base EWCLoRA]
        └── ewclora_youyue_fitArchitechture_gam.py  [Layer 2 — Explicit decoupled GAM]
              └── as1_normfisher_gam.py              [Layer 3 — Normalized Fisher + dual]
```

---

## Layer 1: Base EWCLoRA (`ewclora_youyue_fitArchitechture`)

**Model name in config:** `ewclora_youyue_fitarchitecture`

**What it does:**
- Defines ΔW = B @ A infrastructure: `_iter_delta_terms` yields `ΔW_Q`, `ΔW_V` per block
- Computes past-task Fisher in ΔW space: `grad²/N` accumulated over training batches
- EWC penalty: `0.5 * λ_ewc * Σ_k (F_k + η) * (ΔW_k − ΔW_k*)²`
- When `optimizer_type=gam`: passes `L_task + EWC_penalty` into GAM's closure

**Critical bug (for our purpose):** GAM perturbs to find the worst point of `L_task + EWC_penalty` jointly. So the AS^1 flatness direction is computed on a mixed objective — Fisher contaminates the perturbation. AS^1 and Fisher are **coupled**.

**Config files:** `layer1_mixed_gam_rawfisher_cub200_t20.yaml`  
**Key params:** `ewc_lambda=20, ewc_eta=1e-8`

---

## Layer 2: Explicit Decoupled GAM (`ewclora_youyue_fitArchitechture_gam`)

**Model name in config:** `ewclora_youyue_fitarchitecture_gam`

**What it does (completely overrides `_step_batch`):**
1. `g_clean`: gradient of `L_task` only (no Fisher)
2. `g_gam`: GAM's perturbed gradient (GAM closure contains `L_task` only — Fisher excluded)
3. `g_fisher`: gradient of Fisher penalty, computed separately via `_compute_fisher_grad_cache`
4. Final update: `g_clean + λ_flat * (g_gam − g_clean) + g_fisher`

**This is the theoretically correct implementation:** AS^1 = `g_gam − g_clean` measures purely current-task loss curvature. Fisher is an independent additive term for old-task stability.

**Mechanism tracking:** Logs `cos_flat_fisher`, `fisher_to_clean_norm_ratio`, `flat_to_clean_norm_ratio` every batch.

**Config files:** `layer2_explicit_gam_rawfisher_cub200_t20.yaml`, ablation configs  
**Key params:** `lambda_flat=1.0, ewc_lambda=20, mechanism_eval=true`

**Problem identified:** Raw Fisher in ΔW space has values ~1e-6 (mean). With `ewc_lambda=20`, Fisher penalty ≈ 2.4e-4, task loss ≈ 0.37, ratio = **0.06%**. Fisher gradient is structurally inactive regardless of λ.

---

## Layer 3: Normalized Fisher + Dual Ascent (`as1_normfisher_gam` / `ewclora_normfisher_gam`)

**Model names in config:** `as1_normfisher_gam` OR `ewclora_normfisher_gam` (both registered in factory.py)

**Adds over Layer 2:**

### Trace-Normalized Fisher
```
F̃ = F / (Σ_k trace(F_k) + eps)
```
After normalization, `||F̃||` is scale-invariant. With `λ_ewc=500–2000`, the Fisher penalty now contributes a predictable fraction of the task loss. No need for λ ≈ 31,000 (which would be required with raw Fisher to get 1% penalty ratio).

Override: `_compute_delta_fisher` → calls `super()` then divides by total trace.

### Adaptive Dual Ascent (optional, `ewc_dual_ascent=true`)
```
λ_new = clip(λ_old + α * (D^Fisher_normalized − δ), λ_min, λ_max)
```
Where `D^Fisher = Σ_k F_k * (ΔW_k − ΔW_k*)² / ||ΔW − ΔW*||²` is the normalized drift.
Makes the method a penalty-method approximation to the constrained problem:
```
min  L_task + λ_flat * S^(1)_Δ     s.t.  D^Fisher_normalized ≤ δ
```

Override: `after_task` → computes drift metrics, then calls `super().after_task()`, then updates λ.

**Key params (current best):** `lambda_flat=1.0, ewc_lambda=2000, ewc_normalize_fisher=true, ewc_dual_ascent=false`  
**Config files:** `as1_normfisher_lam{500,2000,5000}_cub200_t20.yaml`, `as1_normfisher_dual_cub200_t20.yaml`

---

## Comparison to Original EWC-LoRA Paper (Zheng et al., ICLR 2026)

### Code file mapping

| File | What it is |
|------|-----------|
| `ewclora_youyue_github.py` | GitHub-style wrapper. `EWCLoRA(BaseLearner)` uses raw (A,B) Fisher hooks and `accumulate_and_reset_lora()` — **not** aligned with paper's F_ΔW. The bottom `Learner` just delegates to `ewclora.py`. |
| `EWC_Yaoyue_LoRA_liying.py` | Our adaptation: temporarily enables grad on frozen qkv weights for Fisher, handles head EWC, but **does not merge LoRA** after each task (diverges from Algorithm 1 Step 4). |
| `ewclora_youyue_fitArchitechture.py` | **Used for reproduction**: F_ΔW via ΔW hooks, same penalty form. But see lifecycle difference below. |

### Critical lifecycle difference: merge vs no-merge

| Aspect | Paper `low-rank-cl` | Our `ewclora_youyue_fitArchitechture` |
|--------|--------------------|------------------------------------|
| After each task | `accumulate_and_reset_lora()`: merge B@A into backbone, reset B=0 | No merge. A,B persist and accumulate across all tasks. |
| Penalty | `F_k * ΔW_k²` (relative to implicit 0 after reset) | `F_k * (ΔW_k − ΔW_ref_k)²` (ref snapshotted at task start) |
| Optimizer state | Fresh Adam after each task (new LoRA params) | Adam momentum carries over all tasks |
| Mathematical intent | Equivalent — both penalise task-t drift from starting point | Same |
| Practical difference | Task-1 gradients don't contaminate Task-2 Adam buffer | May converge differently at later tasks |

**Consequence**: Our EWC-LoRA is a valid variant, not bit-for-bit identical to the paper. Results should be comparable but may differ by a few pp. We run it as our **internal EWC-LoRA baseline**, not a claim of exact reproduction.

### Paper's exact ImageNet-R config (Table 8, Appendix A.2.2)

| Setting | Paper (low-rank-cl) | Our reproduction config | Our method (Layer 3) |
|---------|--------------------|-----------------------|---------------------|
| Optimizer | Adam (β1=0.9, β2=0.99) | Adam (β2=0.999, minor diff) | GAM (SGD-based) |
| LR | 0.0005 | 0.0005 | 0.01 |
| Epochs | 50 | 50 | 20 |
| Rank | 10 | 10 | 16 |
| Weight decay | 0.005 | 0.005 | 0 |
| λ (ewc_lambda) | **1e7** | 1e7 | 2000 (norm'd) |
| Fisher gamma | 1.0 (code) | 1.0 | 1.0 |
| Tasks | 10 × 20 classes | 10 × 20 classes | 20 × 10 classes |
| Fisher type | F_ΔW (hook-based) | F_ΔW (hook-based) | Trace-normalized |
| Flatness | None | None | AS^1 (GAM) |
| LoRA lifecycle | merge+reset per task | no merge (drift ref) | no merge (drift ref) |

**Paper result**: A10=72.86±0.79%, Avg=78.95±0.86%

### Reproduction configs and run scripts

| Dataset | Config | Paper split | Epochs |
|---------|--------|-------------|--------|
| ImageNet-R | `exp_paperA_reproduce_ewclora/ewclora_reproduce_imagenetr_t10_r10.yaml` | 10t×20c | 50 |
| CIFAR-100 | `exp_paperA_reproduce_ewclora/ewclora_reproduce_cifar100_t10_r10.yaml` | 10t×10c | 20 |
| ImageNet-A | `exp_paperA_reproduce_ewclora/ewclora_reproduce_imageneta_t10_r10.yaml` | 10t×20c | 20 |

All on GPU 5 sequentially:
```bash
# Runs cifar100 → imageneta → imagenetr in sequence on GPU 5
bash scripts/exp_paperA/run_reproduce_ewclora_all.sh
# Or parallel on multiple GPUs:
GPU_INR=1 GPU_C100=5 GPU_INA=6 bash scripts/exp_paperA/run_reproduce_ewclora_all.sh
```
Logs: `logs_paperA/reproduce_ewclora/`

**Dataset name**: CIFAR-100 uses `"cifar224"` (applies `build_transform` → `RandomResizedCrop(224)` for ViT). Plain `"cifar100"` stays at 32×32 and crashes ViT.

### What our method adds over EWC-LoRA

| Aspect | EWC-LoRA (paper) | Our Layer 3 |
|--------|-----------------|-------------|
| Update rule | `L_task + λ * EWC` single backward | `g_clean + λ_flat(g_gam−g_clean) + g_fisher` |
| Flatness | None | AS^1 on current-task geometry only |
| Fisher normalization | Raw (λ=1e7 needed) | Trace-normalized (λ=2000 sufficient) |
| Fisher scale issue | Overcome by huge λ | Overcome by normalization |
| Gradient decoupling | Fisher mixed into loss | Fisher added as separate direction |
| Mechanism tracking | None | cos_flat_fisher, fisher/clean ratio |
| Lagrangian | Penalty only | Optional dual ascent |
| Optimizer | Adam | GAM (requires SGD base) |

---

## Mechanism Statistics Explained

Logged at end of each task by Layer 2 and Layer 3:

| Stat | Meaning | Observed value |
|------|---------|---------------|
| `cos_flat_fisher` | Cosine between `g_flat=(g_gam−g_clean)` and `g_fisher` | −0.02 to −0.04 |
| `fisher_to_clean_norm_ratio` | `||g_fisher|| / ||g_clean||` | 0.003–0.06 |
| `flat_to_clean_norm_ratio` | `||g_flat|| / ||g_clean||` | ~4.3 |
| `delta_tensors` | Number of ΔW tensors with valid Fisher | 24 (= 12 blocks × Q,V) |
| `ewc_penalty_value` | Scalar Fisher penalty at end of task | ~2.4e-4 (raw), active with norm |
| `normalized_fisher_drift` | `Σ F_k(ΔW_k−ref)² / Σ(ΔW_k−ref)²` | Used for dual ascent |

**Key finding:** `cos_flat_fisher ≈ −0.02` across all tasks and datasets. This is near-orthogonal — AS^1 and Fisher target structurally different subspaces of the gradient, making the joint design principled rather than heuristic.
