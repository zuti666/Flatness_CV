## Usage Guide

### Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the main experiment script:

   ```bash
   bash scripts/exp1_script/run_q1_seed.sh 
   bash scripts/exp2_script_rwp_imager_16_cosine/run_exp2.sh
   ```

   The corresponding settings live in `exp_1_sam-sgd_imageR_r16_t20_seed_42/`, `exp2_sam-rwp-flat_imageR_r16_t20`. You can change the seed, dataset, LoRA method, and optimizer there.

### Experiment 1: SAM/SGD baselines (ImageNet-R, r16, t20)

Run:

```bash
bash scripts/exp1_script/run_q1_seed.sh
```

Configs are in `exp_1_sam-sgd_imageR_r16_t20_seed_42/`. Edit the YAMLs to switch dataset, seed, LoRA method, or optimizer.

### Experiment 2: RWP/GAM/CFlat flatness study (ImageNet-R, r16, t20)

Run:

```bash
bash scripts/exp2_script_rwp_imager_16_cosine/run_exp2.sh
```

Configs are in `exp2_sam-rwp-flat_imageR_r16_t20/`. Update the YAMLs to change optimizer variants (sam/rwp/gam/cflat), seeds, or datasets.

### Ablation Experiments

Run the length ablation script:

```bash
bash scripts/exp4_script/run_imagenet-r_length_r16_l40_cosine_1993.sh
```

The settings are in `exp4_ablation_length/`. Edit the seed, dataset, LoRA method, and optimizer in those YAML files.

### Evaluation

## Evaluation & Metrics (quick map)

Below are the main metrics, formulas, and the functions to call. All evaluators live in
`evaluation_sharpness` (flatness/curvature) or `evaluation_performance` (accuracy/probe).

- **Cross-task accuracy matrices (CNN/NME)**
  - Function: `trainer.py` → `log_matrix` / `compute_sequence_metrics`
  - Data: R[i,j] = acc on task j after learning task i (lower triangle)

- **Flatness / Sharpness (weight space)**
  - File: `evaluation_sharpness/eval_flatness_weight_Loss.py`
  - Entry: `evaluate_flatness_metrics(model, loader, config, params_override=None)`
  - Key formulas:
    - Grad norm (Sh¹): \(\|\nabla_w L\|\)
    - Stochastic sharpness (E-Sh): \(\mathbb{E}_{\epsilon\sim\mathcal{N}(0,\sigma^2 I)}[L(w+\epsilon)-L(w)]\)
    - Hessian/GGN/Fisher λ_max, trace via MVP + power/Lanczos
    - Loss landscape: 1D/2D slices \(L(w+\alpha d)\)

- **Feature-space flatness (EFM)**
  - File: `evaluation_sharpness/eval_flat_feature.py`
  - Entry: `evaluate_feature_metrics(network, loader, config)`
  - Formula: EFM \(E_f = E_y[g_y g_y^T]\) with \(g_y = W_y - \sum_c p_c W_c\)
  - Metrics: trace, spectral radius, Frobenius norm, effective rank, top eigenvalues.
  - First-vs-last comparison: `compare_first_last_features(...)` (CKA + prototype drift).

- **Curvature localization / delta-W analyses**
  - File: `evaluation_sharpness/curv_localization.py`
  - Functions: `_curvature_localization_metrics`, `_delta_w_projection_eval`, `_delta_w_full_projection_eval`, `_w_delta_alignment_eval`
  - Idea: project eigenvectors onto LoRA subspace or QKV blocks; measure Rayleigh stats / alignment.

- **Loss landscape utilities**
  - File: `evaluation_sharpness/loss_landscape.py`
  - Functions: `_loss_landscape_1d`, `_loss_landscape_2d`, `compute_full_vs_lora_curvature_1d`

- **Linear probe evaluations**
  - File: `evaluation_performance/probe.py`
  - Functions: `fit_linear_probe_softmax_head`, `evaluate_linear_probe_softmax_with_head`
  - Used in `src/trainer.py` for per-task and joint-seen probes.

- **Optimizers (perturbation-based)**
  - Package: `optimer_PerturabtionType`
  - Files: `optimer_sam.py`, `ARWP_cos.py`, `gam.py`, `c_flat.py`, `util.py`
  - Example formula (SAM): \(\epsilon = \rho \frac{g}{\|g\|}\), update on \(w+\epsilon\).
  - Example formula (RWP): \(\tilde w = w + \epsilon,\ \epsilon\sim\mathcal{N}(0,\sigma\|w\|)\); optional Fisher scaling.

## How to run
```bash
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/disk0/users/liying/miniconda3/lib:${LD_LIBRARY_PATH:-}"  # fix PIL/libstdc++
python -m src.main --config config_exps/exp_weight/your_cfg.yaml --mode inc \
  --override outputs_root=outputs_logs
```

- Flatness hooks toggle via config flags: `flat_eval`, `feature_flat_eval`, `attention_probe_eval`, etc.
- Use `params_override` in `evaluate_flatness_metrics` to restrict evaluation to LoRA or selected params.

## File map (core)
- Training entry: `src/main.py`, trainers `src/trainer.py`, `src/trainer_allData.py`
- Accuracy metrics: `evaluation_performance/metrics.py`
- Probes: `evaluation_performance/probe.py`
- Flatness/curvature: `evaluation_sharpness/*`
- Optimizers: `optimer_PerturabtionType/*`




## Methods Overview

### LoRA-Based Continual Learning (loraCL)

The LoRA-based CL methods are implemented in `loraCL/`. In the PECL setting, the backbone is frozen and only the
LoRA adapter parameters are trained. A LoRA update is parameterized as a low-rank residual:

$$
W = W_0 + \Delta W,\ \ \Delta W = B A
$$

$$
A \in \mathbb{R}^{r \times d_{in}},\ \ B \in \mathbb{R}^{d_{out} \times r},\ \ r \ll \min(d_{in}, d_{out})
$$



- `seqlora` (SeqLoRA): one shared LoRA adapter (A, B) is trained sequentially across tasks, maximizing sharing but
  also increasing interference across tasks.

$$
  W_T = W_0 + A B
$$


- `inclora` (IncLoRA): a new LoRA branch (A_t, B_t) is added per task; previous LoRA parameters are frozen and only
  the current task branch is trained.


$$
  W_T = W_0 + \sum_{t=0}^{T} A_t B_t
$$

- `olora` (OLoRA): incremental LoRA with an orthogonality regularizer between the new and previous LoRA A factors,
  plus L2 on the new LoRA params. The training objective follows:


$$
  L = L_{cls} + \lambda_1 \sum_{i < t} \lVert A_t A_i^T \rVert_F^2 + \lambda_2 \lVert \Delta W_t \rVert_F^2
$$

  (see `loraCL/olora.py` and `backbone/lora.py::compute_ortho_loss`).

### Sharpness/Optimizer Methods (optimer)

Sharpness-related optimizers are implemented in `optimer/`. In PECL, these operate on adapter parameters unless
explicitly configured otherwise (see Flat-CL discussion in the PDF).

- `sam`: Sharpness-Aware Minimization. Solve:

  
$$
  \min_w\ \max_{\lVert \epsilon \rVert \le \rho}\ L(w + \epsilon),
  \epsilon \approx \rho \cdot \frac{g}{\lVert g \rVert + 1e{-12}},\ \ g = \nabla_w L(w)
$$

  This encourages flat minima in the adapter subspace.
  (see `optimer/optimer_sam.py`).

- `rwp` / `arwp`: Robust Weight Perturbation with stochastic noise. The implementation perturbs weights as:

  


$$
  \tilde{w} = w + \epsilon,\ \ \epsilon \sim \mathcal{N}(0,\ \sigma \lVert w \rVert)
$$

  with optional Fisher scaling `eps <- eps / sqrt(1 + eta * F)` and then updates using the base optimizer
  (see `optimer/ARWP_cos.py`).

- `gam`: Gradient-Aligned Minimization. Uses two perturbation radii (rho, rho') and gradient decomposition:



$$
  \epsilon_0 = \rho \cdot \frac{g_0}{\lVert g_0 \rVert + \epsilon}
$$


$$
  \epsilon_1 = \rho' \cdot \frac{g_1 - g_0}{\lVert g_1 - g_0 \rVert + \epsilon}
$$


$$
  g = \beta_1 g_1 + \beta_3 g_2 - \gamma v_{\perp}
$$

  (see `optimer/gam.py` for the exact steps and weights).

- `cflat`: Composite-Flatness optimizer. Aggregates gradients from a perturbation and norm-ascent path:

  
$$
  g = g_1 + \lambda (g - g_2)
$$

  (see `optimer/c_flat.py`).

### Flatness Evaluation (eval_flat)

`eval_flat/eval_flatness_weight_Loss.py` implements weight-space flatness metrics, including sharpness and Hessian-based
statistics. These metrics are computed on a selected parameter set (e.g., LoRA-only) and align with the adapter-subspace
flatness analysis in the PDF. Key proxies include:

- Zeroth-order sharpness along gradient:

  


$$
  Sh_0(\rho) = L\left(w + \rho \frac{g}{\lVert g \rVert}\right) - L(w)
$$

- First-order sharpness:

  


$$
  Sh_1(\rho) = \rho \lVert g \rVert
$$

- Expected sharpness (E-Sh) under random perturbations:

  


$$
  ESh = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma^2 I)} \left[ L(w + \epsilon) - L(w) \right]
$$

- Hessian spectral proxies:

  


$$
  \lambda_{max}(H)\ \text{via power iteration}
$$


$$
  \mathrm{tr}(H) \approx \frac{1}{K} \sum_k v_k^T H v_k,\ \ v_k \in \{+1, -1\}^d
$$

Use the config fields in your experiment YAMLs to enable/disable these metrics.
