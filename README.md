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
Evaluation utilities are in `eval_flat/eval_flatness_weight_Loss.py`. You can switch on/off specific metrics by editing the config files used in your experiment.

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
- `inclora` (IncLoRA): a new LoRA branch (A_t, B_t) is added per task; previous LoRA parameters are frozen and only
  the current task branch is trained.
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
  \min_w\ \max_{\lVert \epsilon \rVert \le \rho}\ L(w + \epsilon)
  $$
  $$
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
