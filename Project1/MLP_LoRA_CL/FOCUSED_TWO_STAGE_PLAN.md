# 聚焦验证计划：LoRA 额外收益、端点/轨迹分解与两层 HVP 诊断

## 1. 研究顺序

当前不以 rank、safe plasticity 或 future-space coverage 为起点。研究门控顺序固定为：

$$
\text{LoRA 是否放大 flatness optimizer 的相对收益}
\rightarrow
\text{收益来自端点还是未来轨迹}
\rightarrow
\text{GAM 的 FD-HVP 数值诊断}.
$$

前两个门控不成立时，停止构建 LoRA-specific flatness 机制故事。有限差分误差只作为 GAM 附录解释，不用于替代前两个门控。

## 2. Stage 1：参数化 × 优化器交互

### 2.1 主问题

需要检验的是 difference-in-differences，而不是直接比较 LoRA+SAM 和 Dense+SAM：

$$
G_{\mathcal O}^A(P)=A_{\mathcal O,P}-A_{\mathrm{SGD},P},
$$

$$
G_{\mathcal O}^F(P)=F_{\mathrm{SGD},P}-F_{\mathcal O,P},
$$

$$
I_{\mathcal O}^{A/F}
=G_{\mathcal O}^{A/F}(\mathrm{FactorLoRA})
-G_{\mathcal O}^{A/F}(\mathrm{Dense}).
$$

正的 $I$ 表示 flatness optimizer 相对各自 SGD baseline 的收益在 LoRA 中更大。

### 2.2 最小模型和矩阵

- 模型：`784 -> 64 -> 64 -> 10`；
- 只更新同一个 `64 x 64` middle matrix；
- 任务：Rotated MNIST，角度 `[0, 15, 30, 45, 60]`；
- 公共 base 在任务序列外的辅助域 `-30°` 上预训练，避免把第一个连续任务提前学过；
- 参数化：Dense、Fixed random linear subspace、Factor LoRA；
- LoRA rank：只用 `r=4`；
- 优化器：SGD、SAM、GAM-FD；
- seeds：10 个严格配对种子；
- 顺序确认：ascending 与 reverse 两套任务顺序。

主结论只由 Dense 与 Factor LoRA 决定。Fixed subspace 用于判断额外收益来自低维约束还是 LoRA 双线性参数化。

### 2.3 公平性

每个 paired seed 必须共享：

- base checkpoint、数据划分和任务顺序；
- base checkpoint 中校验辅助预训练角度，防止误复用 `0°` checkpoint；
- minibatch shuffle seed；
- 训练步数和评价 checkpoint；
- 当前任务训练数据；
- effective-weight perturbation radius。

SAM/GAM 不直接匹配 raw factor radius，而通过一维搜索满足：

$$
\|W(\theta+\epsilon)-W(\theta)\|_F=\rho_W.
$$

每个 epoch 记录实际 effective perturbation norm，检查 Dense/LoRA 是否真正匹配。

### 2.4 两种报告口径

1. 固定训练步数：报告 FAA、immediate forgetting、average forgetting；
2. 当前任务损失匹配：在所有分支都达到的共同 loss threshold $q$ 上，比较旧任务损伤。

同时检查最终有效漂移 $\|W_t-W_{t-1}\|_F$。如果 LoRA+Flat 只是移动更少，应在漂移匹配后重新比较。

### 2.5 Stage 1 通过标准

进入 Stage 2 至少需要：

- $I_{SAM}^{A}$ 或 $I_{SAM}^{F}$ 的 paired 95% CI 在两个任务顺序中方向一致；
- 当前任务 loss 匹配后仍存在；
- effective drift 匹配后不完全消失；
- 不是单一学习率或单一 $\rho_W$ 才出现。

SAM 先作为主门控，因为它不依赖显式 FD-HVP。GAM 作为复现。

## 3. Stage 2：端点形成还是未来轨迹

### 3.1 2 × 2 分叉

在共同 task-$t$ 起点处分叉形成两个 endpoint，然后再次分叉训练 task $t+1$：

| task $t$ endpoint | task $t+1$ optimizer | branch |
|---|---|---|
| SGD | SGD | 00 |
| SAM | SGD | 10 |
| SGD | SAM | 01 |
| SAM | SAM | 11 |

Dense 和 Factor LoRA 分别运行完整四分支，所有未来分支使用相同 minibatch 序列。Pilot 先选择中间 transition，例如 task 2 -> task 3，避免 task 0 的初始化特殊性。

### 3.2 轨迹记录

每个 future step 或固定间隔记录：

- 新任务 loss/accuracy；
- 旧任务 loss/accuracy；
- 相对各自 endpoint 的 effective drift；
- 当前 effective weight；
- 实际单步 $\Delta W$。

旧任务损伤定义为：

$$
Y_{ab}(s)=L_t(W_{t+1}^{ab}(s))-L_t(W_t^a).
$$

### 3.3 两种匹配

在四条曲线的共同支持区间内插值：

- 新任务进展匹配：$Y_{ab}(q)$，其中新任务 loss 相同；
- 漂移匹配：$Y_{ab}(r)$，其中 $\|W_{t+1}^{ab}-W_t^a\|_F=r$。

不能外推到某条分支没有到达的 $q$ 或 $r$。

### 3.4 效应

$$
E=Y_{10}-Y_{00}
$$

是 endpoint formation effect；

$$
T=Y_{01}-Y_{00}
$$

是 future optimizer/trajectory effect；

$$
J=Y_{11}-Y_{10}-Y_{01}+Y_{00}
$$

是交互效应。由于 $Y$ 是旧任务损伤，负的 $E/T$ 表示保护作用。

### 3.5 两个必要反事实

共同方向 endpoint sensitivity：从真实未来轨迹收集同一组单位方向 $u_j$，分别作用于 SGD/SAM endpoint：

$$
S_a(u_j,\alpha)=L_t(W_t^a+\alpha u_j)-L_t(W_t^a).
$$

只有相同方向下 SAM endpoint 仍更稳定，才能声称 endpoint 本身更不敏感。

同状态一步分叉：在完全相同的 future state 和 minibatch 上分别生成 SGD step 与 SAM step，然后分别匹配：

- 相同 $\|\Delta W\|_F$；
- 相同新任务 loss decrease。

再比较一步旧任务损伤。这直接检测 SAM 是否选择了更安全的未来方向。

## 4. HVP-A：弱命题——有效 W 空间、真实未来方向

该诊断复用 Stage 2 保存的真实方向：

$$
u_k=\Delta W_k/(\|\Delta W_k\|_F+\epsilon).
$$

在 Dense/LoRA 形成的 endpoint 或 trajectory checkpoint 上计算：

$$
q_{exact}^W=H_Wu_k,
$$

$$
q_{FD}^W(\rho_W)
=\frac{\nabla_WL(W+\rho_Wu_k)-\nabla_WL(W)}{\rho_W}.
$$

主报告 forward difference；central difference 仅作为数值 sanity check。必须同时做 own-direction 与 shared-direction-pool 比较，避免 LoRA 只因选择了更容易近似的方向而看起来误差更低。

输出：absolute error、relative error、angle error、$\|H_Wu\|$、Taylor residual。它只能支持：LoRA 训练产生的 endpoint/方向组合在有效 W 空间更线性。

## 5. HVP-B：强命题——实际训练坐标、GAM 梯度方向

只针对 GAM 的实际 FD-HVP，不把 SAM 简化成 HVP estimator。

对 trainable coordinates $\phi$：

$$
v_\phi=g_\phi/\|g_\phi\|,
\qquad
q_{exact}^\phi=H_\phi v_\phi.
$$

通过一维搜索选择 raw step $c$，使 Dense/LoRA 满足相同：

$$
\|W(\phi+cv_\phi)-W(\phi)\|_F=\rho_W.
$$

然后使用 GAM 实际 forward difference：

$$
q_{FD}^\phi
=\frac{g_\phi(\phi+cv_\phi)-g_\phi(\phi)}{c}.
$$

报告：

$$
E_{rel}=\frac{\|q_{FD}-q_{exact}\|}{\|q_{exact}\|+\epsilon},
\qquad
E_{angle}=1-\cos(q_{FD},q_{exact}).
$$

checkpoint 选择 task-$t+1$ 的 early/middle/late 三个阶段，使用相同 minibatch、augmentation、dropout state，并在实际半径附近扫描多个 $\rho_W$。

### 5.1 数值误差能否解释收益

训练配对：Dense/LoRA × SGD/GAM-FD/GAM-exact。若 Dense 的 FD error 更大，只有当 GAM-exact 相对 GAM-FD 明显修复 Dense 的性能、从而缩小 LoRA interaction，才能认为数值精度解释了额外收益。

若 Dense FD error 较大但 GAM-exact 没有改善 Dense，误差存在却不是性能机制。

## 6. 结果模式

| 结果 | 解释 |
|---|---|
| Stage 1 interaction 约为 0 | 停止 LoRA-specific flatness 故事 |
| $E<0,T\approx0$ 且共同方向下 SAM endpoint 更稳 | endpoint stability 主导 |
| $E\approx0,T<0$ 且同状态 SAM step 更安全 | trajectory regulation 主导 |
| $E<0,T<0$ | 两种机制共同作用 |
| HVP-A 成立、HVP-B 不成立 | 有效 W 路径更线性，但实际 GAM 数值近似没有 LoRA 优势 |
| HVP-B 成立但 GAM-exact 不缩小性能交互 | 数值差异存在，但不能解释 optimizer gain |
| HVP-B 成立且 GAM-exact 修复 Dense | FD accuracy 是 LoRA 额外 GAM 收益的候选机制 |

## 7. 当前实现入口

- Stage 1 forward：[configs/focus_stage1.yaml](configs/focus_stage1.yaml)
- Stage 1 reverse：[configs/focus_stage1_reverse.yaml](configs/focus_stage1_reverse.yaml)
- 两个正式配置均启用 MNIST 自动下载；`run_grid.py --prepare-data-only` 可在训练前单独下载并校验 train/test splits。
- Download-free implementation check：[configs/focus_stage1_smoke.yaml](configs/focus_stage1_smoke.yaml)
- 配对 interaction：[analyze_stage1_interaction.py](analyze_stage1_interaction.py)
- SAM/GAM 已支持 `training.perturbation_metric: effective_weight`，并记录实际 effective perturbation norm。
- HVP-A 的 forward/central FD 与 HVP-B 的 actual-trainable-coordinate 核心计算：[small_cl/diagnostics.py](small_cl/diagnostics.py)。

Stage 2 的四分支轨迹 runner、loss/drift interpolation，以及 HVP-B 在 early/middle/late checkpoint 的自动记录，是下一实现单元；在 Stage 1 门控通过前不运行 rank/safe-route 大网格。
