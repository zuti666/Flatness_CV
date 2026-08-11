# 方法与理论

[返回首页](README.md)

## 1. 问题设定

持续学习共有任务序列 `1,...,T`。学习任务 `t` 后，在任务 `j` 上的准确率记为 `a[t,j]`。Project1 聚焦 frozen-backbone PECL：

```text
W_t = W_frozen,t + DeltaW_t
DeltaW_t belongs to W_Delta,t
```

其中 `W_Delta,t` 不是整个参数空间，而是当前 LoRA 参数化在解附近能够实现的局部 weight-update tangent geometry。

代码采用：

```text
DeltaW = B @ A
d(DeltaW) = B @ dA + dB @ A
```

因此局部可达支撑为 `{B dA + dB A}`。论文正文部分位置写成 `AB`；写作和代码核对时应统一到代码真实 convention `B @ A`。保持 `DeltaW` 不变的缩放应写为 `A' = A / c, B' = cB`，或等价地反向定义 `c`，但全文必须一致。

## 2. LoRA 持续学习组织

| 组织 | 更新方式 | 几何解释 | 主要实现 |
|---|---|---|---|
| SeqLoRA | 所有任务顺序更新同一 LoRA | 强共享、干扰可能最大 | `models_LoRAbasedCL/seqlora.py` |
| IncLoRA | 新任务增加 LoRA，旧模块冻结 | admissible geometry 随任务扩张 | `models_LoRAbasedCL/inclora.py` |
| OLoRA | 扩张并加入正交约束 | 限制跨任务方向重叠 | `models_LoRAbasedCL/olora.py` |
| InfLoRA | 用梯度/投影机制约束更新 | 控制新更新进入的梯度子空间 | `models_LoRAbasedCL/inflora.py` |
| SDLoRA | shared/task-specific 低秩分解 | 分离共享与任务专属方向 | `models_LoRAbasedCL/sdlora.py` |
| SeqFT / Linear Probe | 全量顺序微调 / 只训练分类头 | full-space 与最小更新基线 | `models_CL/finetune1.py` 等 |

主论文的几何分析重点是 SeqLoRA、IncLoRA、OLoRA；InfLoRA、SDLoRA 主要用于 rebuttal 的外部有效性验证。

## 3. 三类 sharpness 目标

### 3.1 RS(0)：random zeroth-order sharpness

```text
RS(0)_S(W, Sigma)
  = E_epsilon[L_S(W + epsilon) - L_S(W)]
```

实验对应 RWP/random weight perturbation。它测试随机邻域敏感性，但不寻找最坏方向。

### 3.2 AS(0)：adversarial zeroth-order sharpness

```text
AS(0)_S(W, rho)
  = max_{||epsilon|| <= rho} [L_S(W + epsilon) - L_S(W)]
```

实验对应 SAM。实现通过当前 loss gradient 构造归一化 ascent perturbation，再对扰动后的 loss 做第二次反向传播。

### 3.3 AS(1)：adversarial first-order sharpness

```text
AS(1)_S(W, rho)
  = max_{||epsilon|| <= rho} ||grad L_S(W + epsilon)||
```

实验对应 GAM。它直接压制邻域中的梯度增长，论文将其解释为更强的 curvature-aware flatness 目标。

### 3.4 名称映射

| 论文符号 | 配置/日志名 | 方法 |
|---|---|---|
| SGD | `sgd` | 普通 SGD；语言实验通常用 Adam baseline |
| RS(0) | `rwp`、部分新实验中的 `random_*` | RWP 或支撑匹配的 Gaussian random perturbation |
| AS(0) | `sam`、`sam_*` | SAM |
| AS(1) | `gam` | GAM |
| C-Flat | `cflat` | SAM/GAM 组合型历史对照，不是当前最核心结论 |
| Flat-LoRA style | `flatlora_full` | 对 merged/full effective weight 注入噪声的全范围对照 |

`RWP` 与 Exp E 的 `random_*` 都使用随机方向，但实现目的不同，不能把所有 `random` 结果都直接标为同一个算法。

## 4. Direction × support 分解

Exp B/E 将“如何选方向”和“在哪里扰动”拆开。

```text
SAM:    epsilon_S = rho * Proj_S(grad L) / ||Proj_S(grad L)||
Random: z_S ~ N_S(0, I), epsilon_S = rho * z_S / ||z_S||
```

| 后缀 | 支撑 | 含义 |
|---|---|---|
| `factor` | raw LoRA `(A,B)` coordinates | 参数化相关；扰动后有效权重含二阶项 `dB dA` |
| `delta` | `{B dA + dB A}` tangent | effective adapter-update geometry，理论对齐且对正比例缩放的列空间不变 |
| `full` | merged effective qkv weight | 在完整 qkv effective weight 上扰动 |
| `all` | raw all-parameter coordinates | 包括冻结参数坐标；只在 ascent 阶段临时开启 |
| `frozen` | frozen backbone coordinates | 不可训练方向负对照 |

必须区分：

- `sam_random`：把 SAM gradient 投影到随机 matched tangent support。
- `random_delta`：在真实 delta tangent support 内采样 Gaussian direction。

二者不是同一实验轴；Exp E 的 11 个配置不包含 `sam_random`。

所有 scoped 变体最终仍只通过 base optimizer 更新常规 trainable parameters（LoRA A/B 与 classifier head）；support 只改变 SAM/RWP 临时扰动的位置。

## 5. 理论主线

当前理论使用 sequential hierarchical PAC-Bayes：

```text
average population risk
 <= average empirical posterior risk
  + sqrt((within-task KL + hyperposterior drift KL + log(1/delta)) / sample size)
```

复杂度分为：

1. `KL(Q_t || P_t)`：任务内 posterior 相对当前 prior 的适应代价。
2. `KL(Pcal_t || Pcal_{t-1})`：任务间 prior-generation mechanism 的漂移。

在 frozen-backbone PECL 中，若 posterior 仅在 adapter-induced update geometry 上变化，KL 与 perturbation-smoothed risk 应缩减到 `W_Delta,t` 支撑。理论意图是把 flatness 从泛化的 ambient property，改写为“沿后续更新实际可达方向的 task-conditioned sensitivity”。

## 6. 可检验预测

| 预测 | 预期 | 对应实验 |
|---|---|---|
| P1 Scope relevance | adapter/admissible support 优于 full/frozen mismatch | scope sweep、Exp B/E |
| P2 Curvature localization | adapter curvature concentration 越强，scope effect 越明显 | `c_t`、`alpha_t`、loss/accuracy trajectory |
| P3 Perturbation type | 理想排序 AS(1) > AS(0) > RS(0) > SGD | ImageNet-R/C/P、跨域表 |
| P4 Trajectory effect | 在后续任务使用 SAM 应改善旧任务 retention | Exp D/F |
| P5 Parameterization invariance | `Sh_AB` 随 `(A,B)` rescale 变化，`Sh_Delta` 稳定 | Exp 7 RQ2、Exp G/Q2 |

P3 是总体趋势而非每个数据集/方法都严格成立；P5 仍缺少本 Wiki 可确认的完整结果。

## 7. 结论边界

- 可以说：SAM direction 在所有五类支撑上都明显优于 Gaussian random direction。
- 可以说：later-task SAM 改变 trajectory，并减少旧任务 degradation。
- 可以说：跨域 vision 中 AS(1) 最常成为最佳方法。
- 不应说：Exp E 已证明 raw factor-space 是理论正确对象。
- 不应说：lower sharpness 已被实验确定为 forgetting 的单一因果原因。
- 不应说：SAM/GAM 在所有语言模型和所有 LoRA 组织上都必然提升。

