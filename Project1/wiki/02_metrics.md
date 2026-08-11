# 评估指标

[返回首页](README.md)

## 1. Accuracy matrix 是统一数据源

令 `R[t,j] = a[t,j]` 表示训练完任务 `t` 后，在任务 `j` 上的准确率。有效区域是下三角：`j <= t`。

```text
          eval task
          0      1      2
train 0  a00    NaN    NaN
train 1  a10    a11    NaN
train 2  a20    a21    a22
```

代码的 canonical orientation 是 `time_by_task`：行是训练时间，列是评估任务。原始矩阵保存在 consolidated metrics JSON 的：

```text
cnn.matrices.final
nme.matrices.final
```

## 2. 持续学习性能指标

设共有 `T` 个任务。

| 指标 | 公式 | 解释 | 趋势 |
|---|---|---|---|
| Current accuracy `CA_t` | `R[t,t]` | 刚学完当前任务的准确率 | ↑ |
| Seen-task average `Acc_t` | `mean(R[t,0:t+1])` | 时间 `t` 的已见任务平均准确率 | ↑ |
| FAA / final `Acc` | `mean(R[T-1,0:T])` | 最终任务结束后的平均准确率 | ↑ |
| AAA | 下三角所有有效 `R[t,j]` 的均值 | 整个学习过程的 anytime performance | ↑ |
| BWT at `t` | `mean_i<t(R[t,i]-R[i,i])` | 新任务对旧任务的回溯影响 | ↑；负值越接近 0 越好 |
| Final BWT | `mean_i<T-1(R[T-1,i]-R[i,i])` | 序列结束时的 backward transfer | ↑ |
| Forgetting | `mean_i<T-1(max_t R[t,i]-R[T-1,i])` | 每个旧任务最佳值到最终值的下降 | ↓ |
| ACA | `mean(diag(R))` | 各任务刚学完时的平均 current accuracy | ↑ |
| ABWT | 各时间步 BWT 的平均 | 全 trajectory 的 backward transfer | ↑ |
| FWT/AFG | strict upper triangle / future-task score | 未学习任务上的 forward generalization；上三角缺失时不可用 | ↑ |

代码定义见 `evaluation_performance/metrics.py`。其中：

- `FAA == Acc`。
- `AAA` 是整个下三角的等权平均，等价于按每个有效 `(t,j)` 单元等权；它不等于先对每个时间步求平均再对时间等权，除非另行定义。
- Forgetting 排除最后一个任务，因为最后任务没有后续学习阶段。

旧 rebuttal 文档把 AAA 文字说明成“per-step running average 的均值”，但实际表值使用下三角等权定义。例如 CUB200 SeqLoRA-SGD 的 10 个 row mean 再平均为约 70.00，而报告 AAA 为 65.48，与下三角 55 个单元的等权平均一致。复用旧表时以矩阵和值为准，不沿用那句文字定义。

## 3. CNN 与 NME/NCM

| 指标族 | 预测方式 | 主要反映 |
|---|---|---|
| CNN / CLS | 训练得到的 classifier head | 端到端分类表现 |
| NME / NCM | 特征归一化后按类别 prototype 最近邻分类 | representation quality，较少依赖 classifier head |

论文使用 `Acc^cls` 与 `Acc^ncm`；代码和旧日志同时出现 `CNN`、`NME`、`NCM`。汇总时应先确认来源字段，不要仅按图标题猜测。

## 4. Exp D/F 的 trajectory 专用指标

两任务 Exp D：

```text
A_1_1 = R[0,0]
A_1_2 = R[1,0]
A_2_2 = R[1,1]
F1_acc = A_1_1 - A_1_2
FinalAvg = (A_1_2 + A_2_2) / 2
```

20 任务 Exp F 以前 10 个任务为 prefix：

```text
prefix_forget = mean_i<10(R[i,i] - R[19,i])
prefix_retention = mean(R[19,0:10])
suffix_final_acc = mean(R[19,10:20])
```

Exp F 的关键是 checkpoint-matched branch comparison，不是只比较最终单个数字。

## 5. Sharpness 指标

| 字段/符号 | 支撑 | 定义或用途 |
|---|---|---|
| `Sh_param_full` / `sh0_max` | raw parameter space | 诊断性 full parameter sharpness；不等于 merged effective-weight sharpness |
| `Sh_AB` / `sh_ab_max` | LoRA A/B factor coordinates | 参数化相关的 factor-space sharpness |
| `Sh_Delta/W_tangent` / `Sh_Delta_tangent` / `sh_delta_max` | effective update tangent | `{B dA + dB A}` 上的理论对齐 sharpness |
| `Sh_rand/W_tangent` | matched random tangent | 支撑结构负对照 |
| `Sh_frozen_coords` | frozen coordinates | 不可训练方向负对照 |
| `E-Sh` | Gaussian distribution | expected loss increase |
| AS(0) | adversarial neighborhood | 最大 loss increase |
| AS(1) | adversarial neighborhood | 最大 gradient norm |

所有 sharpness 数字必须同时报告：评估 loss 对应哪个 task、参数状态 `theta_t`、support、radius `rho`、采样/gradient batches。不同 support 使用相同坐标半径 `rho` 时，其 induced `||delta DeltaW||_F` 不一定相同，不能当作完全 norm-equalized comparison。

## 6. Curvature 与几何指标

| 指标 | 含义 | 趋势 |
|---|---|---|
| `lambda_max(H)` | Hessian 最大特征值，最尖锐局部方向 | ↓ |
| `tr(H)` | Hessian aggregate curvature | ↓ |
| `lambda_max(F)` | empirical Fisher 最大特征值 | ↓ |
| `tr(F)` | Fisher aggregate curvature | ↓ |
| `lambda_max(GGN)` / `tr(GGN)` | generalized Gauss-Newton curvature | ↓ |
| `c_t` | adapter geometry 与 frozen reference 的 curvature projection ratio | 论文用 `>1` 表示更多曲率集中于 adapter geometry |
| `alpha_t` | adapter update 对对应奇异方向的 amplification factor | 用于解释 adapter geometry 的强化 |

Hessian/Fisher/GGN 是代理量，不应与 sharpness 完全等同。评估是否限制到 LoRA 参数由 `flat_eval_param_names`、`flat_eval_include_frozen` 等配置决定。

## 7. Feature 与鲁棒性指标

| 指标 | 解释 |
|---|---|
| Prototype L2 drift | 类 prototype 随任务的移动量，越小通常表示表征更稳定 |
| Linear CKA | 初始/最终或任务间 feature geometry 相似度，越高越稳定 |
| Feature covariance trace | 表征总方差 |
| Effective rank | `((sum lambda_i)^2)/(sum lambda_i^2)`，谱分散程度 |
| Anisotropy | 表征集中在少数方向的程度 |
| ImageNet-C/P accuracy curve | corruption/perturbation shift 下逐任务 `Acc_t` |
| 1D/2D loss landscape | 沿指定 random/eigen/adversarial direction 的局部 loss slice |

当前多数新 Exp E/F YAML 将 `flat_eval`、feature、attention、OOD 总开关设为 `false`，所以 summary 中这些字段为空是预期行为，不代表汇总脚本失效。

## 8. 报告规范

每张表至少写清：

- classifier 还是 NME/NCM；
- FAA、AAA 还是某个时间点 `Acc_t`；
- accuracy matrix orientation；
- task split、seed、rank、epoch；
- sharpness 的 support、direction、radius 和 evaluated task loss；
- mean/std 的 seed 数；单 seed 不写成稳定统计结论。
