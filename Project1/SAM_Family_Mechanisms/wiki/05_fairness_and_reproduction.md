# 05｜公平性、复现与结论边界

## 1. 公平比较的三个层次

同一张表里出现方法名称，不代表比较已经公平。至少要分开报告：

1. **相同外层 optimizer steps**：回答固定训练步数时的行为；
2. **相同 gradient/backward-equivalent 预算**：回答固定计算量时的行为；
3. **相同修正强度**：回答曲率方向本身是否更有效，而不是谁的 $\lVert c\rVert$ 更大。

E001 没有训练轨迹，但仍应在 manifest 和 summary 中记录每个对象的 gradient/HVP/backward equivalents。由于 E001 使用解析 NumPy 算子，这里记录的是当前 surrogate 所访问的唯一梯度点数量，不是 faithful optimizer 的实际 autograd、数据访问或墙钟成本；例如 path-mean surrogate 的 $k+1$ 不能冒充完整 slow-weight 实现的实测 backward 数。E002/E003 必须同时给出 same-steps、same examples/data passes 和 same measured compute 三套 estimand。

## 2. 两种方法强度协议

### 2.1 原生超参数协议

各方法使用自身调优后的 $\rho,\alpha,k$，用于回答“该方法按常规设置能达到什么行为”。调优空间、选择准则和验证数据必须预先记录。

### 2.2 修正范数匹配协议

定义

\[
r_c=\frac{\lVert c^{(m)}\rVert}{\lVert g\rVert+\epsilon}.
\]

在共同的目标 $r_c$ 下比较方向和谱结构，用来隔离“修正指向哪里”与“修正有多强”。E001-S 已使用 $r_c\in\{0.1,0.25,0.5\}$：对每个方法二分求解其原生半径并重新运行 operator，而不是事后线性缩放修正向量。最大匹配相对误差和方法特定半径都进入产物。

E001 的 matched-SAM 是更专门的局部控制：它匹配 Lookbehind 一阶近似下的有效 SAM 半径，但不等同于严格匹配 $\lVert c\rVert$。两种控制必须分别标记；cosine 还必须配合修正范数比与向量相对误差。

## 3. E001 的路径公平性

每条 MS-SAM/Lookbehind 记录必须同时包含：

```text
protocol, rho_step, inner_steps, endpoint_radius, path_radius, path_budget, oracle_radius
```

- fixed-step 用来保留每一步相同半径，但总预算随 $k$ 增加；
- fixed-budget 用来固定总路径长度预算，隔离更多路径点的作用；
- fixed-budget 下 $\rho_{\mathrm{eff}}=(k+1)\rho/(2k)$ 仍随 $k$ 改变；E001-S 的 fixed-effective-radius 是补充控制；
- $Q_0/Q_1$ 的 oracle 必须使用同一记录的 `oracle_radius`；对路径记录它等于 `path_budget`；
- Lookbehind 与 matched-SAM 使用 $\rho_{\mathrm{eff}}=(k+1)\rho_{\mathrm{step}}/2$；
- 不得把 `R_path`、`R_end` 和 `rho_eff` 当作同一个半径。

若 fixed-step 优于单步方法，而 fixed-budget、fixed-effective-radius 和严格修正匹配后差异大幅减弱，合理边界是“效应主要受更大路径预算或一阶修正强度驱动”；这不支持“发现了新的固定 Hessian 阶数”。

## 4. 对象语义公平性

以下对象不能互相替代：

| 对象 | 语义 | 合理对照 |
| --- | --- | --- |
| SAM final correction | 外层方向中的 $H\hat g$ 型修正 | GAM final regularizer、matched-SAM、其他 final correction |
| GAM probe direction | 归一化的 Hessian 定向方向；其 $\rho$ 倍才是内层候选扰动 | 一阶内层 oracle、其他 probe direction |
| GAM probe increment | 探测点的梯度变化，$H^2$ 型 | 其他 probe/increment 对象 |
| GAM final regularizer | 一阶 flatness 的最终正则项，最低阶 $H$ 型 | SAM final correction |
| MS-SAM $g_k$ | 路径最远端梯度 | Lookbehind 路径平均梯度 |
| 路径端点 $z_k-w$ | 内层候选扰动 | 同预算的 $Q_0/Q_1$ oracle |

特别地，不能用 GAM probe increment 的谱斜率给 GAM final update 贴上 $H^2$ 标签，也不能把 final regularizer 行的 `associated_perturbation_radius` 误解为该行承担 $Q_0/Q_1$。GAM 的内层质量只记录在 `gam_probe_direction` 行。

## 5. 固定的数据与随机性条件

标准 E001 是确定性的：固定同一个 $H,w$、float64、归一化约定和 epsilon。E001-S 使用 `seed` 生成可复现的 random-$w$/random-$g$ ensemble；重复运行同一解析配置应得到一致的核心 CSV。

E002/E003 必须进一步固定：

- 同一初始参数；
- 同一 mini-batch 顺序和相同 batch 内容；
- 同一数据增强；
- 同一基础学习率与 weight decay；
- 同一随机种子集合；
- checkpoint 虚拟更新中所有方法面对完全相同的参数点和 batch。

神经网络实验除了按 epoch/step 对齐，还要按相同训练损失对齐，例如预注册 $L_{\mathrm{train}}\in\{0.8,0.5,0.2,0.1\}$。否则更小 Hessian 可能只表示优化更慢。

## 6. 可复现运行清单

每次 E001 运行应按以下顺序执行：

1. 从干净的输出目录或新的 run ID 启动标准 CLI；
2. 保存解析后的完整配置，而不是只保存传入的 YAML；
3. 在 `manifest.json` 记录 schema、dtype、命令、运行环境、对象语义、代码指纹和 algorithmic-equivalent 计算预算；
4. 在 `arrays.npz` 保存重算指标所需的原始数组；
5. 由同一份数值数据生成 CSV、JSON 与图，避免多条计算路径漂移；
6. 严格序列化 JSON：未定义值为 `null`，禁止 NaN/Infinity；
7. 检查解析恒等式、GAM 三行语义、路径预算、oracle 可行性、嵌套拟合和严格配置拒绝规则；
8. 用相同命令复跑并比较 `hvp_scan.csv`、`spectral_gain.csv` 和 `method_summary.csv`；
9. 只在全部验收完成后，向 [日志](log.md) 增加观测结果和产物路径。

推荐把临时复现输出写到 `/tmp/quadratic_operator`；若写到项目内 `outputs/`，应把它视为生成物，不手工编辑。

## 7. 结果报告模板

每条机制结论必须包含：

```text
证据标签：解析恒等式 / 验收目标 / 观测结果 / 待验证假设
实验编号与 run ID：
对象：key + method + object_kind + inner_steps + protocol
半径：rho_scale + rho_step + path_budget + oracle_radius（适用时）
主要指标与对应产物：
可以支持的结论：
不能支持的结论：
```

不要只贴一张图或只报告最终 $\lambda_{\max}$。谱响应、内层目标、修正强度和计算预算需要一起解释。

## 8. 当前研究结论边界

E001 与 E001-S 通过验收后，最多可以支持以下类型的表述：

- “在该精确二次算子与给定半径协议下，某对象的谱响应与 $H^p\hat g$ 对齐”；
- “在匹配路径预算、一阶有效半径、matched-SAM 或严格修正强度后，Lookbehind 是否仍有不能由单次 $H\hat g$ 解释的残差”；
- “某个候选扰动对零阶或一阶内层问题的相对求解质量如何”。

E001 不能支持：

- 某方法选择了更好的或更平坦的 basin；
- 某方法改善了 mini-batch 噪声结构或 $\operatorname{Tr}(H\Sigma)$；
- LookSAM 的正交修正具有可复用的时间相关性；
- 更小 sharpness 导致更高测试准确率；
- 在 FashionMNIST、CIFAR 或更大模型上会有泛化收益。

这些问题分别需要 E002 的非凸轨迹证据和 E003 的端点证据。planned 不能写成 completed。
