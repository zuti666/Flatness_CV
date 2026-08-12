# 06｜E001 结果、敏感性与假设审计

本页只解释两个可追溯运行：标准算子运行 `E001-20260811-7814f2cb` 与敏感性运行 `E001S-20260811-ec752981`。所有结论都限定在同一个确定性、正定、恒 Hessian 的二次族内。

## 1. 先给结论

E001 是一个合格的算子单元测试，但不是足以确认神经网络机制的科学实验。当前结果支持三条窄结论：

1. MS-SAM 的最远端梯度比 Lookbehind 的路径均值更偏向顶部谱；
2. 固定总路径预算后，Lookbehind 大部分修正仍与 matched-SAM 同向，所谓“新信息”很弱；
3. GAM probe 在本构造下更接近一阶梯度范数目标，而 SAM 更接近零阶损失目标。

与此同时，半径、条件数和初始化会显著改变效应大小。特别是，$\rho/\lVert w\rVert$ 不是跨 Hessian 谱的公平强度控制；fixed-budget 也没有完全固定 Lookbehind 的一阶有效半径。因此原始实验思路应保留三级框架，但必须收紧假设措辞和因果控制。

## 2. 哪些是解析恒等式，哪些才是观测

### 2.1 解析正对照

下列结果由二次模型定义必然推出，只能用于验收实现：

- SAM 的修正谱斜率为 1，且 $R_1^2=1$；
- GAM probe direction 的谱斜率为 1；
- GAM probe increment 的谱斜率为 2；
- matched-SAM 的修正方向与 $H\hat g$ 完全一致；
- HVP 半径扫描的 cosine 为 1，误差只反映浮点消去；
- 负曲率指标为 0，因为 $H\succ0$。

标准运行中，SAM 恒等式相对误差为 $5.84\times10^{-16}$，GAM probe/$H^2$ 相对误差为 $3.47\times10^{-16}$。这些数值证明代码与解析式一致，不证明真实非凸损失在相同半径下仍是局部二次的。

### 2.2 标准点的非平凡观测

默认设置为 $d=20$、谱区间 $[0.1,10]$、$\rho=0.01\lVert w\rVert$。

| 对象 | log-log 谱斜率 | Krylov $R_1^2$ | 顶部 E1 | 正 Rayleigh | $\lVert c\rVert/\lVert g\rVert$ | $Q_0/Q_1$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SAM correction | 1.000 | 1.000 | .384 | 7.435 | .130 | .994/.536 |
| GAM final regularizer | 1.098 | .957 | .561 | 8.426 | .186 | — |
| GAM probe increment | 2.000 | .893 | .621 | 8.838 | — | — |
| MS k=5 fixed-step | 1.102 | .980 | .498 | 8.129 | .866 | .896/.514 |
| LB k=5 fixed-step | 1.070 | .990 | .464 | 7.937 | .477 | .896/.514 |
| MS k=5 fixed-budget | 1.024 | .999 | .412 | 7.615 | .140 | .997/.565 |
| LB k=5 fixed-budget | 1.016 | .999 | .403 | 7.557 | .082 | .997/.565 |

这里的 1.098 只是全谱描述性回归斜率，不能把 GAM final 称为“$H^{1.098}$ 算子”。其高端谱存在弯曲，而且 Krylov 增量解释度不等于唯一的多项式系数。

MS 与 LB 的 $Q_0/Q_1$ 完全相同不是两次独立证据：两者共享同一个路径端点，$Q$ 只评价内层候选，无法评价 LB 的外层梯度聚合。

## 3. 对初始假设的判定

| 初始假设 | 当前判定 | 证据与改写 |
| --- | --- | --- |
| SAM 在 E001 中提取 $H\hat g$ | 解析成立 | 保留为实现正对照；到 E002 才能把有限半径误差当经验量 |
| GAM probe 强化 $H^2$ 谱 | 解析成立 | 只适用于 probe increment；不能转写为 GAM final update 是 $H^2g$ |
| Multistep 的 $Q_0$ 高于 SAM | 无条件版本不成立 | fixed-budget 仅从 .994 小幅升至 .996/.997；fixed-step 在更大 oracle 半径下反降至 .982/.896 |
| MS 比 LB 更强地放大顶部谱 | 本实例支持 | 每个配对中 MS 的斜率、E1、Rayleigh 与 H1 残差都高于 LB |
| LB 获得本质不同的高阶信息 | 当前证据弱 | fixed-budget 的 LB/matched correction cosine 为 .99990/.99971，H1 残差仅 .0144/.0239 |
| LB 主要是更强的 SAM 加聚合 | 部分支持 | 方向很接近，但不能忽略幅度：k=5 fixed-step 的修正范数比为 1.221、向量相对误差为 .203 |
| 路径聚合降低随机方差 | E001 不可检验 | 当前只有一条确定性路径；只能说 LB 的顶部谱放大弱于 MS |
| RQ3 的最终性能差异可由协方差解释 | E001 不可检验 | E002 的一步分解也只能给贡献关联；因果“来源”需要噪声干预与完整轨迹 |

## 4. 半径敏感性：高阶外观主要随局部强度增长

E001-S 对所有方法扫描 $\rho\in\{10^{-4},10^{-3},10^{-2},0.05,0.1\}$，而不再只扫描 SAM HVP。

| $\rho/\lVert w\rVert$ | GAM final 斜率 | GAM $\lVert c\rVert/\lVert g\rVert$ | LB k=5 step：斜率/H1 残差/强度 | LB k=5 budget：斜率/H1 残差/强度 |
| ---: | ---: | ---: | ---: | ---: |
| $10^{-4}$ | 1.001 | .0013 | 1.001/.0012/.0039 | 1.000/.00025/.00078 |
| $10^{-3}$ | 1.012 | .0136 | 1.008/.0122/.0400 | 1.002/.0025/.0078 |
| $10^{-2}$ | 1.098 | .186 | 1.070/.101/.477 | 1.016/.0239/.0819 |
| .05 | 1.297 | 1.391 | 1.195/.244/3.112 | 1.070/.101/.477 |
| .10 | 1.428 | 3.038 | 1.251/.288/6.710 | 1.118/.163/1.073 |

因此，“更高阶”与“更非局部”主要随曲率-半径强度共同增长。默认 $\rho=.01$ 处于局部但不完全无穷小的区间；k=5 fixed-step 已有 $\lVert c\rVert/\lVert g\rVert=.477$，MS 更达到 .866，不能当作弱扰动比较。

SAM 的 $Q_0$ 从小半径近 1 降至 $\rho=.1$ 时的 .644；GAM probe 的 $Q_1$ 从近 1 降至 .813。这也说明默认点的目标对齐不能代表整个扫描区间。

## 5. 条件数：原始半径协议不是公平控制

条件数扫描固定几何平均曲率为 1，即 $\lambda_{\min}=1/\sqrt\kappa$、$\lambda_{\max}=\sqrt\kappa$。

| $\kappa$ | SAM 强度 | GAM final 强度 | LB k=5 step 强度 | LB k=5 budget 强度 | LB budget H1 残差/cosine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | .0139 | .0141 | .0420 | .0084 | .00134/.999999 |
| 100 | .130 | .186 | .477 | .0819 | .0239/.999714 |
| 10,000 | 8.055 | 26.915 | 58.349 | 11.052 | .307/.9518 |

同一个 $\rho/\lVert w\rVert=.01$ 在不同谱上对应完全不同的局部修正强度。后续必须同时报告

\[
r_c=\frac{\lVert c\rVert}{\lVert g\rVert},
\qquad
\tau_{\max}=\frac{\rho\lambda_{\max}}{\lVert g\rVert},
\]

并提供按 $r_c$ 或预注册的无量纲半径匹配结果。只按参数范数缩放 $\rho$ 不足以作跨条件数或跨模型比较。

## 6. 维度与初始化

### 6.1 20 维不是主要风险

在 $d=10,20,50,100$ 上，GAM final 的全谱斜率为 1.108、1.098、1.092、1.091；LB k=5 fixed-budget 为 1.018、1.016、1.015、1.014。默认 $d=20$ 足以作为便宜的谱夹具。需要注意，E1/E5/E10 的含义会随 $q/d$ 改变，跨维度不应直接比较绝对顶部能量。

### 6.2 等梯度是放大差异的夹具

E001-S 使用固定 seed 的 100 个 random-$w$ 与 100 个 random-$g$ 起点。下表给中位数，括号为 10%–90% 分位区间。

| 初始化 | SAM $Q_1$ | GAM probe $Q_0$ | LB k=5 budget H1 残差 | SAM 强度 |
| --- | ---: | ---: | ---: | ---: |
| equal-gradient | .536 | .738 | .0239 | .130 |
| random-$w$ | .942 (.879–.973) | .947 (.893–.975) | .00135 (.00062–.00262) | .0224 (.0167–.0315) |
| random-$g$ | .538 (.391–.688) | .735 (.664–.802) | .0202 (.0124–.0295) | .113 (.0749–.179) |

$w_i\propto1/\lambda_i$ 的优点是所有模态都被激发，缺点是它刻意暴露并放大谱差异。合理表述是“diagnostic fixture”，不是“典型训练初始化”。random-$w$ 更容易由顶部曲率主导，所以 SAM 与 GAM probe 看起来同时接近各自 oracle；这进一步说明单一起点不能支撑普遍机制结论。

## 7. 三种 $k$ 控制与严格修正强度匹配

fixed-budget 固定总路径长，但其一阶有效半径

\[
\rho_{\mathrm{eff}}=\frac{k+1}{2k}\rho
\]

仍随 $k$ 改变。E001-S 因此增加第三个诊断协议：

\[
\rho_{\mathrm{step}}=\frac{2\rho_{\mathrm{eff}}}{k+1},
\]

使 matched-SAM 的 $\rho_{\mathrm{eff}}=.01$ 对所有 $k$ 恒定。

| $k$ | LB H1 残差 fixed-step | fixed-budget | fixed-effective-radius | fixed-effective 的修正 cosine/强度 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 1.000/.130 |
| 2 | .0269 | .0144 | .0188 | .99982/.135 |
| 3 | .0530 | .0196 | .0286 | .99959/.138 |
| 5 | .1013 | .0239 | .0388 | .99925/.141 |
| 10 | .1961 | .0272 | .0484 | .99883/.143 |

在固定一阶有效半径后，增加 $k$ 确实产生小幅非 H1 残差，但到 $k=10$ 仍与 matched-SAM 高度同向。这比“fixed-step 下残差上升”更接近路径分辨率本身的证据。

严格强度匹配不是事后缩放向量；实现对每个方法用二分求解原生 `primary_rho`，使 $r_c\in\{.1,.25,.5\}$。最大匹配相对误差为 $8.57\times10^{-11}$。在 $r_c=.5$ 时，单位更新范数下降量依次为 SAM .2661、MS k=5 .2647、LB k=5 .2647、GAM final .2618。差异存在但远小于原生半径下的修正范数差异，说明原始比较中“谁更强”是主要混杂之一。

fixed-step 与 fixed-budget 在严格匹配后会得到相同的实际 path step，只是求得的 `primary_rho` 相差 $k$ 倍；协议标签本身不是一种不同的算子机制。

## 8. $Q_0/Q_1$ 的半径分解

当前 `Q1` 实际是“梯度范数增量的归一化比”，更准确的名字应是 $\Delta Q_1$。它不是原始一阶 sharpness 目标值之比，且在接近驻点或极小半径时分母可能病态。E002 应同时报告原始目标、增量比和 absolute regret。

E001-S 将每个候选拆成三种口径：原始注册 oracle 半径、将相同方向径向投影到 oracle 边界、以及以候选自身范数作为 oracle 半径。

| 候选 | 半径利用率 | raw $Q_0/Q_1$ | boundary $Q_0/Q_1$ | own-radius $Q_0/Q_1$ |
| --- | ---: | ---: | ---: | ---: |
| matched k=5 fixed-step | .600 | .472/.223 | .845/.393 | .941/.431 |
| matched k=5 fixed-budget | .600 | .587/.315 | .994/.536 | .998/.573 |
| MS/LB k=5 fixed-budget endpoint | 1.000 | .997/.565 | .998/.565 | .998/.565 |

因此 matched-SAM 的低 raw Q 同时来自没有用满配对路径球和有限半径下的方向误差。它不应与用满路径预算的端点直接排成“内层求解器优劣榜”。MS 与 LB 的共享端点也应在图中去重，外层聚合另用方向与一步下降指标评价。

## 9. 设置审计

### 合理并应保留

- 三级证据链：算子、轨迹、端点分别回答不同问题；
- $d=20$、等梯度构造作为谱单元测试；
- GAM 三对象拆分；
- 精确 trust-region oracle；
- fixed-step 与 fixed-budget 同时报告；
- 保存 $g/c/d$、原始数组、路径与 algorithmic-equivalent 预算。

### 已修订或必须改写

- `$H^p$ 阶数识别` 改称“有序 Krylov 子空间可压缩性”；QR 只改善数值稳定性，不能唯一归因物理 Hessian 阶数；
- `PathNovelty` 只用于真实路径对象；其他对象相对 $H\hat g$ 的残差写作 `h1_residual`；
- Lookbehind 当前对象是 `path_mean_surrogate`，不是包含 slow-weight interpolation、学习率与动量状态的完整训练 optimizer；
- GAM 当前是同一损失、exact-HVP、权重系数 1 的理想化算子；E002 必须显式暴露实际系数与 batch 语义；
- cosine 必须配合修正范数比、拟合半径比和向量相对误差；
- $Q$ 必须同时区分 path budget、endpoint radius 和径向利用率；
- `same steps`、`same unique gradients`、`same measured compute/data access` 是三个不同 estimand。

### E001 无法覆盖

- 非定 Hessian、负曲率和 signed spectral transfer；
- Hessian 随位置变化造成的真正非局部效应；
- mini-batch 方差、$\operatorname{Tr}(H\Sigma)$ 与时间自相关；
- faithful LookSAM/Lookbehind 的状态与计算成本；
- basin selection、完整轨迹和泛化。

## 10. 进入 E002 前的门槛

1. **方法保真门槛**：为 SAM/GAM/Lookbehind/LookSAM 建立 one-step reference delta 测试；把 path-mean surrogate 与 faithful slow-weight update 分行；实测 backward、sample access 和峰值内存。
2. **谱诊断门槛**：增加一个小型不定二次 saddle 单元测试，验证 signed transfer、$H_+/H_-$ 与负模态；E002 中对小投影分母使用 mask/谱带聚合。
3. **统计门槛**：shared-anchor 与 on-policy checkpoint 分开；mini-batch 数量由 paired bootstrap 的 CI 决定，而不是把 128 当固定真理。
4. **Taylor 门槛**：先扫描学习率；只有预注册区间内二阶预测误差足够小，才解释均值/协方差分解。
5. **公平性门槛**：同时报告 same outer steps、same examples/data passes、same measured compute；性能调参不能使用 test set。
6. **因果门槛**：若 RQ3 保留“来源/导致”语言，必须加入保持均值、匹配 $\operatorname{Tr}(\Sigma)$ 后旋转/白化/替换 centered noise 的完整轨迹干预；否则只写“一步贡献相关”。

在这些门槛完成前，合理下一步是 E002 pilot 的方法保真与数值校验，而不是直接比较最终准确率。

## 11. 可复现入口与产物

```bash
python Project1/SAM_Family_Mechanisms/run_e001_sensitivity.py \
  --output-dir Project1/SAM_Family_Mechanisms/outputs/e001_sensitivity
```

顶层产物包括：

- `sensitivity_summary.csv`：17 个 OFAT 配置、259 条对象记录；
- `strength_matched.csv`：42 条原生半径严格强度匹配记录；
- `fixed_effective_radius.csv`：固定 $\rho_{\mathrm{eff}}$ 的 $k$ 扫描；
- `quality_decomposition.csv`：raw/boundary/own-radius 三种 $Q$；
- `initialization_samples.csv` 与 `initialization_summary.csv`：两个 100 样本随机初始化族；
- 三张汇总图和每个 OFAT 子运行的完整 E001 产物。

敏感性 manifest 的代码指纹为 `ec7529815c6fdafb22afa89d55c831ba59c23c175a8b49832dd1f13aeb0defb3`。生成物位于 `outputs/`，不应手工编辑或当作源代码提交。
