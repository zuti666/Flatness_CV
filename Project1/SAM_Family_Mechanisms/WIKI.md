# Wiki：SAM 家族的曲率信息、路径与端点机制

## 研究主线

本项目不从大数据集最终准确率倒推机制，而采用三级证据链：

```text
E001 算子层：20 维二次函数
    ↓ 先确认每种方法真正提取的 Hessian 信息
E002-P 轨迹层：Two Moons 非凸 MLP（GPU 5 pilot 已运行）
    ↓ 再分析方向、随机性、路径和一步损失变化
E003 端点层：小型 FashionMNIST（planned）
      最后检验局部机制能否解释平坦性与泛化
```

当前已实现 E001 标准算子实验、E001-S 敏感性审计，以及 E002-P 的 shared-anchor/on-policy 校准运行。E002-P 通过方法保真、Hessian 数值、协方差 PSD 与 Taylor 门槛，但因一个预注册训练 smoke 和少数 CI 精度门槛未通过，只能标为 pilot，不能写成正式轨迹/泛化结论。E003 仍只是路线图。

## 统一语言

在固定参数 $w$ 上，令

\[
g=\nabla L(w),\qquad
d^{(m)}=\text{方法 }m\text{ 实际用于下降的方向},\qquad
c^{(m)}=d^{(m)}-g.
\]

解释时必须分开报告原始梯度 $g$、额外修正 $c$ 和总方向 $d$。谱与拟合表还使用 `object_kind` 指明当前分析向量：普通方法是 update correction；GAM 则严格拆成 `probe_direction`、`probe_increment` 与 `final_regularizer` 三行。对多步方法还必须记录路径端点、路径总长和路径梯度。

## 当前比较集合

E001 的曲率方法是：

\[
\mathrm{SAM},\quad
\mathrm{GAM},\quad
\mathrm{MS\text{-}SAM}_{k\in\{2,5\}},\quad
\mathrm{Lookbehind}_{k\in\{2,5\}}.
\]

SGD 只作为 $c=0$ 的参考基线。LookSAM、SAM-$k$、Noise-only、更新协方差与训练轨迹需要时间或 mini-batch 维度，放在 E002；它们不是 E001 的伪实现项。

## 页面索引

1. [范围与研究问题](wiki/01_scope_and_rqs.md)：三级实验为什么分开，以及每一级能支持什么结论。
2. [方法与对象](wiki/02_methods_and_objects.md)：二次问题、SAM/GAM/MS-SAM/Lookbehind，以及两个路径半径协议。
3. [指标](wiki/03_metrics.md)：谱增益、Krylov 子空间拟合、$Q_0/Q_1$、matched-SAM 和验收不变量。
4. [实验](wiki/04_experiments.md)：E001/E002 配置、CLI、输出契约，以及 planned 的 E003。
5. [公平性与复现](wiki/05_fairness_and_reproduction.md)：计算预算、强度匹配、确定性与研究结论边界。
6. [E001 结果与敏感性](wiki/06_e001_results_and_sensitivity.md)：标准结果、OFAT 扫描、假设判定与 E002 门槛。
7. [E002 GPU 5 pilot](wiki/07_e002_gpu5_pilot.md)：shared-anchor、Taylor、协方差、路径、时间复用、on-policy sanity 与失败门槛。
8. [日志](wiki/log.md)：只登记真实运行，不把预期写成结果。

## 证据标签

后续写结论时统一使用以下标签：

- **解析恒等式**：由二次模型直接推出，不是经验结果；
- **验收目标**：用来检查实现是否正确，不是实验发现；
- **观测结果**：必须能追溯到日志中的 run ID 和产物；
- **假设/待验证**：只能指导后续实验，不能写成已证实结论。

这一约定尤其用于避免把 “GAM 的 probe increment 具有 $H^2$ 型响应”误写为“GAM 的最终更新等价于 $H^2g$”。
