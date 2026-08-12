# 01｜研究范围与研究问题

## 为什么需要三级实验

不同层级回答不同问题，不能由一个层级替代另一个层级。

| 层级 | 实验 | 主要观测对象 | 可以回答 | 不能回答 |
| --- | --- | --- | --- | --- |
| 算子层 | E001：20 维二次函数 | 精确 $H$、$g/c/d$、扰动与路径 | 方法的 Hessian 谱滤波、有限差分、路径半径、内层 oracle 质量 | basin selection、mini-batch 噪声、时间复用、泛化 |
| 轨迹层 | E002：Two Moons MLP | checkpoint、更新均值/协方差、路径与时间相关 | 曲率信息如何改变方向、随机性和训练轨迹 | 较大视觉任务上的普适泛化 |
| 端点层 | E003：FashionMNIST 小网络 | 训练/测试端点、top Hessian 谱、sharpness | 前两层机制能否外推到小型视觉网络 | 大规模模型和数据集上的普遍结论 |

**当前实现边界：只做 E001。** E001 已通过工程验收；E002 和 E003 均为 planned。Wiki 描述后两者的设计是为了固定路线图，不代表相关代码、运行或结果已经存在。

## 三个总研究问题

### RQ1：方法实际产生了怎样的 Hessian 谱滤波？

关注固定参数点和固定目标下，不同方法的修正 $c^{(m)}$ 以及 GAM 的 probe direction/probe increment 在 Hessian 特征方向上的有效响应，并检验这些语义不同的对象在有序 Krylov 子空间 $\operatorname{span}(H\hat g,\ldots,H^p\hat g)$ 中是否可压缩。这里的增量拟合不能唯一识别“真实 Hessian 阶数”。

E001 可直接回答：

- SAM 修正是否与 $H\hat g$ 一致；
- GAM 的 probe increment 是否呈 $H^2\hat g$ 型谱响应；
- MS-SAM 与 Lookbehind 的谱增益是否仅来自更大的有效半径。

### RQ2：多步探测改变的是信息本身，还是强度、稳定性与时间尺度？

E001 只考察“路径信息”和“有效半径”两部分：

- fixed-step 与 fixed-budget 两个路径协议是否给出不同结论；
- Lookbehind 与 matched-SAM 的差异是否在匹配一阶有效半径后仍存在；
- 路径聚合相对最后一步梯度是否含有超出 $H\hat g$ 的残差；
- 在 matched radius、严格修正范数和固定一阶有效半径后，残差是否仍存在。

“时间尺度”必须等到 E002 才能通过 LookSAM 正交修正的自相关、陈旧误差与刷新频率来检验。E001 不能回答时间复用。

### RQ3：一步变化中确定性修正与更新协方差各占多少？

E001 是确定性全批量二次模型，只能建立确定性算子基线。以下对象均留给 E002/E003：

\[
\Sigma_m,\qquad
\operatorname{Tr}(H_+\Sigma_m),\qquad
\text{一步 Taylor 的随机二阶项},\qquad
\text{泛化差异}.
\]

因此，E001 的任何图都不能用于回答 RQ3 的协方差或泛化部分。即使 E002 完成 fixed-checkpoint 的 Taylor 分解，也只能描述一步贡献；若要使用“最终性能来自/导致”这样的因果措辞，还需要保持均值并干预 centered noise 结构的完整轨迹实验。

## E001 的四个具体问题

1. **单步有限差分是否忠实提取 $H\hat g$？** 对 SAM 扫描多个相对半径，比较有限差分 HVP 与解析 HVP。
2. **GAM 的不同阶段对应哪个 Hessian 阶数？** 分别分析 probe direction、probe increment 和 final regularizer，禁止只看总更新后宣称 $H^2$。
3. **多步路径是新信息还是放大？** 用 QR 后的嵌套 Krylov 拟合、H1 residual、matched-SAM、严格修正强度和固定有效半径检验。
4. **内层扰动解决了哪个 sharpness 目标？** 用同半径精确 oracle 归一化的 $Q_0/Q_1$ 比较零阶与一阶目标。

## 当前明确不做的内容

- 不训练神经网络，不使用 CIFAR，也不报告准确率或泛化间隙；
- 不实现 LookSAM、SAM-$k$、Noise-only 或 mini-batch 协方差；
- 不用二次模型结果声称某方法会选择更平坦的 basin；
- 不把理论预期、验收阈值或图表占位符写成实验结果；
- 不用单一 $\lambda_{\max}$ 代替方法自身对应的零阶或一阶 sharpness 目标。

## 升级到下一层的门槛

E001 的工程门槛已经通过，但 [敏感性审计](06_e001_results_and_sensitivity.md) 表明科学门槛尚未完全通过。正式 E002 前还需完成方法 one-step 保真、path-mean surrogate 与 faithful Lookbehind 的语义拆分、不定二次 signed-spectrum 单元测试，以及 shared-anchor/on-policy、bootstrap 和 Taylor 误差阈值的预注册。门槛约束实现和推断有效性，不要求结果符合某个预期方向。
