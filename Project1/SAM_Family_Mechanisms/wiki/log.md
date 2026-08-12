# 实验日志

本页只登记实际发生的实现、运行和验收。理论预期放在方法/指标页，尚未运行的数值不得填入结果栏。

## 状态看板

| 编号 | 实验 | 实现状态 | 运行状态 | 结果状态 |
| --- | --- | --- | --- | --- |
| E001 | 20 维精确 Hessian 二次算子实验 | 已实现；标准与扩展合计 17 项 unittest 通过 | 标准运行 passed | 已作算子层解读；无跨层结论 |
| E001-S | E001 敏感性与假设审计 | 已实现 | 17 个 OFAT 配置、强度/半径/初始化对照 passed | 已解读；科学门槛仍有缺口 |
| E002 | Two Moons 非凸轨迹实验 | planned，尚未实现 | 未运行 | 无结果 |
| E003 | 小型 FashionMNIST 端点验证 | planned，尚未实现 | 未运行 | 无结果 |

## 2026-08-11｜Wiki 初始化

- 建立三级证据链：E001 算子层、E002 轨迹层、E003 端点层；
- 将本次范围锁定为 E001，E002/E003 仅保留 planned 设计；
- 固定统一 $g/c/d$ 对象和 GAM 三对象语义；
- 固定 fixed-step、fixed-budget 与 matched-SAM 半径协议；
- 固定精确 $Q_0/Q_1$ oracle、CLI 入口和输出 schema；
- 尚未在此日志登记任何运行得到的具体数值。

## 2026-08-11｜E001 工程验收

- 状态：`passed`；
- 标准命令：`python run_quadratic.py --config configs/quadratic.yaml --output-dir outputs/e001_default`；
- 输出目录：`outputs/e001_default`；
- 必需产物：10/10 完整，包括 JSON、NPZ、三份核心 CSV 和四张图；
- 自动测试：17 项 unittest 通过（含后续敏感性入口）；
- 机器可读性：JSON 值均为 finite 或 `null`；
- 确定性：相同配置重复运行时，三份核心 CSV 字节一致；
- 覆盖的工程不变量：SAM/GAM 解析恒等式、GAM 三对象语义、两种路径预算、$k=1$ 退化、trust-region $Q_0/Q_1$ 上界、嵌套 $R^2$ 单调性和严格配置校验；
- 证据边界：以上是实现与复现验收，不是方法优劣的科学结果；当前未登记具体机制指标值，也未形成 E002/E003 所需的轨迹或泛化结论。

## 2026-08-11｜E001 标准结果解读

- run ID：`E001-20260811-7814f2cb`；
- 代码指纹：`7814f2cbf9154e0f8367b62a1274718980228a2cdce1c6040f2943c1f9229904`；
- 输出目录：`outputs/e001_default`；
- 解析正对照：SAM/HVP 与 GAM probe/$H^2$ 恒等式通过；它们没有被记作新机制发现；
- 观测：fixed-budget 后 LB k=2/5 与 matched-SAM 的 correction cosine 为 .99990/.99971，H1 residual 为 .0144/.0239；
- 观测：MS 在每个配对中的谱斜率、顶部能量和 H1 residual 均高于 LB，支持“路径平均软化最远端梯度”的实例内表述；
- 否定/修订：`Q0(Multistep)>Q0(SAM)` 不能无条件保留；fixed-step 的 k=2/5 反而为 .982/.896，对应 oracle 半径也不同；
- 完整表格与解释边界见 [结果页](06_e001_results_and_sensitivity.md)。

## 2026-08-11｜E001-S 敏感性运行

- run ID：`E001S-20260811-ec752981`；
- 代码指纹：`ec7529815c6fdafb22afa89d55c831ba59c23c175a8b49832dd1f13aeb0defb3`；
- 命令：`python run_e001_sensitivity.py --output-dir outputs/e001_sensitivity`；
- 输出目录：`outputs/e001_sensitivity`；
- 规模：17 个 OFAT 配置、259 条对象记录、42 条严格修正强度匹配记录、两个各 100 样本的随机初始化族；
- 新对照：fixed-effective-radius、raw/boundary/own-radius Q 分解、原生半径二分强度匹配；
- 最强设置告警：条件数从 4 增到 10,000 时，同一 $\rho/\lVert w\rVert=.01$ 下 SAM 的 $\lVert c\rVert/\lVert g\rVert$ 从 .0139 增至 8.055；
- 初始化告警：random-$w$ 的 LB k=5 fixed-budget H1 residual 中位数仅 .00135，而 engineered equal-gradient 为 .0239；
- 工程验收：17 项 unittest 全部通过，严格强度匹配最大相对误差 $8.57\times10^{-11}$；
- 结论边界：仍然只是 PSD 恒 Hessian 二次族；不支持负曲率、随机协方差、轨迹或泛化结论。

## E001 后续解读待办

- [x] 完成标准配置运行与产物完整性检查；
- [x] 完成解析恒等式、路径、oracle、嵌套拟合和确定性测试；
- [x] 从 `manifest.json` 登记 code fingerprint 与唯一 run ID；
- [x] 在不越过算子层边界的前提下解读四张主图与 summary；
- [x] 对每条观测结论记录对象、半径协议、matched-SAM 对照及不能支持的结论；
- [x] 补全全方法半径、条件数、维度、$k$ 与初始化敏感性；
- [x] 补严格修正强度、固定有效半径和 Q 半径分解；
- [ ] 增加不定二次 signed-spectrum 单元测试；
- [ ] 完成 E002 方法保真与统计协议门槛。

## 运行记录模板

复制以下小节，为每次真实运行建立唯一记录。

```markdown
### E001 / RUN-YYYYMMDD-HHMM-短标识

- 状态：running / passed / failed
- 代码版本：
- 完整命令：
- 配置文件：
- 输出目录：
- manifest schema：
- dtype：
- 解析后 rho_scales / primary_rho_scale：
- inner_steps / path_protocols：
- 计算预算核对：
- 产物完整性：
- 解析恒等式验收：
- 路径与 oracle 验收：
- 确定性复跑：
- 观测结果（只填真实数值）：
- 可以支持的结论：
- 不能支持的结论：
- 异常与后续动作：
```

## E002/E003 启动规则

E002 只有在 [结果页列出的启动门槛](06_e001_results_and_sensitivity.md#10-进入-e002-前的门槛) 完成后才从 `planned` 改为 `in progress`。E001 工程验收通过本身不再视为充分条件。E003 只有在 E002 形成可复核的稳定机制假设后才启动。状态变化必须附日期、责任范围和可追溯产物，不能因 Wiki 已描述设计就标记完成。
