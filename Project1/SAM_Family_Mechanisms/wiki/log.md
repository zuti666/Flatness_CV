# 实验日志

本页只登记实际发生的实现、运行和验收。理论预期放在方法/指标页，尚未运行的数值不得填入结果栏。

## 状态看板

| 编号 | 实验 | 实现状态 | 运行状态 | 结果状态 |
| --- | --- | --- | --- | --- |
| E001 | 20 维精确 Hessian 二次算子实验 | 已实现；标准与扩展合计 17 项 unittest 通过 | 标准运行 passed | 已作算子层解读；无跨层结论 |
| E001-S | E001 敏感性与假设审计 | 已实现 | 17 个 OFAT 配置、强度/半径/初始化对照 passed | 已解读；科学门槛仍有缺口 |
| E002-P | Two Moons 非凸轨迹 pilot | 已实现；25 项全套 unittest 通过 | GPU 5 正式运行 completed | 局部描述已解读；formal gate 未通过 |
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
- [x] 增加不定二次 signed-spectrum 单元测试；
- [x] 完成 E002-P 方法保真、shared-anchor/on-policy 与 Taylor 校准；
- [ ] 扩大协方差 probe、补 paired contrast CI 和外部 reference fidelity 后进入 formal E002。

## 2026-08-12｜E002-P GPU 5 正式 pilot

- 状态：`completed as pilot; engineering_passed=false; formal_e002_ready=false`；
- 命令：`CUDA_VISIBLE_DEVICES=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_e002_pilot.py --config configs/e002_pilot.yaml --output-dir outputs/e002_gpu5_pilot`；
- 输出目录：`outputs/e002_gpu5_pilot`；
- 代码指纹：`516705290632608a06172beba67fda62c4cc2fe3b45028113c25761364af6b5a`；
- 设备：物理 GPU 5，RTX 6000 Ada，UUID `GPU-6eac7f06-8173-e1a3-c938-239f6f6eb19e`；身份验证 passed；
- 环境：Python 3.11.15、PyTorch 2.4.1+cu121、CUDA 12.1、float64、deterministic algorithms；
- 用时/显存：423.0 秒；自身 peak allocated 85,664,256 bytes；GPU 5 同期已有其他满载进程，墙钟不可比较方法成本；
- 规模：2 seeds × 5 shared SGD checkpoints × 64 paired probe batches；6 条 on-policy 轨迹各 800 steps；
- 产物：除 Matplotlib cache 外 93 个文件；9 份 CSV、6 张图、3 份 NPZ、shared/on-policy checkpoints、manifest/metrics/integrity 完整；
- 自动测试：全项目 25/25 unittest passed；包括 E002 quick CLI、不定二次 signed spectrum、HVP 与方法退化契约；
- passed gates：内部方法 fidelity、Hessian symmetry/eigen reconstruction、covariance PSD、主/最小 $\eta$ Taylor；
- failed gate：seed 3408 的 SGD train loss ratio .829 > .7；其绝对 loss 与 accuracy 仍改善，故登记为脆弱相对门槛失败而非发散；
- precision warning：140 条 covariance 记录中 4 条 CI 相对半宽 > .25；
- Taylor 观测：$\eta=.05$ 最坏 row median .00303、最坏 p90 .00798；允许本设置的一步分解；
- Hessian 观测：10 个 anchor 全部不定，负模态 25–33 个，$\lambda_{\max}$ 没有随训练下降；
- 协方差观测：SAM/GAM/MS/LB 的 raw $\operatorname{Tr}(H_+\Sigma)$ 均高于 SGD，但总方差同时放大，NHA 只小幅变化；
- 路径观测：$k=2$ fixed-budget 的 misalignment 很小，以最后梯度为分母，路径平均 trace 低约 5.6%–6.9%；
- 时间观测：fixed-data SGD path 上 orthogonal correction 的 lag-20 cosine 约 .967/.975，但相对 drift 仍 .322/.243；
- 结论边界：不能作方法排名、GAM endpoint 结论、LookSAM cache 因果结论、协方差导致泛化结论或 E003 外推；完整解释见 [E002-P 结果页](07_e002_gpu5_pilot.md)。

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

E002-P 已从 `planned` 推进到已运行 pilot，但因上述失败/精度门槛不能标为 formal completed。formal E002 必须先完成 [E002-P 结果页的下一步 gate](07_e002_gpu5_pilot.md#6-下一步-gate)。E003 只有在 formal E002 形成可复核的稳定机制假设后才启动。状态变化必须附日期、责任范围和可追溯产物，不能因 Wiki 已描述设计就标记完成。
