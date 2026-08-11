# Project1 Wiki：LoRA-PECL 中的 Flatness 与 Sharpness

> 整理日期：2026-08-06  
> 核对范围：`Project1/`、`RebuttalReply/`、`config_exps_paper1_PAC/`、`outputs_logs/`、`logs_exp1_rebuttal*`、`summaries/` 及当前评估代码。  
> 当前论文题目：*Revisiting Sharpness in Low-rank Subspaces for Continual Learning*。

## 1. 一句话结论

Project1 研究的不是“再提出一个优化器”，而是：在冻结 backbone、只允许 LoRA 低秩更新的参数高效持续学习（PECL）中，什么范围、什么方向的 flatness 才与泛化和遗忘相关。

现有证据支持三点：

1. 扰动应围绕 PECL 实际可达的 adapter-update geometry 来解释，而不是无条件使用全参数空间。
2. adversarial sharpness-aware direction 是主要有效成分；同支撑上的 Gaussian random direction 基本不能复现增益。
3. 该作用会改变后续 posterior trajectory，并降低旧任务遗忘，不只是改善当前任务的局部拟合。

需要保留的边界是：默认 LoRA 参数化下 `sam_factor` 的表现强于 `sam_delta`，因此当前结果不能单独证明 raw factor-space 是正确理论对象；重参数化控制仍是关键实验。

## 2. Wiki 导航

| 页面 | 内容 |
|---|---|
| [方法与理论](01_method.md) | PECL 几何、三类 sharpness、扰动 direction/support、PAC-Bayes 主线 |
| [评估指标](02_metrics.md) | Accuracy matrix、FAA/AAA/BWT/Forgetting、NME、flatness、curvature 与 feature 指标 |
| [实验与配置](03_experiments_and_configs.md) | 主实验、消融、rebuttal 扩展、配置入口和当前有效设置 |
| [日志与复现](04_logs_and_reproduction.md) | 原始日志、metrics JSON、summary CSV、运行/汇总命令和数据追踪规则 |
| [结果总结](05_results.md) | ImageNet-R、跨域、Het5、T5/Llama 的结果、结论强度与尚未完成项 |
| [绘图代码](06_plotting.md) | 统一绘图脚本、输入数据、输出图和扩展方法 |
| [结果数据说明](data/README.md) | Wiki 快照 CSV 的字段、来源和优先级 |
| [MLP–LoRA 新故事与实验计划](../MLP_LoRA_CL/WIKI.md) | constrained reachable geometry、merge-reset 对照、精确方向曲率与小模型实现 |

## 3. 研究问题与证据对应

| 研究问题 | 主要实验 | 当前结论 |
|---|---|---|
| 扰动范围是否重要？ | 论文 scope sweep、Exp B/E | LoRA/admissible scope 优于 full/frozen；但 factor 与 invariant delta geometry 的比较仍需重参数化控制 |
| adversarial direction 是否重要？ | Exp E support × direction | 每个 support 上 SAM 都明显优于 Gaussian random |
| 是否只是当前任务局部效应？ | Exp D、Exp F forked trajectory | 从同一 checkpoint 分叉后，后续 SAM 显著降低 prefix forgetting；random-factor 与 SGD 几乎相同 |
| 结论是否跨方法/数据成立？ | ImageNet-R/C/P、5 个细粒度数据集、Het5、T5/Llama | SAM 广泛有效；GAM/AS(1) 在 vision 多数单元格最强，但 RWP 很不稳定 |
| 理论对象是否参数化不变？ | Exp 7 RQ2、Exp G/Q2 | 已设计，现有 Wiki 未发现可用于定论的完整结果 |

## 4. 资料口径

仓库同时保存了多代论文、配置和日志。本 Wiki 按以下优先级引用：

1. 原始 `*_cl_metrics.json` accuracy matrix 和实验 summary CSV。
2. 当前活跃论文 `Project1/main_new1 (8).tex` 中未注释的表格。
3. `RebuttalReply/ExpSummaryExpForRebuttal/` 的自动核对表。
4. 历史 PDF、历史 TeX 和目录名，只作为追溯材料。

结果值与“当前 YAML”分开记录。YAML 会继续修改，旧日志开头打印的最终参数才代表该次实际运行。详细差异见[实验与配置](03_experiments_and_configs.md#8-已知配置与文稿漂移)和[日志与复现](04_logs_and_reproduction.md#5-结果追踪规则)。

## 5. 最可信的现有数字

- Exp E，ImageNet-R，SeqLoRA，`r=16, T=20, seed=1993`：`sam_factor` FAA 67.94、BWT -6.81；SGD 为 58.29、-18.99；`random_factor` 为 58.79、-18.47。
- Exp F，同一 task-9 SGD checkpoint 分叉：suffix `sam_factor` 将 FAA 从 61.72 提升到 66.38，并把 prefix forgetting 从 17.19 降到 11.03；suffix `random_factor` 与 SGD 基本一致。
- 当前论文跨域 AAA 表中，AS(0)/SAM 在 30 个 method × dataset 单元格中均高于 SGD；AS(1)/GAM 在绝大多数单元格最强，但 Aircraft-OLoRA 和 CUB-InfLoRA 是明确例外。
- T5-large 上，SAM 对 SeqLoRA、IncLoRA、OLoRA 的三顺序平均 FAA 分别提升 4.79、6.16、0.58 个点；T5-small/IncLoRA 为负增益 -2.61，不能写成无条件跨架构一致提升。

## 6. 与已有 Wiki 的关系

- `research-wiki/`：ICML 阶段的 claim/idea/paper 图谱。
- `Project1/wiki_neurips2026/`：后续 NeurIPS theory paper 的 pathwise PAC-Bayes 与 Exp E/F 结构。
- `Project1/MLP_LoRA_CL/`：本次新增的小模型因果验证包；把 Dense/LoRA、rank、生命周期与精确 HVP 放进同一受控实验。
- 本目录：面向 Project1 全流程的工程 Wiki，集中覆盖方法、指标、配置、日志、结果与可复现绘图。
