# 结果总结

[返回首页](README.md)

## 1. ImageNet-R：support × direction（Exp E）

设置：SeqLoRA、ViT-B/16、rank 16、20×10 classes、seed 1993、20 epoch/task。

| Variant | FAA | AAA | BWT | Forget |
|---|---:|---:|---:|---:|
| SGD | 58.29 | 67.05 | -18.99 | 19.09 |
| SAM-factor | **67.94** | **70.60** | **-6.81** | **7.10** |
| SAM-full | 64.67 | 69.04 | -12.01 | 12.04 |
| SAM-delta | 65.41 | 69.63 | -11.56 | 11.66 |
| SAM-all | 63.19 | 67.55 | -12.28 | 12.28 |
| SAM-frozen | 62.91 | 67.29 | -12.37 | 12.46 |
| Random-factor | 58.79 | 67.12 | -18.47 | 18.57 |
| Random-full | 58.15 | 67.01 | -19.24 | 19.34 |
| Random-delta | 58.26 | 67.04 | -19.08 | 19.18 |
| Random-all | 58.32 | 67.04 | -19.01 | 19.11 |
| Random-frozen | 58.24 | 67.05 | -19.10 | 19.20 |

关键对比：

- SAM direction 在 factor/full/delta/all/frozen 每个 support 上都比对应 random direction 高 4.67–9.14 FAA。
- 所有 `random_*` 都接近 SGD，因此收益不是“加噪声本身”。
- `sam_factor` 比 SGD 高 9.64 FAA，final BWT 改善 12.19 点。
- `sam_delta` 明显有效，但弱于 `sam_factor`。该结果支持 adversarial direction，不足以完成 parameterization-invariant geometry 的证明。

## 2. ImageNet-R：forked trajectory（Exp F）

| Prefix → Suffix | FAA | AAA | BWT | Prefix Forget | Suffix Acc |
|---|---:|---:|---:|---:|---:|
| SGD → SGD | 61.72 | 67.60 | -15.32 | 17.19 | 64.33 |
| SGD → SAM-factor | 66.38 | 69.51 | -9.97 | 11.03 | 67.49 |
| SGD → Random-factor | 61.72 | 67.59 | -15.32 | 17.19 | 64.33 |
| SAM-factor → SGD | 65.29 | 69.84 | -10.38 | 11.87 | 66.46 |
| SAM-factor → SAM-factor | **67.50** | **70.42** | **-7.94** | **8.72** | **67.72** |

从完全相同的 SGD task-9 checkpoint 出发：

- suffix SAM-factor：FAA +4.66，prefix forgetting -6.15。
- suffix random-factor：与 suffix SGD 的 FAA 只差 -0.003，prefix forgetting 只差 +0.006。

因此 Exp F 是当前最直接的 trajectory-level 证据：later-task sharpness-aware update 改变旧任务 retention，而普通 matched random noise 不产生同样效果。

## 3. CIFAR10 两任务诊断

### Exp C

`sam_factor` 的 forgetting 最低（2.76）；`sam_random` 的 FAA/AAA 最高（97.11/97.78）；`sam_frozen` 最差。该实验适合做机制 pilot，不适合支撑大规模主结论。

### Exp D

仅在 task 2 使用 SAM-factor，将旧任务最终 accuracy `A_1,2` 相对 SGD 提升 0.36；仅在 task 1 使用只提升 0.06。random-factor 两种介入都只有约 0.02。结论与 Exp F 一致：后续 update path 比“早期任务得到一个 flat solution”更能解释 retention 增益。

## 4. Scope sweep

ImageNet-R 的 `Delta Acc^cls`（相对各自 SGD）：

| Scope | Strength | SeqLoRA | IncLoRA | OLoRA |
|---|---:|---:|---:|---:|
| Full | 0.05 | -3.50 | -2.00 | -2.62 |
| Full | 0.10 | -4.82 | -4.74 | -5.53 |
| Full | 0.15 | -6.72 | -6.84 | -6.49 |
| LoRA | 0.05 | +0.27 | +0.79 | +0.12 |
| LoRA | 0.10 | +0.70 | +0.86 | +0.28 |
| LoRA | 0.15 | +0.41 | +0.77 | +0.27 |

全参数 perturbation 随强度增大持续恶化，LoRA-only perturbation 保持小幅正增益。该实验支持“scope mismatch 有害”，但 full 与 LoRA 使用相同坐标强度并不保证相同 effective `DeltaW` norm。

## 5. 五个细粒度数据集与 Het5

当前活跃论文的 AAA 表显示：

- AS(0)/SAM 在 5 methods × 6 datasets 的 30 个单元格中均高于 SGD（按论文一位小数值核对）。
- AS(1)/GAM 在绝大多数单元格达到最高或并列最高。
- 明确例外：Aircraft-OLoRA 中 AS(0) 38.3 高于 AS(1) 35.5；CUB-InfLoRA 中 RS(0) 76.2 高于 AS(1) 68.3。
- RS(0)/RWP 高度不稳定：例如 IncLoRA-Cars 从 SGD 38.7 降至 21.4，但 IncLoRA-Flowers 从 57.5 升至 80.0。

因此最准确的表述是“AS(1) most often strongest，SAM broadly improves”，而不是“每个组合严格满足 AS(1)>AS(0)>RS(0)>SGD”。

完整数据见 [`data/vision_cross_domain_aaa.csv`](data/vision_cross_domain_aaa.csv)，FAA/AAA 双指标和 task matrix 见 `RebuttalReply/ExpSummaryExpForRebuttal/`。

## 6. NLP 外部有效性

### T5 三任务顺序平均 FAA

| Model | Method | Adam | SAM | Gain |
|---|---|---:|---:|---:|
| T5-large | SeqLoRA | 64.17 | 68.96 | +4.79 |
| T5-large | IncLoRA | 65.26 | 71.42 | +6.16 |
| T5-large | OLoRA | 77.21 | 77.79 | +0.58 |
| T5-small | SeqLoRA | 42.93 | 47.60 | +4.67 |
| T5-small | IncLoRA | 53.77 | 51.16 | **-2.61** |
| T5-small | OLoRA | 60.50 | 60.89 | +0.39 |

### Llama-3.2 三顺序平均 AAA

SAM 对 1B/3B 的 SeqLoRA、IncLoRA、OLoRA 六个已完成组合均提升 4.92–6.86 点；已有 GAM 的四个 IncLoRA/OLoRA 组合又均高于 SAM。SeqLoRA GAM 尚未完成。

语言结果支持 portability，但尚无语言侧 support comparison、curvature localization 或完整同尺度控制，不能视为理论的完整跨架构验证。

## 7. 已完成与未完成的 claim

| Claim | 状态 | 证据/缺口 |
|---|---|---|
| adversarial direction 优于 Gaussian random | 强支持 | Exp E 每个 support 一致；Exp F random control |
| effect 是 trajectory-level | 强支持 | Exp D/F |
| scope mismatch 会伤害 PECL | 支持 | scope sweep；full/frozen 较弱 |
| AS(1) 总体最强 | 趋势支持 | vision 大多数单元格；存在明确例外 |
| `Sh_Delta` 比 `Sh_AB` 更参数化不变 | 理论/实现支持，实证未闭环 | 需要 rescaling measurement/training control |
| per-task `Sh_Delta` 预测 future forgetting | 未完成 | Q1 post-hoc correlation 尚缺可确认结果 |
| multi-seed 稳健性 | 不完整 | Exp E/F 主表均是 seed 1993 |
| non-vacuous PAC-Bayes bound | 未完成 | 当前主要是结构性/定性 support reduction |

## 8. 写作建议

推荐主结论：

> Under frozen-backbone PECL, the effect of sharpness-aware training depends jointly on perturbation direction and admissible update support. Controlled support-matched experiments show that adversarial directions, rather than generic random noise, change the later posterior trajectory and reduce forgetting.

避免：

- “flat minima 一定导致不遗忘”；
- “SAM-factor 的最佳结果证明 raw factor-space 是正确理论空间”；
- “所有架构/数据集都严格满足同一个 optimizer 排序”；
- 用单 seed 的 Exp E/F 声称统计显著性。

