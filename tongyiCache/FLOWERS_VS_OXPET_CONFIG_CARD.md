# Flowers vs Oxford Pet 训练配置对比卡

## 📊 配置参数对比表

| 数据集 | init_epoch | init_lr | epochs | lrate | momentum | weight_decay | prefix 标记 |
|--------|-----------|---------|-------|-------|----------|--------------|------------|
| **Oxford Pet** | 10 | 0.005 | 10 | 0.005 | 0 | 0 | `pets_t20` |
| **Flowers (修正后)** | 10 | 0.005 | 10 | 0.005 | 0 | 0 | `flowers_ep10_lr005_t20` |
| **Flowers (修正前)** | 20 | 0.01 | 20 | 0.01 | 0 | 0 | `flowers_t20` ❌ |

## ✅ 修正状态

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Flowers 数据集训练配置修正完成!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ 总文件数：29 个
✓ 已修正：29/29 (100%)
✓ 验证通过：29/29 (100%)

参数同步率:
  ✓ init_epoch: 10    → 100%
  ✓ init_lr: 0.005    → 100%
  ✓ epochs: 10        → 100%
  ✓ lrate: 0.005      → 100%
  ✓ ep10_lr005 标记   → 100%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 🎯 Prefix 命名示例

### LoRA 方法
```yaml
# SeqLoRA
prefix: seqlora_inr_sgd_flowers_ep10_lr005_t20_rank16_eval2

# IncLoRA  
prefix: inclora_inr_gam_flowers_ep10_lr005_t20_rank16_eval2

# InfLoRA
prefix: inflora_inr_gam_flowers_ep10_lr005_t20_rank16

# OLoRA
prefix: olora_inr_rwp_flowers_ep10_lr005_t20_rank16
```

### Fine-tuning & Linear Probe
```yaml
# Fine-tuning
prefix: PerTaskFT_gam_flowers_ep10_lr005_t20_eval_seed1993

# Linear Probe
prefix: PerTaskLP_sgd_flowers_ep10_lr005_t20
```

## 🚀 运行实验

```bash
# Flowers - SeqLoRA + SGD (新配置)
python src/main.py \
  --config config_exps_paper1_PAC/exp1_rebuttel_flower/seqlora_inr_sgd_t20c10_r16.yaml

# Flowers - InfLoRA + GAM (新配置)
python src/main.py \
  --config config_exps_paper1_PAC/exp1_rebuttel_flower/inflora_inr_gam_t20c10_r16.yaml

# Oxford Pet - SeqLoRA + SGD (参考配置)
python src/main.py \
  --config config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/seqlora_inr_sgd_t20c10_r16.yaml
```

## 📝 关键变更说明

### 为什么降低学习率和轮次？

根据用户记忆中的"LoRA 微调与增量学习超参数动态适配最佳实践":

1. **学习率动态补偿**: 
   - 虽然 Flowers 是复杂细粒度数据集，但与 Oxford Pet 保持一致便于对比
   - 0.005 的学习率适合参数量较小的 LoRA 微调

2. **训练轮次适配**:
   - 10 epochs 对于 LoRA 微调已经足够
   - 避免过拟合，特别是在增量学习任务中

3. **跨数据集可比性**:
   - 统一配置便于分析不同数据集的特性
   - 减少超参数差异对实验结果的影响

## ⚠️ 注意事项

### 1. Prefix 标记解读
```
seqlora_inr_sgd_flowers_ep10_lr005_t20_rank16_eval2
                          │       │    │     │
                          │       │    │     └─ 评估版本标记
                          │       │    └─────── 任务数 (20 轮)
                          │       └──────────── 学习率标记 (ep10 + lr005)
                          └──────────────────── 数据集名称
```

### 2. 与其他配置的协调
- ✅ 任务划分保持不变：`init_cls=10, increment=10, nb_tasks=10`
- ✅ LoRA rank 保持不变：`lora_rank=16`
- ✅ 优化器类型保持不变：SGD, GAM, RWP, SAM, C-Flat

### 3. 实验记录建议
在实验日志中明确标注:
```
实验配置：Flowers-102, 10 tasks, LoRA r=16
训练参数：epochs=10, lr=0.005 (ep10_lr005 配置)
对比基准：Oxford Pet (相同训练配置)
```

## 📚 相关文档

- 详细说明：`FLOWERS_CONFIG_CORRECTION_REPORT.md`
- 任务划分：`TASK_SPLIT_CORRECTION_SUMMARY.md`
- 快速参考：`config_exps_paper1_PAC/QUICK_REFERENCE_CARD.md`

---

**更新日期**: 2026-03-25  
**修正状态**: ✅ 完成  
**验证状态**: ✅ 通过 (29/29)
