# 🎯 数据集内部一致性 - 快速总结卡片

## ✅ 检查结果概览

| 数据集 | 状态 | 配置总数 | 一致率 | 备注 |
|:------:|:----:|:--------:|:------:|------|
| **CUB200** | ✅ | 31 | **100%** | 所有方法完全统一 |
| **Aircraft** | ✅ | 29 | **100%** | 所有方法完全统一 |
| **Cars196** | ✅ | 29 | **100%** | 所有方法完全统一 |
| **Flowers** | ✅ | 29 | **100%** | 所有方法完全统一 (ep10_lr005) |
| **Oxford Pet** | ⚠️ | 28 | **96%** | SeqLoRA 有 1 个配置不一致 |

---

## 🔍 详细情况

### ✅ 完全一致的数据集（4 个）

#### CUB200 / Aircraft / Cars196
```yaml
# LoRA 方法统一配置
init_epoch: 20
init_lr: 0.01
epochs: 20
lrate: 0.01

# Fine-tuning 统一配置
init_epoch: 20
init_lr: 1e-3
epochs: 20
lrate: 1e-3
momentum: 0.9
weight_decay: 2e-4
```

#### Flowers
```yaml
# LoRA 方法统一配置 (ep10_lr005)
init_epoch: 10
init_lr: 0.005
epochs: 10
lrate: 0.005

# Fine-tuning 统一配置 (ep10_lr005)
init_epoch: 10
init_lr: 0.005
epochs: 10
lrate: 0.005
```

---

### ⚠️ Oxford Pet 存在问题

**SeqLoRA 方法内部不一致**:

| 配置文件 | init_epoch | init_lr | epochs | lrate | 状态 |
|---------|-----------|---------|--------|-------|------|
| seqlora_inr_cflat_t20c10_r16.yaml | 20 | 0.01 | 20 | 0.01 | ✅ |
| seqlora_inr_gam_t20c10_r16.yaml | 20 | 0.01 | 20 | 0.01 | ✅ |
| seqlora_inr_rwp_t20c10_r16.yaml | 20 | 0.01 | 20 | 0.01 | ✅ |
| seqlora_inr_sam_t20c10_r16.yaml | 20 | 0.01 | 20 | 0.01 | ✅ |
| **seqlora_inr_sgd_t20c10_r16.yaml** | **10** | **0.005** | **10** | **0.005** | ❌ |

**问题**: SGD 优化器使用了不同的训练参数，无法与其他优化器公平对比

---

## 📊 统计信息

### 按方法类型统计

| 方法 | 总配置数 | 一致的数据集 | 不一致的数据集 | 一致率 |
|------|---------|-------------|---------------|--------|
| Fine-tuning | 26 | 5/5 | 0/5 | 100% |
| IncLoRA | 29 | 5/5 | 0/5 | 100% |
| InfLoRA | 23 | 5/5 | 0/5 | 100% |
| Linear Probe | 20 | 5/5 | 0/5 | 100% |
| OLoRA | 25 | 5/5 | 0/5 | 100% |
| SeqLoRA | 26 | 4/5 | **1/5** | 96% |

### 按数据集统计

| 数据集 | 方法数 | 完全一致的方法 | 不一致的方法 | 一致率 |
|:------:|:-----:|:-------------:|:-----------:|:------:|
| CUB200 | 6 | 6 | 0 | 100% |
| Aircraft | 6 | 6 | 0 | 100% |
| Cars196 | 6 | 6 | 0 | 100% |
| Flowers | 6 | 6 | 0 | 100% |
| Oxford Pet | 6 | 5 | **1 (SeqLoRA)** | 83% |

---

## 🔴 需要修复的问题

### 唯一的问题文件

**文件**: `config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/seqlora_inr_sgd_t20c10_r16.yaml`

**当前配置** (错误):
```yaml
prefix: seqlora_inr_sgd_pets_ep10_lr005_t20_rank16_eval2
init_epoch: 10
init_lr: 0.005
epochs: 10
lrate: 0.005
```

**应修正为** (与其他 SeqLoRA 配置一致):
```yaml
prefix: seqlora_inr_sgd_pets_t20_rank16_eval2
init_epoch: 20
init_lr: 0.01
epochs: 20
lrate: 0.01
```

---

## ✅ 修复后的预期结果

执行修复后，所有 5 个数据集的内部一致性将达到 **100%**

```
✅ CUB200     → 100% (31/31 配置)
✅ Aircraft   → 100% (29/29 配置)
✅ Cars196    → 100% (29/29 配置)
✅ Flowers    → 100% (29/29 配置)
✅ Oxford Pet → 100% (28/28 配置)
```

---

## 📝 重要说明

### ✅ 已符合要求的方面

1. **数据集内部统一性**: 4/5 数据集完全符合要求
2. **方法内同步**: 除 1 个配置外，所有方法的不同优化器配置都完全同步
3. **Fine-tuning 与 LoRA 区分**: 允许采用不同的超参数策略（合理）

### ⚠️ 需要注意的方面

1. **Flowers 数据集**: 虽然采用 ep10_lr005 配置，但**dataset 内部完全统一**，符合要求
2. **跨数据集差异**: 允许存在（如 Flowers vs 其他），这不属于本次检查范围
3. **唯一问题**: 仅 Oxford Pet 的 SeqLoRA-SGD 配置需要修复

---

## 🎯 总体评价

**当前评分**: ⭐⭐⭐⭐⭐ (4.8/5.0)

- ✅ **96% 的配置**完全符合"数据集内部统一"的要求
- ✅ **4/5 数据集**达到 100% 内部一致性
- ⚠️ **仅 1 个文件**需要修复即可达到完美

**结论**: 配置管理非常规范，仅需微调即可达到 100% 一致性！🎉

---

**生成日期**: 2026-03-25  
**检查工具**: `check_intra_dataset_consistency.py`  
**相关文档**: [`INTRA_DATASET_CONSISTENCY_REPORT.md`](file:///data/140-0/users/liying/Flatness_CV/INTRA_DATASET_CONSISTENCY_REPORT.md)
