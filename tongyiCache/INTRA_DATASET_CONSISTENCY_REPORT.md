# 📊 数据集内部训练超参数一致性检查报告

## ✅ 检查结论

**检查范围**: 每个数据集**内部**的训练超参数一致性  
**检查目标**: 确保同一数据集内，相同方法的所有优化器配置完全统一  
**评价标准**: 不要求跨数据集统一，允许不同数据集采用不同的训练策略

---

## 📈 总体结果

| 数据集 | 内部一致性状态 | 问题方法 | 问题配置数 |
|--------|--------------|----------|-----------|
| **CUB200** | ✅ **完全一致** | 无 | 0/31 |
| **Aircraft** | ✅ **完全一致** | 无 | 0/29 |
| **Cars196** | ✅ **完全一致** | 无 | 0/29 |
| **Flowers** | ✅ **完全一致** | 无 | 0/29 |
| **Oxford Pet** | ⚠️ **存在不一致** | SeqLoRA | 1/28 |

**总体通过率**: 4/5 数据集 (80%) 完全一致

---

## ✅ 完全一致的数据集（4 个）

### 1. CUB200 数据集

**配置统计**: 31 个配置文件，6 种方法

| 方法 | 配置数 | init_epoch | init_lr | epochs | lrate | momentum | weight_decay | 状态 |
|------|--------|-----------|---------|--------|-------|----------|--------------|------|
| Fine-tuning | 6 | 20 | 1e-3 | 20 | 1e-3 | 0.9 | 2e-4 | ✅ |
| IncLoRA | 5 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |
| InfLoRA | 5 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |
| Linear Probe | 5 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |
| OLoRA | 5 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |
| SeqLoRA | 6 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |

**结论**: ✅ 所有方法内部配置完全统一

---

### 2. Aircraft 数据集

**配置统计**: 29 个配置文件，6 种方法

| 方法 | 配置数 | init_epoch | init_lr | epochs | lrate | momentum | weight_decay | 状态 |
|------|--------|-----------|---------|--------|-------|----------|--------------|------|
| Fine-tuning | 5 | 20 | 1e-3 | 20 | 1e-3 | 0.9 | 2e-4 | ✅ |
| IncLoRA | 6 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |
| InfLoRA | 5 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |
| Linear Probe | 3 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |
| OLoRA | 5 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |
| SeqLoRA | 5 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |

**结论**: ✅ 所有方法内部配置完全统一

---

### 3. Cars196 数据集

**配置统计**: 29 个配置文件，6 种方法

| 方法 | 配置数 | init_epoch | init_lr | epochs | lrate | momentum | weight_decay | 状态 |
|------|--------|-----------|---------|--------|-------|----------|--------------|------|
| Fine-tuning | 5 | 20 | 1e-3 | 20 | 1e-3 | 0.9 | 2e-4 | ✅ |
| IncLoRA | 6 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |
| InfLoRA | 5 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |
| Linear Probe | 3 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |
| OLoRA | 5 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |
| SeqLoRA | 5 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | ✅ |

**结论**: ✅ 所有方法内部配置完全统一

---

### 4. Flowers 数据集

**配置统计**: 29 个配置文件，6 种方法

| 方法 | 配置数 | init_epoch | init_lr | epochs | lrate | momentum | weight_decay | 状态 |
|------|--------|-----------|---------|--------|-------|----------|--------------|------|
| Fine-tuning | 5 | 10 | 0.005 | 10 | 0.005 | 0 | 0 | ✅ |
| IncLoRA | 6 | 10 | 0.005 | 10 | 0.005 | 0 | 0 | ✅ |
| InfLoRA | 4 | 10 | 0.005 | 10 | 0.005 | 0 | 0 | ✅ |
| Linear Probe | 4 | 10 | 0.005 | 10 | 0.005 | 0 | 0 | ✅ |
| OLoRA | 5 | 10 | 0.005 | 10 | 0.005 | 0 | 0 | ✅ |
| SeqLoRA | 5 | 10 | 0.005 | 10 | 0.005 | 0 | 0 | ✅ |

**结论**: ✅ 所有方法内部配置完全统一（ep10_lr005 配置）

**说明**: Flowers 数据集虽然采用了与其他数据集不同的 `ep10_lr005` 配置，但** dataset 内部所有方法完全统一**，符合"数据集内部统一"的要求。

---

## ⚠️ 存在不一致的数据集（1 个）

### 5. Oxford Pet 数据集

**配置统计**: 28 个配置文件，6 种方法

| 方法 | 配置数 | 一致性状态 | 问题描述 |
|------|--------|-----------|----------|
| Fine-tuning | 5 | ✅ 一致 | - |
| IncLoRA | 6 | ✅ 一致 | - |
| InfLoRA | 4 | ✅ 一致 | - |
| Linear Probe | 4 | ✅ 一致 | - |
| OLoRA | 5 | ✅ 一致 | - |
| **SeqLoRA** | **5** | ❌ **不一致** | **1 个配置与其他 4 个不同** |

---

### 🔴 SeqLoRA 方法配置不一致详情

**问题文件**: [`seqlora_inr_sgd_t20c10_r16.yaml`](config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/seqlora_inr_sgd_t20c10_r16.yaml)

**异常配置值**:
```yaml
init_epoch: 10      # ❌ 其他 4 个配置为 20
init_lr: 0.005      # ❌ 其他 4 个配置为 0.01
epochs: 10          # ❌ 其他 4 个配置为 20
lrate: 0.005        # ❌ 其他 4 个配置为 0.01
```

**正常配置** (其他 4 个 SeqLoRA 配置):
```yaml
# seqlora_inr_cflat_t20c10_r16.yaml
# seqlora_inr_gam_t20c10_r16.yaml
# seqlora_inr_rwp_t20c10_r16.yaml
# seqlora_inr_sam_t20c10_r16.yaml
init_epoch: 20
init_lr: 0.01
epochs: 20
lrate: 0.01
momentum: 0
weight_decay: 0
```

**影响分析**:
- ❌ 违反"全量同步原则"（用户记忆规范）
- ❌ 同一方法下不同优化器配置不一致
- ❌ 无法公平对比 SeqLoRA 方法下不同优化器的性能
- ❌ SGD 优化器使用不同的训练策略，导致实验结果不可比

---

## 🔧 修复建议（仅 Oxford Pet）

### 需要修复的问题

**目标**: 将 [seqlora_inr_sgd_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/seqlora_inr_sgd_t20c10_r16.yaml) 回退到与其他 4 个 SeqLoRA 配置一致

**修复命令**:
```bash
cd /data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_oxfordPet

# 回退训练参数
sed -i 's/init_epoch: 10$/init_epoch: 20/' seqlora_inr_sgd_t20c10_r16.yaml
sed -i 's/init_lr: 0.005$/init_lr: 0.01/' seqlora_inr_sgd_t20c10_r16.yaml
sed -i 's/epochs: 10$/epochs: 20/' seqlora_inr_sgd_t20c10_r16.yaml
sed -i 's/lrate: 0.005$/lrate: 0.01/' seqlora_inr_sgd_t20c10_r16.yaml

# 移除 prefix 中的 ep10_lr005 标记
sed -i 's/_ep10_lr005//g' seqlora_inr_sgd_t20c10_r16.yaml
```

**预期修复后的配置**:
```yaml
prefix: seqlora_inr_sgd_pets_t20_rank16_eval2

init_epoch: 20
init_lr: 0.01
epochs: 20
lrate: 0.01
momentum: 0
weight_decay: 0
```

---

## 📊 修复后预期结果

| 数据集 | 修复前状态 | 修复后状态 | 改进 |
|--------|-----------|-----------|------|
| CUB200 | ✅ 100% | ✅ 100% | - |
| Aircraft | ✅ 100% | ✅ 100% | - |
| Cars196 | ✅ 100% | ✅ 100% | - |
| Flowers | ✅ 100% | ✅ 100% | - |
| Oxford Pet | ⚠️ 96% | ✅ 100% | +4% |

**总体通过率**: 80% → **100%** ✅

---

## ✅ 验证脚本

修复后运行以下命令验证：

```bash
cd /data/140-0/users/liying/Flatness_CV
python check_intra_dataset_consistency.py
```

**预期输出**:
```
✅ CUB200 数据集内部所有配置完全一致!
✅ Aircraft 数据集内部所有配置完全一致!
✅ Cars196 数据集内部所有配置完全一致!
✅ Flower 数据集内部所有配置完全一致!
✅ Oxfordpet 数据集内部所有配置完全一致!
```

---

## 📝 总结

### 当前状态

- ✅ **4/5 数据集** (80%) 内部配置完全统一
- ⚠️ **Oxford Pet** 的 SeqLoRA 方法存在 1 个配置不一致
- ✅ **其他所有数据集和方法**都符合"数据集内部统一"的要求

### 需要修复

- 🔴 **仅 1 个文件**需要修复：[seqlora_inr_sgd_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/seqlora_inr_sgd_t20c10_r16.yaml) (Oxford Pet)
- 🔴 **工作量**: 约 5 分钟即可完成修复和验证

### 符合规范

✅ 符合用户记忆中的"**全量同步原则**":
> 当调整关键超参数时，必须遍历该方法下所有优化器的配置文件，确保全部同步更新

✅ 符合"**实验配置完整性、命名与同步校验规范**":
> 防止因配置不一致导致对比实验失效

---

**检查日期**: 2026-03-25  
**检查工具**: `check_intra_dataset_consistency.py`  
**整体评分**: 4.8/5.0 (96% 通过)
