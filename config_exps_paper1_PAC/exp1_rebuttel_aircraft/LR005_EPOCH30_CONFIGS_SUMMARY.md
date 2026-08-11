# Aircraft LoRA 配置文件 lr005_epoch30 更新总结

## 📋 更新概述

为所有使用 `init_lr: 0.05` 和 `lrate: 0.05` 的 LoRA 方法配置文件在 prefix 中添加了 **`lr005_epoch30`** 标识，并将训练轮次从 20 提升到 **30 epochs**。

---

## ✅ 已更新的配置文件（共 16 个）

### IncLoRA 系列（5 个中的 4 个）

| 配置文件 | 原 prefix | 新 prefix | LR | Epochs |
|---------|----------|----------|-----|--------|
| [inclora_inr_gam_aircraft_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inclora_inr_gam_aircraft_t20c10_r16.yaml) | `inclora_inr_gam_aircraft_t20_rank16_eval2` | ✅ `inclora_inr_gam_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [inclora_inr_gam_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inclora_inr_gam_t20c10_r16.yaml) | `inclora_inr_gam_aircraft_t20_rank16_eval2` | ✅ `inclora_inr_gam_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [inclora_inr_rwp_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inclora_inr_rwp_t20c10_r16.yaml) | `inclora_inr_rwp_aircraft_t20_rank16_1993_SH` | ✅ `inclora_inr_rwp_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [inclora_inr_sam_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inclora_inr_sam_t20c10_r16.yaml) | `inclora_inr_sam_aircraft_t20_rank16_eval2` | ✅ `inclora_inr_sam_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [inclora_inr_sgd_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inclora_inr_sgd_t20c10_r16.yaml) | `inclora_inr_sgd_aircraft_t20_rank16_eval2` | ✅ `inclora_inr_sgd_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [inclora_inr_cflat_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inclora_inr_cflat_t20c10_r16.yaml) | `inclora_inr_cflat_aircraft_t20_rank16` | ⚠️ 未更新（lr=0.01） | 0.01 | 20 |

### InfLoRA 系列（4 个）

| 配置文件 | 原 prefix | 新 prefix | LR | Epochs |
|---------|----------|----------|-----|--------|
| [inflora_inr_gam_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inflora_inr_gam_t20c10_r16.yaml) | `inflora_inr_gam_aircraft_t20_rank16` | ✅ `inflora_inr_gam_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [inflora_inr_rwp_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inflora_inr_rwp_t20c10_r16.yaml) | `inflora_inr_rwp_aircraft_t20_rank16` | ✅ `inflora_inr_rwp_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [inflora_inr_sgd_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inflora_inr_sgd_t20c10_r16.yaml) | `inflora_inr_sgd_aircraft_t20_rank16` | ✅ `inflora_inr_sgd_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [inflora_inr_cflat_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inflora_inr_cflat_t20c10_r16.yaml) | `inflora_inr_cflat_aircraft_t20_rank16` | ⚠️ 未更新（lr=0.01） | 0.01 | 20 |

### OLoRA 系列（4 个）

| 配置文件 | 原 prefix | 新 prefix | LR | Epochs |
|---------|----------|----------|-----|--------|
| [olora_inr_gam_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/olora_inr_gam_t20c10_r16.yaml) | `olora_inr_gam_aircraft_t20_rank16_eval2` | ✅ `olora_inr_gam_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [olora_inr_rwp_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/olora_inr_rwp_t20c10_r16.yaml) | `olora_inr_rwp_aircraft_t20_rank16_1993_SH` | ✅ `olora_inr_rwp_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [olora_inr_sam_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/olora_inr_sam_t20c10_r16.yaml) | `olora_inr_sam_aircraft_t20_rank16_eval2` | ✅ `olora_inr_sam_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [olora_inr_sgd_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/olora_inr_sgd_t20c10_r16.yaml) | `olora_inr_sgd_aircraft_t20_rank16_eval2` | ✅ `olora_inr_sgd_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [olora_inr_cflat_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/olora_inr_cflat_t20c10_r16.yaml) | `olora_inr_cflat_aircraft_t20_rank16` | ⚠️ 未更新（lr=0.01） | 0.01 | 20 |

### SeqLoRA 系列（4 个）

| 配置文件 | 原 prefix | 新 prefix | LR | Epochs |
|---------|----------|----------|-----|--------|
| [seqlora_inr_gam_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/seqlora_inr_gam_t20c10_r16.yaml) | `seqlora_inr_gam_aircraft_t20_rank16_eval2` | ✅ `seqlora_inr_gam_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [seqlora_inr_rwp_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/seqlora_inr_rwp_t20c10_r16.yaml) | `seqlora_inr_rwp_aircraft_t20_rank16_1993_SH_eval2` | ✅ `seqlora_inr_rwp_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [seqlora_inr_sam_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/seqlora_inr_sam_t20c10_r16.yaml) | `seqlora_inr_sam_aircraft_t20_rank16_eval2` | ✅ `seqlora_inr_sam_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [seqlora_inr_sgd_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/seqlora_inr_sgd_t20c10_r16.yaml) | `seqlora_inr_sgd_aircraft_t20_rank16_test` | ✅ `seqlora_inr_sgd_aircraft_t20_rank16_lr005_epoch30` | 0.05 | 30 |
| [seqlora_inr_cflat_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/seqlora_inr_cflat_t20c10_r16.yaml) | `seqlora_inr_cflat_aircraft_t20_rank16` | ⚠️ 未更新（lr=0.01） | 0.01 | 20 |

---

## ⚠️ 未更新的配置文件（lr=0.01，共 4 个）

以下配置文件使用 `init_lr: 0.01` 和 `lrate: 0.01`，保持原样，**未添加 lr005_epoch30 标记**：

| 方法 | 配置文件 | 优化器 | LR | Epochs |
|-----|---------|-------|-----|--------|
| **IncLoRA** | [inclora_inr_cflat_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inclora_inr_cflat_t20c10_r16.yaml) | C-Flat | 0.01 | 20 |
| **InfLoRA** | [inflora_inr_cflat_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inflora_inr_cflat_t20c10_r16.yaml) | C-Flat | 0.01 | 20 |
| **OLoRA** | [olora_inr_cflat_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/olora_inr_cflat_t20c10_r16.yaml) | C-Flat | 0.01 | 20 |
| **SeqLoRA** | [seqlora_inr_cflat_t20c10_r16.yaml](file:///data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/seqlora_inr_cflat_t20c10_r16.yaml) | C-Flat | 0.01 | 20 |

**注意**: 这些配置的学习率为 0.01，如需要也应添加相应的 `lr001` 标记。

---

## 📊 学习率与 Epoch 统计

### lr=0.05, epoch=30（已更新，共 16 个）⭐

- ✅ **IncLoRA**: 4 个（SGD, GAM, SAM, RWP）
- ✅ **InfLoRA**: 3 个（SGD, GAM, RWP）
- ✅ **OLoRA**: 4 个（SGD, GAM, SAM, RWP）
- ✅ **SeqLoRA**: 4 个（SGD, GAM, SAM, RWP）

### lr=0.01, epoch=20（未更新，共 4 个）

- ⚠️ **C-Flat 系列**: 4 个（所有方法的 C-Flat 变体）

---

## 🔧 修改详情

### 超参数修改

```yaml
# 修改前
init_epoch: 20
init_lr: 0.01
epochs: 20
lrate: 0.01

# 修改后
init_epoch: 30    # 增加 50% 训练轮次
init_lr: 0.05     # 提升 5 倍学习率
epochs: 30        # 增加 50% 训练轮次
lrate: 0.05       # 提升 5 倍学习率
```

### Prefix 命名规范

**格式**: `{method}_{variant}_{optimizer}_{dataset}_t{tasks}_rank{r}_lr005_epoch30`

**示例**:
```
seqlora_inr_gam_aircraft_t20_rank16_lr005_epoch30
```

---

## 🎯 更新原因

根据之前的分析（参见 [AIRCRAFT_LOSS_ANALYSIS_AND_FIXES.md](file:///data/140-0/users/liying/Flatness_CV/logs_exp1_rebuttal/AIRCRAFT_LOSS_ANALYSIS_AND_FIXES.md)）：

### 问题诊断
1. **学习率过低 (0.01)** → 参数更新缓慢，无法适应新任务
2. **训练轮次不足 (20 epochs)** → Aircraft 极端细粒度分类需要更多训练
3. **梯度信号弱** → 每任务仅 10 类，需要更高学习率补偿

### 预期改进效果

| 指标 | 修改前 | 修改后（预期） | 改善幅度 |
|-----|-------|--------------|---------|
| **最终 Loss** | ~2.0 | ~1.0-1.2 | ↓ 40-50% |
| **Train Acc** | ~28% | ~70-80% | ↑ 2.5-3 倍 |
| **Test Acc** | ~25% | ~65-75% | ↑ 2.5-3 倍 |
| **收敛状态** | 未收敛 | 充分收敛 | ✅ |

---

## ✅ 验证清单

- [x] ✅ 所有 lr=0.05 的 LoRA 配置已添加 lr005_epoch30 标记（16 个）
- [x] ✅ lr=0.01 的配置保持原样（4 个 C-Flat）
- [x] ✅ init_lr 和 lrate 同时设置为 0.05
- [x] ✅ init_epoch 和 epochs 同时设置为 30
- [x] ✅ 所有修改的配置文件语法检查通过

---

## 📝 后续建议

1. **考虑更新 lr=0.01 的配置**: 为这 4 个 C-Flat 配置添加 `lr001` 标记以保持一致性
2. **统一命名规范**: 未来创建新配置时直接使用包含 lr 和 epoch 标记的 prefix
3. **监控训练效果**: 重新运行实验，观察 Loss 下降曲线是否符合预期

---

**更新时间**: 2026-03-25  
**更新者**: AI Assistant  
**状态**: ✅ 已完成 16 个配置文件的 lr005_epoch30 标记更新
