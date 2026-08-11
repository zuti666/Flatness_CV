# 配置文件命名规范说明

## ⚠️ 重要提示

**当前问题**: 配置文件中存在命名不一致的情况，导致脚本引用错误。

---

## 📋 命名规则

### ✅ 标准命名格式（推荐）

```
{method}_{variant}_{optimizer}_{task_config}.yaml
```

**示例**:
- `seqlora_inr_rwp_t20c10_r16.yaml`
- `inclora_inr_gam_t20c10_r16.yaml`
- `olora_inr_sgd_t20c10_r16.yaml`
- `inflora_inr_rwp_t20c10_r16.yaml`

**特点**:
- ❌ **不包含** 数据集名称（如 `aircraft`, `flowers`）
- ✅ **通用性强**，可在不同数据集间复用
- ✅ **简洁清晰**，文件名长度适中

---

### ❌ 不推荐的命名格式

```
{method}_{variant}_{optimizer}_{dataset}_{task_config}.yaml
```

**示例** (应避免):
- `seqlora_inr_rwp_aircraft_t20c10_r16.yaml` ← **不存在，会导致错误**
- `inclora_inr_gam_aircraft_t20c10_r16.yaml` ← **例外情况，仅此一例**

**问题**:
- 文件名过长
- 无法在不同数据集间复用
- 容易导致脚本引用错误

---

## 🔍 当前状态统计

### exp1_rebuttel_aircraft 目录

| 命名模式 | 文件数量 | 示例 |
|---------|---------|------|
| **标准命名（无数据集名）** | 28 个 | `seqlora_inr_rwp_t20c10_r16.yaml` |
| **非标准命名（含数据集名）** | 1 个 | `inclora_inr_gam_aircraft_t20c10_r16.yaml` |

**注意**: 唯一的例外是 `inclora_inr_gam_aircraft_t20c10_r16.yaml`，这是历史遗留问题。

---

## 🛠️ 修复方案

### 已修复的脚本

**文件**: `scripts/exp1_rebuttel_aircraft/run_sgd_1gpu_test.sh`

**修改前**:
```bash
VIT_CONFIGS=(
  seqlora_inr_rwp_aircraft_t20c10_r16.yaml  # ❌ 文件不存在
)
```

**修改后**:
```bash
VIT_CONFIGS=(
  seqlora_inr_rwp_t20c10_r16.yaml  # ✅ 正确的文件名
  inclora_inr_rwp_t20c10_r16.yaml
  olora_inr_rwp_t20c10_r16.yaml
  inflora_inr_rwp_t20c10_r16.yaml
)
```

---

## 📝 脚本编写规范

### ✅ 正确的引用方式

在所有 bash 脚本中，应使用**标准命名**的配置文件：

```bash
# Flowers 数据集脚本
bash scripts/exp1_rebuttel_flowers/run_gam_3gpu.sh
# 引用：seqlora_inr_gam_flower_t20c10_r16.yaml (如果存在)
# 或：seqlora_inr_gam_t20c10_r16.yaml (推荐)

# Stanford Pet 数据集脚本
bash scripts/exp1_rebuttel_stanfordpet/run_gam_3gpu.sh
# 引用：seqlora_inr_gam_oxfordPet_t20c10_r16.yaml (如果存在)
# 或：seqlora_inr_gam_t20c10_r16.yaml (推荐)

# Aircraft 数据集脚本
bash scripts/exp1_rebuttel_aircraft/run_gam_3gpu.sh
# 引用：seqlora_inr_gam_t20c10_r16.yaml ✅
```

---

## 🎯 统一命名建议

### 方案 A: 保持现状（推荐）

**原则**: 大多数文件已经是标准命名，保持即可。

**行动**:
1. ✅ 所有脚本使用标准命名（无数据集名）
2. ✅ 新创建的配置文件遵循标准命名
3. ⚠️ 唯一的例外文件 `inclora_inr_gam_aircraft_t20c10_r16.yaml` 保持不变

### 方案 B: 完全统一（可选）

如果需要完全一致，可以重命名那个例外文件：

```bash
# 备份后重命名
cd config_exps_paper1_PAC/exp1_rebuttel_aircraft
mv inclora_inr_gam_aircraft_t20c10_r16.yaml inclora_inr_gam_t20c10_r16.yaml
```

**注意**: 需要同时更新所有引用该文件的脚本。

---

## ✅ 检查清单

在创建新的配置文件或脚本时，请检查：

- [ ] 配置文件名**不包含**数据集名称
- [ ] 脚本中引用的文件名与实际存在的文件**完全匹配**
- [ ] 使用 tab 补全验证文件名是否正确
- [ ] 在多个数据集目录间保持一致的命名

**验证命令**:
```bash
# 检查配置文件是否存在
ls config_exps_paper1_PAC/exp1_rebuttel_aircraft/seqlora_inr_rwp_t20c10_r16.yaml

# 列出所有配置文件（不含数据集名）
ls config_exps_paper1_PAC/exp1_rebuttel_aircraft/*.yaml | grep -v "_aircraft_" | grep -v "_flower_" | grep -v "_oxfordPet_"
```

---

## 📊 最佳实践

### 配置文件组织

```
config_exps_paper1_PAC/
├── exp1_rebuttel_aircraft/
│   ├── seqlora_inr_gam_t20c10_r16.yaml      # ✅ 标准命名
│   ├── inclora_inr_gam_t20c10_r16.yaml      # ✅ 标准命名
│   └── FT_gam_aircraft_t20.yaml             # ⚠️ Fine-tuning 可包含数据集名
│
├── exp1_rebuttel_flower/
│   ├── seqlora_inr_gam_t20c10_r16.yaml      # ✅ 与 aircraft 共用相同命名
│   └── FT_gam_flower_t20.yaml               # ⚠️ Fine-tuning 可包含数据集名
│
└── exp1_rebuttel_oxfordPet/
    ├── seqlora_inr_gam_t20c10_r16.yaml      # ✅ 统一命名
    └── FT_gam_oxfordPet_t20.yaml            # ⚠️ Fine-tuning 可包含数据集名
```

### 脚本中的配置列表

```bash
# ✅ 推荐：清晰的注释和分组
VIT_CONFIGS=(
  # LoRA 方法（RWP 优化器）
  seqlora_inr_rwp_t20c10_r16.yaml
  inclora_inr_rwp_t20c10_r16.yaml
  olora_inr_rwp_t20c10_r16.yaml
  
  # InfLoRA 方法（RWP 优化器）
  inflora_inr_rwp_t20c10_r16.yaml
)

# ❌ 避免：包含数据集名
VIT_CONFIGS=(
  seqlora_inr_rwp_aircraft_t20c10_r16.yaml  # 错误！
)
```

---

## 🔗 相关文档

1. **NEW_DATASETS_CONFIGS_README.md** - 新数据集配置总览
2. **EXP1_REBUTTEL_NEW_DATASETS_SCRIPTS.md** - 运行脚本使用说明
3. **LORA_METHODS_ANALYSIS_REPORT.md** - LoRA 方法对比分析

---

**文档创建时间**: 2026-03-25  
**维护者**: Flatness_CV Team  
**目的**: 避免因配置文件命名不一致导致的脚本错误
