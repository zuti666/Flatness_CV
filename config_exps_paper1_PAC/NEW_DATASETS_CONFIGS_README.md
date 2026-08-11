# 新数据集配置文件说明

## 概述
已成功创建三个新数据集的完整配置文件集合，用于支持 FGVC-Aircraft、Flowers-102 和 Oxford-IIIT-Pet 数据集的实验。

## 创建的文件夹

### 1. `exp1_rebuttel_aircraft/` - FGVC-Aircraft 数据集
- **类别数**: 100 类
- **任务划分**: 10 个任务 (init_cls: 10, increment: 10)
- **数据集名称**: `aircraft`
- **配置文件数量**: 29 个

### 2. `exp1_rebuttel_flower/` - Flowers-102 数据集
- **类别数**: 102 类
- **任务划分**: 10 个任务 (init_cls: 10, increment: 10)
- **数据集名称**: `flowers`
- **配置文件数量**: 29 个

### 3. `exp1_rebuttel_oxfordPet/` - Oxford-IIIT-Pet 数据集
- **类别数**: 37 类
- **任务划分**: 10 个任务 (init_cls: 4, increment: 3)
- **数据集名称**: `pets`
- **配置文件数量**: 29 个

## 配置文件类型

每个文件夹包含以下类型的配置文件：

### Fine-tuning 方法 (FT_*.yaml)
- `FT_gam_<dataset>_t20.yaml` - GAM 优化器
- `FT_sgd_<dataset>_t20.yaml` - 标准 SGD
- `FT_rwp_<dataset>_t20.yaml` - RWP 优化器
- `FT_cflat_<dataset>_t20.yaml` - C-Flat 优化器
- `LP_*` - Linear Probe 变体

### LoRA 方法 (inclora_inr_*.yaml, inflora_inr_*.yaml, olora_inr_*.yaml, seqlora_inr_*.yaml)
- `inclora_inr_gam_<dataset>_t20c10_r16.yaml` - iLoRA with GAM
- `inclora_inr_sgd_<dataset>_t20c10_r16.yaml` - iLoRA with SGD
- `inclora_inr_rwp_<dataset>_t20c10_r16.yaml` - iLoRA with RWP
- `inclora_inr_sam_<dataset>_t20c10_r16.yaml` - iLoRA with SAM
- `inclora_inr_cflat_<dataset>_t20c10_r16.yaml` - iLoRA with C-Flat
- 以及 inflora, olora, seqlora 的相应变体

## 关键修改点

### 1. 数据集设置
```yaml
# Aircraft
dataset: "aircraft"
init_cls: 10
increment: 10

# Flowers
dataset: "flowers"
init_cls: 10
increment: 10

# Oxford Pet
dataset: "pets"
init_cls: 4
increment: 3
```

### 2. 保存路径前缀
配置文件的 `prefix` 字段已更新为对应的数据集名称：
- `PerTaskFT_gam_aircraft_t20_eval_seed1993`
- `PerTaskFT_gam_flowers_t20_eval_seed1993`
- `PerTaskFT_gam_pets_t20_eval_seed1993`

## 使用示例

### 运行 Aircraft 实验
```bash
# Fine-tuning with GAM
python src/main.py --config config_exps_paper1_PAC/exp1_rebuttel_aircraft/FT_gam_aircraft_t20.yaml

# iLoRA with GAM
python src/main.py --config config_exps_paper1_PAC/exp1_rebuttel_aircraft/inclora_inr_gam_aircraft_t20c10_r16.yaml
```

### 运行 Flowers 实验
```bash
# Fine-tuning with GAM
python src/main.py --config config_exps_paper1_PAC/exp1_rebuttel_flower/FT_gam_flower_t20.yaml

# iLoRA with GAM
python src/main.py --config config_exps_paper1_PAC/exp1_rebuttel_flower/inclora_inr_gam_flower_t20c10_r16.yaml
```

### 运行 Oxford Pet 实验
```bash
# Fine-tuning with GAM
python src/main.py --config config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/FT_gam_oxfordPet_t20.yaml

# iLoRA with GAM
python src/main.py --config config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/inclora_inr_gam_oxfordPet_t20c10_r16.yaml
```

## 数据路径要求

确保数据集已下载到以下路径之一：
- `/data/140-0/datasets/fgvc-aircraft-2013b/`
- `/data/140-0/datasets/flowers-102/`
- `/data/140-0/datasets/oxford-iiit-pet/`

或通过以下方式指定数据根目录：
```bash
export DATA_ROOT=/your/custom/data/path
```

## 注意事项

1. **标签映射已修复**: Aircraft 数据集的标签映射已在 `utils/data.py` 中修复，确保 train/val/test 共享全局统一的 label_map。

2. **Scipy 依赖**: Flowers-102 需要 scipy 库来读取 .mat 文件：
   ```bash
   pip install scipy
   ```

3. **任务划分**:
   - Aircraft: 10 + 10×9 = 100 类 ✓
   - Flowers: 10 + 10×9 = 100 类 (实际 102 类，最后剩余 2 类)
   - Oxford Pet: 4 + 3×9 = 31 类 (实际 37 类，最后剩余 6 类)

4. **评估设置**: 所有配置默认启用 flatness evaluation、feature flatness、CKA 等评估指标。

## 相关文件修改

- ✅ `utils/data.py` - 修复了 iAircraft 的标签映射逻辑
- ✅ `utils/data_manager.py` - 已包含三个新数据集的映射
- ✅ `config_exps_paper1_PAC/exp1_rebuttel_aircraft/` - 新建 29 个配置
- ✅ `config_exps_paper1_PAC/exp1_rebuttel_flower/` - 新建 29 个配置
- ✅ `config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/` - 新建 29 个配置

## 总计
- **新增配置文件**: 87 个
- **支持数据集**: 3 个
- **支持方法**: Fine-tuning, iLoRA, IncLoRA, OLoRA, SeqLoRA 等
- **支持优化器**: SGD, GAM, RWP, SAM, C-Flat
