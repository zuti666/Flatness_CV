# 📊 Exp1 Rebuttal - ViT Backbone 五数据集训练超参数配置分析报告

## 🎯 分析范围

**五个细粒度数据集 (ViT Backbone)**:
1. ✅ **CUB200** - 200 类，任务划分：20 + 20×9 = 200 类
2. ✅ **Aircraft** - 100 类，任务划分：10 + 10×9 = 100 类  
3. ✅ **Cars196** - 196 类，任务划分：20 + 19×9 = 191 类 (+5 余数)
4. ⚠️  **Flowers** - 102 类，任务划分：10 + 10×9 = 100 类 (+2 余数) **[已修正]**
5. ✅ **Oxford Pet** - 37 类，任务划分：4 + 4×9 = 36 类 (+1 余数)

---

## 📈 当前配置状态总结

### 方法覆盖情况

| 数据集 | SeqLoRA | IncLoRA | InfLoRA | OLoRA | Fine-tuning | Linear Probe | 总配置数 |
|--------|---------|---------|---------|-------|-------------|--------------|----------|
| **CUB200** | 5 | 5 | 5 | 5 | 6 | 5 | 31 |
| **Aircraft** | 5 | 6 | 5 | 5 | 5 | 3 | 29 |
| **Cars196** | 5 | 6 | 5 | 5 | 5 | 3 | 29 |
| **Flowers** | 5 | 6 | 4 | 5 | 5 | 4 | 29 |
| **Oxford Pet** | 5 | 5 | 4 | 5 | 5 | 4 | 28 |

---

## 🔍 训练超参数对比（按方法）

### 1️⃣ LoRA 方法 (SeqLoRA/IncLoRA/InfLoRA/OLoRA)

#### 当前配置分组

| 数据集组 | init_epoch | init_lr | epochs | lrate | momentum | weight_decay | 配置文件数 |
|----------|-----------|---------|--------|-------|----------|--------------|------------|
| **CUB200** | 20 | 0.01 | 20 | 0.01 | 0 | 0 | 20 |
| **Aircraft** | 20 | 0.01 | 20 | 0.01 | 0 | 0 | 21 |
| **Cars196** | 20 | 0.01 | 20 | 0.01 | 0 | 0 | 21 |
| **Flowers** ✨ | **10** | **0.005** | **10** | **0.005** | 0 | 0 | 20 |
| **Oxford Pet** | 20 | 0.01 | 20 | 0.01 | 0 | 0 | 19 |

**一致性分析**:
- ❌ **Flowers 数据集与其他 4 个数据集不一致**
- Flowers 使用 `ep10_lr005` 配置（低学习率、少轮次）
- 其他 4 个数据集使用标准配置（epochs=20, lr=0.01）

#### 共同特征
✅ **Batch Size**: 所有数据集统一为 128  
✅ **Scheduler**: 所有数据集统一为 cosine  
✅ **Momentum**: 所有 LoRA 方法统一为 0  
✅ **Weight Decay**: 所有 LoRA 方法统一为 0  

---

### 2️⃣ Fine-tuning 方法

#### 当前配置分组

| 数据集组 | init_epoch | init_lr | epochs | lrate | momentum | weight_decay | scheduler |
|----------|-----------|---------|--------|-------|----------|--------------|-----------|
| **CUB200** | 20 | 1e-3 | 20 | 1e-3 | 0.9 | 2e-4 | cosine |
| **Aircraft** | 20 | 1e-3 | 20 | 1e-3 | 0.9 | 2e-4 | cosine |
| **Cars196** | 20 | 1e-3 | 20 | 1e-3 | 0.9 | 2e-4 | cosine |
| **Flowers** ✨ | **10** | **0.005** | **10** | **0.005** | 0 | 0 | cosine |
| **Oxford Pet** | 20 | 1e-3 | 20 | 1e-3 | 0.9 | 2e-4 | cosine |

**一致性分析**:
- ❌ **Flowers 数据集与其他 4 个数据集严重不一致**
- Flowers 的 Fine-tuning 配置也使用了 `ep10_lr005`
- 其他 4 个数据集使用标准 FT 配置（lr=1e-3, momentum=0.9, wd=2e-4）

---

### 3️⃣ Linear Probe 方法

#### 当前配置分组

| 数据集组 | init_epoch | init_lr | epochs | lrate | momentum | weight_decay |
|----------|-----------|---------|--------|-------|----------|--------------|
| **CUB200** | 20 | 0.01 | 20 | 0.01 | 0 | 0 |
| **Aircraft** | 20 | 0.01 | 20 | 0.01 | 0 | 0 |
| **Cars196** | 20 | 0.01 | 20 | 0.01 | 0 | 0 |
| **Flowers** ✨ | **10** | **0.005** | **10** | **0.005** | 0 | 0 |
| **Oxford Pet** | 20 | 0.01 | 20 | 0.01 | 0 | 0 |

**一致性分析**:
- ❌ **Flowers 数据集与其他 4 个数据集不一致**
- 模式与 LoRA 方法相同

---

## ⚠️ 发现的问题

### 问题 1: Flowers 数据集配置孤立

**现状**:
- Flowers 数据集的所有方法（LoRA/FT/LP）都使用了 `ep10_lr005` 配置
- 其他 4 个数据集（CUB200, Aircraft, Cars196, Oxford Pet）保持一致
- **造成跨数据集对比实验的不公平性**

**影响**:
```
❌ 无法公平对比不同数据集上的性能
❌ 无法确定性能差异来自数据集特性还是超参数设置
❌ 违反"控制变量法"实验原则
```

### 问题 2: Fine-tuning 与 LoRA 方法超参数差异

**现状**:
- **Fine-tuning**: lr=1e-3, momentum=0.9, wd=2e-4（标准 SGD 配置）
- **LoRA 方法**: lr=0.01, momentum=0, wd=0（简化配置）

**合理性分析**:
✅ **可以接受**: Fine-tuning 和 LoRA 由于参数量和学习动态不同，可以采用不同的超参数策略
- Fine-tuning 需要更强的正则化（momentum=0.9, wd=2e-4）
- LoRA 参数量小，学习更简单，可以采用简化的优化策略

---

## 💡 建议方案

### 方案 A: 统一所有数据集为当前标准配置（推荐）

**目标配置**:
```yaml
# LoRA 方法 (SeqLoRA/IncLoRA/InfLoRA/OLoRA)
init_epoch: 20
init_lr: 0.01
epochs: 20
lrate: 0.01
momentum: 0
weight_decay: 0
batch_size: 128
scheduler: cosine

# Fine-tuning
init_epoch: 20
init_lr: 1e-3
epochs: 20
lrate: 1e-3
momentum: 0.9
weight_decay: 2e-4
batch_size: 128
scheduler: cosine

# Linear Probe
init_epoch: 20
init_lr: 0.01
epochs: 20
lrate: 0.01
momentum: 0
weight_decay: 0
batch_size: 128
scheduler: cosine
```

**操作步骤**:
1. 将 Flowers 数据集的配置**回退**到与其他数据集一致
2. 移除 prefix 中的 `ep10_lr005` 标记
3. 确保所有数据集的 Fine-tuning 配置统一

**优点**:
- ✅ 保证跨数据集对比的公平性
- ✅ 符合用户记忆中"全量同步原则"
- ✅ 便于解释实验结果

**缺点**:
- ⚠️ 需要重新运行 Flowers 数据集的所有实验

---

### 方案 B: 将所有数据集调整为 Flowers 配置

**目标配置**:
```yaml
# LoRA 方法
init_epoch: 10
init_lr: 0.005
epochs: 10
lrate: 0.005
momentum: 0
weight_decay: 0

# Fine-tuning
init_epoch: 10
init_lr: 0.005
epochs: 10
lrate: 0.005
momentum: 0
weight_decay: 0
```

**操作步骤**:
1. 修改 CUB200, Aircraft, Cars196, Oxford Pet 的配置
2. 在所有配置的 prefix 中添加 `ep10_lr005` 标记
3. 批量替换训练参数

**优点**:
- ✅ 采用更低的学习率和更少轮次，可能避免过拟合
- ✅ 减少训练时间成本

**缺点**:
- ❌ 需要修改大量配置文件（~100 个）
- ❌ 需要重新运行所有 4 个数据集的实验
- ⚠️ 学习率过低可能导致收敛困难（参考用户记忆）

---

### 方案 C: 保持现状，分组报告（不推荐）

**策略**:
- 将 Flowers 单独作为一组（ep10_lr005 配置）
- 其他 4 个数据集作为另一组（标准配置）
- 在论文中明确说明配置差异

**缺点**:
- ❌ 违反实验设计的"控制变量"原则
- ❌ 难以解释跨数据集的性能差异
- ❌ 可能被审稿人质疑实验严谨性

**唯一适用场景**:
- 如果 Flowers 数据集在标准配置下出现严重过拟合或训练不稳定

---

## 📋 决策建议矩阵

| 评估维度 | 方案 A | 方案 B | 方案 C |
|----------|--------|--------|--------|
| **实验公平性** | ✅ 优秀 | ✅ 优秀 | ❌ 差 |
| **工作量** | ✅ 小（仅修改 Flowers） | ❌ 大（修改 4 个数据集） | ✅ 无修改 |
| **时间成本** | ✅ 低（重跑 Flowers） | ❌ 高（重跑 4 个数据集） | ✅ 无额外成本 |
| **论文可解释性** | ✅ 清晰 | ✅ 清晰 | ❌ 复杂 |
| **符合规范程度** | ✅ 完全符合 | ✅ 符合 | ❌ 不符合 |

**🎯 推荐选择**: **方案 A** - 统一所有数据集为当前标准配置

---

## 🔧 执行脚本（方案 A）

如果需要将 Flowers 回退到标准配置，可以使用以下脚本：

```bash
cd /data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower

# 回退训练参数
sed -i 's/init_epoch: 10/init_epoch: 20/g' *.yaml
sed -i 's/init_lr: 0.005/init_lr: 0.01/g' *.yaml
sed -i 's/epochs: 10/epochs: 20/g' *.yaml
sed -i 's/lrate: 0.005/lrate: 0.01/g' *.yaml

# 移除 prefix 中的 ep10_lr005 标记
sed -i 's/_ep10_lr005//g' *.yaml
```

---

## 📊 最终配置对比表（方案 A 执行后）

### LoRA 方法统一配置

| 数据集 | init_epoch | init_lr | epochs | lrate | momentum | wd | batch_size | scheduler |
|--------|-----------|---------|--------|-------|----------|----|------------|-----------|
| CUB200 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | 128 | cosine |
| Aircraft | 20 | 0.01 | 20 | 0.01 | 0 | 0 | 128 | cosine |
| Cars196 | 20 | 0.01 | 20 | 0.01 | 0 | 0 | 128 | cosine |
| **Flowers (调整后)** | **20** | **0.01** | **20** | **0.01** | 0 | 0 | 128 | cosine |
| Oxford Pet | 20 | 0.01 | 20 | 0.01 | 0 | 0 | 128 | cosine |
| **一致性** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Fine-tuning 统一配置

| 数据集 | init_epoch | init_lr | epochs | lrate | momentum | wd | batch_size | scheduler |
|--------|-----------|---------|--------|-------|----------|----|------------|-----------|
| CUB200 | 20 | 1e-3 | 20 | 1e-3 | 0.9 | 2e-4 | 128 | cosine |
| Aircraft | 20 | 1e-3 | 20 | 1e-3 | 0.9 | 2e-4 | 128 | cosine |
| Cars196 | 20 | 1e-3 | 20 | 1e-3 | 0.9 | 2e-4 | 128 | cosine |
| **Flowers (调整后)** | **20** | **1e-3** | **20** | **1e-3** | **0.9** | **2e-4** | **128** | **cosine** |
| Oxford Pet | 20 | 1e-3 | 20 | 1e-3 | 0.9 | 2e-4 | 128 | cosine |
| **一致性** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## ✅ 验证清单

执行方案 A 后，请验证以下内容：

- [ ] Flowers 数据集所有 LoRA 方法的 `init_epoch=20, init_lr=0.01`
- [ ] Flowers 数据集所有 LoRA 方法的 `epochs=20, lrate=0.01`
- [ ] Flowers 数据集所有 Fine-tuning 的 `momentum=0.9, weight_decay=2e-4`
- [ ] Flowers 数据集所有配置的 prefix 中不含 `ep10_lr005` 标记
- [ ] 运行验证脚本确认所有参数一致性达到 100%

---

## 📝 结论

**当前状态**: Flowers 数据集的训练配置与其他 4 个数据集**不一致**

**建议行动**: 
1. 采用**方案 A**，将 Flowers 回退到标准配置
2. 确保所有数据集的超参数完全一致
3. 重新运行 Flowers 数据集的实验
4. 在论文中明确说明所有数据集使用统一的训练配置

**理论依据**: 
- 符合用户记忆中的"全量同步原则"和"实验配置完整性、命名与同步校验规范"
- 遵循科学实验的"控制变量法"原则
- 提高论文的可信度和可复现性

---

**分析日期**: 2026-03-25  
**分析师**: AI Assistant  
**建议优先级**: 高（影响实验结论的可信度）
