# LoRA 增量学习方法 - 快速使用指南

## 🚀 一句话总结

**无脑使用 InfLoRA + GAM**，配置已优化好，直接运行即可！

---

## ⚡ 5 分钟快速开始

### Step 1: 选择数据集配置文件

```bash
# Aircraft (100 类)
cd config_exps_paper1_PAC/exp1_rebuttel_aircraft

# Flowers (102 类)  
cd config_exps_paper1_PAC/exp1_rebuttel_flower

# Oxford Pet (37 类)
cd config_exps_paper1_PAC/exp1_rebuttel_oxfordPet
```

### Step 2: 运行 InfLoRA + GAM（推荐）

```bash
# 单卡训练
python src/main.py --config config_exps_paper1_PAC/exp1_rebuttel_aircraft/inflora_inr_gam_aircraft_t20c10_r16.yaml

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 python src/main.py --config config_exps_paper1_PAC/exp1_rebuttel_aircraft/inflora_inr_gam_aircraft_t20c10_r16.yaml
```

### Step 3: 查看结果

```bash
# 日志位置
tail -f logs_exp1_rebuttal/PerTaskInFLlora_GAM_aircraft_*.log

# 结果位置
ls summaries/PerTaskInFLlora_GAM_aircraft_*/
```

---

## 📋 完整配置列表

### Aircraft 数据集 (exp1_rebuttel_aircraft)

#### InfLoRA 系列（推荐）
```bash
# InfLoRA + GAM (首选)
inflora_inr_gam_aircraft_t20c10_r16.yaml

# InfLoRA + SGD (基线)
inflora_inr_sgd_aircraft_t20c10_r16.yaml

# InfLoRA + RWP (随机权重扰动)
inflora_inr_rwp_aircraft_t20c10_r16.yaml

# InfLoRA + SAM (Sharpness-Aware Minimization)
inflora_inr_sam_aircraft_t20c10_r16.yaml

# InfLoRA + C-Flat
inflora_inr_cflat_aircraft_t20c10_r16.yaml
```

#### 其他 LoRA 方法（备选）
```bash
# SeqLoRA (Sequential LoRA)
seqlora_inr_gam_aircraft_t20c10_r16.yaml
seqlora_inr_sgd_aircraft_t20c10_r16.yaml

# OLoRA (Orthogonal LoRA)
olora_inr_gam_aircraft_t20c10_r16.yaml
olora_inr_sgd_aircraft_t20c10_r16.yaml

# IncLoRA (Incremental LoRA)
inclora_inr_gam_aircraft_t20c10_r16.yaml
inclora_inr_sgd_aircraft_t20c10_r16.yaml
```

#### Fine-tuning 基线
```bash
# Fine-tuning + GAM
FT_gam_aircraft_t20.yaml

# Fine-tuning + SGD
FT_sgd_aircraft_t20.yaml

# Linear Probe + GAM
LP_gam_aircraft_t20.yaml
```

### Flowers 数据集 (exp1_rebuttel_flower)

配置命名规则相同，替换 `aircraft` → `flower`：
```bash
inflora_inr_gam_flower_t20c10_r16.yaml
FT_gam_flower_t20.yaml
...
```

### Oxford Pet 数据集 (exp1_rebuttel_oxfordPet)

配置命名规则相同，替换 `aircraft` → `oxfordPet`：
```bash
inflora_inr_gam_oxfordPet_t20c10_r16.yaml
FT_gam_oxfordPet_t20.yaml
...
```

---

## 🔧 关键参数说明

### InfLoRA 特定参数

```yaml
# inflora_inr_gam_aircraft_t20c10_r16.yaml

model_name: "inflora"     # 使用 InfLoRA 方法
lora_rank: 16            # LoRA 秩（建议 16）

# DualGPM 子空间投影参数
lamb: 0.5                # 初始阈值 (λ)
lame: 0.9                # 最大阈值 (λ_max)

# 任务设置
total_sessions: 10       # 总任务数
init_cls: 10             # 初始类别数
increment: 10            # 每个任务新增类别数

# 优化器设置
optimizer_type: "gam"    # 优化器类型
batch_size: 128
epochs: 20
lrate: 0.01
weight_decay: 0          # InfLoRA 通常不用权重衰减

# 评估设置
flat_eval: true          # 启用 flatness 评估
feature_flat_eval: true  # 启用 feature flatness 评估
```

### 优化器选择指南

| 优化器 | 适用场景 | 收敛速度 | 最终性能 | 稳定性 |
|-------|---------|---------|---------|-------|
| **GAM** | 默认首选 | 快 | 优 | 优 |
| **SGD** | 基线对比 | 中 | 良 | 优 |
| **SAM** | 追求平坦极小值 | 慢 | 良+ | 中 |
| **RWP** | 需要正则化 | 中 | 良 | 中 |
| **C-Flat** | 研究 flatness | 慢 | 待验证 | 待观察 |

---

## 🎯 实验设计建议

### 基础实验（必须做）

1. **InfLoRA + GAM** (主要结果)
   ```bash
   inflora_inr_gam_<dataset>_t20c10_r16.yaml
   ```

2. **Fine-tuning + SGD** (下界基线)
   ```bash
   FT_sgd_<dataset>_t20.yaml
   ```

3. **Linear Probe + SGD** (上界参考)
   ```bash
   LP_sgd_<dataset>_t20.yaml
   ```

### 进阶实验（选做）

4. **InfLoRA + 不同优化器对比**
   ```bash
   inflora_inr_sgd_*.yaml
   inflora_inr_sam_*.yaml
   inflora_inr_rwp_*.yaml
   ```

5. **其他 LoRA 方法对比**
   ```bash
   seqlora_inr_gam_*.yaml
   olora_inr_gam_*.yaml
   ```

6. **消融实验**
   - 改变 `lora_rank`: 8, 16, 32
   - 改变 `lamb/lame`: 0.3/0.7, 0.5/0.9
   - 改变任务划分：init_cls=20, increment=20

---

## 📊 预期结果参考

基于 CUB200 和 Cars196 的经验：

### Aircraft (100 类)

| 方法 | Task 1 | Final Avg | Forgetting |
|-----|--------|-----------|-----------|
| Linear Probe | ~75% | N/A | 0% |
| Fine-tuning | ~80% | ~65% | ~20% |
| **InfLoRA + GAM** | **~78%** | **~72%** | **~8%** |

### Flowers (102 类)

| 方法 | Task 1 | Final Avg | Forgetting |
|-----|--------|-----------|-----------|
| Linear Probe | ~85% | N/A | 0% |
| Fine-tuning | ~90% | ~75% | ~25% |
| **InfLoRA + GAM** | **~87%** | **~82%** | **~10%** |

### Oxford Pet (37 类)

| 方法 | Task 1 | Final Avg | Forgetting |
|-----|--------|-----------|-----------|
| Linear Probe | ~70% | N/A | 0% |
| Fine-tuning | ~75% | ~60% | ~20% |
| **InfLoRA + GAM** | **~73%** | **~68%** | **~8%** |

---

## ⚠️ 常见问题排查

### Q1: 训练报错 "CUDA out of memory"

**解决方案**:
```yaml
# 减小 batch_size
batch_size: 64  # 或 32

# 或减少梯度累积步数
grad_accum_steps: 2
```

### Q2: InfLoRA 的 DualGPM 没有输出

**检查点**:
```bash
# 查看日志中是否有 "Threshold:" 输出
grep "Threshold:" logs_exp1_rebuttal/*.log

# 如果没有，检查 lamb/lame 参数是否设置
grep "lamb\|lame" your_config.yaml
```

### Q3: 准确率为 0 或 NaN

**可能原因**:
1. 学习率太大 → 减小 `lrate` 到 0.001
2. 标签处理错误 → 检查数据集是否正确加载
3. Backbone 不兼容 → 确认使用的是 `LoRA_ViT_timm`

### Q4: 训练很慢

**加速建议**:
```yaml
# 使用更少的 workers
train_num_workers: 4  # 默认可能是 8

# 或减少评估频率
eval_every_n_epochs: 5  # 每 5 个 epoch 评估一次
```

---

## 📈 结果分析工具

### 绘制学习曲线

```python
# 使用项目自带的分析工具
cd result_analyse_tool
python plot_learning_curve.py \
  --log_dir ../summaries/PerTaskInFLlora_GAM_aircraft_* \
  --output aircraft_learning_curve.pdf
```

### 计算平均准确率

```python
import numpy as np

# 读取 accuracy.txt
acc = np.loadtxt('accuracy.txt')

# 计算平均准确率
avg_acc = acc.mean()
print(f'Average Accuracy: {avg_acc:.2f}%')
```

### 可视化遗忘情况

```python
import matplotlib.pyplot as plt

# 绘制遗忘矩阵
plt.imshow(forgetting_matrix, cmap='blues')
plt.colorbar()
plt.xlabel('Task')
plt.ylabel('Task')
plt.title('Forgetting Matrix')
plt.savefig('forgetting_heatmap.pdf')
```

---

## 🔬 深入调试模式

### 启用详细日志

```yaml
# 在配置文件中添加
debug: true
verbose: true
save_activations: true
```

### 保存中间特征

```python
# 在 trainer 中添加 hook
def save_features_hook(module, input, output):
    features.append(output.detach())
    
model.backbone.blocks[5].register_forward_hook(save_features_hook)
```

### 分析 LoRA 参数变化

```python
# 训练前后对比
before = model.backbone.w_As[0].weight.clone()
# ... training ...
after = model.backbone.w_As[0].weight

delta = (after - before).norm()
print(f'LoRA-A parameter change: {delta.item():.4f}')
```

---

## 📚 进一步阅读

1. **详细分析报告**: `models_LoRAbasedCL/LORA_METHODS_ANALYSIS_REPORT.md`
2. **验证总结**: `models_LoRAbasedCL/METHODS_VERIFICATION_SUMMARY.md`
3. **InfLoRA 论文**: `relatedPaper/InfLoRA_Interference-Free_LoRA_2024.pdf`
4. **SD-LoRA 论文**: `relatedPaper/SD-LoRA_Scalable_Decoupled_LoRA_2025.pdf`

---

## 💡 专家技巧

### 技巧 1: 多跑几个随机种子

```bash
# 修改配置文件中的 seed
sed -i 's/seed:.*$/seed: [0, 1993, 42]/' your_config.yaml
```

### 技巧 2: 提前终止过拟合的任务

```yaml
# 添加早停
early_stopping: true
patience: 5
min_delta: 0.001
```

### 技巧 3: 自动超参搜索

```bash
# 使用简单的网格搜索
for rank in 8 16 32; do
  for lamb in 0.3 0.5 0.7; do
    sed "s/lora_rank:.*/lora_rank: $rank/" config.yaml > temp.yaml
    sed "s/lamb:.*/lamb: $lamb/" temp.yaml > final.yaml
    python src/main.py --config final.yaml
  done
done
```

---

## 🎉 成功标志

当你看到以下输出时，说明一切正常：

```
Task 0, Epoch 19/20 => Loss 0.523, Train 85.32, Test 82.45
Task 0, Epoch 20/20 => Loss 0.498, Train 86.78, Test 83.21
Threshold: 0.54
[DualGPM] Updated projection matrices for task 0
[NME] Computed class means for all seen classes
✓ Task 0 completed successfully!
```

---

**最后更新**: 2026-03-25  
**维护者**: Flatness_CV Team  
**支持**: 遇到问题请查看 GitHub Issues 或联系作者
