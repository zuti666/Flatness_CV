# 细粒度数据集任务划分 - 快速参考卡

## 📦 数据集配置速查表

| 数据集 | 类别数 | init_cls | increment | 任务数 | 覆盖 |
|--------|--------|----------|-----------|--------|------|
| CUB200 | 200 | 20 | 20 | 10 | ✅ 200/200 |
| Cars196 | 196 | 20 | 19 | 10 | ✅ 196/196 |
| Aircraft | 100 | 10 | 10 | 10 | ✅ 100/100 |
| Flowers | 102 | 10 | 10 | 10 | ⚠️ 100/102 (+2) |
| Oxford Pet | 37 | 4 | 4 | 10 | ⚠️ 37/37 (+1) |

## 🔍 快速验证命令

```bash
# 查看所有配置
for dataset in cub200 cars196 aircraft flower oxfordPet; do
  dir="config_exps_paper1_PAC/exp1_rebuttel_${dataset}"
  echo "=== $dataset ==="
  grep -E "^init_cls:|^increment:|^nb_tasks:" $dir/seqlora_inr_sgd_t20c10_r16.yaml
done
```

## 📋 典型配置文件示例

### Oxford Pet (修正后)
```yaml
dataset: "pets"
init_cls: 4
increment: 4    # ← 从 3 改为 4
nb_tasks: 10
```

### Cars196 (修正后)
```yaml
dataset: "cars196"
init_cls: 20
increment: 19   # ← 从 20 改为 19
nb_tasks: 10
```

## ⚡ 运行实验

```bash
# Oxford Pet - SeqLoRA + SGD
python src/main.py --config config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/seqlora_inr_sgd_t20c10_r16.yaml

# Cars196 - InfLoRA + GAM
python src/main.py --config config_exps_paper1_PAC/exp1_rebuttel_cars196/inflora_inr_gam_t20c10_r16.yaml

# Aircraft - SeqLoRA + RWP
python src/main.py --config config_exps_paper1_PAC/exp1_rebuttel_aircraft/seqlora_inr_rwp_t20c10_r16.yaml
```

## 🎯 关键要点

✅ **所有数据集统一为 10 个任务**  
✅ **DataManager 自动处理余数**  
✅ **最后一个任务包含剩余类别**  
✅ **标签映射全局统一**  

## 📚 相关文档

- 详细说明：`config_exps_paper1_PAC/TASK_SPLIT_CONFIG_FIX.md`
- 总结报告：`TASK_SPLIT_CORRECTION_SUMMARY.md`
- 验证脚本：`verify_task_split_configs.sh`
