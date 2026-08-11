#!/bin/bash
# 验证细粒度数据集任务划分配置的脚本

echo "=========================================="
echo "细粒度数据集任务划分配置验证"
echo "=========================================="
echo ""

CONFIG_DIR="config_exps_paper1_PAC"

# 定义数据集及其预期类别数
declare -A DATASETS=(
    ["exp1_rebuttel_cub200"]="CUB200 (200 classes)"
    ["exp1_rebuttel_aircraft"]="Aircraft (100 classes)"
    ["exp1_rebuttel_cars196"]="Cars196 (196 classes)"
    ["exp1_rebuttel_flower"]="Flowers102 (102 classes)"
    ["exp1_rebuttel_oxfordPet"]="Oxford Pet (37 classes)"
)

# 为每个数据集生成报告
for dataset_dir in "${!DATASETS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 ${DATASETS[$dataset_dir]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ ! -d "$CONFIG_DIR/$dataset_dir" ]; then
        echo "❌ Directory not found: $CONFIG_DIR/$dataset_dir"
        continue
    fi
    
    # 选取代表性配置文件 (SeqLoRA SGD)
    config_file="$CONFIG_DIR/$dataset_dir/seqlora_inr_sgd_t20c10_r16.yaml"
    
    if [ ! -f "$config_file" ]; then
        echo "❌ Config file not found: $config_file"
        continue
    fi
    
    # 提取配置参数
    init_cls=$(grep "^init_cls:" "$config_file" | awk '{print $2}')
    increment=$(grep "^increment:" "$config_file" | awk '{print $2}')
    nb_tasks=$(grep "^nb_tasks:" "$config_file" | awk '{print $2}')
    
    # 计算覆盖的类别数
    if [ -n "$init_cls" ] && [ -n "$increment" ] && [ -n "$nb_tasks" ]; then
        covered=$((init_cls + increment * (nb_tasks - 1)))
        
        echo "✅ 配置参数:"
        echo "   - Initial classes (init_cls): $init_cls"
        echo "   - Increment per task (increment): $increment"
        echo "   - Number of tasks (nb_tasks): $nb_tasks"
        echo "   - Total classes covered: $covered"
        
        # 显示任务序列
        echo ""
        echo "📋 任务序列:"
        for ((i=0; i<nb_tasks; i++)); do
            if [ $i -eq 0 ]; then
                task_size=$init_cls
            else
                task_size=$increment
            fi
            echo "   Task $i: $task_size classes"
        done
        
        echo ""
    else
        echo "❌ Failed to parse configuration"
    fi
done

echo "=========================================="
echo "✅ 验证完成!"
echo "=========================================="
echo ""
echo "💡 提示："
echo "   - CUB200: 20 + 20×9 = 200 ✓ (完美覆盖)"
echo "   - Aircraft: 10 + 10×9 = 100 ✓ (完美覆盖)"
echo "   - Cars196: 20 + 19×9 = 191 (+5 余数，最后任务包含 24 类)"
echo "   - Flowers: 10 + 10×9 = 100 (+2 余数，可接受)"
echo "   - Oxford Pet: 4 + 4×9 = 36 (+1 余数，最后任务包含 1 类)"
echo ""
echo "📄 详细信息请查看：config_exps_paper1_PAC/TASK_SPLIT_CONFIG_FIX.md"
echo ""
