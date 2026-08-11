#!/usr/bin/env python3
"""
清理并验证所有评估配置，确保无重复且正确
"""

import os
import re

config_base = "/data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC"

datasets = [
    "exp1_rebuttel_cub200",
    "exp1_rebuttel_aircraft",
    "exp1_rebuttel_cars196",
    "exp1_rebuttel_flower",
    "exp1_rebuttel_oxfordPet"
]

def clean_and_verify(filepath):
    """清理配置文件中的重复评估配置"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到 For evaluation 部分
    eval_start = -1
    for i, line in enumerate(lines):
        if 'For evaluation' in line:
            eval_start = i
            break
    
    if eval_start == -1:
        print(f"⚠️  No evaluation section in {os.path.basename(filepath)}")
        return False
    
    # 保留 For evaluation 之前的内容
    new_lines = lines[:eval_start+1]
    
    # 收集评估配置（去重）
    eval_config_lines = []
    seen_params = set()
    
    for line in lines[eval_start+1:]:
        # 检查是否是配置行
        if ':' in line and not line.strip().startswith('#'):
            param_name = line.split(':')[0].strip()
            if param_name in seen_params:
                continue  # 跳过重复项
            seen_params.add(param_name)
        eval_config_lines.append(line)
    
    new_lines.extend(eval_config_lines)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✓ Cleaned: {os.path.basename(filepath)}")
    return True

# 处理所有数据集
total_cleaned = 0
for dataset_dir in datasets:
    dir_path = os.path.join(config_base, dataset_dir)
    
    if not os.path.exists(dir_path):
        continue
    
    yaml_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.yaml')])
    
    print(f"\n{'='*80}")
    print(f"🧹 清理 {dataset_dir}: {len(yaml_files)} 个文件")
    print(f"{'='*80}")
    
    for filename in yaml_files:
        filepath = os.path.join(dir_path, filename)
        try:
            if clean_and_verify(filepath):
                total_cleaned += 1
        except Exception as e:
            print(f"✗ Error cleaning {filename}: {e}")

print(f"\n{'='*80}")
print(f"✅ 清理完成！总计处理：{total_cleaned} 个文件")
print(f"{'='*80}")
