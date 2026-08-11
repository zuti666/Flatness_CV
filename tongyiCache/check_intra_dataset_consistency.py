#!/usr/bin/env python3
"""
检查每个数据集内部的训练超参数一致性
不要求跨数据集统一，只检查同一数据集内部不同方法间的配置
"""

import os
import re
from collections import defaultdict

config_base = "/data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC"

datasets = [
    "exp1_rebuttel_cub200",
    "exp1_rebuttel_aircraft", 
    "exp1_rebuttel_cars196",
    "exp1_rebuttel_flower",
    "exp1_rebuttel_oxfordPet"
]

train_params = ['init_epoch', 'init_lr', 'epochs', 'lrate', 'momentum', 'weight_decay', 'batch_size', 'scheduler']

print("=" * 120)
print("📊 数据集内部训练超参数一致性检查报告")
print("=" * 120)
print()

for dataset_dir in datasets:
    dataset_name = dataset_dir.replace("exp1_rebuttel_", "").title()
    dir_path = os.path.join(config_base, dataset_dir)
    
    if not os.path.exists(dir_path):
        continue
    
    print(f"\n{'='*120}")
    print(f"📁 数据集：{dataset_name}")
    print(f"📂 目录：{dataset_dir}")
    print(f"{'='*120}")
    
    yaml_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.yaml')])
    
    # 按方法分类并收集配置
    method_configs = defaultdict(list)
    
    for filename in yaml_files:
        filepath = os.path.join(dir_path, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取方法类型
        if filename.startswith('seqlora'):
            method = 'SeqLoRA'
        elif filename.startswith('inclora'):
            method = 'IncLoRA'
        elif filename.startswith('inflora'):
            method = 'InfLoRA'
        elif filename.startswith('olora'):
            method = 'OLoRA'
        elif filename.startswith('FT_') or 'PerTaskFT' in content:
            method = 'Fine-tuning'
        elif filename.startswith('LP_') or 'PerTaskLP' in content:
            method = 'Linear Probe'
        else:
            method = 'Other'
        
        # 提取训练参数
        config = {'filename': filename}
        for param in train_params:
            match = re.search(rf'^{param}:\s*(.+)$', content, re.MULTILINE)
            if match:
                value = match.group(1).strip()
                if '#' in value and not value.startswith('"'):
                    value = value.split('#')[0].strip()
                config[param] = value
        
        method_configs[method].append(config)
    
    # 检查每个方法内部的配置一致性
    all_consistent = True
    
    for method, configs in sorted(method_configs.items()):
        if len(configs) == 0:
            continue
        
        print(f"\n  🔹 方法：{method} ({len(configs)} 个配置)")
        print(f"  {'-'*110}")
        
        # 检查参数一致性
        param_values = defaultdict(set)
        for cfg in configs:
            for param in train_params:
                if param in cfg:
                    param_values[param].add(cfg[param])
        
        # 显示不一致的参数
        method_consistent = True
        for param in train_params:
            values = param_values[param]
            if len(values) > 1:
                method_consistent = False
                all_consistent = False
                print(f"    ❌ {param:20s} 存在 {len(values)} 种值:")
                for val in sorted(values):
                    # 找出使用该值的文件
                    files_using = [c['filename'] for c in configs if c.get(param) == val]
                    print(f"       - {val:15s} (使用：{', '.join(files_using[:3])}{'...' if len(files_using)>3 else ''})")
            elif len(values) == 1:
                val = list(values)[0]
                print(f"    ✅ {param:20s} = {val:15s} (所有配置一致)")
        
        if method_consistent:
            print(f"    🎯 该方法下所有配置完全一致!")
        else:
            print(f"    ⚠️  该方法下存在配置不一致!")
    
    print(f"\n  {'─'*110}")
    if all_consistent:
        print(f"  ✅ {dataset_name} 数据集内部所有配置完全一致!")
    else:
        print(f"  ❌ {dataset_name} 数据集内部存在配置不一致!")
    
    print()

print("\n" + "=" * 120)
print("📋 总体总结")
print("=" * 120)
print()
print("检查范围：每个数据集内部的不同方法配置")
print("检查目标：确保同一数据集内，相同方法的所有优化器配置一致")
print()
print("注意：不要求跨数据集统一，允许不同数据集采用不同的训练策略")
print("=" * 120)
