#!/usr/bin/env python3
"""
分析 exp1_rebuttel 系列数据集中 ViT backbone 的训练超参数配置
检查五个数据集：CUB200, Aircraft, Cars196, Flowers, Oxford Pet
"""

import os
import re
from collections import defaultdict

config_base = "/data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC"

# 五个目标数据集
datasets = [
    "exp1_rebuttel_cub200",
    "exp1_rebuttel_aircraft", 
    "exp1_rebuttel_cars196",
    "exp1_rebuttel_flower",
    "exp1_rebuttel_oxfordPet"
]

# 要提取的训练参数
train_params = [
    'init_epoch',
    'init_lr',
    'epochs',
    'lrate',
    'momentum',
    'weight_decay',
    'batch_size',
    'scheduler'
]

# 任务划分参数
task_params = ['init_cls', 'increment', 'nb_tasks']

print("=" * 100)
print("Exp1 Rebuttal - ViT Backbone 五数据集训练超参数配置分析报告")
print("=" * 100)
print()

# 统计信息
all_configs = []

for dataset_dir in datasets:
    dataset_name = dataset_dir.replace("exp1_rebuttel_", "").title()
    dir_path = os.path.join(config_base, dataset_dir)
    
    if not os.path.exists(dir_path):
        print(f"⚠️  Directory not found: {dir_path}")
        continue
    
    print(f"\n{'='*100}")
    print(f"📊 数据集：{dataset_name}")
    print(f"📁 目录：{dataset_dir}")
    print(f"{'='*100}")
    
    yaml_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.yaml')])
    print(f"配置文件数量：{len(yaml_files)}")
    print()
    
    # 按方法分类
    methods = defaultdict(list)
    for filename in yaml_files:
        # 提取方法名
        if filename.startswith('seqlora'):
            methods['SeqLoRA'].append(filename)
        elif filename.startswith('inclora'):
            methods['IncLoRA'].append(filename)
        elif filename.startswith('inflora'):
            methods['InfLoRA'].append(filename)
        elif filename.startswith('olora'):
            methods['OLoRA'].append(filename)
        elif filename.startswith('FT_') or filename.startswith('PerTaskFT'):
            methods['Fine-tuning'].append(filename)
        elif filename.startswith('LP_') or filename.startswith('PerTaskLP'):
            methods['Linear Probe'].append(filename)
    
    # 分析每个方法的配置
    for method, files in sorted(methods.items()):
        print(f"\n  🔹 方法：{method} ({len(files)} 个配置)")
        print(f"  {'-'*90}")
        
        # 随机选择一个代表性配置进行详细分析
        sample_file = files[0]
        filepath = os.path.join(dir_path, sample_file)
        
        config_data = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 提取 prefix
            prefix_match = re.search(r'^prefix:\s*(.+)$', content, re.MULTILINE)
            if prefix_match:
                config_data['prefix'] = prefix_match.group(1).strip()
            
            # 提取训练参数
            for param in train_params + task_params:
                match = re.search(rf'^{param}:\s*(.+)$', content, re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    # 移除注释
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    config_data[param] = value
        
        # 显示配置详情
        print(f"  示例配置：{sample_file}")
        if 'prefix' in config_data:
            print(f"  Prefix: {config_data['prefix']}")
        print()
        print(f"  训练超参数:")
        for param in train_params:
            value = config_data.get(param, 'N/A')
            print(f"    {param:20s} = {value}")
        
        print()
        print(f"  任务划分参数:")
        for param in task_params:
            value = config_data.get(param, 'N/A')
            print(f"    {param:20s} = {value}")
        
        # 计算覆盖类别数
        try:
            init_cls = int(config_data.get('init_cls', 0))
            increment = int(config_data.get('increment', 0))
            nb_tasks = int(config_data.get('nb_tasks', 0))
            total_classes = init_cls + increment * (nb_tasks - 1)
            print(f"\n  覆盖类别数：{init_cls} + {increment}×{nb_tasks-1} = {total_classes} 类")
        except:
            pass
        
        all_configs.append({
            'dataset': dataset_name,
            'method': method,
            'config': config_data
        })
        
        print()

# 跨数据集一致性分析
print("\n" + "=" * 100)
print("🔍 跨数据集超参数一致性分析")
print("=" * 100)

# 按方法分组比较
methods_to_compare = ['SeqLoRA', 'IncLoRA', 'InfLoRA', 'OLoRA', 'Fine-tuning']

for method in methods_to_compare:
    method_configs = [c for c in all_configs if c['method'] == method]
    
    if len(method_configs) < 2:
        continue
    
    print(f"\n{'='*100}")
    print(f"方法：{method}")
    print(f"{'='*100}")
    
    # 收集各数据集的参数
    param_values = defaultdict(dict)
    
    for config_info in method_configs:
        dataset = config_info['dataset']
        cfg = config_info['config']
        
        for param in train_params[:4]:  # 只关注主要参数
            value = cfg.get(param, 'N/A')
            param_values[param][dataset] = value
    
    # 检查一致性
    print("\n参数值对比:")
    print(f"{'参数':<20} | ", end="")
    datasets_in_method = [c['dataset'] for c in method_configs]
    for ds in datasets_in_method:
        print(f"{ds:<15} | ", end="")
    print()
    print("-" * 20 + "-+-" + "-+-".join(["-"*15]*len(datasets_in_method)) + "-")
    
    for param in train_params[:4]:
        print(f"{param:<20} | ", end="")
        values = []
        for ds in datasets_in_method:
            val = param_values[param].get(ds, 'N/A')
            values.append(val)
            print(f"{val:<15} | ", end="")
        print()
        
        # 检查是否一致
        unique_values = set(values)
        if len(unique_values) == 1:
            print(f"  ✅ {param}: 所有数据集一致 ({list(unique_values)[0]})")
        else:
            print(f"  ⚠️  {param}: 存在差异 -> {unique_values}")
        print()

print("\n" + "=" * 100)
print("📋 总结报告")
print("=" * 100)

# 统计完全一致的配置
consistent_count = 0
total_checks = 0

for method in methods_to_compare:
    method_configs = [c for c in all_configs if c['method'] == method]
    if len(method_configs) < 2:
        continue
    
    for param in train_params[:4]:
        values = [c['config'].get(param, 'N/A') for c in method_configs]
        total_checks += 1
        if len(set(values)) == 1:
            consistent_count += 1

consistency_rate = (consistent_count / total_checks * 100) if total_checks > 0 else 0

print(f"\n总体一致性比率：{consistent_count}/{total_checks} ({consistency_rate:.1f}%)")

if consistency_rate == 100:
    print("\n✅ 恭喜！所有数据集的训练超参数配置完全一致！")
elif consistency_rate >= 80:
    print(f"\n⚠️  大部分配置一致，但有 {total_checks - consistent_count} 项存在差异")
else:
    print(f"\n❌ 配置一致性较低，建议检查并统一超参数设置")

print("\n" + "=" * 100)
print("分析完成!")
print("=" * 100)
