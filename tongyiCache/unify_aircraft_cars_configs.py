#!/usr/bin/env python3
"""
统一 Aircraft 和 Cars196 数据集的 LoRA 方法训练超参数配置
"""

import os
import re

config_base = "/data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC"

# Aircraft 目标配置
aircraft_config = {
    'init_epoch:': 'init_epoch: 30',
    'init_lr:': 'init_lr: 0.05',
    'epochs:': 'epochs: 30',
    'lrate:': 'lrate: 0.05',
}

# Cars196 目标配置
cars_config = {
    'init_epoch:': 'init_epoch: 20',
    'init_lr:': 'init_lr: 0.05',
    'epochs:': 'epochs: 20',
    'lrate:': 'lrate: 0.05',
}

def update_configs(dataset_dir, target_config, lora_patterns):
    """Update configs for a specific dataset."""
    dir_path = os.path.join(config_base, dataset_dir)
    
    if not os.path.exists(dir_path):
        print(f"❌ Directory not found: {dir_path}")
        return 0
    
    yaml_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.yaml')])
    lora_files = [f for f in yaml_files if any(p in f for p in lora_patterns)]
    
    print(f"\n{'='*80}")
    print(f"📊 处理 {dataset_dir}: {len(lora_files)} 个 LoRA 配置文件")
    print(f"{'='*80}")
    
    updated_count = 0
    for filename in lora_files:
        filepath = os.path.join(dir_path, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            updated = False
            if not line.strip().startswith('#'):
                for key, value in target_config.items():
                    if line.startswith(key) and not line.strip().startswith('#'):
                        if line.strip() == value:
                            new_lines.append(line)
                        else:
                            new_lines.append(value + '\n')
                        updated = True
                        break
            
            if not updated:
                new_lines.append(line)
        
        # 更新 prefix（如果需要）
        content = ''.join(new_lines)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Updated: {filename}")
        updated_count += 1
    
    print(f"\n✅ 完成！已更新 {updated_count}/{len(lora_files)} 个文件")
    return updated_count

# 处理 Aircraft 数据集
print("\n" + "="*80)
print("🔧 开始统一 Aircraft 数据集配置 (epochs=30, lr=0.05)")
print("="*80)
lora_methods = ['seqlora', 'inclora', 'inflora', 'olora']
update_configs('exp1_rebuttel_aircraft', aircraft_config, lora_methods)

# 处理 Cars196 数据集
print("\n" + "="*80)
print("🔧 开始统一 Cars196 数据集配置 (epochs=20, lr=0.05)")
print("="*80)
update_configs('exp1_rebuttel_cars196', cars_config, lora_methods)

print("\n" + "="*80)
print("🎉 所有数据集配置统一化完成!")
print("="*80)
