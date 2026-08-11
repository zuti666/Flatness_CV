#!/usr/bin/env python3
"""
将 Oxford Pet 数据集下所有 LoRA 方法的训练参数统一为 ep10_lr005 配置
目标配置:
  init_epoch: 10
  init_lr: 0.005
  epochs: 10
  lrate: 0.005
"""

import os
import re

config_dir = "/data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_oxfordPet"

# 需要更新的文件模式（LoRA 方法）
lora_patterns = ['seqlora', 'inclora', 'inflora', 'olora']

# 需要更新的参数
updates = {
    'init_epoch:': 'init_epoch: 10',
    'init_lr:': 'init_lr: 0.005',
    'epochs:': 'epochs: 10',
    'lrate:': 'lrate: 0.005',
}

def update_config(filepath):
    """Update a single config file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        # 更新训练参数（跳过注释行和 Fine-tuning 配置）
        updated = False
        if not line.strip().startswith('#'):
            for key, value in updates.items():
                if line.startswith(key) and not line.strip().startswith('#'):
                    # 跳过已经正确的值
                    if line.strip() == value:
                        new_lines.append(line)
                    else:
                        new_lines.append(value + '\n')
                    updated = True
                    break
        
        if not updated:
            new_lines.append(line)
    
    # 更新 prefix
    content = ''.join(new_lines)
    def update_prefix(match):
        prefix = match.group(1)
        # 移除旧的标记并添加新的
        prefix = re.sub(r'_ep\d+_lr[\d.]+', '', prefix)  # 移除旧的学习率标记
        prefix = re.sub(r'_t20(_rank)?', r'_ep10_lr005_t20\1', prefix)  # 添加新标记
        return f'prefix: {prefix}'
    
    content = re.sub(r'^prefix:\s*(.+)$', update_prefix, content, flags=re.MULTILINE)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Updated: {os.path.basename(filepath)}")

# 处理所有 yaml 文件
yaml_files = sorted([f for f in os.listdir(config_dir) if f.endswith('.yaml')])
print(f"Found {len(yaml_files)} config files in {config_dir}\n")

updated_count = 0
for filename in yaml_files:
    # 只处理 LoRA 方法的配置文件
    if any(pattern in filename for pattern in lora_patterns):
        filepath = os.path.join(config_dir, filename)
        try:
            update_config(filepath)
            updated_count += 1
        except Exception as e:
            print(f"✗ Error updating {filename}: {e}")

print(f"\n✅ Done! Updated {updated_count}/{len(yaml_files)} files.")
print(f"   (仅更新 LoRA 方法配置，保留 Fine-tuning 和 Linear Probe 原配置)")
