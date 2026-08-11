#!/usr/bin/env python3
"""
批量更新 Flowers 数据集配置文件
- 训练参数：init_epoch=10, init_lr=0.005, epochs=10, lrate=0.005
- 在 prefix 中添加 ep10_lr005 标记
"""

import os
import re

config_dir = "/data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower"

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
        # 更新训练参数
        updated = False
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
    # 匹配 prefix 行并添加 ep10_lr005 标记
    def update_prefix(match):
        prefix = match.group(1)
        # 避免重复添加
        if 'ep10_lr005' in prefix:
            return match.group(0)
        # 在 _t20_ 或 _t20_rank 前插入标记
        new_prefix = re.sub(r'_t20(_rank)?', r'_ep10_lr005_t20\1', prefix)
        return f'prefix: {new_prefix}\n'
    
    content = re.sub(r'^prefix:\s*(.+)$', update_prefix, content, flags=re.MULTILINE)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Updated: {os.path.basename(filepath)}")

# 处理所有 yaml 文件
yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
print(f"Found {len(yaml_files)} config files in {config_dir}\n")

for filename in sorted(yaml_files):
    filepath = os.path.join(config_dir, filename)
    try:
        update_config(filepath)
    except Exception as e:
        print(f"✗ Error updating {filename}: {e}")

print(f"\n✅ Done! Updated {len(yaml_files)} files.")
