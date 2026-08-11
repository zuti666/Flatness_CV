#!/usr/bin/env python3
"""修复 prefix 中重复的 ep10_lr005 标记"""

import os
import re

config_dir = "/data/140-0/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower"

def fix_prefix(filepath):
    """Fix duplicate ep10_lr005 in prefix."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换重复的标记
    def fix_duplicate(match):
        prefix = match.group(1)
        # 将连续的 ep10_lr005 替换为单个
        fixed = re.sub(r'(ep10_lr005_)+', r'\1', prefix)
        return f'prefix: {fixed}'
    
    content = re.sub(r'^prefix:\s*(.+)$', fix_duplicate, content, flags=re.MULTILINE)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Fixed: {os.path.basename(filepath)}")

yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
for filename in sorted(yaml_files):
    filepath = os.path.join(config_dir, filename)
    try:
        fix_prefix(filepath)
    except Exception as e:
        print(f"✗ Error fixing {filename}: {e}")

print(f"\n✅ Done! Fixed {len(yaml_files)} files.")
