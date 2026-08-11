#!/usr/bin/env python3
"""
统一所有 5 个数据集的评估设置配置
目标目录：exp1_rebuttel_cub200, exp1_rebuttel_aircraft, exp1_rebuttel_cars196, 
         exp1_rebuttel_flower, exp1_rebuttel_oxfordPet
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

# 新的评估配置内容
new_eval_config = """#-------------- # Linear Probe Evaluation , False------------------
linear_probe_eval_num_workers: 0
linear_probe_softmax_joint_seen_eval: true
linear_probe_softmax_per_task_eval: false
probe_train_mode: train
probe_test_mode: test

probe_fit_epochs: 20       
probe_fit_lr: 0.015
probe_fit_moment: 0 
probe_fit_wd: 0.0005
probe_fit_train_batch_size: 128




# #-------------- #  Weight Flatness ------------------

flat_eval: true
flat_eval_sharpness: true
flat_eval_hessian: true
flat_eval_GGN: true 
flat_eval_fisher: true


flat_eval_batch_size: 32
flat_eval_dataset_fraction: 0.1  #  10% of TestDataset for flatness/lossland 

# First and zero order Sharpness definition

flat_eval_sharpness_radius: 0.05

# E-sh definition

flat_eval_esh_gaussian_std: null
flat_eval_esh_samples: 100  

# Hessian 

eval_hessian: true
flat_eval_task_indices: "-1"        # 只在最后一个 task 执行 weight flatness
feature_flat_task_indices: "-1"     # 只在最后一个 task 执行 feature flatness

flat_eval_hessian_power_iters: 100   
flat_eval_hessian_trace_samples: 100  

#  Loss landscape

weight_loss_land_1d: false
weight_loss_land_2d: false
weight_loss_land_radius: 1            
weight_loss_land_num_points: 41
weight_loss_land_filter_norm: false     
loss_land_modes: "full"
loss_land_basis: "random"
eig_save_vectors: false                   


#-------------- # Feature Roubtness（EFM）------------------
feature_flat_eval: false

# feature_flat_max_batches: None

feature_flat_topk: 200
feature_flat_eps: 1e-12
feature_flat_rank_tol: 1e-06
feature_flat_save_path: null

# CKA eval

feature_cka_eval: false
feature_cka_max_batches: 32         # Max batches for CKA anchor sampling
feature_cka_max_samples: 2048       # Max samples for CKA anchor sampling
feature_sep_max_samples: 2048       # Sampling cap for separability metric
feature_margin_max_samples: 4096    # Sampling cap for margin statistics

# Prptotype drift

feature_proto_eval: false
feature_proto_max_batches: 32        # Max batches for prototype drift stats
feature_proto_max_samples: 2048     # Max samples for prototype drift stats

# for attention probe

attention_probe_eval: false
"""

def remove_old_eval_config(content):
    """移除旧的评估配置部分"""
    # 定义需要移除的配置块模式
    patterns_to_remove = [
        # Linear Probe Evaluation
        (r'#-{0,10}.*Linear Probe Evaluation.*?(\n(?=#|-{10,}|[a-z_]+:)|\Z)', ''),
        # Weight Flatness
        (r'#-{0,10}.*Weight Flatness.*?(\n(?=#|-{10,}|[a-z_]+:)|\Z)', ''),
        # Feature Robustness
        (r'#-{0,10}.*Feature Rou?btness.*?(\n(?=#|-{10,}|[a-z_]+:)|\Z)', ''),
        # Feature flatness
        (r'feature_flat_eval:.*?(\n(?=#|[a-z_]+:)|\Z)', '', re.DOTALL),
        # flat_eval related
        (r'flat_eval:.*?(\n(?=#|[a-z_]+:)|\Z)', '', re.DOTALL),
        # probe_fit related
        (r'probe_fit_epochs:.*?(\n(?=#|[a-z_]+:)|\Z)', '', re.DOTALL),
        # loss landscape
        (r'weight_loss_land_1d:.*?(\n(?=#|[a-z_]+:)|\Z)', '', re.DOTALL),
        # CKA eval
        (r'feature_cka_eval:.*?(\n(?=#|[a-z_]+:)|\Z)', '', re.DOTALL),
        # attention probe
        (r'attention_probe_eval:.*?(\n(?=#|[a-z_]+:)|\Z)', '', re.DOTALL),
    ]
    
    for pattern, replacement, *flags in patterns_to_remove:
        if flags:
            content = re.sub(pattern, replacement, content, flags=flags[0])
        else:
            content = re.sub(pattern, replacement, content)
    
    return content

def update_config_file(filepath):
    """更新单个配置文件的评估设置"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 移除旧的评估配置
    content = remove_old_eval_config(content)
    
    # 查找插入位置（在 For evaluation 部分之后，或在文件末尾）
    eval_marker = '#---------------------------- For evaluation ------------------------------------'
    if eval_marker in content:
        # 找到标记位置
        marker_pos = content.find(eval_marker)
        # 找到标记后的第一个空行或下一个配置项
        next_line_pos = content.find('\n', marker_pos + len(eval_marker))
        if next_line_pos != -1:
            # 在标记后插入新配置
            new_content = content[:next_line_pos+1] + '\n' + new_eval_config
            # 保留标记后的其他内容（如果有）
            remaining = content[next_line_pos+1:]
            # 移除剩余部分中的重复评估配置
            remaining = remove_old_eval_config(remaining)
            content = new_content + remaining
    else:
        # 如果没有找到标记，在文件末尾添加
        content = content.rstrip() + '\n\n' + new_eval_config
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Updated: {os.path.basename(filepath)}")

# 处理所有数据集
total_updated = 0
for dataset_dir in datasets:
    dir_path = os.path.join(config_base, dataset_dir)
    
    if not os.path.exists(dir_path):
        print(f"❌ Directory not found: {dir_path}")
        continue
    
    yaml_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.yaml')])
    
    print(f"\n{'='*80}")
    print(f"📊 处理 {dataset_dir}: {len(yaml_files)} 个配置文件")
    print(f"{'='*80}")
    
    updated_count = 0
    for filename in yaml_files:
        filepath = os.path.join(dir_path, filename)
        try:
            update_config_file(filepath)
            updated_count += 1
            total_updated += 1
        except Exception as e:
            print(f"✗ Error updating {filename}: {e}")
    
    print(f"\n✅ 完成！已更新 {updated_count}/{len(yaml_files)} 个文件")

print(f"\n{'='*80}")
print(f"🎉 所有数据集评估配置统一化完成!")
print(f"   总计更新：{total_updated} 个配置文件")
print(f"{'='*80}")
