# 日志、结果文件与复现

[返回首页](README.md)

## 1. 输出层级

一次当前训练通常写到：

```text
outputs_logs/
└── logs_inc_lora/<method>/<optimizer-tag>/<dataset>/<seed>/<prefix>/<mode>/
    ├── checkpoints/
    └── <increment>/
        ├── <prefix>_<backbone>.log
        ├── <prefix>_<backbone>_cl_metrics.json
        ├── *.csv / *.npy
        └── flatness/                    # 仅 flat_eval 开启时
```

launcher 自己的 stdout/stderr 位于：

```text
outputs_logs/<experiment>_launcher_logs/
```

历史 rebuttal 日志位于：

- `logs_exp1_rebuttal/`
- `logs_exp1_rebuttal_5datasets/`
- `logs_exp1_rebuttal_het5/`
- `logs_exp0_redoFlatLoRA/`
- `logs_exp1_newDESIGN/`

这些历史目录名可能与日志内部真实 `dataset`/`config` 不一致，目录名不能作为最终证据。

## 2. Consolidated metrics JSON

`*_cl_metrics.json` 是当前最重要的机器可读结果：

```text
cnn.steps.<task>
cnn.matrices.t00 ... t19
cnn.matrices.final
cnn.final.{FAA,AAA,BWT_final_avg,Forget_avg,...}

nme.steps.<task>
nme.matrices.final
nme.final.{...}
```

写入与 matrix assembly 在 `utils/metrics_book.py`，指标计算在 `evaluation_performance/metrics.py`。

JSON 中允许 `NaN`，属于 Python JSON 的宽松扩展；某些严格 JSON 工具会拒绝读取。汇总脚本当前使用 Python `json`，可正常处理。

## 3. 主要汇总脚本

| 实验 | 汇总命令 | 主要输出 |
|---|---|---|
| Exp C | `python config_exps_paper1_PAC/exp_C_cifar10_task_conditioned/summarize_cifar10_task_conditioned.py` | `summary_by_variant.csv`、task-conditioned flatness long table |
| Exp D | `python config_exps_paper1_PAC/exp_D_taskwise_sam_trajectory/summarize_taskwise_sam_trajectory.py` | variant summary、`contrasts_by_seed.csv` |
| Exp E | `python config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction/summarize_exp_E_support_direction.py` | performance 与 per-task flatness CSV |
| Exp E plot | `python config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction/plot_exp_E_support_direction.py` | pairwise CSV、FAA/BWT 图 |
| Exp F | `python config_exps_paper1_PAC/exp_F_imagenetr_r16_t20_forked_taskwise/summarize_forked_taskwise.py` | full/prefix summary |
| Exp F plot | `python config_exps_paper1_PAC/exp_F_imagenetr_r16_t20_forked_taskwise/plot_forked_taskwise_curves.py` | trajectory/forgetting curves |
| Exp 7 | `python config_exps_paper1_PAC/exp_7_RQ1RQ2/summarize_rq1rq2.py` | rescaling 与 sharpness summary |

Project1 Wiki 的统一出图入口见[绘图代码](06_plotting.md)。它读取 Wiki 的审计快照，避免修改原实验脚本。

## 4. 当前已核对的 summary 路径

### Exp E

```text
outputs_logs/exp_E_imagenetr_r16_t20_support_direction_summary/
├── summary_by_variant.csv
├── per_task_flatness.csv
└── figures/
    ├── support_direction_pairwise.csv
    ├── support_direction_faa_bwt.png
    └── support_direction_faa_bwt.pdf
```

当前 `per_task_flatness.csv` 没有有效行，`summary_by_variant.csv` 的 `num_flat_tasks=0`，与 Exp E YAML 的 `flat_eval=false` 一致。

### Exp F

```text
outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_summary/
├── summary_by_variant.csv
├── summary_official_and_prefix.csv
├── prefix_full_aaa_faa_bwt_table.csv
└── figures/
    ├── per_task_curves_long.csv
    ├── *_time_curves.{png,pdf}
    ├── *_aaa_curve.{png,pdf}
    └── *_final_forgetting.{png,pdf}
```

### Rebuttal 扩展

```text
RebuttalReply/ExpSummaryExpForRebuttal/
├── combine_CNN_detailed.md
├── table_rebuttal_vision_onepage.md
├── taskmatrix_{cub200,cars196,aircraft,flowers,pets}.md
├── combine_LLM_detailed.md
├── table_rebuttal_nlp_t5.md
└── taskmatrix_nlp_o{1,2,3}.md
```

对应原始/中间 Excel 多数位于 `summaries/`。

## 5. 结果追踪规则

每个报告数字应能反向追到：

```text
paper/wiki number
  -> summary CSV/Markdown row
  -> *_cl_metrics.json or source Excel
  -> accuracy matrix / raw task-wise records
  -> training .log header
  -> effective config + code revision
```

推荐优先级：

1. 用 accuracy matrix 重新计算 FAA/AAA/BWT/Forget。
2. 与 summary CSV 交叉核对，容差只允许显示舍入误差。
3. 检查日志头打印的 `dataset/model/seed/rank/LR/epoch/optimizer_type`。
4. 最后才用配置文件补充未打印字段。

不要：

- 按日志文件夹名字推断 dataset；
- 按图文件名中的 `rank16` 推断真实 rank；
- 把当前被修改过的 YAML 当成历史 run snapshot；
- 对不同 task split 的矩阵直接求平均；
- 把缺失/NaN 写成 0。

## 6. 最小复现流程

```bash
# 1. 进入包含 torch/timm/torchvision 的实验环境
export FLATNESS_CV_DATA_ROOT=/data/140-1/datasets

# 2. 运行受控实验（示例：Exp E 子集）
PYTHON_BIN=/path/to/env/bin/python \
EXP_E_GPUS="0 1 2" \
EXP_E_VARIANTS="sgd sam_factor random_factor" \
bash config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction/run_exp_E_support_direction.sh

# 3. 汇总
python config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction/summarize_exp_E_support_direction.py
python config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction/plot_exp_E_support_direction.py

# 4. 生成 Wiki 图
MPLCONFIGDIR=/tmp/project1-mpl \
python Project1/wiki/scripts/plot_project1_results.py --figure all
```

## 7. 新实验建议保存内容

每个 run 目录应增加：

- 实际生效配置快照，而不是只存配置路径；
- Git commit/hash 和 `git diff --stat`；
- Python、PyTorch、CUDA、timm 版本；
- 数据 root、dataset loader 名、task boundaries；
- checkpoint hash；
- consolidated metrics JSON；
- summary script version 与生成时间。

这能消除当前最主要的“配置更新后旧结果失去精确对应关系”问题。

