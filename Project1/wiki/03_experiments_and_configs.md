# 实验与配置

[返回首页](README.md)

## 1. 主实验地图

| 组别 | 目的 | 数据/模型 | 配置入口 | 状态 |
|---|---|---|---|---|
| 原始 ImageNet-R main | SGD/SAM/RWP/GAM、LoRA 组织、flatness | ImageNet-R，ViT-B/16 | `config_exps_paper1_PAC/exp_1_sam-sgd_imageR_r16_t20/`、`exp2_sam-rwp-flat_imageR_r16_t20/` | 已有结果；配置含历史变体 |
| Scope sweep | LoRA-only 与 full-parameter perturbation | ImageNet-R，Seq/Inc/OLoRA | `exp1_redo_FlatLoRA/`、`exp5_full0lora/` | 论文已报告 scope delta |
| OOD | ImageNet-C/P robustness | Tiny ImageNet-C/P，ViT | `exp3_dataset_imagec/`、`exp3_dataset_imagep/` | 已有曲线；当前 YAML 与论文命名有漂移 |
| Rank/length ablation | adapter capacity、sequence length | ImageNet-R | `exp4_ablation_rank/`、`exp4_ablation_length/` | 已报告 |
| Cross-domain rebuttal | 5 个细粒度 benchmark | CUB/Cars/Aircraft/Flowers/Pets | `exp1_rebuttel_*` | 已报告，旧日志与当前 YAML 不完全一致 |
| Het5 | 单一跨数据集 stream | 5 datasets，635 classes | `exp1_rebuttel_5datacombine/` | 已报告，多代结果并存 |
| Exp C | task-conditioned support pilot | CIFAR10 2 tasks | `exp_C_cifar10_task_conditioned/` | 完成 |
| Exp D | task-wise optimizer factorial | CIFAR10 2 tasks | `exp_D_taskwise_sam_trajectory/` | 完成 |
| Exp E | support × direction | ImageNet-R，SeqLoRA，T=20 | `exp_E_imagenetr_r16_t20_support_direction/` | 完成，最强 direction 证据 |
| Exp F | same-checkpoint forked trajectory | ImageNet-R，前10/后10 tasks | `exp_F_imagenetr_r16_t20_forked_taskwise/` | 完成，最强 trajectory 证据 |
| Exp 7 / G / Q2 | LoRA rescaling invariance | ImageNet-R | `exp_7_RQ1RQ2/` | 设计与代码具备；结果尚不足以定论 |
| NLP | 跨架构外部有效性 | T5-small/large、Llama-3.2 | 结果在 `RebuttalReply`/论文；训练工程部分在外部 O-LoRA 项目 | 部分完成 |

`config_exps_paper1_PAC/exp_paperA_*` 是后续方法论文分支，不纳入本 Wiki 的 Project1 核心结论。

## 2. ImageNet-R Exp E/F 的 canonical 设置

| 参数 | 值 |
|---|---|
| Dataset | `imagenetr`，200 classes |
| Split | `init_cls=10, increment=10`，20 tasks |
| Class order | `class_shuffle=false` |
| Backbone | `vit_base_patch16_224`，ImageNet-21K pretrained |
| Method | SeqLoRA |
| LoRA rank | 16 |
| Replay | `memory_size=0`，无 replay |
| Batch size | 128 |
| Seed | 1993 |
| Optimizer | base SGD，momentum 0，weight decay 0 |
| LR/scheduler | 0.01，cosine |
| Epochs | 20/task |
| SAM/random radius | 0.05 |
| Flat evaluation | 当前 Exp E/F YAML 为 `false` |

Exp E 有 11 个独立 YAML：SGD；`sam_{factor,full,delta,all,frozen}`；`random_{factor,full,delta,all,frozen}`。

Exp F 先训练 SGD prefix 和 SAM-factor prefix 到 task 9，再复制相同 checkpoint，按 task-wise optimizer schedule 继续 task 10–19。launcher 会验证 checkpoint 与 metrics 文件存在；完整 hash 一致性记录见 `Project1/experiment_summary_flatness_pecl_2026-04-30.md`。

## 3. CIFAR10 Exp C/D 设置

| 参数 | 值 |
|---|---|
| Dataset | `cifar10_224` |
| Split | classes 0–4，然后 5–9 |
| Backbone | ViT-B/16 |
| Method/rank | SeqLoRA，rank 16 |
| Exp C | 比较 support，并输出 `theta_t × loss_task` 的 task-conditioned flatness |
| Exp D | 比较 task1/task2 分别使用 SGD、SAM-factor、random-factor 的 factorial schedules |

Exp D 的受控对比应优先读 `contrasts_by_seed.csv`，而不是只读 FinalAvg。

## 4. 五个细粒度数据集与 Het5

### 4.1 当前代码的 task split

| Dataset | 类数 | 当前 split | 任务数 |
|---|---:|---|---:|
| CUB200 | 200 | 20 + 20×9 | 10 |
| Aircraft | 100 | 10×10 | 10 |
| Cars196 | 196 | 16 + 20×9 | 10 |
| Flowers102 | 102 | 12 + 10×9 | 10 |
| Oxford-IIIT Pet | 37 | 5 + 4×8 | 9 |

Het5 当前 loader 顺序是：

```text
CUB200(200) -> Aircraft(100) -> Cars196(196) -> Flowers102(102) -> Pets(37)
```

总计 635 类，`task_splits=[200,100,196,102,37]`。部分 rebuttal/论文文稿写成 Aircraft 开头的不同顺序；复现当前代码必须以后者代码顺序为准。

完整数据根目录、loader、transform、日志实跑参数与脚本风险已核对在仓库根目录的 `5dataset-wiki.md`；本页只保留主索引。

### 4.2 结果表对应的历史报告设置

`RebuttalReply/ExpSummaryExpForRebuttal/combine_CNN_detailed.md` 报告：ViT、rank 16、40 epoch/task、seed 0，并列出历史 LR：CUB 0.01、Cars/Aircraft 0.005、Flowers/Pets 0.00025。

当前 YAML 和 2026-03 日志已发生系统漂移：当前主 YAML 多为 Cars/Aircraft 0.05、Flowers/Pets 0.0025；旧日志又包含 10–30 epoch 的运行。引用结果时必须把“表格报告设置”与“当前可运行设置”分开。

## 5. 原始 ImageNet-R/C/P 与消融配置

当前目录核对结果：

- `exp_1_sam-sgd_imageR_r16_t20`：ImageNet-R，10-class increments，20 epochs；混有 ViT/ResNet、LoRA/FT/LP 和不同 seed。
- `exp2_sam-rwp-flat_imageR_r16_t20`：RWP/GAM/C-Flat；混有 seed 521/1024/1993/42 和 `flat_eval` 开关。
- `exp3_dataset_imagec` / `imagep`：当前 LoRA YAML 的 `lora_rank=8`，但部分结果文件名和论文文字写 `rank16`。未核对原始日志头前，不得仅凭目录名确定 rank。
- `exp4_ablation_rank`：当前 YAML 主要是 `rank=2` 的补充点，不代表完整 rank sweep；完整结果来自多代日志/summary。
- `exp4_ablation_length`：当前 YAML 为 20-class increments、10 tasks、seed 2048；图中其它 length 来自历史运行。

## 6. NLP 设置

T5 结果覆盖三个任务顺序：

| Order | Sequence |
|---|---|
| o1 | DBpedia → Amazon → Yahoo → AGNews |
| o2 | DBpedia → Amazon → AGNews → Yahoo |
| o3 | Yahoo → Amazon → AGNews → DBpedia |

模型为 T5-large/T5-small，方法为 SeqLoRA/IncLoRA/OLoRA，比较 Adam 与 SAM。T5 表中的 GAM/RWP 尚未提供。

当前论文另报告 Llama-3.2-1B/3B 的三顺序平均 AAA；SeqLoRA GAM 尚未完成，IncLoRA/OLoRA 有 Adam/SAM/GAM。

## 7. 运行入口

单配置：

```bash
python -m src.main --config path/to/config.yaml
```

覆盖字段：

```bash
python -m src.main \
  --config path/to/config.yaml \
  --override data_root=/path/to/data flat_eval=false
```

主要 launcher：

```bash
bash config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction/run_exp_E_support_direction.sh
bash config_exps_paper1_PAC/exp_F_imagenetr_r16_t20_forked_taskwise/run_forked_taskwise_sam.sh
bash config_exps_paper1_PAC/exp_C_cifar10_task_conditioned/run_cifar10_task_conditioned.sh
bash config_exps_paper1_PAC/exp_D_taskwise_sam_trajectory/run_taskwise_sam_trajectory.sh
```

建议显式提供 `PYTHON_BIN` 与 GPU 列表，避免 launcher 自动选择到缺少 PyTorch 的环境。

## 8. 已知配置与文稿漂移

| 风险 | 影响 | 处理 |
|---|---|---|
| 当前 YAML 与旧日志的 LR/epoch 不同 | 无法用当前文件名复述历史实验 | 以日志头或 run 内 YAML snapshot 为准 |
| Het5 顺序在文稿与代码中不同 | benchmark 定义变化 | 报告 `task_splits` 和实际 label ranges |
| 论文写 `DeltaW=AB`、代码是 `B@A` | rescaling 公式可能写反 | 统一 code convention |
| OOD 当前 YAML rank 8、结果名写 rank16 | 图的配置不确定 | 追查图对应原始日志，不按文件名推断 |
| Exp E/F `flat_eval=false` | sharpness summary 为 NaN | 不把 NaN 解释成零；另跑 post-hoc eval |
| 当前论文引用的主 flatness 表处于注释状态 | 文稿有 dangling table reference | 恢复真实表或删去引用，禁止引用 placeholder 数值 |
| 旧 rebuttal Het5 表有缺失，当前论文表已补值 | 两代表不一致 | 最新论文表用于现状，旧表保留为历史证据 |
| 同一 `rho` 在不同坐标支撑上不等价 | support comparison 非 norm-equalized | 明确 fixed coordinate budget，必要时增加 `||delta DeltaW||` 对齐实验 |
| Aircraft `olora_inr_rwp_t20c10_r16.yaml` 含 `olora_lamda_1=0.5` | 当前文件不能被 `yaml.safe_load` 解析 | 改为合法的 `key: value` 后再运行，并保留旧结果配置快照 |
| 5-dataset/Het5 launcher 的数据集或方法列表与注释不一致 | 批量任务数和日志目录可能错误 | 运行前打印 resolved job matrix；详细核对见根目录 `5dataset-wiki.md` |
