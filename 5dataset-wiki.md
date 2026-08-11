# 5-Dataset / Het5 实验设置 Wiki

> 核对日期：2026-08-06  
> 核对范围：`config_exps_paper1_PAC`、`logs_exp1_rebuttal_5datasets`、`logs_exp1_rebuttal_het5`、当前数据加载代码和批量启动脚本。  
> 说明：本文把“当前 YAML/代码定义的设置”和“2026-03 日志中实际运行过的设置”分开记录。二者已经发生漂移，不能混为同一组实验。

## 1. 实验的两种含义

| 设置 | 数据流 | 当前实际任务数 | 目的 |
|---|---|---:|---|
| 5 个数据集分别训练 | CUB200、Aircraft、Cars196、Flowers102、Oxford-IIIT Pet 各自形成一条 class-incremental 流 | 10、10、10、10、9 | 比较方法在不同细粒度数据集上的表现 |
| `het5` 联合流 | CUB200 → Aircraft → Cars196 → Flowers102 → Oxford-IIIT Pet | 5 | 一个模型依次学习 5 个异构数据集，每个数据集是一个 task |

两套设置使用相同的 5 个原始数据集，但含义不同：前者训练 5 个独立模型/实验流；后者只训练一条跨数据集的连续学习流。

## 2. 相关文件

### 配置

- 独立数据集：
  - `config_exps_paper1_PAC/exp1_rebuttel_cub200/`
  - `config_exps_paper1_PAC/exp1_rebuttel_aircraft/`
  - `config_exps_paper1_PAC/exp1_rebuttel_cars196/`
  - `config_exps_paper1_PAC/exp1_rebuttel_flower/`
  - `config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/`
- 联合流：`config_exps_paper1_PAC/exp1_rebuttel_5datacombine/`

### 日志

- 独立数据集：`logs_exp1_rebuttal_5datasets/`
- 联合流：`logs_exp1_rebuttal_het5/`

### 入口与启动脚本

- 单实验入口：`python -m src.main --config=<yaml>`
- 独立数据集批量脚本：`scripts/exp1_rebuttel_5datasets/run_all_methods_8gpu.sh`
- Het5 批量脚本：`scripts/exp1_rebuttel_5datacombine/run_het5.sh`

## 3. 数据根目录如何解析

数据集名称通过 `utils/data_manager.py::_get_idata()` 映射到 `utils/data.py` 中的具体 loader。根目录候选由 `utils/data.py::_candidate_data_roots()` 生成。

### 3.1 根目录优先级

1. 配置或命令行 override 中的 `data_root`、`dataset_root`、`datasets_root`、`data_dir`、`dataset_dir`。
2. 环境变量 `FLATNESS_CV_DATA_ROOT`、`DATA_ROOT`、`DATASET_ROOT`。
3. 代码默认目录 `/data/140-0/datasets`。
4. 仓库内的 `data/`。

需要特别注意：`DataManager` 在没有 `data_root` 时会先注入 `/data/140-0/datasets`，因此它会出现在环境变量之前；loader 会按顺序选择第一个包含目标数据集结构的根目录。

### 3.2 当前机器上的真实位置

核对时，5 个数据集均位于：

```text
/data/140-1/datasets
├── cub200/{train,test}
├── cars196/{cars_train,cars_test,devkit}
├── fgvc-aircraft-2013b/data
├── flowers-102
└── oxford-iiit-pet
```

代码默认的 `/data/140-0/datasets` 当前不存在，当前 shell 也没有设置上述三个数据环境变量。因此直接执行现有批量脚本会找不到数据。复现前至少应执行：

```bash
export FLATNESS_CV_DATA_ROOT=/data/140-1/datasets
```

单实验可以更明确地强制指定：

```bash
python -m src.main \
  --config=config_exps_paper1_PAC/exp1_rebuttel_5datacombine/inclora_inr_sgd_het5_r16.yaml \
  --override data_root=/data/140-1/datasets
```

当前非实验 Python 环境中没有 `torch`，运行前还需要切换到包含 PyTorch、torchvision、timm、PyYAML 和 scipy 的实验环境。Flowers102 及 Cars196 raw-layout loader 都依赖 scipy。

## 4. 每个数据集的加载方式

| 数据集 | YAML 名称 | 类数 | 当前查找路径（相对 data root） | train/test 定义 |
|---|---|---:|---|---|
| CUB200 | `cub200` | 200 | `cub/{train,test}`、`cub200/{train,test}` 或 `cub-200/{train,test}` | torchvision `ImageFolder` |
| FGVC-Aircraft | `aircraft` | 100 | `fgvc-aircraft-2013b/data` | `images_variant_train.txt + images_variant_val.txt` 作为 train；`images_variant_test.txt` 作为 test |
| Stanford Cars | `cars196` | 196 | 优先 `cars196/{train,test}` 等 ImageFolder 布局；否则使用 `cars196/{cars_train,cars_test,devkit}` raw 布局 | 当前机器命中 raw 布局，使用 `.mat` 标注，标签从 1-based 转为 0-based |
| Flowers102 | `flowers` | 102 | `flowers-102` | `trnid + valid` 作为 train，`tstid` 作为 test；标签来自 `imagelabels.mat` 并转为 0-based |
| Oxford-IIIT Pet | `pets` | 37 | `oxford-iiit-pet` | `annotations/trainval.txt` 作为 train，`annotations/test.txt` 作为 test |

Aircraft loader 会从 train/val/test 三个 split 共同建立全局 `label_map`，避免不同 split 中同一类别被映射为不同标签。

### 4.1 图像预处理

- CUB200、Flowers102、Oxford Pet：train 为 bicubic resize 到 `256×256`、random crop `224×224`、random horizontal flip；test 为 resize 后 center crop；最后使用 ImageNet mean/std normalize。
- Aircraft：几何增强相同，但先通过 torchvision tensor loader 读取，再 `ConvertImageDtype(float32)` 和 ImageNet normalize。
- Cars196：短边 resize 到 224、center crop 224；train 增加 horizontal flip；配置中的 normalize 为 mean `(0,0,0)`、std `(1,1,1)`，即不改变数值范围。
- `het5` 合并后统一采用 `iHet5Datasets` 的 CUB transform。因此 het5 中包括 Cars196 和 Aircraft 在内的所有图片，实际都走同一套 PIL + CUB/ImageNet-normalized 预处理，而不是各自独立 loader 的 transform。

## 5. Task 划分

`nb_tasks` 并不完全由 YAML 中的同名字段决定。普通数据集由 `DataManager` 根据类别总数、`init_cls` 和 `increment` 重新计算；`het5` 则直接采用 loader 暴露的 `task_splits`。

### 5.1 当前独立数据集设置

下表是当前 4×4 LoRA 主矩阵 YAML 的设置。

| 数据集 | `class_shuffle` | `init_cls` | `increment` | DataManager 实际 increments | 当前主设置 LR | 当前主设置 epoch |
|---|---:|---:|---:|---|---:|---:|
| CUB200 | false | 20 | 20 | `[20] × 10` | 0.01 | 40/task |
| Aircraft | false | 10 | 10 | `[10] × 10` | 0.05 | 40/task |
| Cars196 | false | 16 | 20 | `[16, 20, 20, 20, 20, 20, 20, 20, 20, 20]` | 0.05 | 40/task |
| Flowers102 | false | 12 | 10 | `[12, 10, 10, 10, 10, 10, 10, 10, 10, 10]` | 0.0025 | 40/task |
| Oxford Pet | false | 5 | 4 | `[5, 4, 4, 4, 4, 4, 4, 4, 4]` | 0.0025 | 40/task |

当前 YAML 中有三个重要例外：

- Cars196 `olora_inr_rwp_t20c10_r16.yaml` 使用 LR 0.02、80 epoch。
- Aircraft `olora_inr_gam_t20c10_r16.yaml` 使用 LR 0.02、40 epoch。
- Aircraft `olora_inr_rwp_t20c10_r16.yaml` 当前不是合法 YAML：`olora_lamda_1=0.5` 应为 YAML 的 `key: value` 形式。它会在训练开始前由 `yaml.safe_load()` 报错；文件内其余字段写的是 LR 0.025、80 epoch，但目前不会生效。

### 5.2 当前 Het5 设置

当前 `utils/data.py::_HET5_REGISTRY` 的真实顺序与标签空间为：

| Task | 数据集 | 类数 | 全局标签范围 | offset |
|---:|---|---:|---|---:|
| 0 | CUB200 | 200 | 0–199 | 0 |
| 1 | Aircraft | 100 | 200–299 | 200 |
| 2 | Cars196 | 196 | 300–495 | 300 |
| 3 | Flowers102 | 102 | 496–597 | 496 |
| 4 | Oxford Pet | 37 | 598–634 | 598 |

总类别数为 635，`task_splits=[200,100,196,102,37]`，标签在拼接前先转为 `int64` 再加 offset。Het5 YAML 中的 `init_cls: 200`、`increment: 1` 主要用于兼容公共入口；实际 task 大小由 `task_splits` 覆盖，绝不是每次只增加 1 类。

仓库部分 rebuttal 文档写成 `Aircraft → Cars196 → CUB200 → Flowers → OxfordPet`，但当前 loader、启动脚本注释和 2026-03 日志中的 `Learning on 0-200, 200-300, ...` 一致表明，实际运行顺序是 `CUB200 → Aircraft → Cars196 → Flowers102 → OxfordPet`。复现现有日志时应以后者为准。

## 6. 模型与训练公共设置

参考日志的主实验矩阵为：

- 方法：SeqLoRA、IncLoRA、OLoRA、InfLoRA。
- 优化器策略：SGD、SAM、RWP、GAM。
- 独立数据集理论任务数：`5 datasets × 4 methods × 4 optimizers = 80`。
- Het5 理论任务数：`4 methods × 4 optimizers = 16`。

配置目录中还存在 SDLoRA、C-Flat、fine-tune 和 linear-probe 配置，但它们不属于上述两组参考日志的完整主矩阵。

共同训练参数：

| 参数 | 设置 |
|---|---|
| Backbone | `vit_base_patch16_224`，timm pretrained ViT-B/16，输出维度 768 |
| 日志中解析到的权重 | `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` |
| Backbone 更新方式 | 冻结 base ViT；LoRA 注入每个 block 的 attention Q/V |
| LoRA rank | 16 |
| Batch size | 128 |
| Seed | 0 |
| Scheduler | cosine |
| Base optimizer | SGD |
| Momentum | 0 |
| Weight decay | 0 |
| Replay memory | `memory_size=0`、`memory_per_class=0`，无 replay |
| 模式 | `all_or_inc=inc` |

YAML 中的 `optimizer: sgd` 表示底层更新器；`optimizer_type` 才选择 `sgd/sam/rwp/gam` 的训练分支。

### 6.1 优化器/扰动参数

- SGD：普通 SGD。
- SAM：主矩阵通常为 `sam_rho=0.05`、`sam_adaptive=false`。
- RWP：主矩阵通常为 `rwp_std=0.01`、`rwp_eta=0.1`、`rwp_beta=0.99`、`rwp_std_follow_lr=true`、`rwp_noise_type=Gauss_standard`。Aircraft SeqLoRA-RWP 当前配置把 `rwp_std` 设为 0.05。
- GAM：如果 YAML 没显式写入，四个 LoRA 实现当前默认使用 `gam_grad_rho=0.2`、`gam_grad_norm_rho=0.2`、`gam_grad_gamma=0.1`。部分当前 YAML 显式覆盖：
  - Aircraft SeqLoRA：0.01 / 0.1 / 0.02。
  - Aircraft OLoRA、InfLoRA：0.01 / 0.1 / 0.03。
  - Cars196 SeqLoRA、IncLoRA、OLoRA：0.05 / 0.2 / 0.03。
  - Cars196 InfLoRA：0.01 / 0.1 / 0.03。
  - Flowers IncLoRA、OLoRA、InfLoRA：0.05 / 0.2 / 0.03。

由于 GAM 参数在数据集和方法之间并不完全统一，报告结果时不能只写“GAM”，还应记录有效的三个超参数。

## 7. 2026-03 日志中的实际运行设置

以下内容来自每个日志开头由 `trainer.py` 打印的最终参数，而不是根据当前 YAML 或日志目录名推测。日志目录名并不可靠：例如 `logs_exp1_rebuttal_5datasets/cub200_inflora_gam/gpu7.log` 的日志头实际写的是 `dataset: aircraft`、`config: .../exp1_rebuttel_aircraft/...`。

| 日志数据集 | 日志中的 split | LR | epoch/task | `flat_eval` | 与当前 YAML 的主要差异 |
|---|---|---:|---:|---:|---|
| CUB200 | 20 + 20，10 tasks | 0.01 | 20 | true | 当前改为 40 epoch 且 `flat_eval=false` |
| Aircraft | 10 + 10，10 tasks | 0.05 | 30 | false | 当前主配置改为 40 epoch |
| Cars196 | 16 + 20，10 tasks | 0.05 | 20 | false | 当前主配置改为 40 epoch |
| Flowers102 | 日志同时存在旧的 10+10 和修正后的 12+10 | 0.005 | 10 | false | 当前改为 12+10、LR 0.0025、40 epoch |
| Oxford Pet | 大部分为 5+4、9 tasks；有一个 InfLoRA-SGD 日志为旧的 4+4、10 tasks | 0.005 | 10 | false | 当前改为 5+4、LR 0.0025、40 epoch |
| Het5 | loader 实际为 `[200,100,196,102,37]` | 0.02 | 30 | false | 当前 SeqLoRA/IncLoRA 仍为 30；OLoRA/InfLoRA YAML 已改为 40 |

Het5 日志中还存在一次 SeqLoRA-GAM、LR 0.05 的额外尝试；标准记录为 LR 0.02。由此可见，已有日志不能直接用当前 YAML 文件名反推超参数，汇总结果时应优先读取日志头部或对应的 consolidated metrics JSON。

## 8. 评估与输出

当前主矩阵 YAML 中：

- `flat_eval=false`
- `feature_flat_eval=false`
- `feature_cka_eval=false`
- `feature_proto_eval=false`
- `attention_probe_eval=false`
- linear-probe softmax joint/per-task evaluation 均为 false

虽然 `flat_eval_sharpness`、`flat_eval_hessian`、`flat_eval_GGN` 和 `flat_eval_fisher` 等子开关仍为 true，但总开关 `flat_eval=false` 时不会执行这组 flatness evaluation。2026-03 的 CUB200 日志是例外，当时 `flat_eval=true`。

训练器在每个 task 后输出 CNN 和 NME 的 per-task/seen-class accuracy，并在最终阶段计算 accuracy matrix、average accuracy 和 forgetting。默认输出结构为：

```text
outputs_logs/
└── logs_inc_lora/<method>/<optimizer-tag>/<dataset>/<seed>/<prefix>/<mode>/
    ├── checkpoints/
    └── <increment>/
        ├── <prefix>_<backbone>.log
        └── <prefix>_<backbone>_cl_metrics.json
```

RWP 当前会生成类似 `rwp_None/.../rwp/` 的路径，因为这些配置没有显式设置 `rwp_range`。

## 9. 当前复现阻塞项与一致性风险

| 优先级 | 问题 | 影响 |
|---|---|---|
| P0 | 默认 data root 是不存在的 `/data/140-0/datasets`，真实数据在 `/data/140-1/datasets` | 不设置环境变量/override 会直接找不到数据 |
| P0 | 当前默认 `python` 环境缺少 `torch` | 必须先进入正确实验环境 |
| P0 | Aircraft OLoRA-RWP YAML 使用 `olora_lamda_1=0.5` | YAML 无法解析，该组合不能启动 |
| P0 | `run_all_methods_8gpu.sh` 的 `DATASETS` 注释掉 CUB，但 `DATASET_NAMES` 仍保留 CUB | 已造成部分日志目录与日志内的真实 dataset/config 错位；按当前脚本再跑时 CUB 也不会运行 |
| P1 | `run_het5.sh` 注释称运行 4 个方法，但当前 `LORA_METHODS` 只有 `sdlora` 和 `inflora` | `all` 模式实际只启动 8 个任务，不是注释中的 16 个 |
| P1 | Het5 脚本的 `_reap_done_jobs` 在 `gpu=$(wait_for_gpu)` 的 stdout 中打印完成消息 | 已造成 `logs_exp1_rebuttal_het5` 中包含换行和完成消息的异常日志文件名 |
| P1 | 当前 YAML 的 LR/epoch 与 3 月日志有系统性差异 | “复现旧结果”和“按当前配置重跑”是两套不同实验 |
| P1 | rebuttal 文档中的 Het5 顺序与当前代码/日志不一致 | 论文描述和实现可能报告成不同 benchmark |
| P2 | Flowers 旧日志混有 10+10 split，Pet 有一个 4+4 split | 汇总时必须过滤旧 split，不能按目录名盲目平均 |

## 10. 建议的复现口径

如果目标是复现 2026-03 日志：以日志头打印的参数为准，使用 CUB→Aircraft→Cars196→Flowers→Pet 的 Het5 顺序，并过滤 Flowers/Pet 的旧 split 和 Het5 的 LR 0.05 额外尝试。

如果目标是按当前配置重新实验：以第 5、6 节的当前 YAML 为准，但应先完成以下操作：

1. 指定 `/data/140-1/datasets`。
2. 激活正确的 PyTorch 环境。
3. 修正 Aircraft OLoRA-RWP YAML。
4. 修正两个批量脚本的方法/数据集列表及 Het5 stdout 污染。
5. 明确并统一论文中的 Het5 数据集顺序。
6. 为每次运行保留 YAML 快照或把完整有效配置写入结果目录，避免后续修改 YAML 后无法对应历史日志。
