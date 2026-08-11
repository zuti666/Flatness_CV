# EWCLoRA 实现对照说明

本文档对照当前工程中的两个 EWCLoRA 入口和论文原始代码，说明它们在算法思想、训练流程、LoRA 参数组织和工程适配上的联系与差异。

## 对照文件

当前工程实现：

- `models_LoRAbasedCL/ewclora_youyue_github.py`
- `models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py`

原始实现：

- `low-rank-cl/methods/ewclora.py`
- `low-rank-cl/models/vit_ewclora.py`
- `low-rank-cl/models/net_ewclora.py`

当前工程中还存在一个被 `ewclora_youyue_github.py` 复用的集成版：

- `models_LoRAbasedCL/ewclora.py`
- `backbone/vit_ewclora.py`
- `backbone/net_ewclora.py`

## 原始 EWCLoRA 的核心逻辑

原始代码把 EWCLoRA 写成一个独立方法和一套专用 ViT backbone：

- `low-rank-cl/methods/ewclora.py` 管理 continual learning 生命周期。
- `low-rank-cl/models/vit_ewclora.py` 在 ViT attention 的 K/V 投影上加入 LoRA。
- `low-rank-cl/models/net_ewclora.py` 封装 ViT encoder 和每个 task 的 `classifier_pool`。

核心流程如下：

1. 每个 attention 层维护两组 LoRA：
   - 历史累积分支：`lora_A_k/lora_B_k` 和 `lora_A_v/lora_B_v`
   - 当前任务新分支：`lora_new_A_k/lora_new_B_k` 和 `lora_new_A_v/lora_new_B_v`
2. 当前任务只训练 `classifier_pool[cur_task]` 和 `lora_new_*`。
3. 训练损失为当前任务交叉熵，加上过去任务 Fisher 加权的 delta-W 正则：

   ```text
   loss = CE + lambda / 2 * sum_i omega_i * (B_new_i @ A_new_i)^2
   ```

4. 每个 task 结束后，通过 hook 捕获 `delta_w_k_new` 和 `delta_w_v_new` 的梯度，估计 empirical Fisher：

   ```text
   F_i = E_batch[(d loss / d delta_W_i)^2]
   omega_i <- gamma * omega_i + F_i
   ```

5. task 结束后调用 `accumulate_and_reset_lora()`，把当前新 LoRA 合并进历史累积分支，再把新分支重新初始化。

这个实现的正则项没有显式的 `(theta - theta*)`。由于新分支在 task 开始时被 reset，原始语义等价于把新任务 delta-W 拉回 0，也就是限制新任务在过去 Fisher 重要方向上的额外改动。

## 当前实现 1：`ewclora_youyue_github.py`

这个文件是对原始代码最直接的移植版本，但已经接入当前工程的 learner/factory/backbone 体系。

### 与原始实现的联系

- 保留原始 EWCLoRA 的 delta-W Fisher 思路。
- 保留 `omega <- gamma * omega + fisher` 的重要性累积方式。
- 保留 `lambda / 2 * sum omega * delta_W^2` 的正则形式。
- 保留 K/V LoRA 命名和结构：`lora_new_A_k/lora_new_B_k`、`lora_new_A_v/lora_new_B_v`。
- 保留 task 结束后 merge current LoRA 到 accumulated LoRA，再 reset current LoRA 的逻辑。
- `backbone/vit_ewclora.py` 和 `backbone/net_ewclora.py` 基本对应原始的 `vit_ewclora.py` 和 `net_ewclora.py`，只是做了当前工程需要的兼容修改。

### 与原始实现的主要差异

- 原始入口是 `EWCLoRA(BaseLearner)`；当前 factory 实际使用的是文件底部的 `Learner` 类。
- `Learner` 继承 `models_LoRAbasedCL.ewclora.Learner`，并用 `IncrementalNet(args, True)` 接入当前框架。
- `utils/inc_net.py` 会在 `model_name == "ewclora_youyue_github"` 时加载 `backbone.net_ewclora.EWC_net`，所以它仍然使用专用 EWCLoRA backbone。
- 当前工程保留 `IncrementalNet.fc` 的扩展逻辑，但 EWCLoRA backbone 自己也维护 `classifier_pool`；实际 forward 返回 backbone 内部的 task head logits。
- 当前实现加入了当前工程的 optimizer 配置路径，例如 `sgd/adam/adamw/sam/cflat/gam` 等分支。
- Fisher 存储和计算更偏工程化：
  - 可通过 `ewc_max_batches` 限制 Fisher batch 数。
  - Fisher tensor 放到 CPU 或做 device/dtype 对齐，降低显存和 dtype 问题。
  - batch 格式同时兼容 `(_, inputs, targets)` 和 `(inputs, targets)`。

### 适合用途

`ewclora_youyue_github.py` 更适合作为“尽量复现原始 EWCLoRA 代码语义”的版本。它的结构和原始论文实现最接近，但因为使用专用 backbone，和当前工程其他 LoRA 方法的架构不完全统一。

## 当前实现 2：`ewclora_youyue_fitArchitechture.py`

这个文件不是逐行移植原始代码，而是把 EWCLoRA 的 Fisher/delta-W 正则思想改写到当前工程的 SeqLoRA 生命周期上。

### 与原始实现的联系

- 仍然在有效 LoRA 更新 `delta_W = B @ A` 空间中估计 Fisher。
- 仍然在训练新任务时用过去 Fisher 对当前 delta-W 变化加权惩罚。
- 仍然在 task 结束后用当前 task 数据计算 Fisher，并用 `gamma` 做历史累积。
- 仍然通过 hook 捕获 delta-W 的梯度，而不是直接对 A/B 参数分别做普通 EWC。

### 与原始实现的主要差异

- 继承 `SeqLoRA`，复用当前工程已有的 LoRA backbone、task lifecycle、evaluation、optimizer 构建和保存逻辑。
- 不使用 `low-rank-cl/models/net_ewclora.py` 那种专用 `classifier_pool`；它沿用当前 `IncrementalNet` 的 expandable classifier，并在训练更新任务时对 logits 做 class offset/slicing。
- 原始实现的专用 ViT 在 K/V 上加 LoRA；当前 SeqLoRA backbone 的 `_LoRA_qkv_timm_train` 在 Q/V 上加 LoRA：
  - 原始：`delta_w_k_new`、`delta_w_v_new`
  - 当前 fitArchitecture：`delta_w_q_new`、`delta_w_v_new`
- 原始实现只正则当前新分支的 `delta_W^2`；fitArchitecture 会在 task 开始时 snapshot 当前有效 delta-W，并正则：

  ```text
  0.5 * lambda * sum_i (F_past_i + eta) * (delta_W_now_i - delta_W_ref_i)^2
  ```

  这里 `delta_W_ref` 是当前 task 开始时的 delta-W reference。

- Fisher 存储从原始的 list 顺序改为 dict key：
  - key 形如 `module_name.q`、`module_name.v`
  - 好处是更稳健，不依赖 module 遍历顺序和 list zip 顺序。
- 它支持更多当前工程 optimizer 分支：
  - 支持：`sgd`、`adam`、`adamw`、`sam`、`cflat`、`gam`、`rwp`、`arwp`
  - 不支持：`flatlora`、`flatlora_full`、`faltlora`、`faltlora_full`、`mergegam`、`mergegam_lora`

### 适合用途

`ewclora_youyue_fitArchitechture.py` 更适合作为“把 EWCLoRA 思想纳入当前统一 LoRA 框架”的版本。它和 SeqLoRA、当前 optimizer、当前 evaluation 更一致，但不再是原始代码的逐结构复现。

## 核心差异表

| 维度 | 原始 `low-rank-cl` | `ewclora_youyue_github.py` | `ewclora_youyue_fitArchitechture.py` |
| --- | --- | --- | --- |
| 目标 | 论文代码的独立 EWCLoRA 实现 | 原始 EWCLoRA 的当前工程移植 | EWCLoRA 思想适配当前 SeqLoRA 架构 |
| Learner 基类 | `methods.base.BaseLearner` | 当前工程 `LoraBaseLearner`/集成 EWCLoRA runner | `SeqLoRA` |
| Backbone | 专用 `models.net_ewclora.Net` | `IncrementalNet` 包装专用 `backbone.net_ewclora.EWC_net` | 当前工程 `LoRA_ViT_timm` / `IncrementalNet` |
| LoRA 位置 | ViT attention K/V | ViT attention K/V | ViT attention Q/V |
| 历史 LoRA 处理 | merge 到 accumulated branch | merge 到 accumulated branch | 由 SeqLoRA 的保存/加载和当前 adapter lifecycle 处理 |
| 分类头 | per-task `classifier_pool` | EWCLoRA backbone 内部 `classifier_pool` | `IncrementalNet` expandable classifier |
| 正则形式 | `omega * delta_W^2` | 基本保留 | `omega * (delta_W_now - delta_W_ref)^2` |
| Fisher 容器 | 按 module 顺序的 list | list，加入工程兼容和限制 batch 选项 | 按 module 名称的 dict |
| Fisher hook | `delta_w_k_new/v_new.register_hook` | 同原始 K/V hook | `_register_delta_hook` 控制 Q/V hook |
| optimizer | 原始 build path | 当前工程部分 optimizer | 当前工程 SeqLoRA optimizer 子集 |
| 与原始数值可比性 | 基准 | 最接近 | 思想相同，但架构和 LoRA 位置不同，不能直接视为逐项复现 |

## 公式层面的对应关系

原始实现和 `ewclora_youyue_github.py`：

```text
delta_W_i = B_new_i @ A_new_i
F_i = E[(dL / d delta_W_i)^2]
omega_i <- gamma * omega_i + F_i
L = CE + lambda / 2 * sum_i omega_i * delta_W_i^2
```

`ewclora_youyue_fitArchitechture.py`：

```text
delta_W_i(t0) = task-start reference
delta_W_i(t)  = current effective LoRA update
F_i = E[(dL / d delta_W_i)^2]
omega_i <- gamma * omega_i + F_i
L = CE + lambda / 2 * sum_i (omega_i + eta) * (delta_W_i(t) - delta_W_i(t0))^2
```

因此两者都遵循“在 LoRA 有效权重 delta-W 空间中估计 Fisher 并约束重要方向”的 EWCLoRA 思想；区别在于 reference point 和 LoRA 架构。

## 使用和命名注意

- 当前 factory 中可用的名字：
  - `ewclora_youyue_github`
  - `ewclora_youyue_fitarchitechture`
  - `ewclora_youyue_fitarchitecture`
- `fitArchitechture` 文件名里 `Architechture` 有拼写历史问题；factory 已兼容正确拼写和历史拼写。
- 如果目标是复现原始论文代码语义，优先看 `ewclora_youyue_github.py`。
- 如果目标是和当前 SeqLoRA、SAM/CFlat/GAM/RWP 等实验体系保持一致，优先看 `ewclora_youyue_fitArchitechture.py`。
- 比较实验结果时要特别注明：
  - K/V LoRA vs Q/V LoRA 不同。
  - per-task classifier_pool vs expandable classifier 不同。
  - `delta_W^2` vs `(delta_W_now - delta_W_ref)^2` 不同。
  - Fisher 默认 batch 数可能不同，尤其 `fitArchitechture` 默认 `ewc_max_batches=100`。

## 推荐解读

可以把当前两个版本理解为两个层级：

1. `ewclora_youyue_github.py`：原始 EWCLoRA 的工程移植版，用来回答“我们是否保留了论文实现的核心机制”。
2. `ewclora_youyue_fitArchitechture.py`：当前代码体系下的架构适配版，用来回答“EWCLoRA 思想如何和现有 SeqLoRA/optimizer/evaluation 管线结合”。

两者不是互相替代的同一实现。前者更接近原始实现，后者更接近当前实验框架。
