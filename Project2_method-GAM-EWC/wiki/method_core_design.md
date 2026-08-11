# 方法核心思路与设计

**最后更新**: 2026-05-04  
**当前有效方法名**: `ewclora_normfisher_gam`  
**方法简称**: AS^1 + Trace-Normalized Fisher for SeqLoRA

---

## 一句话概括

在无 replay 的参数高效持续学习中，单个共享 LoRA adapter 会在新任务上持续漂移，导致旧任务遗忘。我们的方法把“学习新任务的平坦性”和“保留旧任务敏感方向”拆成两个近正交的梯度子问题：用 AS^1/GAM 控制当前任务 loss 的局部尖锐性，用 ΔW 空间的 Fisher-EWC 控制旧任务重要方向的漂移，并且把两者显式解耦后再合成最终更新方向。

---

## 问题设定

实验设置是 frozen ViT-B/16 backbone + 一个共享 LoRA adapter，任务按 class-incremental 顺序到来，`memory_size=0`，没有 replay。

LoRA 的参数是低秩因子 `A, B`，但真正改变 backbone 的量是有效更新：

```text
Delta W = B @ A
```

因此，旧任务约束不应该只看 `A` 或 `B` 的坐标，而应该落在 `Delta W` 空间。这样 Fisher 和 EWC penalty 对 LoRA 分解方式更稳定，也更接近模型功能变化本身。

---

## 核心假设

遗忘来自两类不同方向：

1. **当前任务尖锐方向**：新任务训练如果落在尖锐区域，后续小漂移就会造成性能快速变化。这个方向由当前任务 loss 的局部几何决定。
2. **旧任务敏感方向**：旧任务中对输出最重要的参数方向，应该被 Fisher 加权保护。这个方向由过去任务的 Fisher 信息决定。

实验机制统计显示，AS^1 flatness 梯度和 Fisher 梯度的 cosine 接近 0，典型范围约 `-0.03` 到 `-0.001`。所以它们不是同一个正则项的重复表达，而是分别控制稳定性-可塑性中的两个不同子空间。

---

## 设计 1：在 Delta W 空间做 Fisher-EWC

对每个任务结束时的 LoRA 有效更新保存参考点：

```text
Delta W*_t = B_t @ A_t
```

然后在后续任务上惩罚当前有效更新偏离过去参考点：

```text
L_fisher = 0.5 * lambda_ewc * sum_k (F_k + eta) * (Delta W_k - Delta W*_k)^2
```

其中 `k` 遍历所有 LoRA Q/V projection 的 `Delta W` 张量。当前实现中 ViT-B/16 的 LoRA 插在 12 个 transformer block 的 Q/V 上，因此共有 `24` 个 Delta W terms。

这个设计的关键是：Fisher、drift、penalty 全部在 `Delta W = B @ A` 空间定义，而不是在原始低秩因子坐标上定义。

---

## 设计 2：AS^1 只看当前任务，不混入 Fisher

普通做法容易把任务 loss 和 Fisher penalty 加在一起，再交给 GAM/SAM 这类 sharpness-aware optimizer。这样 GAM 的扰动方向会由：

```text
L_task + L_fisher
```

共同决定，导致当前任务 flatness 和旧任务 Fisher 保护被耦合在一起。这个耦合会让机制解释不清楚：模型变好时无法判断是 flatness 生效，还是 Fisher 生效，或者两者互相干扰。

我们的 Layer 2/Layer 3 设计显式拆开三类梯度：

```text
g_clean  = grad L_task
g_gam    = GAM perturbed gradient of L_task only
g_fisher = grad L_fisher
```

最终每个 batch 的更新方向是：

```text
g_update = g_clean + lambda_flat * (g_gam - g_clean) + g_fisher
```

这里：

- `g_gam - g_clean` 是 AS^1 flatness component，只来自当前任务 loss。
- `g_fisher` 是旧任务稳定项，只来自 Delta W Fisher penalty。
- `lambda_flat` 控制 flatness 强度，默认 `1.0`。

这个结构让“学新任务”和“保旧任务”的梯度来源可分离、可记录、可解释。

---

## 设计 3：Trace-normalized Fisher 解决尺度问题

原始 Delta W Fisher 的数值通常很小，均值大约在 `1e-6` 量级。直接使用 raw Fisher 时，即使用较大的 `ewc_lambda`，Fisher penalty 也可能只占 task loss 的极小比例，实际约束很弱。

因此 Layer 3 对 Fisher 做 trace normalization：

```text
F_tilde = F / (sum_k trace(F_k) + eps)
```

归一化后，`lambda_ewc` 不再需要像原始 EWC-LoRA 那样依赖极大的数值，而是更直接地控制 Fisher penalty 的相对强度。当前主设置使用：

```text
ewc_normalize_fisher = true
ewc_lambda = 2000
ewc_eta = 0.0
```

这也是 `ewclora_normfisher_gam` 相比 Layer 2 raw-Fisher GAM 的主要改动。

---

## 可选设计：Dual ascent

代码中保留了一个可选的自适应 Lagrangian 版本：

```text
lambda_new = clip(lambda_old + alpha * (D_fisher - delta), lambda_min, lambda_max)
```

其中：

```text
D_fisher = sum_k F_k * (Delta W_k - Delta W*_k)^2 / ||Delta W - Delta W*||^2
```

它对应的约束形式是：

```text
min L_task + lambda_flat * AS^1_Delta
s.t. D_fisher <= delta
```

当前主实验默认 `ewc_dual_ascent=false`，因为 CUB200 sweep 中固定 `ewc_lambda=2000` 已经稳定，dual ascent 没有明显额外收益。这个模块适合作为后续分析或 adaptive stability budget 的扩展。

---

## 三层方法关系

| Layer | 配置/模型名 | 核心区别 | 用途 |
|---|---|---|---|
| Layer 1 | `ewclora_youyue_fitarchitecture` | Delta W Fisher-EWC；如果配 GAM，则 GAM 看到 `L_task + Fisher` 的混合目标 | baseline / coupling 对照 |
| Layer 2 | `ewclora_youyue_fitarchitecture_gam` | 显式拆分 `g_clean`, `g_gam`, `g_fisher`；Fisher 仍是 raw scale | decoupling ablation |
| Layer 3 | `ewclora_normfisher_gam` | Layer 2 + trace-normalized Fisher + optional dual ascent | 当前主方法 |

注意：`as1_normfisher_gam` 是代码文件和历史 alias，但当前实验配置必须使用 `ewclora_normfisher_gam`。本代码库里 LoRA backbone 选择依赖 `model_name` 是否包含 `lora`；旧的 `as1_normfisher_gam` 输出会落到 `logs_inc` 而不是 `logs_inc_lora`，不应用作正式方法对比。

---

## 当前推荐配置

跨数据集主设置：

```yaml
model_name: "ewclora_normfisher_gam"
optimizer_type: "gam"
lambda_flat: 1.0
ewc_lambda: 2000
ewc_gamma: 0.9
ewc_eta: 0.0
ewc_normalize_fisher: true
ewc_dual_ascent: false
mechanism_eval: true
mechanism_grad_stats: true
```

Paper4 tuned SGD 设置额外使用：

```yaml
optimizer: "sgd"
gam_base_optimizer: "sgd"
init_lr: 0.01
lrate: 0.01
gam_grad_rho: 0.05
gam_grad_norm_rho: 0.1
ewc_max_batches: 100
optimizer_tag_override: "gam_sgd_lr001_e*_rho005_f100"
```

其中 `e*` 随数据集 epoch 设置变化，例如 CIFAR-100 用 `e30`，ImageNet-R 用 `e50`，DomainNet 用 `e5`。

---

## 机制指标

每个任务结束后记录以下机制统计：

| 指标 | 含义 | 作用 |
|---|---|---|
| `grad_cos_flat_fisher_mean` | `g_gam - g_clean` 与 `g_fisher` 的 cosine | 验证 flatness 与 Fisher 是否近正交 |
| `grad_fisher_to_clean_norm_ratio_mean` | Fisher 梯度相对 clean 梯度的强度 | 判断 Fisher 是否实际生效 |
| `grad_flat_to_clean_norm_ratio_mean` | AS^1 梯度相对 clean 梯度的强度 | 判断 flatness 对更新的影响 |
| `delta_tensors` | 有效 Fisher/Delta W 张量数 | 确认 LoRA Q/V 机制是否启用 |
| `normalized_fisher_drift` | Fisher 加权漂移 | dual ascent 和稳定性分析 |
| `ewc_penalty_value` | 当前 Fisher penalty 标量值 | 判断 penalty 尺度是否合理 |

正式分析中最重要的证据是 `grad_cos_flat_fisher_mean` 接近 0：这说明我们不是简单堆叠两个正则，而是在两个几何上不同的方向上分别控制可塑性和平稳性。

---

## 与 EWC-LoRA 的区别

EWC-LoRA 的核心是用 Fisher 保护旧任务，但它没有显式 flatness 控制，也没有把 current-task sharpness 和 old-task Fisher 分开建模。

我们的 Layer 3 相比 EWC-LoRA 增加三点：

1. **AS^1 flatness**：通过 GAM 直接控制当前任务 loss 的一阶尖锐性。
2. **梯度解耦**：GAM closure 只包含 `L_task`，Fisher 作为独立梯度加入。
3. **Trace-normalized Fisher**：让 Fisher penalty 的尺度可控，避免 raw Fisher 过小导致约束失效。

因此论文叙事不应该写成“在 EWC-LoRA 上调了一个更大的 lambda”，而应该写成：我们把 PECL 中的 adapter drift 分解为 current-task flatness control 和 past-task Fisher protection 两个近正交子问题，并在 Delta W 空间给出可实现的联合更新。

---

## 实现入口

核心代码：

| 文件 | 作用 |
|---|---|
| `models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py` | Layer 1，Delta W Fisher/EWC 基础实现 |
| `models_LoRAbasedCL/ewclora_youyue_fitArchitechture_gam.py` | Layer 2，显式 decoupled GAM + Fisher 更新 |
| `models_LoRAbasedCL/as1_normfisher_gam.py` | Layer 3，trace-normalized Fisher + optional dual ascent |
| `src/trainer.py` | 根据 `optimizer_tag_override` 和 `model_name` 写出结果路径 |

推荐结果和配置入口：

| 目的 | 路径 |
|---|---|
| CUB200 三层对照 | `config_exps_paper1_PAC/exp_paperA_three_methods/` |
| CUB200 Layer 3 lambda sweep | `config_exps_paper1_PAC/exp_paperA_newmethod/` |
| 跨数据集 Layer 3 | `config_exps_paper1_PAC/exp_paperA_cross_dataset/` |
| Paper4 tuned SGD | `config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3_sgd_tuned/` |
| 最新方法结果筛选 | `Project2_method-GAM-EWC/wiki/method_result_selection_2026-05-04.md` |

---

## 论文写法建议

可以把方法段落组织成四个小节：

1. **Adapter-space forgetting geometry**：定义 `Delta W = B @ A`，解释为什么 Fisher 和 drift 都应在 Delta W 空间计算。
2. **Decoupled AS^1 and Fisher directions**：给出 `g_update = g_clean + lambda_flat(g_gam-g_clean) + g_fisher`。
3. **Trace-normalized Fisher**：解释 raw Fisher 尺度问题和归一化后的 `lambda_ewc=2000`。
4. **Mechanism verification**：报告 `cos_flat_fisher ≈ 0`，说明两项控制不同子空间。

一句核心贡献表述：

```text
We formulate continual LoRA adaptation as a coupled geometry problem in the effective update space Delta W, and propose a decoupled update that combines current-task first-order flatness with trace-normalized past-task Fisher protection.
```

