# Project 1-3 Initialization：PGSR-LoRA

> **PGSR-LoRA: Preview-Guided Subspace Reuse for Continual LoRA**
> 状态：shared-\(W_2\) 的五分支 GPU 因果实验已跑通；同函数、不同早期轨迹得到单 checkpoint 支持，但最终性能与遗忘收益未出现。

## 1. 一句话主线

持续 LoRA 保留了最新累计模型，却通常丢弃历史任务学习过程中形成的低秩更新结构。PGSR-LoRA 用少量新任务数据，在**当前累计模型状态**上评价历史更新子空间；通过 prior-adjusted MAP 在 fresh 与历史候选间选择，并用零输出 LoRA 初始化保证在共同校准 head 给定后 predictor 不发生变化。

最准确的英文表述是：

> **State-conditioned selection among functionally identical but optimization-distinct LoRA initializations.**

直观上：

> The learner keeps what it knows, briefly looks at what comes next, and then chooses how to move.

## 2. 主张边界

这项工作的稳健贡献不是“历史 LoRA 初始化普遍减少遗忘”，而是“在有限历史候选中，根据当前状态和新任务 preview 选择早期更新几何”。论文必须主动限制以下主张：

1. 不声称 PGSR 在同一个 preview 梯度的一步训练损失上优于梯度 top-\(r\) 初始化。允许从任意 rank-\(r\) 子空间中选择时，LoRA-One 类方法具有一步最优性。
2. 不声称 preview selection 天然减少遗忘。它首先应改善 adaptation AUC、LA 和 AAA；只有旧任务损伤也下降时，才能讨论遗忘。
3. 不声称首次复用历史 LoRA 子空间。创新点是**当前累计状态上的有限候选选择、fresh abstention 与严格函数保持**。
4. 最可能成立的条件性结论是：preview 很少、任务存在结构复现时，历史有限候选库可充当统计正则化；preview 足够大时，当前梯度初始化可能追平或超过 PGSR。

## 3. 问题定义

任务 \(t\) 到来前，累计模型写为

\[
\bar W_{t-1,\ell}
=W_{0,\ell}+\sum_{s=1}^{t-1}\Delta W_{s,\ell}.
\]

标准持续 LoRA 为新任务重新初始化

\[
A_{t,\ell}^{(0)}\sim\mathcal I,\qquad B_{t,\ell}^{(0)}=0.
\]

它保持当前函数，但没有利用历史任务发现的低秩更新几何。直接复制历史 adapter 则会产生

\[
\bar W_{t-1}+\Delta W_k\neq \bar W_{t-1},
\]

从而立即改变 predictor。任务语义相似也不能保证 \(\Delta W_k\) 在新的累计状态上仍有优化价值。因此真正的问题是：

> 在不访问历史数据、不改变当前模型函数的条件下，能否用少量新任务数据判断哪个历史 LoRA 更新子空间仍是有效的初始更新空间？

## 4. 完整方法

### 4.1 历史有效更新子空间库

任务 \(k\) 学习完成后，每个 LoRA 位置的有效更新是

\[
\Delta W_{k,\ell}=c_\ell B_{k,\ell}A_{k,\ell},
\qquad c_\ell=\alpha_\ell/r.
\]

执行薄 SVD：

\[
\Delta W_{k,\ell}=U_{k,\ell}\Sigma_{k,\ell}V_{k,\ell}^{\top},
\]

仅保留右奇异子空间 \(V_{k,\ell}\in\mathbb R^{d_{in}\times r}\)。不能简单把 \(A_k^\top\) 当作有效子空间，因为 \(A_k\) 中可能有未被 \(B_k\) 使用的方向。

实现没有显式构造 \(B A\in\mathbb R^{d_{out}\times d_{in}}\)，而是使用

\[
B=Q_BR_B,\qquad A^\top=Q_AR_A,
\]

只对 \(r\times r\) 小矩阵 \(R_BR_A^\top\) 做 SVD，再恢复右子空间。这使 bank 提取适用于 ViT 的 24 个 Q/V LoRA 位置。

若有效更新的数值秩 \(q<r\)，实现只保留前 \(q\) 个非零奇异方向，再用按
task/site 固定 seed 的 fresh 正交方向补足到 \(r\) 维；当 \(B=0\) 时，补空间与
未被使用的 \(A\) 完全无关。这样既维持 LoRA 固定 rank 接口，也避免把 SVD 的任意
零奇异补空间误当作历史知识。

### 4.2 Class-balanced preview 与分类头校准

从当前任务训练数据构造弱增强、固定顺序、类别均衡的 preview 集 \(U_t\)。首版实现默认每个任务取 25 个样本。

Class-IL 中不能用随机新分类头计算 backbone 梯度，否则候选得分主要反映随机 head。当前实现先用 preview feature prototype 对新类别分类器做 feature imprinting，所有候选共享完全相同的校准头。

这里必须区分两个操作：feature imprinting 会改变扩展后的新类别 head；函数保持仅指
在该共同校准 head 固定后，切换不同零输出 LoRA 初始化不改变 logits。首版 selector
与 Task \(t>0\) 训练使用 current-task sliced CE，而 Class-IL 评测使用全部已见类别
logits；因此当前 score 直接对应 task-local adaptation，不直接覆盖旧类竞争。

### 4.3 当前累计状态上的 full-weight gradient

在新 LoRA 尚未激活、历史 LoRA 已累计生效时，临时允许 ViT 各层原始 QKV 权重计算梯度：

\[
G_{t,\ell}=\nabla_{W_\ell}\widehat L_{U_t}(\bar W_{t-1}).
\]

这些 full-weight 参数不会进入 optimizer。评分结束后立即清空梯度并恢复冻结状态。Q 和 V 通道分别使用 QKV 梯度的前 \(d\) 行与后 \(d\) 行。

### 4.4 历史候选与 fresh candidate

对每个任务生成一个新的随机正交候选 \(\mathcal V_0\)，允许模型拒绝复用历史方向。每个历史任务是一个 task-level candidate，即它在所有 LoRA 位置上的子空间集合。

候选 \(k\) 的归一化梯度能量为

\[
\widetilde E_{t,k}=
\frac{\sum_\ell \lVert G_{t,\ell}V_{k,\ell}\rVert_F^2}
{\sum_\ell \lVert G_{t,\ell}\rVert_F^2+\varepsilon}.
\]

首版所有位置使用相同 LoRA rank 和初始化尺度，因此公共的 \(c_\ell^2s_\ell^2\) 在比较中约去。后续若不同层使用不同 rank/scale，必须恢复逐层权重。

设 fresh prior 为 \(\alpha_0\)，历史候选数为 \(K\)：

\[
\pi_0=\alpha_0,\qquad
\pi_k=\frac{1-\alpha_0}{K}.
\]

超后验和 MAP 规则是

\[
\rho_{t,k}=
\frac{\pi_k\exp(\gamma\widetilde E_{t,k})}
{\sum_j\pi_j\exp(\gamma\widetilde E_{t,j})},
\]

\[
k_t^\star=\arg\max_k
\left\{\gamma\widetilde E_{t,k}+\log\pi_k\right\}.
\]

当前 pilot 使用 \(\alpha_0=0.5,\gamma=12\)。这两个量必须通过 preview size × bank size 消融确定，不能根据最终测试集调参。

首轮 pilot 暴露了一个重要的 score-scale 问题。某个历史候选要超过 fresh，至少需要

\[
\widetilde E_{history}-\widetilde E_{fresh}>\frac{\log K}{\gamma}.
\]

Task 2 中 \(K=2,\gamma=12\)，门槛约为 0.0578；实际最大能量差仅约 0.00151。因此此次 fresh fallback 几乎完全由 prior 决定。后续必须对 \(\gamma\) 做数量级校准（当前数据暗示需要数百到上千），或者用 fresh null distribution 对 energy 做标准化，不能把当前 fallback 解释成可靠的负迁移检测。

### 4.5 零输出、函数保持初始化

选择候选后设置

\[
A_{t,\ell}^{(0)}=sV_{k_t^\star,\ell}^{\top},
\qquad B_{t,\ell}^{(0)}=0.
\]

因此 \(B^{(0)}A^{(0)}=0\)，所有候选有相同 logits、loss 和 accuracy，但 \(B\) 的初始可更新空间不同。正交 \(A\) 使用

\[
s=1/\sqrt 3,
\]

其期望 Frobenius 范数与 PyTorch `nn.Linear` 的 Kaiming 初始化一致。
代数上，只要 \(B=0\)，对任意输入都严格有新 branch 的 \(\Delta W=0\)。代码另外在
feature-imprinting 之后，以第一个 preview batch 做数值检查：

\[
D_{func}=\max_x\lVert f_{after}(x)-f_{before}(x)\rVert_\infty,
\]

默认要求 \(D_{func}\le 10^{-6}\)。本次 preview 只有一个 batch（25 个样本），因此
raw check 覆盖了全部 preview，但它不是全输入空间的经验验证；全输入函数保持来自
\(B=0\) 的代数性质。

### 4.6 子空间 warmup

前 \(h\) 个优化步骤冻结当前 LoRA 的 \(A\)，具体做法是将 `A.grad` 设为
`None`：

\[
A_t^{(j)}=A_t^{(0)},\qquad 0\le j<h.
\]

此时 \(B\) 与新类别分类头仍正常更新。之后允许 \(A,B\) 联合训练。首版设置
\(h=5\)。当前源码已在 warmup 边界记录 A/B 位移，但首轮 raw run 早于该日志字段，
所以现有产物只能由最终非零位移确认 warmup 后发生过联合更新，不能直接实测前 5 步
A 位移为零。

### 4.7 与仓库 InCLoRA 的对应关系

原始 `inclora.py` 每个任务增加一个新 LoRA branch，冻结旧 branch，并在前向中累加所有历史更新。它不是物理地把 \(BA\) 写回 backbone，但函数上等价于维护累计模型。因此首版 PGSR 保留这一组织方式：

1. Task 0 按标准 InCLoRA 训练；
2. 保存当前任务 A/B checkpoint；
3. 通过 A/B 的有效更新提取并保存 \(V\)；
4. Task \(t>0\) 重建带历史 branch 的累计 backbone；
5. 校准新 head，计算 preview full-weight gradients；
6. 评分 fresh 与历史 task-level candidates；
7. 零输出初始化当前 branch；
8. 冻结 A 的 warmup 后正常联合训练；
9. 保存 A/B、head、selection diagnostics 和新子空间。

## 5. 为什么能预测第一步下降

令 \(P_{k,\ell}=V_{k,\ell}V_{k,\ell}^{\top}\)。因为 \(B=0\)，初始时

\[
\nabla_{B_{t,\ell}}\widehat L
=c_\ell s_\ell G_{t,\ell}V_{k,\ell}.
\]

对 \(B\) 做一步 SGD 后，full-weight 更新是

\[
\delta W_{t,k,\ell}
=-\eta_Bc_\ell^2s_\ell^2G_{t,\ell}P_{k,\ell}.
\]

因此最大化投影梯度能量等价于在**给定有限候选库中**选择局部一步下降保证最大的子空间。但是，如果允许自由选择任意 rank-\(r\) 子空间，最优解是当前梯度的 top-\(r\) 右奇异子空间：

\[
\max_{V^\top V=I_r}\lVert GV\rVert_F^2
=\sum_{j=1}^r\sigma_j^2(G).
\]

PGSR 的价值必须来自有限历史候选带来的低方差、多步适应和 held-out 泛化，而不是声称训练梯度一步最优。

## 6. PAC-Bayes 在论文中的角色

PAC-Bayes 用来刻画 preview-dependent selection 的复杂度，而不是替代一步下降分析。联合 prior/posterior 的 KL 可拆为

\[
\mathrm{KL}(G_{i,t}\Vert H_i)
=\mathrm{KL}(\rho_i\Vert\pi_{i-1})
+\mathbb E_{k\sim\rho_i}\mathrm{KL}(Q_{i,t}(\cdot\mid k)\Vert P_{i,k}).
\]

hard selection 的离散选择代价是

\[
\mathrm{KL}(\delta_{k_i^\star}\Vert\pi)=\log\frac1{\pi_{k_i^\star}}.
\]

均匀历史 bank 下复杂度按 \(\log K\) 增长，产生可检验预测：preview 很小时，扩大 bank 同时提高 oracle coverage 和 selection overfitting；preview 增大后，更大的 bank 才可能稳定有益。

Flatness 不作为本项目主线，只可作为 posterior perturbation 的附录分析。

## 7. 代码结构

| 文件 | 作用 |
|---|---|
| `pgsr/bank.py` | QR + 小矩阵 SVD、FP16 历史 bank、checkpoint 读取与统计 |
| `pgsr/selector.py` | fresh/历史正交补候选、归一化能量、prior/posterior、MAP |
| `pgsr/initialization.py` | Kaiming 范数匹配、`A=sVᵀ, B=0`、函数偏差 |
| `pgsr_inclora.py` | InCLoRA 生命周期、preview、head imprinting、full Q/V 梯度、warmup、日志 |
| `configs/pgsr_inclora_cifar100_gpu_pilot.yaml` | 三任务真实数据 GPU pilot |
| `scripts/run_gpu_pilot.sh` | GPU 状态检查和可复现实验入口 |
| `results/GPU_PILOT_SUMMARY.md` | 已完成 GPU pilot 的结果摘要 |
| `configs/original_idea_cifar100_gpu.yaml` | aquatic mammals → vehicles 1 → fish 的显式类序 |
| `scripts/run_original_idea_gpu.sh` | shared-\(W_2\) 前缀和五路 GPU fork 启动器 |
| `scripts/analyze_original_idea.py` | 一步、AUC、oracle、轨迹与 CL 结果聚合 |
| `results/original_idea_validation/analysis/ORIGINAL_IDEA_RESULTS.md` | 原始思路验证结果 |

仓库 `utils/factory.py` 已增加 `pgsr_lora` / `pgsr_inclora` 动态入口。模型名包含 `lora`，因此沿用仓库已有 LoRA backbone 构造路径。
`pgsr_selection_mode` 支持 `map`、`raw_history`、`fresh`、`latest`、
`random_history`、显式 `history_task` 和 `perpendicular`，用于同一 checkpoint
上的受控选择对照。
PGSR 另外覆盖了 checkpoint resume：fork 到新输出目录时复制 \(0..t\) 的完整
additive A/B 历史、分类头和诊断文件；ViT snapshot 构造也已修正为遵循显式
`task_idx`。相应回归检查位于 `tests/test_pgsr_regressions.py`。

## 8. 已完成的真实 GPU pilot

### 8.1 硬件与运行条件

- 主机：`cudahpc40.cvc.uab.es`
- GPU：物理 GPU 4，NVIDIA RTX 6000 Ada Generation，49,140 MiB
- PyTorch：2.4.1+cu121
- 数据：真实 CIFAR-100，`/data/140-1/datasets`
- Backbone：预训练 ViT-B/16
- 任务：3 个任务，每任务 5 类、2,500 个训练样本
- 类顺序：`class_shuffle=false`，固定 CIFAR-100 类别 0–14
- Replay memory：0
- LoRA：24 个 Q/V 位置，rank 4
- 训练：每任务 1 epoch，batch size 128；约 20 steps/task
- 优化：SGD，lr 0.01，momentum 0，weight decay 0，cosine scheduler
- Preview：每个新任务 25 个样本，每类 5 个
- 选择：fresh prior 0.5，\(\gamma=12\)
- Warmup：前 5 steps 冻结 A；B 与新类别分类头继续更新
- 优化目标：Task 1/2 的 selector 与训练均使用 current-task sliced CE；
  评测及训练 accuracy 使用全部已见类别 logits
- Seed：0

启动命令：

```bash
GPU=4 bash "Project1-3 INitilization/scripts/run_gpu_pilot.sh"
```

### 8.2 工程链路结果

| 检查项 | 结果 |
|---|---|
| Task 0 训练并保存 A/B | 通过 |
| 从有效 \(BA\) 提取子空间 | 通过 |
| 每任务 24 个 basis，shape `[768, 4]` | 通过 |
| FP16 bank 后最大正交误差 | \(<1.0\times10^{-4}\) |
| 新任务 head feature imprinting | 通过，每类 5 preview |
| 24 个 Q/V full-weight gradient | 通过 |
| fresh/history 统一评分 | 通过 |
| 历史复用路径 | Task 1 触发 |
| fresh fallback 路径 | Task 2 触发 |
| 所选 branch 的 preview 数值函数检查 | 两次均 \(D_{func}=0\) |
| warmup 后 A/B 联合更新 | 部分验证；最终 A/B 位移均非零，但旧产物无 step-5 边界字段 |
| checkpoint、selection、metrics 输出 | 通过 |

### 8.3 选择结果

Task 1：

| Candidate | Normalized energy | Prior | Posterior | MAP |
|---|---:|---:|---:|---:|
| fresh | 0.00503984 | 0.50 | 0.49787 | -0.63267 |
| history task 0 | 0.00575132 | 0.50 | 0.50213 | -0.62413 |

选择 `history_task_0`，但优势很小、posterior 接近均匀，不能据此宣称历史复用有效。

Task 2：

| Candidate | Normalized energy | Prior | Posterior | MAP |
|---|---:|---:|---:|---:|
| fresh | 0.00436130 | 0.50 | 0.49551 | -0.64081 |
| history task 0 | 0.00584327 | 0.25 | 0.25220 | -1.31618 |
| history task 1 | 0.00587550 | 0.25 | 0.25230 | -1.31579 |

选择 `fresh`。注意历史候选的能量其实更高，而且两个历史候选的 posterior
合计为 0.50449，略高于 fresh 的 0.49551。fresh 仅因历史 family 的 prior 被拆给
两个候选而取得单候选 MAP。这证明 fallback 代码路径可执行，但不能解释为
evidence-based abstention；后续应比较 family-level abstention 或对分数尺度进行校准。

### 8.4 训练与 CL 输出

- Task 0：训练 loss 0.866，train accuracy 69.76%，测试 94.8%。
- Task 1：20 steps，训练 loss 0.326，train accuracy 90.12%；最终 A 位移 0.00143，B 位移 0.03047。
- Task 2：20 steps，训练 loss 0.356，train accuracy 83.52%；最终 A 位移 0.000461，B 位移 0.01908。

CNN accuracy matrix：

\[
\begin{bmatrix}
94.8 & - & -\\
0.2 & 97.8 & -\\
0.0 & 94.6 & 95.4
\end{bmatrix}
\]

按本文采用的 AAA 公式（先计算每一阶段的 prefix mean，再对阶段平均），三个阶段为
94.8、49.0、63.33，因此 \(\mathrm{AAA}=69.04\)。当前仓库 JSON 中名为
`AAA` 的字段是下三角元素等权平均，值为 63.80；两者权重定义不同，论文实验前
必须统一命名。FAA 为 63.33，ACA 为 96.0；最终时刻的 average BWT 为 -49.0，
而仓库跨阶段 `ABWT` 字段为 -71.8。Task 0 最终遗忘 94.8
个百分点，Task 1 为 3.2 个百分点；这种不均匀下降与 1 epoch、无 replay、task-local
sliced CE 的 smoke 设置一致，但没有对照实验，不能作因果归因。没有同预算基线和
多 seed，因此这些值**不能用于支持 PGSR 的性能主张**。

### 8.5 已修复的环境问题

首轮使用 4 个 DataLoader workers 时，NFS 临时目录清理产生 `.nfs... resource busy` 警告；训练本身成功。当前配置已改为
`train_num_workers=0, eval_num_workers=0`，当前源码还新增了
`selection_mode` 与 warmup-boundary 字段。因此 raw run 与当前 revision 不是
字节级同一版本；下一次运行应保存 resolved config、源码或 git hash、环境信息、
启动命令与 exit code，并直接验证前 5 步 A 位移为零。启动脚本现已改为在训练前
建立时间戳 provenance 目录，并把 host、物理 GPU、`nvidia-smi`、Python/Torch/timm
版本、git 状态和 exit code 一并写入下一次运行日志；旧 raw run 不含这些新增字段。

## 9. 首轮工程 pilot 能说明什么、不能说明什么

已经能说明：

1. PGSR 可嵌入现有 InCLoRA，不需要修改 ViT LoRA 前向实现。
2. 三个任务的历史有效更新均成功提取成 task-level bank，数值秩和正交误差良好。
3. 当前累计模型上的 preview full-weight gradient 可计算。
4. fresh/history 均可参与同一 MAP 规则。
5. `A=sVᵀ, B=0` 在真实 ViT 上严格保持函数。
6. history reuse 和 fresh fallback 两条路径都能继续训练并保存下一任务 bank。

首轮 pilot 本身尚不能说明（其中一部分由第 10 节的新实验继续检验）：

1. Preview score 能预测真实一步下降或多步 AUC。
2. MAP candidate 接近 oracle candidate。
3. 历史复用优于 orthogonal fresh、latest、random history 或 LoRA-One。
4. PGSR 改善 AAA/FAA/BWT。
5. Task 2 的 fresh fallback 是正确拒绝，而不是 prior 过强。

## 10. 原始思路的 shared-\(W_2\) GPU 验证

2026-08-08 在真实 CIFAR-100、预训练 ViT-B/16 和 RTX 6000 Ada 上完成了
“Functionally Equivalent but Optimization-Inequivalent LoRA Initializations”
单 seed 诊断实验。完整机器可读结果与表格位于
results/original_idea_validation/analysis/。

### 10.1 预注册问题与任务

实验从同一累计 checkpoint \(W_2\) 分叉，只检验下面四段链条：

\[
\text{同一初始函数}
\rightarrow
\text{不同切空间/首步}
\rightarrow
\text{不同早期轨迹}
\rightarrow
\text{不同最终任务习得或历史损伤}.
\]

显式 CIFAR-100 raw class 顺序为：

| Task | Coarse group | Raw fine classes |
|---|---|---|
| T1 | aquatic mammals | 4, 30, 55, 72, 95 |
| T2 | vehicles 1 | 8, 13, 48, 58, 90 |
| T3 | fish | 1, 32, 67, 73, 91 |

三组类别两两不交叠，因此训练样本也不可能重叠。T3 的 2,500 个训练样本按类固定
拆成 preview 5/类（25）、held-out 50/类（250）和正式训练 445/类（2,225）；
三组 index 交集均为 0，且 held-out/test 不参与选择或训练。

共享前缀只训练一次 T1→T2，每任务 3 epochs。\(W_1\) 上 T1 accuracy 为
82.0%；共同 \(W_2\) 上 T1/T2 为 1.2%/93.8%。随后从该 \(W_2\) 恢复
完全相同的累计 LoRA 与分类头，分叉训练四个独立候选：

1. history0：T1 历史子空间；
2. history1：T2/latest 历史子空间；
3. fresh：固定 seed 的随机正交子空间；
4. perpendicular：与 T1/T2 历史 span 正交的负对照。

第五路 raw_history 根据历史候选的 raw \(\kappa\) 选择后再训练；它最终选择
history1，因此是 forced history1 的独立重复，不是第五个独立候选。
本 motivating experiment 故意不用 prior-adjusted MAP；同一诊断 JSON 中的
MAP 会选择 fresh，因为 history family 的 prior 被拆给两个候选。

### 10.2 严格控制

- 所有分支先用相同 preview 做相同的 feature-imprinted head 校准。
- 每个候选都用 \(A=sV^\top,B=0\)，总 \(A\) Frobenius norm 约 5.65685。
- 最大 basis 正交误差为 \(8.59\times10^{-5}\)；perpendicular 与历史 span
  的最大重叠为 \(1.50\times10^{-7}\)。
- 300 个 function-check test 样本覆盖 15 类，每类 20 个；五路
  \(300\times15\) logits 的全部 10 个 pair 都 bitwise equal，
  \(d_{\mathrm{logit}}=0\)，初始 held-out loss range 也为 0。
- 机制 probe 只更新 B，使用 SGD、lr 0.01、momentum/weight decay 0，
  head 和 A 不更新；probe 后 B 恢复误差为 0。
- 正式训练从恢复后的 step 0 重新开始，共 3 epochs/54 steps；前 5 steps
  冻结 A，但 B 与分类头正常更新。五路使用相同 seed、batch order、workers=0
  和 task-local sliced CE。

### 10.3 Score、一步与早期 AUC

\(\kappa\) 是 preview full-weight gradient 被候选右子空间捕获的归一化能量。
一步 \(\Delta L\) 为正表示 loss 下降。gain AUC 的精确定义是：在不等距
steps \(0,1,2,5,10,18,36,54\) 上，对 \(L(0)-L(s)\) 作分段线性梯形积分后
除以 54；它是 0–54 步的平均 loss reduction，不是原始 loss AUC。由于
18–36 和 36–54 区间更长，它们在该估计中的权重也更大。

| Candidate | \(\kappa\) | Preview B-only 一步 \(\Delta L\) | Held-out B-only 一步 \(\Delta L\) | Held-out gain AUC |
|---|---:|---:|---:|---:|
| history0 | 0.00553335 | 0.000699024 | -0.0000489502 | 0.114473 |
| history1/latest | 0.00565295 | 0.000714474 | -0.0000485382 | **0.114643** |
| fresh | 0.00462790 | 0.000585508 | **0.0000022736** | 0.113466 |
| perpendicular | 0.00362330 | 0.000458841 | -0.0000037842 | 0.112476 |

四个独立候选的描述性相关性为：

| Relation | Spearman |
|---|---:|
| score vs preview B-only 一步下降 | 1.0 |
| score vs independent held-out B-only 一步下降 | -0.6 |
| score vs held-out gain AUC | 1.0 |
| score vs preview gain AUC | -1.0 |

preview 上的局部 B-only 排序成立，但没有泛化到 held-out 的精确单一步。
正式训练的 held-out gain AUC 在这个 checkpoint 上又按 score 排序；
history-only raw-score 规则选择 history1，其相对四个候选的观测 regret 为 0。
但它恰好等于 trivial latest，而不是找回语义相关的 history0。主方法的
prior-MAP 实际会选择 fresh，其相对 history1 oracle 的观测 regret 为
0.00117722。

四者 held-out gain AUC 的总 spread 只有 0.00216677，history1-fresh 差为
0.00117722；约 0.113 的共同 AUC 主要是所有分支共享的适应收益，不能整体归因
于初始化。相关性也只有 \(n=4\)，只能称为描述性 checkpoint 证据。正式训练的
preview gain AUC 全为负，不能把“probe 的正一步下降”误写成“preview 多步持续改善”。

### 10.4 语义正条件实际上没有成立

32 个随机正交子空间只评分、不训练，其 \(\kappa\) mean 为 0.00520055，
empirical 5th–95th percentile range（\(n=32\)，不是 confidence interval）
为 [0.00476625, 0.00592289]。history0/history1 分别只位于随机 null 的
81.25/84.375 percentile，均未超过 95th percentile；并且

\[
\kappa_{\mathrm{fish},\mathrm{vehicles}}
=0.00565295
>
\kappa_{\mathrm{fish},\mathrm{aquatic}}
=0.00553335.
\]

所以“aquatic mammals 与 fish 语义相近”没有转化为预期的优化几何关系，
不能把本次设置称为已实现的 high-coverage positive condition。raw 规则选择
history1 而不是 history0，说明它遵循实测几何而非语义标签；但 history1 同时
就是 latest，因此本轮也没有证明优于 latest。

本次 fresh 的 \(\kappa=0.00462790\)，甚至低于全部 32 个 random-null
（null minimum 0.00464403）。由于只训练了这个低尾 fresh，没有训练
median/top random-null 或多个 fresh seeds，history 对该 fresh 的微小 AUC 优势
可能只是 fresh lottery，不能归因于历史结构。

### 10.5 轨迹分离与最终结果

四个独立候选的 24 个 LoRA site 使用 factorized 公式计算聚合
\(\lVert BA-B'A'\rVert_F\)：

| Step | Mean pairwise distance | Max pairwise distance |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0.00314888 | 0.00408353 |
| 5 | 0.00807604 | 0.0106263 |
| 18 | 0.0189175 | 0.0251268 |
| 54 | 0.0372665 | 0.0499831 |

step 0 的零距离来自所有候选 \(B=0\)，不是 A basis 相同；step 1 后非零距离
直接证明相同 predictor 已进入不同有效更新轨迹。raw-history 与 forced history1
的 held-out AUC 只差 \(6.36\times10^{-8}\)，step-54 轨迹距离仅
\(3.16\times10^{-6}\)，最终指标完全一致，说明独立分支控制具有数值复现性。

四个独立候选的最终 Class-IL matrix 都完全相同：

\[
\begin{bmatrix}
82.0 & - & -\\
1.2 & 93.8 & -\\
0.4 & 93.8 & 88.0
\end{bmatrix}.
\]

即 T3 test Class-IL=88.0%、旧任务均值=47.1%、FAA=60.7333%、final
average BWT=-40.8，四路 range 全为 0。相对共同 \(W_2\)，T1/T2
transition damage 为 -0.8/0.0 个百分点，平均 -0.4；没有任何候选改善最终
任务习得、FAA、BWT 或历史保持。T1 在进入 T3 前已从 82.0% 降到 1.2%，
也限制了继续比较其保持差异的统计空间。诊断 JSON 中 90.8–91.6% 的
task-aware 数值来自 250 个 train-heldout 样本，不是 T3 test-set 指标。

### 10.6 本轮 go/no-go 结论

| 原始问题 | 本轮结论 |
|---|---|
| 不同初始化是否保持同一函数？ | 是，300 样本五路 pairwise logits exact equal |
| 是否产生不同首步/轨迹？ | 是，step 0 为 0，step 1 起 \(BA\) 距离非零 |
| score 是否预测学习速度？ | preview 一步与 held-out 54-step gain AUC 描述性支持；held-out 精确一步和 preview 多步不支持 |
| 是否改变最终习得或历史损伤？ | 否，所有最终 Class-IL/FAA/BWT 完全相同 |
| 预设 high-coverage 条件是否成立？ | 否，history0 < history1，且两者都未超过 random-null p95 |
| 是否优于 latest/random？ | 否；选择即 latest，random-null 未训练 |

因此，最稳妥的结论是：

> 本 checkpoint 支持“函数等价但优化不等价”的机制主张；score 对早期轨迹有
> 混合但部分正向的描述性证据。完整的“早期差异传播到最终任务习得和历史保持”
> 因果链没有被验证。

本实验仍是单 seed、单 triplet、四个独立候选。它没有记录实测首步 tensor 与
解析式的相对误差，没有训练 32 个 random-null，没有低覆盖 T3 条件，也没有测量
旧任务梯度兼容性/曲率。任何普遍性能或减少遗忘的结论都不成立。

## 11. 下一阶段：把机制证据变成方法证据

本轮之后不应直接宣称方法成立，而应继续固定累计 checkpoint 做以下判别实验：

1. 至少运行 5 seeds、多个预注册 task triplets/checkpoints；增加明确的低覆盖条件，
   high/low coverage 只能由实测 \(\kappa\) 与 random-null 的关系判定。
2. 每个 checkpoint 实际训练多个 independent fresh、random-null median、top-null
   和 perpendicular，先排除本次 low-tail fresh lottery。
3. 在 2–3 个小学习率上只更新 B，直接保存并验证实测首步 tensor 与
   \(-\eta c^2s^2GVV^\top\) 的相对误差。
4. 同预算比较每个历史 task、latest、fresh、random、raw-history、prior-MAP、
   LoRA-One/current-gradient top-\(r\) 和离线 oracle。
5. 固定 batch/augmentation 计划，记录更密的 early curve；同时报告候选相对
   shared adaptation curve 的增量，而非把共同 AUC 归因初始化。
6. 用旧任务测试数据只作离线诊断，测量梯度兼容性与曲率；不得进入选择或训练。
7. 比较 task-local sliced CE、all-seen-logit CE 和 feature-imprinting 敏感性。
8. 使用能在 \(W_2\) 保留合理 T1 accuracy 的前缀或更强 CL 训练，再比较 T3
   transition damage；否则 T1 已接近地板，BWT 对照不敏感。
9. 校准 raw-history 与 prior-MAP/family-level abstention；改变 task order 与
   recurrence，不能只改变不影响类序的普通 seed。

决定方法能否成立的三项指标：

\[
\operatorname{Spearman}(\widetilde E_{t,k},\Delta L_{t,k}^{(1)})>0,
\]

\[
\operatorname{Spearman}(\widetilde E_{t,k},\mathrm{AUC}_{t,k}^{1:H})>0,
\]

\[
\mathrm{Regret}_t=\mathrm{AUC}_{oracle}-\mathrm{AUC}_{selected}
\quad\text{足够小。}
\]

如果只满足第一项而不满足第二、第三项，energy 只是局部梯度诊断，不足以支撑完整方法。

## 12. 完整研究问题与基线

| RQ | 问题 | 主要证据 |
|---|---|---|
| RQ1 | 相同 predictor 是否产生不同轨迹？ | \(D_{func}\)、候选早期 loss curves |
| RQ2 | preview score 是否有效？ | 一步/多步 Spearman、oracle regret |
| RQ3 | 何时优于当前梯度初始化？ | preview size × task recurrence |
| RQ4 | 能否避免负迁移？ | abrupt task 下 fresh fallback |
| RQ5 | 是否改善序列性能？ | AAA、FAA、LA、BWT、Forgetting |
| RQ6 | bank/preview 复杂度预测是否成立？ | bank size × preview size |
| RQ7 | 收益是否真的来自选择？ | latest、random、oracle 对照 |

必须包括的初始化基线：

1. Vanilla LoRA；
2. Orthogonal fresh LoRA；
3. Latest history；
4. Random history；
5. PGSR MAP；
6. Oracle history；
7. LoRA-GA；
8. LoRA-One；
9. SLAO 类历史基初始化；
10. 若使用 replay，可把 SLICE 作为非同资源上界，而非严格公平基线。

所有方法必须共享 rank、数据、head 初始化、preview 数量、数据顺序、优化步数、学习率搜索预算和 seed。

## 13. 指标清单

序列主指标：

- AAA（主指标）
- FAA
- LA
- BWT
- Forgetting

选择质量：

- score vs one-step decrease Spearman
- score vs held-out adaptation AUC Spearman
- oracle Hit@1 / Hit@3
- selection regret
- historical reuse / fresh fallback rate
- posterior entropy
- preview–held-out score gap

机制诊断：

- \(D_{func}\)
- 梯度能量捕获率
- warmup 边界 A/B 位移
- A 子空间 principal angles
- 达到目标准确率所需 steps
- 更新范数和优化路径长度
- 相同新任务准确率下的旧任务损伤
- bank 存储、选择时间和 FLOPs

## 14. 本次实验 GPU 状态记录

2026-08-08 11:07:39 CEST、启动 shared-\(W_2\) 实验前，在
`cudahpc40.cvc.uab.es` 实时检查到 7 张 RTX 6000 Ada（每张 49,140 MiB）。
GPU 0/2/3/4/5/6 均只有 4 MiB 占用且利用率 0%；GPU 1 有其他进程占用
6,814 MiB，因此没有使用 GPU 1。

| GPU | Used / Free MiB | Utilization |
|---:|---:|---:|
| 0 | 4 / 48,534 | 0% |
| 1 | 6,814 / 41,725 | 0% |
| 2 | 4 / 48,534 | 0% |
| 3 | 4 / 48,534 | 0% |
| 4 | 4 / 48,534 | 0% |
| 5 | 4 / 48,534 | 0% |
| 6 | 4 / 48,534 | 0% |

物理 GPU 6 训练唯一的 T1→T2 前缀；五个 T3 分支分别映射为 GPU 0=history0、
GPU 2=history1、GPU 3=fresh、GPU 4=perpendicular、GPU 5=raw_history。
全部分支退出码为 0，分析器于 11:18:54 CEST 完成。环境为 PyTorch
2.4.1+cu121、CUDA 12.1、timm 1.0.9。启动时 GPU 状态、host、git 状态、
resolved config、运行源码与各分支日志保存在
`results/original_idea_launcher/20260808_110739/`。
实验结束后复查状态与上表一致：GPU 0/2/3/4/5/6 均回到 4 MiB、0% 利用率，
说明本实验的五个分支与前缀进程已经释放；GPU 1 的 6,814 MiB 为实验外占用。

GPU 状态可用下列命令检查：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
```

## 15. 结论

现在已有两层证据：旧 pilot 证明 PGSR-InCLoRA 工程链路可执行；新的 shared-\(W_2\)
实验进一步证明，不改变初始 predictor 的不同 LoRA basis 会从第一步开始形成不同
有效更新轨迹。当前允许使用的结论是：

> 在一个固定 seed、共享 \(W_2\) checkpoint 和固定 T3 split 上，四种零输出
> LoRA 初始化产生了完全相同的初始 logits，但从第一步开始形成不同的有效
> \(BA\) 轨迹。preview 梯度能量严格排序了隔离的 B-only preview 一步下降，
> 并与 54-step held-out gain AUC 呈描述性一致排序；但 exact held-out 一步不支持，
> 历史候选没有超出随机 null，所选候选就是 latest，且所有最终 Class-IL、FAA
> 和 BWT 完全相同。

因此，本轮支持“functionally equivalent but optimization-inequivalent”这一机制，
但不支持历史子空间优于 random/latest，不支持主方法 prior-MAP 优于 oracle，也不支持
最终性能或减少遗忘。下一道 go/no-go 门槛是多 seed、多 checkpoint、多个实际训练的
fresh/random 候选：只有 raw/MAP selector 在独立 held-out AUC 上稳定接近 oracle，
并在 matched current-task accuracy 下改善历史损伤，才值得进入完整序列主实验。
