# E002-P GPU 5 精简结果摘要

## 运行身份

- 实验：`E002_two_moons_shared_anchor_pilot`
- 日期：2026-08-12
- 设备：物理 GPU 5，NVIDIA RTX 6000 Ada Generation
- GPU UUID：`GPU-6eac7f06-8173-e1a3-c938-239f6f6eb19e`
- 精度：float64
- 耗时：423.0 秒
- 代码指纹：`516705290632608a06172beba67fda62c4cc2fe3b45028113c25761364af6b5a`

## 核心配置

- Two Moons train/validation/test：512/1024/4096
- 数据噪声：.15；仅训练集标签翻转：10%
- 模型：$2\to16\to2$ tanh MLP，82 参数
- seeds：3407、3408
- batch size：32；outer steps：800；learning rate：.05
- shared checkpoints：80、240、480、720、800
- 每个 checkpoint：64 个 paired probe batches
- 主半径：$\rho=.05$
- 多步路径：$k=2$ fixed-budget

## 门槛结果

| 项目 | 结果 |
| --- | --- |
| GPU 5 identity | passed |
| internal method fidelity | 56/56 passed |
| Hessian symmetry/eigen reconstruction | passed |
| covariance PSD | 140/140 passed |
| primary/minimum-$\eta$ Taylor | passed |
| SGD training smoke | failed；seed 3408 loss ratio .829 > .7 |
| covariance precision | 4/140 rows underpowered |
| `engineering_passed` | `false` |
| `formal_e002_ready` | `false` |

## 局部观测

- 10 个 shared SGD anchors 的 full-train Hessian 全部不定，负模态 25–33 个；$\lambda_{\max}$ 没有随训练一致降低。
- $\eta=.05$ 的 Taylor component-normalized error 最坏 row median 为 .00303，最坏 p90 为 .00798；这只支持本 pilot 的一步分解。
- SAM/GAM-exact/MS-SAM/LB 的 raw $\operatorname{Tr}(H_+\Sigma)$ 点估计高于 SGD，但总方差也同步放大；归一化对齐只小幅变化。
- $k=2$ fixed-budget Lookbehind 路径的 misalignment 很小；以最后梯度为分母，路径平均 trace 低约 5.6%–6.9%。
- fixed-data SGD path 上的 LookSAM orthogonal correction 在 lag 20 仍保持高 cosine，但幅度 drift 为 .322/.243，不能当作实际 stochastic cache 的直接证据。

## 解释边界

这是 2-seed 校准 pilot，不支持优化器排名、GAM on-policy 端点结论、LookSAM cache 因果结论、“协方差导致泛化”或 E003 外推。完整协议和统计限制见 [E002-P 结果页](../wiki/07_e002_gpu5_pilot.md)。

原始 CSV、PNG、NPZ 和 checkpoints 保留在本地 `outputs/e002_gpu5_pilot/`，受 `.gitignore` 保护，不随本摘要提交。
