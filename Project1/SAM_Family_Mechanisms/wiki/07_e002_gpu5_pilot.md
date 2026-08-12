# 07｜E002-P：Two-Moons GPU 5 pilot

## 1. 判定先行

E002-P 已真实运行并生成完整产物，但状态是 **pilot calibration completed；formal E002 not ready**。

通过的门槛：物理 GPU 5 身份、内部方法/退化夹具、完整 Hessian 对称与特征重构、协方差 PSD、主/最小学习率的一步 Taylor 精度。未通过的门槛：seed 3408 的预注册 SGD 相对训练损失降幅，以及 4/140 条协方差记录的 CI 精度。因此 `engineering_passed=false`、`formal_e002_ready=false`；不能为了把状态改成 passed 而事后放宽阈值。

## 2. 运行身份与协议

- 日期：2026-08-12；
- 输出：`outputs/e002_gpu5_pilot`；
- 代码指纹：`516705290632608a06172beba67fda62c4cc2fe3b45028113c25761364af6b5a`；
- Python 3.11.15，PyTorch 2.4.1+cu121，CUDA 12.1，NumPy 2.2.6，float64；
- 物理 GPU 5：RTX 6000 Ada，UUID `GPU-6eac7f06-8173-e1a3-c938-239f6f6eb19e`；进程内映射为 `cuda:0`；
- deterministic algorithms 与 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 已启用；
- 耗时 423.0 秒，实验自身 peak allocated 85,664,256 bytes、peak reserved 96,468,992 bytes；
- 启动时 GPU 5 已有其他任务占用 12,276 MiB，运行结束时总占用 12,937 MiB 且利用率 100%。墙钟时间不能用于方法效率比较。

数据为 Two Moons train/validation/test $512/1024/4096$，噪声 .15，训练标签单独翻转 10%。模型是 82 参数的 $2\to16\to2$ tanh MLP。种子 3407/3408，各训练 800 步，batch 32，plain SGD 学习率 .05；shared checkpoint 为 step 80/240/480/720/800，每点使用同一组 64 个 paired probe batch。

### 两种证据必须分开

1. `shared_anchor`：在 SGD checkpoint 固定 $w$，比较同一 batch 上的虚拟方向、协方差与真实一步 full-train 损失变化。它回答局部描述，不回答 basin selection。
2. `on_policy`：各方法从同一初始化、同一 batch stream 自己训练，只作有限性与轨迹 sanity。它们的参数点和实际计算量已经不同，端点差异不能由 shared 协方差作因果归因。

GAM 只有 `gam_exact_hvp_same_batch_alpha1` shared reference，没有 on-policy 训练。LookSAM shared 的 `age0..4` 是同一当前 probe batch、旧 SGD anchor 上构造 cache 的 counterfactual staleness，不是 on-policy cache 分布。时间表是在 fixed balanced diagnostic subset 上沿 SGD history 重算对象，不能称为随机训练 cache 的直接测量。

## 3. 工程与数值门槛

| 门槛 | 结果 | 判定 |
| --- | ---: | --- |
| GPU 5 identity | UUID 精确匹配 | passed |
| 内部 method fidelity | 56/56 检查通过 | passed |
| Hessian symmetry | 10/10 relative error 0 | passed |
| eigensystem reconstruction | 最坏 $3.65\times10^{-15}$ | passed |
| covariance PSD | 140/140 在容差内 | passed |
| $\eta=.05$ Taylor | 最坏 row median .00303；最坏 p90 .00798 | passed |
| $\eta=.00625$ Taylor | 最坏 row median $6.88\times10^{-5}$；最坏 p90 $4.18\times10^{-4}$ | passed |
| SGD training smoke | seed 3408 loss ratio .829，高于 .7 | failed |
| covariance CI | 4/140 相对半宽 $>.25$ | underpowered subset |

SGD smoke 的失败不是发散：seed 3408 的 train CE 从 .5343 降至 .4431，test accuracy 从 .7632 升至 .8965。它暴露的是“最终/初始 < .7”对初始损失敏感，正式协议应同时预注册绝对收敛和相对降幅，但不能追溯性更改本次判定。

在预注册的 step-240 两个 SGD anchor 上，最小半径 $\rho=.01$ 的 full-train SAM finite-difference HVP 相对误差为 .00781/.00586，cosine 为 .999990/.999985，达到门槛；$\rho=.05$ 的相对误差增至 .0387/.0295，说明实际主半径仍很对齐，但不可把有限半径修正当作精确 $H\hat g$。

## 4. 观测结果

### 4.1 非凸 Hessian 与 Taylor 可解释区间

10 个 full-train Hessian 全部不定：负特征值 25–33 个、近零模态 17 个；$\lambda_{\min}$ 位于 -.0544 到 -.0342，$\lambda_{\max}$ 位于 .7697 到 1.1343。两条 SGD anchor 的 $\lambda_{\max}$ 从 .803/.918 增至 .942/1.134，因此不能写成“训练使 Hessian 变平”。

在所有 shared method×checkpoint 行中（每行跨 64 个 probes），$\eta=.05$ 的 component-normalized Taylor row-median 中位数为 .00113，最坏 p90 仅 .00798。由此可以解释本 pilot 的**一步均值分解**；不能外推到连续多步累计误差。

正曲率随机项在这些 anchor 上占主导；负曲率随机项相对 $H_+$ 项很小。但这只是 SGD anchors 的局部描述，不能说负曲率对非凸训练普遍不重要。

### 4.2 更新协方差

相对同一 anchor 的 SGD，10 个 anchor 上原始 $\operatorname{Tr}(H_+\Sigma)$ 比率为：SAM 1.282–1.376、GAM-exact reference 1.292–1.400、MS-SAM-k2 1.284–1.379、LB path mean 1.209–1.277。总方差 $\operatorname{Tr}(\Sigma)$ 也同步放大；归一化 NHA 只提高约 .7%–3.1%。所以支持的表述是：

> 在本设置的 shared SGD anchors 和所用 64 个 probes 中，几种 sharpness-aware 虚拟方向的 raw curvature-weighted variance 点估计高于 SGD，而主要差异来自整体方差尺度，不是显著更强的归一化 Hessian 对齐。

不能写成“噪声更小”“协方差导致更好泛化”或方法排名。64 个 probe draws 都条件于同一个有限训练集，同一批 probes 又在方法/检查点间复用，所以这些结果相关；当前只对单方法量做边际 bootstrap，没有直接估计 paired contrast CI。bootstrap RNG seed 还包含 method 名，因此完全相同的重复控制也可能由 Monte Carlo 误差得到不同的 CI 和 `underpowered` flag；.25 门槛对这一随机性敏感。4 条 underpowered 记录分别是 seed 3408 的 step 240 LookSAM-age1、step 480 age2/age4，以及 step 800 SGD。formal 必须共用 resample indices 并直接报告 paired contrast。

### 4.3 Lookbehind 路径

$k=2$ 路径的 mean pairwise misalignment 只有 $4.46\times10^{-4}$ 到 $7.72\times10^{-4}$；最后梯度的 batch trace 是路径平均的 1.0597–1.0739 倍（以最后梯度为分母，平均值降低约 5.6%–6.9%）。LB 与 matched-SAM 的 $\operatorname{Tr}(H_+\Sigma)$ 相对差只有 .056%–.144%。这支持“在 $k=2,\rho=.05$ fixed-budget 的窄设置中路径新颖性弱”。

但 $\alpha_{\mathrm{LB}}=.5=1/k$ 使 faithful plain-SGD direction 与 path mean 解析相等，二者重复不是独立经验证据。不能推广到 $k=5$、fixed-step、其他 $\alpha$、momentum 或 weight decay。

### 4.4 LookSAM 时间诊断

在同一 fixed diagnostic subset、沿 SGD history 重算时，orthogonal correction 的 lag-5 median cosine 为 .9865/.9930，相对 drift 为 .197/.143；lag-20 cosine 为 .9675/.9748，drift 为 .322/.243。在 lag $\le20$ 的这项固定数据诊断中，correction 角度 cosine 保持较高，但缓存幅度漂移不可忽略。

这不是实际 stochastic LookSAM cache 的自相关，也没有 random/shuffled/EMA 对照，因而不能声称时间复用带来性能收益。shared age 非刷新方向还会将 correction 重标为 $.7\lVert g\rVert$；其 correction norm 是 fresh SAM 的约 5.1–13.7 倍，所以 covariance 不能直接当作同强度比较。

### 4.5 On-policy endpoint 只作探索性 sanity

| 方法 | seed 3407 test acc | seed 3408 test acc |
| --- | ---: | ---: |
| SGD | .8833 | .8965 |
| SAM | .8787 | .8928 |
| MS-SAM-k2 | .8787 | .8928 |
| Lookbehind-k2, $\alpha=.5$ | .8789 | .8936 |
| LookSAM-5, $\alpha=.7$ | .8567 | .8687 |
| SAM-5 | .8826 | .8962 |

全部轨迹有限。在这一固定超参数 pilot 中，LookSAM 的 train fitting 较弱且 $\alpha=.7$ 未调参，因而其 test 差距不是独立的泛化差距。validation 没有用于调参，test trajectory 仅作 exploratory report；又因只有两种子且方法非等计算量，不能据此发布优化器排名或说 cache 机制导致下降。GAM 不在此表，因为没有运行 GAM on-policy。

### 4.6 Signed-spectrum 与作图限制

`signed_spectral_transfer.csv` 每个 method-anchor 都 mask 17/82 个分母近零的模态，与二分类 softmax 的 common-logit gauge 零空间一致。其他分母虽未过 mask 仍可很小，因此不解读 transfer ratio 极值。`signed_spectrum.png` 实际画的是 signed correction projection versus eigenvalue，只展示 seed 3407 最后 anchor 的四个核心方法；它不是 ratio 图，也不是谱演化证据。

`looksam_temporal.png` 只展示 seed 3407，且 $[-1,1]$ 的 cosine 纵轴会压缩高相似区间；`temporal_reuse.csv` 才是两种子的权威数值表。`on_policy_trajectories.png` 平均了两个异质种子且没有误差带，端点表和原始 CSV 优先于图上均值。

## 5. 产物契约

- 运行身份：`manifest.json`、`integrity.json`、`resolved_config.json`；
- 原始数据：`data.npz`、`batch_indices.npz`、`batch_probes.npz`、`initial_state.pt`；
- checkpoints：`checkpoints/shared/` 与 `checkpoints/on_policy/`；
- 表：`training_history.csv`、`method_fidelity.csv`、`hessian_spectrum.csv`、`covariance_summary.csv`、`taylor_summary.csv`、`signed_spectral_transfer.csv`、`path_metrics.csv`、`temporal_reuse.csv`、`endpoint_summary.csv`；
- 图：`shared_taylor.png`、`covariance_curvature.png`、`signed_spectrum.png`、`lookbehind_path.png`、`looksam_temporal.png`、`on_policy_trajectories.png`；
- 汇总：`metrics.json`。

输出目录除 Matplotlib cache 外共 93 个文件；`integrity.json` 为其余 92 个文件记录 bytes 与 SHA-256。CSV/JSON 均通过 finite serialization；未定义 signed transfer 使用空值而不是 NaN。

## 6. 下一步 gate

1. 将 underpowered 的 probe 扩至 $M=128$，并对方法相对 SGD 的 paired contrast 直接 bootstrap；
2. 补 LookSAM save/resume delta、外部 reference one-step delta，以及逐方法实际 reverse/forward/sample/wall/memory 计数；
3. 加 random/shuffled/EMA orthogonal cache 与 centered-noise rotate/whiten 干预；
4. 扫 $k,\rho,\alpha$，把 $\operatorname{Tr}(\Sigma)$ 或 correction norm 匹配后再解释 NHA；
5. 预注册不依赖初始损失的绝对训练 smoke，并保留本次相对门槛失败记录；
6. formal E002 至少增加随机种子与 loss-aligned checkpoint；通过后才考虑 E003。

## 7. 文献语义锚点

- GAM：[Zhang et al., *Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization*](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Gradient_Norm_Aware_Minimization_Seeks_First-Order_Flatness_and_Improves_Generalization_CVPR_2023_paper.html), CVPR 2023；
- Lookbehind-SAM：[Mordido et al., *Lookbehind-SAM: k Steps Back, 1 Step Forward*](https://proceedings.mlr.press/v235/mordido24a.html), ICML 2024；
- LookSAM：[Liu et al., *Towards Efficient and Scalable Sharpness-Aware Minimization*](https://arxiv.org/abs/2203.02714), CVPR 2022。

这些引用只固定算法语义；本页数值来自本仓库 E002-P，不是论文复现实验。
