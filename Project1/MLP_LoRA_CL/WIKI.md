# MLP–LoRA 机制实验 Wiki：Factor Pullback Geometry、真实轨迹与方向性遗忘

> **2026-08-09 更新：** 已完成 Stage-1A–2D、P0/P1 正式分析、P2–P2d，以及 P3 exact-step-normalization pilot 320/320 branches。证据不支持“LoRA 普遍放大 SAM”“低维使有限差分 HVP 更准”“静态 flatness/pullback 决定遗忘”，也不支持“SAM 的每一步在等步幅下都更安全”。当前最严格的结论是：**Factor chart 同时改变有效步幅、塑性速度和单位更新方向。逐步归一化消除了 raw gauge 的运动预算差异并大幅削弱部分 progress-gauge effect，但没有消除全部 chart effect；Identity Factor-SAM 在 `−30→+30` 的相同学习进度下仍有收益，在 `+45→−45` 则不稳定。SAM 的 clean-to-perturbed 修正主要降低一阶干扰，同时提高方向曲率，净单步安全性尚未显著。** P3 formal 已冻结并等待空闲 GPU。

## 快速阅读：从现象到当前结论的完整实验逻辑

### A. 统一问题、符号和判断标准

研究问题不是“LoRA 的参数更少，为什么还需要 flatness”这一句直觉，而是依次区分四件事：

1. LoRA 是否真的比 Dense 获得更大的 SAM/GAM **相对收益**；
2. 如果只在部分任务方向成立，收益发生在旧任务留下的平坦性，还是学习新任务的真实轨迹；
3. 轨迹差异来自低维可达集合、移动的 rank manifold，还是 $BA$ 因子坐标；
4. factor chart 的效应来自方向、曲率，还是更直接的有效步幅与 plasticity speed 改变。

统一使用以下指标。所有统计先在同一 seed 内作差，再对 seed bootstrap；多个 diagnostic batches 是 seed 内重复测量，不能充当额外 seeds。

| 符号/指标 | 计算 | 回答的问题 |
|---|---|---|
| 旧任务损伤 $D$ | $L_{old}^{end}-L_{old}^{start}$ | 新任务训练实际增加了多少旧任务 loss；越小越好 |
| SAM benefit $B$ | $D_{SGD}-D_{SAM}$ | 正值表示 SAM 比 SGD 少遗忘 |
| 参数化/gauge interaction | $B_{Factor/gauge}-B_{control/identity}$ | SAM 的相对收益是否被 Factor chart 改变 |
| 一阶干扰 $I_k$ | $g_{old,k}^{T}\delta W_k$ | 当前真实更新是否与旧任务梯度冲突 |
| 方向曲率 $C_k$ | $\frac12\delta W_k^TH_{old,k}\delta W_k$ | 旧任务沿真实未来更新方向暴露了多少二阶损伤 |
| Taylor 余项 $R_k$ | $\Delta L_{old,k}-I_k-C_k$ | 局部二阶展开还遗漏多少非局部/高阶效应 |
| Pullback $\mathcal M[G]$ | $s^2(BB^TG+GA^TA)$ | factor gradient 在有效权重空间诱导的局部 preconditioner |
| Step path | $\sum_k\|\delta W_k\|_F$ | 实际运动预算；区别于可能因往返抵消而很小的 net drift |
| Current-loss matched | 在共同 current-loss crossing 插值 $D$ | 相同新任务进度下谁更安全 |
| Drift/path matched | 在共同 net drift 或累计 step path 插值 $D$ | 相同位移或运动预算下谁更安全 |

### B. 全部实验的因果链总表

| 实验 | 目的 | 核心设置 | 主指标 | 关键结果 | 分析、结论与下一步 |
|---|---|---|---|---|---|
| Stage-1A | 检查“LoRA 普遍放大 flatness optimizer”是否存在 | `64×64` middle；Dense/Random/Factor；SGD/SAM/GAM-FD；正反五任务；10 seeds，180 runs | FAA、forgetting、LoRA-minus-Dense interaction | 8 个预定义 interaction CI 全跨 0 | 普遍命题不成立；先做 lr/radius 与匹配稳健性，不进入 HVP 故事 |
| Stage-1B | 排除单一 lr/radius 和 endpoint 比较造成的假阴性 | lr `{.01,.03,.1}`、rho `{.005,.02,.05}`；pilot 240 + held-out 60 | endpoint、current-loss matched、drift matched interaction | held-out 仅 reverse 的 loss-matched interaction 稳定为正；forward 不复现 | 收益是任务顺序条件性的，不是 LoRA 通则 |
| Stage-1C | 区分旧任务 SAM imprint 与新任务 SAM trajectory | 对称 `−30↔+30`；Task-A/Task-B SGD/SAM 四种 schedule；Dense/Factor；160 runs | prospective Hessian/GGN、$I/C/R$、三种 matched damage | forward Factor 的收益主要来自 Task-B SAM 轨迹；旧任务 prospective flatness 下降但不必然少遗忘 | Flatness 是中间性质而非充分条件；研究对象转向真实未来路径 |
| Stage-1D | 检查 merge-reset 是否消除收益 | persistent 与每任务 merge-reset；40 runs | loss/drift-matched SAM benefit、merge invariance | merge-reset 没有消除收益，lifecycle difference CI 跨 0 | 问题不是简单重复使用未合并 factors |
| Stage-1E | 排除“纯低维所以 SAM 更有效” | Dense、Random-128、Fixed-LoRA-tangent、Factor；80 新 runs | matched damage、pathwise $I/C$ | 两个固定低维控制没有复现 Factor，Factor forward 为正 | 纯维度不充分；转向动态 tangent、rank manifold 与 factor geometry |
| Stage-2A | 排除 Dense/Factor 的 Task-A endpoint 不同 | 每 seed 共享精确 $W_A$；Dense/Factor Task-B；20 seeds/order，160 runs | endpoint hash/logit audit、matched interaction、$I/C$/GGN | forward Factor interaction 在共同 endpoint 后仍显著 | 现象不是旧任务起点混杂 |
| Stage-2B | 区分固定低维、移动 rank manifold 与 $BA$ 坐标 | Random、Fixed mature、Projected rank、Balanced Factor、Standard Factor；680 runs | matched damage、radius audit、path $C$ | Random/Fixed/Projected/Balanced 均未稳定复现 Standard Factor | rank manifold 本身仍不够；自由 factor chart 至少参与机制 |
| Stage-2C | 检验任务方向是否调节净收益 | 六个有向 Rotated-MNIST pairs；Dense/Factor×SGD/SAM；20 seeds，480 runs | $I,C$, GGN、gradient alignment、三种 matched benefit | `+45→−45` 中 $C$/GGN 改善但 $I$ 恶化，最终 SAM 更差 | 方向曲率下降不是充分条件；必须分析 $I+C+R$ 的净和 |
| Stage-2D | 用函数保持 intervention 检验 factor gauge | 初始 $B=0$；$c=.3,1,3$；同 $W$/logits/radius；240 runs | gauge interaction、$I/C$/GGN、匹配损伤 | forward 收益随 gauge 显著且非单调；reverse 不稳定 | 支持 factor-coordinate geometry，但 $B=0$ 是奇异初始化，需成熟状态 P1 |
| P0 | 检验真实路径项能否跨方向解释遗忘 | 重分析 Stage-2C 480 runs；六方向；direction+seed 双重留出 | path、$\Delta I$、$\Delta C$ 的 LOSO/LODO MAE、$R^2$、AUC | Factor：path-only $R^2=-.059$，加 $I$ 为 `.338`，加 $I+C$ 为 `.612` | $C$ 在 path+$I$ 后有增量信息，但仍是预测证据且 $R$ 不可忽略；进入因果 P1 |
| P1 | 在成熟非零 factors 上因果改变 chart | Dense Task A 5 epochs；Factor warmup 1 epoch；6 gauges；SGD/SAM；20 seeds/order，480 formal | 函数/tangent 审计、damage、$I/C$、pullback、condition、bilinear、LOSO | orthogonal 完全等价；非正交 gauge 改变轨迹与 damage；静态 pullback $R^2<0$ | chart 确实有因果作用，但不知道是方向还是步幅；进入动态、多批次 P2 |
| P2 | 检验 P1 的跨 batch 稳健性和 time-varying pullback 预测 | 4 个非重叠 batches；epoch 0–4 动态记录；5 seeds/order，100 runs | batch sign、static/dynamic LOSO、固定 $I+C$ | forward gauge 符号复现；dynamic pullback 两方向均未优于 static；reverse $I+C$ 失败 | 否定“动态 pullback 均值足以预测终点”；按停止规则不跑 formal，改查时间分辨率 |
| P2b | 区分公式错误与 epoch 分段过粗 | 每 10 steps checkpoint，约 32 段；Identity/Scalar `.5/2`；3 seeds，36 runs | clean-pullback+bilinear 一步误差、coarse/fine $R$ | SGD 相对误差约 $10^{-6}$；SAM 约 `1%–2.3%`；Identity reverse $|R|$ 从 `.119` 降到 `.009` | 局部公式正确；部分失败来自分段过粗，但非正交 gauge 仍有大余项 |
| P2c | 恢复真实 per-step path，检查 chord 是否隐藏振荡 | reverse 相同 seeds；保存全部 316 optimizer steps；18 replay runs | 累计 path、最大单步 norm、per-step $I/C/R$ | path：Identity `48.3`、Scalar-.5 `108.2`、Scalar-2 `71.5`；非正交单步可达 `.9–1.1` | 10-step chord 隐藏往返；某些单步本身已非局部。下一步先控制有效运动预算 |
| P2d | 检验巨大 gauge forgetting 是否主要由 path/plasticity 尺度造成 | 独立 seeds 708–712；lr：Identity `.1`、Scalar-.5 `.046`、Scalar-2 `.062`、Aniso-4 `.063`；100 runs | path ratio、raw/current-loss/path/drift-matched gauge damage、matched SAM interaction | path ratio 压到约 `.85–1.14`；所有 current-loss-matched SGD gauge-damage CI 跨 0；SAM interaction 随 path/loss matching 改变符号 | 强烈提示 raw gauge forgetting 受隐式步幅和 plasticity speed 主导，但 5-seed CI 跨 0 不是等价证明；进入逐步归一化 P3 |

### C. P0–P2d 的统一六字段实验卡片

#### P0：真实方向性 Taylor 项的预测价值

- **目的**：确认遗忘是否真的由未来任务走过的方向解释，而不是由更新范数或全局 sharpness 单独解释。
- **设置**：不增加训练；重分析 Stage-2C 的 480 runs，覆盖 Dense/Factor、六个有向任务对、20 seeds；同时留出待测 direction 和 seed。
- **指标**：SAM benefit $B$；path-length difference；$\Delta I$；$\Delta C$；Taylor residual；LODO/LOSO MAE、$R^2$、sign accuracy；共同 current-loss support 上的 damage AUC。
- **结果**：Factor 的 $R^2$ 从 path-only `−.059` 提升到 path+$I$ `.338`、path+$I+C$ `.612`；加入 $C$ 的 direction-block MAE 改善为 `.00344 [.00049,.00660]`。
- **分析**：真实方向曲率提供独立于路径长度和一阶干扰的解释力，但 p90 absolute residual 仍为 `.0373`，不能把二阶近似当成完整因果模型。
- **结论/下一步**：支持“遗忘看真实未来方向”，不支持“全局 flatness 单指标”；P1 用函数保持 intervention 检验 factor chart 的因果作用。

#### P1：成熟 Factor chart 的因果作用

- **目的**：排除 Stage-2D 的 $B=0$ 奇异初始化，判断保持 $W$、logits 和 tangent range 不变时，factor coordinate metric 是否仍改变轨迹。
- **设置**：共享 Dense Task-A endpoint；Task-B SGD warmup 1 epoch 得到 $A_*,B_*\neq0$；Identity、orthogonal、Scalar `.5/2`、Anisotropic `2/4`；SGD/SAM `.005`；两方向各 20 formal seeds。
- **指标**：weight/logit/tangent projector audit；old damage；current-loss/drift/path matching；$I,C,R$/GGN；start pullback、Jacobian condition；实际 linear/bilinear step closure；LOSO endpoint prediction。
- **结果**：Identity/orthogonal 的 endpoint 差约 $10^{-7}$；非正交 gauges 可把 SGD damage 改变 `.05–.12`；Identity 的 SAM benefit 为 forward `.00330`、reverse `.00420`；静态 start-pullback 的 $R^2$ 为 `−.022/−.369`。
- **分析**：在函数和可达 tangent range 相同的条件下，chart 确实因果改变优化；但 condition、起点 pullback、bilinear ratio 都不能单独预测终点。P1 本身还不能区分方向变化与隐式有效 learning-rate/path-scale 变化。
- **结论/下一步**：接受“factor chart 参与机制”，拒绝“静态 $JJ^T$ 已解释收益”；P2 检验 time-resolved 指标与多批次重复性。

#### P2：动态 pullback 与多 diagnostic batches

- **目的**：检验 P1 是否依赖单个 batch，以及沿轨迹记录 $\mathcal M_k$/condition 是否比起点指标更能预测 gauge damage。
- **设置**：两个任务方向、5 gauges、SGD/SAM、seeds 700–704；每任务 4 个互不重叠 128-sample batches；continuation epoch 0–4 记录动态 geometry，共 100 branches。
- **指标**：四 batch 符号一致性；static/dynamic pullback LOSO $R^2$/MAE；dynamic+condition；post-hoc bilinear；固定系数 $I+C$；orthogonal negative control。
- **结果**：forward Scalar-2 interaction `−.00576`、Scalar-.5 `+.00164`，均 4/4 batches 同号；但 dynamic pullback 的 forward/reverse SGD $R^2$ 为 `−.404/−1.051`，均不如 static `−.244/−.149`；reverse 固定 $I+C$ 也为 `−1.708`。
- **分析**：chart sensitivity 可重复，但 full-batch、epoch-mean pullback 不是终点充分统计量；reverse 的大 Taylor residual 指向非局部路径，而不是简单缺少 condition 特征。
- **结论/下一步**：P2 formal 按预注册停止；P2b 直接审计真实 minibatch 一步公式与更细路径。

#### P2b：10-step 分辨率与一步公式审计

- **目的**：判断 P2 失败来自 pullback/双线性公式错误，还是 epoch 段过长。
- **设置**：seeds 705–707；两方向；Identity/Scalar `.5/2`；SGD/SAM；每 10 steps 保存 $W$，约 32 个 path segments；36 branches、1,152 个 sampled steps。
- **指标**：$I_{actual}$ 与 clean-pullback linear prediction + actual bilinear correction 的绝对/相对误差；4-segment 与 32-segment $|R|$；endpoint closure。
- **结果**：coarse/fine endpoint loss 完全一致；SGD 最大一步误差 $2.65\times10^{-7}$、相对 MAE 约 $10^{-6}$；SAM 相对 MAE `1.2%–2.3%`。Identity reverse $|R|$ 从 `.1194/.1081` 降到 `.0089/.0084`。
- **分析**：factor pullback 的局部一步公式成立，SAM 只产生较小的 perturbed-gradient correction；但 Scalar-.5 reverse 的 fine residual 仍约 `.15`，说明 10-step chord 仍可能隐藏振荡。
- **结论/下一步**：公式不是主要失败点；P2c 保存全部真实 optimizer steps。

#### P2c：真实 per-step 路径与非局部单步

- **目的**：检查更粗 checkpoint 连接形成的直线 chord 是否把实际往返路径压短，从而人为改善 Taylor residual。
- **设置**：重放 P2b reverse seeds 705–707；相同三 gauges/两优化器；保存全部 316 步，共 18 branches、5,688 个一步审计。
- **指标**：实际累计 $\sum\|\delta W_k\|$、最大单步 norm、per-step $I/C/R$、clean prediction closure。
- **结果**：平均 path 为 Identity `48.3`、Scalar-.5 `108.2`、Scalar-2 `71.5`；平均最大单步约 `.48/1.07/.89`。Identity $|R|$ 降到 `.014–.017`，非正交 gauges 并不单调下降。
- **分析**：10-step chord 的确通过抵消隐藏大量往返运动；相同 factor lr 在不同 gauge 下对应完全不同的有效权重步幅，部分单步已不满足“足够局部”的二阶展开前提。
- **结论/下一步**：P1/P2 的 raw gauge effect 混合了 coordinate direction 与 effective step budget；P2d 用独立 seeds 做 lr/path calibration。

#### P2d：Path calibration 与 plasticity matching

- **目的**：判断 P1 中巨大的 raw gauge forgetting，在近似匹配累计有效路径和新任务进度后是否仍存在。
- **设置**：用 P2 cohort 冻结 gauge-specific lr，再在独立 seeds 708–712 验证；Identity/Orthogonal `.1`、Scalar-.5 `.046`、Scalar-2 `.062`、Aniso-4 `.063`；两方向、SGD/SAM，共 100 branches。
- **指标**：相对 Identity 的 path ratio；raw、step-path、current-loss、drift-matched gauge damage；四种对应的 SAM-benefit interaction；orthogonal audit。
- **结果**：path ratio 从原先 `1.57–2.24` 压到约 `.85–1.14`；所有非正交 current-loss-matched SGD gauge-damage CI 跨 0。Forward 的非正交 SAM interaction 在 step-path matching 后全部跨 0；reverse Scalar-.5/Aniso-4 在相同 path 下为正，而 current-loss matching 下部分 interaction 为负。
- **分析**：结果强烈提示 raw gauge forgetting 的相当一部分来自隐式步幅、振荡路径和 plasticity speed，而非“相同学习进度下 chart 必然更伤旧任务”。但 P2d 只有 5 seeds、路径仅近似匹配，CI 跨 0 不能证明方向效应等价于零。SAM 可以以不同路径效率达到相同运动预算或相同 current loss，因此两类 matching 不应混写。
- **结论/下一步**：factor chart 的剩余作用是条件性的轨迹效率/干扰修正。下一步必须逐步 normalize effective step，并增加 dense current-loss checkpoints；通过后再做 20-seed formal 和五任务外推。

### D. 假设演化、当前可写结论与不可写结论

| 假设 | 当前状态 | 证据 |
|---|---|---|
| LoRA 普遍放大 SAM/GAM | **否定为普遍命题** | Stage-1A/B 和六任务方向均不一致 |
| LoRA 低维使 FD-HVP 更准确，因此收益更大 | **无支持，未作为当前主故事** | SAM 主结果不依赖 FD；固定低维控制不复现；尚无 GAM-FD/exact 的因果中介证据 |
| SAM 先把旧任务训练得更平坦，所以未来少遗忘 | **中间现象存在，但非充分条件** | Stage-1C prospective curvature 可下降而遗忘不下降 |
| 纯低维可达空间使 SAM 有效 | **不充分** | Random/Fixed tangent、Projected rank 未复现 Standard Factor |
| Factor chart 会改变实际优化轨迹 | **因果支持** | P1/P2 orthogonal negative control 与非正交 interventions |
| 静态或 epoch-mean pullback 可预测终点遗忘 | **否定** | P1/P2 LOSO $R^2$ 为负或不稳定 |
| 真实 $I+C$ 比 path norm 更能解释遗忘 | **支持但有边界** | P0 跨方向增量预测；reverse/大步长时 $R$ 仍很大 |
| 巨大 raw gauge forgetting 表示 chart 方向本身不安全 | **不支持** | P2d 中 current-loss-matched gauge damage 全部不显著 |
| SAM 在 Factor LoRA 中仍可通过轨迹修正减少遗忘 | **条件性支持** | forward common endpoint、P0 AUC、P1 identity 及 P2d matched interactions；效应依赖方向和 estimand |

当前论文可以写成：

> **Factor LoRA 并不因为低维而免于真实有效权重路径上的干扰与曲率。其因子坐标诱导随训练变化的隐式 preconditioner，显著改变有效步幅、塑性速度和振荡路径；SAM 还会改变 factor-gradient 更新，但其在有效权重空间中的真实修正幅度、方向效用及遗忘收益必须在相同步幅和相同学习进度下分别检验。**

当前不能写成：

- “LoRA 普遍比 Dense 更受益于 SAM/GAM”；
- “LoRA 维度更低，所以有限差分 HVP 更准确”；
- “flatness/方向曲率下降必然减少遗忘”；
- “某个静态 pullback、condition number 或 gauge scalar 可以预测最终收益”；
- “raw forgetting 更低就表示更安全”，除非同时控制 current-task progress 与实际运动预算。

## 0. 聚焦 Stage 1：LoRA 是否放大 flatness optimizer 的相对收益

### 0.1 研究问题与统计对比

第一阶段只回答基本现象是否存在，不先解释 rank、safe plasticity 或 HVP：

\[
I_{\mathcal O}^{A/F}
=
\left(M_{\mathcal O}-M_{\mathrm{SGD}}\right)_{\mathrm{FactorLoRA}}
-
\left(M_{\mathcal O}-M_{\mathrm{SGD}}\right)_{\mathrm{Dense}},
\qquad \mathcal O\in\{\mathrm{SAM},\mathrm{GAM\mbox{-}FD}\}.
\]

其中准确率指标使用 `Flat − SGD`，遗忘指标使用 `SGD − Flat`，所以正的 interaction 都表示 flatness optimizer 在 LoRA 下获得更大的相对收益。每个 contrast 使用相同 seed 配对，并以 seed bootstrap 95% CI 报告。

### 0.2 正式配置

| 项目 | Stage-1 设置 |
|---|---|
| 模型 | `784 → 64 → 64 → 10`，GELU |
| 可训练权重 | 仅同一个 `64 × 64` middle matrix；外层、分类头和 bias 冻结 |
| 公共初始化 | 在任务序列外的 `−30°` MNIST 上预训练 10 epochs；所有条件共享同一 checkpoint |
| 连续任务 | Rotated MNIST `[0°,15°,30°,45°,60°]`；另做 `[60°,45°,30°,15°,0°]` 反向顺序 |
| 每任务数据 | 固定随机子集：10,000 train、2,000 test；旧任务数据只评价，不参与新任务训练 |
| 参数化 | Dense、Random linear subspace、Factor LoRA；后两者 `r=4` |
| 优化器 | SGD、SAM、GAM-FD |
| 训练 | 每任务 5 epochs，batch 128，lr 0.03，momentum 0.9，weight decay 0 |
| Flatness 参数 | `sam_rho=0.02`，`gam_radius=0.02`，`gam_weight=0.05` |
| 公平扰动 | Dense/LoRA 都通过一维搜索匹配 `||W(θ+ε)−W(θ)||_F=0.02`，而不是匹配 raw factor norm |
| 随机种子 | 0–9，严格配对 |
| 总规模 | 3 parameterizations × 3 optimizers × 10 seeds × 2 task orders = 180 runs |
| 主指标 | FAA、average forgetting；随后补 current-task-loss matching 和 effective-drift matching |
| 主统计 | LoRA-minus-Dense difference-in-differences、paired seed bootstrap 95% CI |

正式入口：

- forward：[configs/focus_stage1.yaml](configs/focus_stage1.yaml)；
- reverse：[configs/focus_stage1_reverse.yaml](configs/focus_stage1_reverse.yaml)；
- 配对分析：[analyze_stage1_interaction.py](analyze_stage1_interaction.py)。

### 0.3 启动状态与运行环境

正式网格于 **2026-08-06** 启动并完成。MNIST 已完成 integrity check：60,000 train、10,000 test。当前执行环境没有可用 CUDA，因此使用 CPU；为避免小矩阵训练的线程过度竞争，运行 8 个互不重叠的 worker，每 worker 8 个计算线程，DataLoader `num_workers=0`。

| task order | 已完成/计划 | 状态 |
|---|---:|---|
| forward | 90/90 | 完成 |
| reverse | 90/90 | 完成 |
| 合计 | 180/180 | 完成 |

完整性检查结果：每个 task order 的 9 个 parameterization × optimizer cells 都恰好包含 seeds 0–9；FAA、forgetting 与 elapsed time 均为有限值。共检查 3,000 个 flatness epoch 记录，实际 effective-weight radius 与目标 0.02 的最大绝对误差小于 $1.9\times10^{-8}$。

### 0.4 完整性能结果

表中每格为 `FAA / average forgetting` 的 mean ± sample SD，单位均为百分比点；FAA 越高越好，forgetting 越低越好。

| order | parameterization | SGD | SAM | GAM-FD |
|---|---|---:|---:|---:|
| forward | Dense | 70.329 ± 0.859 / 27.914 ± 1.166 | 70.442 ± 0.874 / 27.761 ± 1.176 | 70.558 ± 0.933 / 27.573 ± 1.251 |
| forward | Random subspace | 62.594 ± 1.468 / 18.090 ± 1.545 | 62.592 ± 1.473 / 18.095 ± 1.536 | 62.571 ± 1.481 / 18.135 ± 1.567 |
| forward | Factor LoRA | 62.269 ± 1.480 / 30.070 ± 1.628 | 62.276 ± 1.278 / 30.086 ± 1.450 | 62.518 ± 1.134 / 29.793 ± 1.513 |
| reverse | Dense | 73.409 ± 0.757 / 23.309 ± 0.891 | 73.521 ± 0.828 / 23.155 ± 0.997 | 73.588 ± 0.819 / 23.066 ± 0.944 |
| reverse | Random subspace | 61.950 ± 1.743 / 17.531 ± 2.109 | 61.946 ± 1.736 / 17.545 ± 2.128 | 61.928 ± 1.746 / 17.559 ± 2.146 |
| reverse | Factor LoRA | 61.937 ± 2.402 / 25.468 ± 2.409 | 62.658 ± 1.307 / 26.071 ± 2.035 | 60.686 ± 4.788 / 22.528 ± 6.682 |

Random subspace 的 forgetting 较低但 FAA 同时显著较低，说明它主要是低 plasticity/underfitting control，不能把较低 forgetting 单独解释为更安全。Reverse Factor-LoRA + GAM-FD 的 seed variance 很大，也不应只报告其平均 forgetting 改善。

![Stage-1 grouped FAA and forgetting](figures/stage1_group_metrics.png)

### 0.5 LoRA-minus-Dense interaction

下表单位为百分比点。正数表示 flatness optimizer 相对各自 SGD baseline 的收益在 Factor LoRA 中更大。

| order | optimizer | FAA interaction [paired 95% CI] | forgetting-reduction interaction [paired 95% CI] |
|---|---|---:|---:|
| forward | SAM | −0.106 [−0.531, 0.455] | −0.169 [−0.627, 0.313] |
| forward | GAM-FD | 0.020 [−0.525, 0.618] | −0.064 [−0.633, 0.523] |
| reverse | SAM | 0.609 [−0.346, 1.790] | −0.758 [−2.340, 0.536] |
| reverse | GAM-FD | −1.430 [−3.483, 0.248] | 2.697 [−0.915, 6.960] |

![Stage-1 interaction confidence intervals](figures/stage1_interaction_ci.png)

完整配对表：

- forward：[analysis/focus_stage1_forward.csv](analysis/focus_stage1_forward.csv)；
- reverse：[analysis/focus_stage1_reverse.csv](analysis/focus_stage1_reverse.csv)；
- 绘图代码：[plot_stage1_results.py](plot_stage1_results.py)。

### 0.6 第一阶段判断

8 个预先定义的 interaction 置信区间全部跨过 0，且效应方向没有在 forward/reverse 间保持一致。因此，**当前固定设置 `r=4, lr=0.03, rho_W=0.02` 不支持“LoRA 放大 SAM/GAM 相对收益”这一基本现象**。

这不是对所有 LoRA/optimizer 超参数的全局否定：Factor LoRA 的 FAA 比 Dense 低约 8–13 个点，reverse GAM-FD 还表现出明显 seed instability，说明当前可能存在 plasticity 或超参数失配。按预注册门控，后续只做小范围的 learning-rate/radius robustness、current-task-loss matching 与 effective-drift matching；该实验现已完成，结果见下一节。

详细的聚焦研究顺序、Stage-2 端点/轨迹分解和 HVP-A/HVP-B 设计见 [FOCUSED_TWO_STAGE_PLAN.md](FOCUSED_TWO_STAGE_PLAN.md)。

### 0.7 Stage-1B：learning-rate/radius robustness 与匹配分析

Stage-1B 于 **2026-08-06** 启动。由于 Stage-1A 的 SAM interaction 门控未通过，本轮只以不依赖显式 FD-HVP 的 SAM 做主门控，不扩张 GAM 网格。

| 项目 | Stage-1B pilot |
|---|---|
| 参数化 | Dense、Factor LoRA (`r=4`) |
| 优化器 | SGD、SAM |
| learning rates | `{0.01, 0.03, 0.1}` |
| effective-weight SAM radii | `{0.005, 0.02, 0.05}`；SGD 自动折叠 radius 重复 |
| seeds | pilot seeds 0–4；候选设置再用 seeds 5–9 确认 |
| task order | forward 与 reverse |
| pilot 规模 | 每个 order 120 runs，共 240 runs |
| 轨迹记录 | 每个 task 的 epoch 0–5：current loss/accuracy、即时旧任务 loss/accuracy damage、task-start effective drift |

除固定 epoch 的 FAA/forgetting interaction 外，本轮在每个 seed、transition 内计算：

1. **current-loss matched**：取 SGD/SAM 都达到的最深共同 current-task loss，沿 epoch 轨迹线性插值旧任务损伤；
2. **effective-drift matched**：取两条轨迹都达到的最大共同 $\|W-W_{task\ start}\|_F$，插值旧任务损伤；
3. 对每个 seed 先平均四个即时 transitions，再计算 LoRA-minus-Dense interaction 和 paired seed bootstrap CI。

配置入口：[configs/focus_stage1b_sam_robustness_forward.yaml](configs/focus_stage1b_sam_robustness_forward.yaml)、[configs/focus_stage1b_sam_robustness_reverse.yaml](configs/focus_stage1b_sam_robustness_reverse.yaml)。匹配分析代码：[analyze_stage1b_matching.py](analyze_stage1b_matching.py)。

Pilot 状态：**240/240 完成**。每个 order 包含 6 个 SGD cells 和 18 个 SAM cells，每 cell 5 seeds；共记录 7,200 个 trajectory points。匹配汇总位于 [analysis/focus_stage1b_matching.csv](analysis/focus_stage1b_matching.csv)。

Pilot 中最一致的区域是 `lr=0.01`：

| rho | endpoint FAA interaction F/R | loss-matched old-loss interaction F/R | drift-matched old-loss interaction F/R |
|---:|---:|---:|---:|
| 0.005 | −0.00036 / +0.00516 | +0.00139 / +0.00428 | +0.00053 / +0.00165 |
| 0.020 | +0.00076 / +0.00306 | +0.00691 / +0.00385 | +0.00266 / +0.00243 |
| 0.050 | +0.01014 / +0.00260 | +0.00868 / −0.00632 | +0.00795 / −0.02100 |

其中正值表示 SAM 的相对保护作用在 Factor LoRA 下更强。`rho=0.05` 在 reverse 的 matched 指标上翻转，因此不进入确认。确认阶段冻结 `lr=0.01`，以 `rho=0.02` 为主候选、`rho=0.005` 为小半径对照，使用未参与选择的 seeds 5–9；forward/reverse 各 30 runs，共 60 runs。配置：[focus_stage1b_confirm_forward.yaml](configs/focus_stage1b_confirm_forward.yaml)、[focus_stage1b_confirm_reverse.yaml](configs/focus_stage1b_confirm_reverse.yaml)。

![Stage-1B pilot loss-matched robustness](figures/stage1b_pilot_loss_matched_heatmap.png)

#### Held-out confirmation（seeds 5–9）

确认实验已 **60/60 完成**。每个顺序包含 6 个完整 cells，每 cell 5 个配对 seeds；共记录 1,800 个轨迹点。SAM 的实际 effective-weight radius 与目标值最大绝对误差为 $1.14\times10^{-8}$。

下表报告 `Factor LoRA minus Dense` interaction 的 mean `[paired bootstrap 95% CI]`。accuracy 指标使用 `[0,1]` 比例，old-loss 指标使用交叉熵 loss；所有列均以正值表示 SAM 在 LoRA 下比在 Dense 下提供更强的相对保护。

| order / $\rho_W$ | endpoint FAA | endpoint forgetting reduction | current-loss-matched old-loss reduction | drift-matched old-loss reduction |
|---|---:|---:|---:|---:|
| forward / 0.005 | −0.000300 [−0.001220, +0.000520] | −0.000750 [−0.001975, +0.000425] | −0.000080 [−0.001223, +0.001155] | −0.000561 [−0.001176, +0.000063] |
| reverse / 0.005 | +0.009720 [−0.003740, +0.028300] | +0.002650 [−0.005100, +0.011425] | **+0.005479 [+0.002432, +0.008525]** | +0.001380 [−0.001914, +0.005016] |
| forward / 0.020 | −0.001720 [−0.004600, +0.000000] | **−0.001825 [−0.004025, −0.000175]** | −0.000874 [−0.003097, +0.001348] | −0.001295 [−0.003041, +0.000573] |
| reverse / 0.020 | **+0.011780 [+0.002080, +0.030180]** | +0.006800 [−0.000725, +0.016425] | **+0.010156 [+0.005558, +0.014754]** | +0.004956 [−0.001651, +0.009755] |

![Stage-1B held-out confirmation interactions](figures/stage1b_confirmation_interactions.png)

#### Stage-1B 判断

1. **current-task-loss matching 后只在 reverse 复现。** Reverse 在两个 radius 上的 loss-matched CI 都完全大于 0；forward 均接近 0 或略为负，且 CI 跨 0。
2. **effective-drift matching 没有得到确定证据。** 四个 held-out drift-matched interaction 的 CI 全部跨 0；因此不能把 reverse 的端点收益稳定归因于相同有效更新距离下更安全的 LoRA 轨迹。
3. **主门控失败。** 主候选 `lr=0.01, rho_W=0.02` 在 reverse 有正的 FAA 和 loss-matched interaction，但 forward 没有复现，forward 的 endpoint forgetting interaction 反而显著为负。这不支持“LoRA 普遍放大 SAM 收益”的任务顺序无关命题。
4. **Reverse 信号不是 Dense 退化造成的。** 在 `rho_W=0.02` 的 held-out loss matching 中，forward 的 SAM reduction 为 Dense `+0.000147`、LoRA `−0.000727`；reverse 为 Dense `+0.000269`、LoRA `+0.010425`。因此观察到的是 LoRA 条件本身随顺序发生改变。

需要注意，公共 checkpoint 在 `−30°` 预训练：forward 首任务为 `0°`，reverse 首任务为 `60°`。因此这里的“顺序”同时改变了首任务到 base 的距离和初始适配难度，并不是纯粹的 task permutation。若以后研究 reverse-specific 现象，需要重新设计对称初始化或成对 order；这将是新的条件性故事，而不是当前中心假设的确认。

按照预先设定的停止规则，**不进入通用的 LoRA-specific HVP/curvature Stage 2**。Pilot 与 held-out 数据可以合并用于描述性估计，但不能覆盖超参数选择后的确认失败；10-seed 描述性汇总见 [analysis/focus_stage1b_combined10.csv](analysis/focus_stage1b_combined10.csv)。

结果与复现入口：

- pilot 配对结果：[analysis/focus_stage1b_matching.csv](analysis/focus_stage1b_matching.csv)；
- held-out confirmation：[analysis/focus_stage1b_confirmation.csv](analysis/focus_stage1b_confirmation.csv)；
- 匹配分析代码：[analyze_stage1b_matching.py](analyze_stage1b_matching.py)；
- 绘图代码：[plot_stage1b_matching.py](plot_stage1b_matching.py)。

### 0.8 Stage-1C：平坦性发生在“保护过去”还是“学习未来”

Stage-1C 不重新扩大 HVP 网格，而是解决全程 `SGD` 对全程 `SAM` 无法回答的时间归因问题。对两个连续任务 $A\to B$，交叉设置：

| Task A | Task B | 主要解释 |
|---|---|---|
| SGD | SGD | 基线 |
| SAM | SGD | **Protection/imprint**：SAM 是否把旧任务训练得更耐未来更新 |
| SGD | SAM | **Trajectory**：SAM 是否让新任务走一条更少损伤旧任务的路线 |
| SAM | SAM | 两种作用是否叠加 |

#### 对称实验设置

| 项目 | Stage-1C 设置 |
|---|---|
| 模型 | `784 → 32 → 32 → 10`，仅训练 $32\times32$ middle matrix |
| 参数化 | Dense、Factor LoRA (`r=4`, persistent) |
| 公共 base | $0^\circ$ MNIST 预训练 10 epochs，测试准确率约 96.8% |
| 对称任务 | forward `[-30°, +30°]`；reverse `[+30°, −30°]` |
| 优化 | `lr=0.01`，`rho_W=0.02`，effective-weight radius matching |
| 训练 | 每任务 5 epochs，10,000 train / 2,000 test |
| Pilot | seeds 0–4，两个顺序各 40 runs，共 80 |
| Held-out confirmation | seeds 5–9，两个顺序各 40 runs，共 80 |
| 总规模 | 160 runs；每阶段、每顺序 2 parameterizations × 4 schedules × 5 seeds |

每个 task-B epoch 记录 current loss、旧任务损伤和 $\|W-W_{B,start}\|_F$，并做 current-loss/effective-drift matching。对旧任务还精确计算：

\[
I_k=g_{A,k}^{\top}\delta_k,
\qquad
C_k=\frac12\delta_k^{\top}H_{A,k}\delta_k,
\qquad
\sum_k(I_k+C_k).
\]

在 task B 开始、尚未执行新任务更新时，额外取新任务梯度在当前可达空间中的投影 $v_B$，测量：

\[
\kappa_{A\mid B}^{\mathrm{prospective}}
=v_B^\top H_Av_B,
\qquad
\kappa_{A\mid B}^{\mathrm{GGN}}
=v_B^\top G_Av_B.
\]

这使 protection curvature 不受 task-B 优化器轨迹污染。配置：[focus_stage1c_temporal_forward.yaml](configs/focus_stage1c_temporal_forward.yaml)、[focus_stage1c_temporal_reverse.yaml](configs/focus_stage1c_temporal_reverse.yaml)、[focus_stage1c_confirm_forward.yaml](configs/focus_stage1c_confirm_forward.yaml)、[focus_stage1c_confirm_reverse.yaml](configs/focus_stage1c_confirm_reverse.yaml)。

#### Held-out trajectory effect：固定旧任务为 SGD，只改变新任务优化器

下表使用 seeds 5–9。每个值为 `SGD-on-B damage − SAM-on-B damage` 的 mean `[95% CI]`；LoRA−Dense 为两种参数化的配对差分。正值表示新任务 SAM 更安全。

| order / contrast | endpoint old-loss reduction | current-loss matched | effective-drift matched |
|---|---:|---:|---:|
| forward / Dense | −0.006692 [−0.010893, −0.003502] | −0.004805 [−0.009360, −0.001824] | +0.002316 [−0.000041, +0.005126] |
| forward / LoRA | **+0.008281 [+0.005928, +0.010152]** | **+0.012027 [+0.007243, +0.017645]** | **+0.010528 [+0.007664, +0.012882]** |
| forward / LoRA−Dense | **+0.014973 [+0.011831, +0.018836]** | **+0.016831 [+0.011697, +0.021965]** | **+0.008212 [+0.006413, +0.010012]** |
| reverse / Dense | −0.000127 [−0.001107, +0.000987] | −0.001286 [−0.003647, +0.000939] | **+0.005879 [+0.004841, +0.006917]** |
| reverse / LoRA | +0.001508 [−0.005460, +0.010028] | +0.005571 [−0.000701, +0.014308] | **+0.005040 [+0.000303, +0.012474]** |
| reverse / LoRA−Dense | +0.001636 [−0.005554, +0.010514] | +0.006857 [−0.000018, +0.017828] | −0.000839 [−0.006291, +0.006384] |

结论是：forward 中的 LoRA 安全轨迹效应通过了 endpoint、current-loss matching 和 drift matching 三个门控；reverse 中 LoRA 在相同 drift 下也得到正收益，但 Dense 同样受益，因此没有确认 LoRA-specific interaction。Pilot+confirmation 的 10-seed 描述性估计中，loss-matched LoRA effect 在 forward/reverse 分别为 `+0.009704` 和 `+0.005331`，但 reverse 的 held-out CI 略跨 0，不能用合并结果替代确认结论。

#### Protection/imprint：旧任务 SAM 是否留下更低的未来方向曲率

固定 task B 为 SGD，比较 task A 使用 SGD/SAM。Held-out prospective GGN curvature reduction 为：

| order / contrast | prospective GGN curvature reduction | current-loss-matched old-loss reduction |
|---|---:|---:|
| forward / Dense | **+0.005868 [+0.003272, +0.010041]** | **−0.004201 [−0.007436, −0.000966]** |
| forward / LoRA | +0.007263 [−0.004962, +0.019209] | +0.010087 [−0.004306, +0.025865] |
| forward / LoRA−Dense | +0.001395 [−0.010829, +0.013619] | +0.014288 [−0.000055, +0.031047] |
| reverse / Dense | **+0.003731 [+0.001138, +0.006031]** | **+0.004725 [+0.003114, +0.006717]** |
| reverse / LoRA | **+0.024352 [+0.016053, +0.032228]** | −0.001406 [−0.035030, +0.035367] |
| reverse / LoRA−Dense | **+0.020622 [+0.010612, +0.030480]** | −0.006131 [−0.041354, +0.031648] |

SAM 确实能让旧任务在“下一任务可达梯度方向”上更平坦，尤其是 reverse LoRA；但曲率下降没有稳定转化为实际遗忘下降。因此，**prospective flatness 是存在的中间机制，但不是充分条件**。未来方向的梯度对齐、实际轨迹和高阶项仍然决定最终遗忘。

#### Pathwise Taylor insight

在 forward held-out 的 trajectory contrast 中，LoRA−Dense 的诊断 batch old-loss reduction 为 `+0.015904`；其 pathwise 一阶、二阶和预测差分分别为：

\[
\Delta I=+0.008246,
\qquad
\Delta C=+0.005720,
\qquad
\Delta(I+C)=+0.013966,
\]

三者 CI 均完全大于 0。也就是说，在该任务方向上，SAM 同时改善了 LoRA 相对于 Dense 的一阶干扰和方向曲率暴露。Reverse held-out 中对应的 Taylor interaction 没有复现，并且一阶 interaction 为负。因此目前最准确的表述是：

> **平坦型优化在 LoRA 中的主要可见作用发生在新任务轨迹，而不是仅仅把旧任务放进一个全局平坦极小值；它可以在特定任务几何下同时修正一阶干扰和方向曲率，但该作用具有任务方向依赖。**

对称 base 已排除“两个顺序首任务到 base 距离不同”这一混杂，但仍不能排除旋转方向、具体类别样本和 LoRA tangent 随 seed 改变造成的条件性。

![Stage-1C held-out temporal mechanisms](figures/stage1c_confirmation_mechanisms.png)

结果与代码：

- pilot：[analysis/focus_stage1c_effects.csv](analysis/focus_stage1c_effects.csv)；
- held-out confirmation：[analysis/focus_stage1c_confirmation_effects.csv](analysis/focus_stage1c_confirmation_effects.csv)；
- 10-seed 描述性结果：[analysis/focus_stage1c_combined10_effects.csv](analysis/focus_stage1c_combined10_effects.csv)；
- 分析与绘图：[analyze_stage1c_temporal.py](analyze_stage1c_temporal.py)、[plot_stage1c_temporal.py](plot_stage1c_temporal.py)。

### 0.9 Stage-1D：每任务 merge-reset 是否消除 SAM 的 LoRA 收益

合并后旧任务历史进入 $W_t$，下一任务仍产生 $B_{t+1}A_{t+1}$。为检验收益是否只来自持续复用同一 factors/tangent，Stage-1D 固定 task A 为 SGD，只比较 task B 的 SGD/SAM，并在每任务结束执行 `merge_and_reset()`：

- Factor LoRA `r=4`；
- forward/reverse 对称两任务；
- seeds 0–9；
- 每个顺序 2 schedules × 10 seeds = 20 runs，共 40 runs；
- merge 前后最大 logit 误差为 0；effective SAM radius 最大误差小于 $8.3\times10^{-9}$。

下表比较 persistent 与 merge-reset 的新任务 SAM trajectory effect。`merge−persistent` 的正值表示 merge-reset 放大 SAM 收益。

| order / lifecycle | current-loss matched old-loss reduction | drift-matched old-loss reduction |
|---|---:|---:|
| forward / persistent | **+0.009704 [+0.006408, +0.013399]** | **+0.009903 [+0.007508, +0.012162]** |
| forward / merge-reset | **+0.010865 [+0.004227, +0.017588]** | +0.001095 [−0.013255, +0.012171] |
| forward / merge−persistent | +0.001161 [−0.005673, +0.007605] | −0.008809 [−0.021766, +0.001205] |
| reverse / persistent | **+0.005331 [+0.001483, +0.010137]** | **+0.006356 [+0.003120, +0.010285]** |
| reverse / merge-reset | +0.005586 [−0.001966, +0.012742] | **+0.010037 [+0.003828, +0.016457]** |
| reverse / merge−persistent | +0.000255 [−0.006080, +0.006254] | +0.003681 [−0.001111, +0.008172] |

**Merge-reset 没有消除 SAM 收益。** 两个顺序的 current-loss matched 均值与 persistent 几乎一致，所有 lifecycle difference CI 都跨 0。Merge 改变了下一任务 tangent，并增加了部分 seed variance，但旧有效权重仍需要承受下一次低秩更新。因此不能把机制简化为“未合并 factors 被重复使用”；更符合证据的是有效权重空间中的条件性安全轨迹。

![Stage-1D persistent versus merge-reset](figures/stage1d_lifecycle_matching.png)

配置、结果与代码：[focus_stage1d_merge_forward.yaml](configs/focus_stage1d_merge_forward.yaml)、[focus_stage1d_merge_reverse.yaml](configs/focus_stage1d_merge_reverse.yaml)、[analysis/focus_stage1d_lifecycle.csv](analysis/focus_stage1d_lifecycle.csv)、[analyze_stage1d_lifecycle.py](analyze_stage1d_lifecycle.py)、[plot_stage1d_lifecycle.py](plot_stage1d_lifecycle.py)。

### 0.10 Stage-1E：纯低维约束还是 Factor LoRA 动态几何

为判断 Stage-1C 的 trajectory benefit 是否只是低维效应，增加两个与标准 LoRA **初始 tangent 维度相同**的线性控制：

- Random linear subspace：随机固定的 $32r=128$ 维有效权重子空间；
- Fixed LoRA tangent：固定标准 LoRA 在 $B=0$ 初始化时的 $\{\mathrm dB A_0\}$ 空间；
- 对照已有 Dense 和 Factor LoRA；Factor LoRA 的局部 tangent 可随训练扩张到 rank-$r$ 流形维度 $r(64-r)=240$；
- 固定 task A 为 SGD，只比较 task B 的 SGD/SAM；
- forward/reverse、seeds 0–9，共新增 80 runs。

下表是 10-seed trajectory effect。正值表示 task-B SAM 比 SGD 更少损伤 task A。

| parameterization | forward current-loss matched | forward drift matched | reverse current-loss matched | reverse drift matched |
|---|---:|---:|---:|---:|
| Dense | **−0.004954 [−0.007421, −0.002834]** | **+0.002073 [+0.000480, +0.003807]** | −0.000685 [−0.002260, +0.000907] | **+0.005752 [+0.004272, +0.007019]** |
| Random subspace | **−0.000529 [−0.000687, −0.000365]** | +0.000016 [−0.000124, +0.000160] | **−0.000625 [−0.000911, −0.000379]** | −0.000050 [−0.000247, +0.000166] |
| Fixed LoRA tangent | **−0.000228 [−0.000425, −0.000024]** | +0.000070 [−0.000139, +0.000287] | **−0.000267 [−0.000461, −0.000082]** | +0.000003 [−0.000169, +0.000165] |
| Factor LoRA | **+0.009704 [+0.006475, +0.013444]** | **+0.009903 [+0.007479, +0.012157]** | **+0.005331 [+0.001539, +0.010055]** | **+0.006356 [+0.003107, +0.010090]** |

结果非常明确：Random 和 Fixed 两个低维线性控制没有复现 Factor LoRA 的收益，current-loss matched effect 反而稳定为小幅负值。它们的 pathwise Taylor reductions 在两个顺序也都为负，而 Factor LoRA 在 forward 为正。

因此可以排除一个过于简单的故事：

> “只要参数空间维度低，SAM 就更有效。”

证据更符合：

> **Factor LoRA 的动态 tangent、双线性 $BA$ 映射或由此产生的自适应条件数，使 SAM 能改变有效权重轨迹；固定低维空间本身没有这项能力。**

但两个固定控制的 SGD current-task loss 明显更高：forward 分别约 `0.659/0.755`，Factor LoRA 约 `0.435`；reverse 分别约 `0.695/0.871`，Factor LoRA 约 `0.437`。因此 Stage-1E 可以排除“纯维度充分解释”，却还不能在“动态 tangent 扩张”和“factor conditioning/bilinear geometry”之间做最终选择。下一项最有判别力的实验应是 Balanced Factor LoRA 或直接的 rank-manifold 参数化，并匹配 current loss、drift 和可达维度；在此之前不应回到有限差分 HVP 精度故事。

![Stage-1E parameterization controls](figures/stage1e_parameterization_controls.png)

配置、结果与代码：[focus_stage1e_parameterization_forward.yaml](configs/focus_stage1e_parameterization_forward.yaml)、[focus_stage1e_parameterization_reverse.yaml](configs/focus_stage1e_parameterization_reverse.yaml)、[analysis/focus_stage1e_parameterizations.csv](analysis/focus_stage1e_parameterizations.csv)、[plot_stage1e_parameterizations.py](plot_stage1e_parameterizations.py)。

### 0.11 Stage-2A：严格公共 Task-A endpoint

Stage-1 的 Dense 与 Factor LoRA 先分别学习 Task A，起点并不完全相同。Stage-2A 消除这一混杂：每个 seed 只训练一次 Dense Task A，保存精确有效权重 $W_A$，随后把同一个 $W_A$ 无误差地重参数化为 Dense 或 Factor LoRA，再训练 Task B。审计同时检查 weight SHA256、最大权重误差和最大 logit 误差；正式 40 个 `order × seed` endpoint groups 全部通过。

每个顺序使用 20 个配对 seeds（100–119），总计 160 runs。下表报告 Factor-minus-Dense 的 SAM interaction；定义为“SGD 旧任务损伤减去 SAM 旧任务损伤”，正值表示 Factor LoRA 中 SAM 相对更安全。

| order | endpoint old-loss | current-loss matched | effective-drift matched | path $I$ | path $C$ | path GGN cost |
|---|---:|---:|---:|---:|---:|---:|
| forward | **+.01999 [.01471,.02504]** | **+.01870 [.01096,.02574]** | **+.00857 [.00397,.01332]** | **+.00793 [.00382,.01228]** | **+.00729 [.00182,.01275]** | **+.01913 [.00914,.02915]** |
| reverse | **+.00646 [.00274,.01025]** | **+.02484 [.00817,.05259]** | +.00047 [−.00310,.00423] | −.00309 [−.00801,.00221] | **+.00865 [.00368,.01315]** | **+.02329 [.01503,.03102]** |

公共 endpoint 后 forward 效应仍存在，因此它不是“Task A 被不同参数化训练到不同位置”的产物。Reverse 在 drift matching 后消失，继续表明它是条件性机制，而非普遍定律。

结果与审计：[focus_stage2a_formal_effects.csv](analysis/focus_stage2a_formal_effects.csv)、[focus_stage2a_formal_endpoint_audit.csv](analysis/focus_stage2a_formal_endpoint_audit.csv)。

![Common-endpoint interaction](figures/stage2a_common_endpoint_interaction.png)

### 0.12 Stage-2B：低维、动态 tangent、rank manifold 与 gauge 的分解

所有条件继续共享同一个 $W_A$，只改变 Task B 的参数化。正式 cohort 为每顺序 20 个 seeds（排除只用于数值稳定性选择的 210/212）。主网格包含 480 个选定 runs，128 维控制 160 个 runs，另加 40 个 Factor 小半径 SAM runs。

| 参数化 | 有效自由度 | 判别的解释 |
|---|---:|---|
| Dense | 1024 | 完整空间基线 |
| Random subspace | 240；另做 128 | 纯维度/任意固定路线 |
| Fixed mature tangent | 240 | 固定在成熟 LoRA tangent，排除 tangent 漂移 |
| Fixed initial tangent | 128 | 固定 $B=0$ 的初始 tangent |
| Projected direct rank | rank-$4$ manifold | 允许 rank manifold 移动，但不用 $A,B$ 坐标 |
| Balanced Factor LoRA | $BA$，每步 SVD canonical gauge | 保留 factorization、抑制 gauge imbalance |
| Standard Factor LoRA | $BA$ | 动态 tangent、双线性映射和自由 gauge 全部保留 |

先按参数化只用 current-task loss 选择 Task-B learning rate：Dense/Random-240/Fixed-mature/Projected 用 0.3，Factor/Balanced 用 0.1。SAM 使用有效权重扰动半径；Factor 为 .02，Balanced 因 .01 在工程 seeds 上不稳定而预先固定为最大稳定值 .005。主分析始终同时报告 endpoint、current-loss matching 和 drift matching。

Forward 的主要 old-loss reductions 如下：

| 参数化 | endpoint | current-loss matched | drift matched | 解释 |
|---|---:|---:|---:|---|
| Dense | **−.01310** | **−.02860** | **+.01171** | endpoint/loss matched 反而更差 |
| Random-240 | **−.00335** | **−.00334** | ≈0 | 纯固定低维不够 |
| Fixed mature-240 | **−.00272** | **−.01023** | ≈0 | 成熟 tangent 维度不够 |
| Projected rank manifold | **−.01209** | **−.02195** | −.00581（CI 跨 0） | 可移动 rank manifold 仍不够 |
| Balanced Factor, $\rho=.005$ | **+.00215** | **−.02020** | −.00089（CI 跨 0） | canonical gauge 未复现标准 Factor |
| Standard Factor, $\rho=.02$ | **+.01496** | **+.00891** | **+.01324** | 三个主指标均改善 |

Factor 的 current-task loss 约 .47；Balanced/Random-240/Projected 分别约 .48/.47/.43，因此关键否定不是由这些控制明显欠拟合造成。128 维 Fixed/Random 控制确实欠拟合，只作为“低容量下界”，不用于强机制比较。Reverse 的 Factor endpoint 为 +.00248，三个主 CI 均跨 0，再次显示方向依赖。

由于主网格 Factor 与 Balanced 半径不同，又补做同半径 $\rho_W=.005$：

| forward contrast | endpoint | current-loss matched | drift matched | path $C$ interaction |
|---|---:|---:|---:|---:|
| Factor | **+.00379 [.00254,.00504]** | **+.00250 [.00068,.00423]** | **+.00340 [.00215,.00457]** | — |
| Balanced | **+.00215 [.00015,.00394]** | **−.02020 [−.05271,−.00008]** | −.00089 [−.00443,.00218] | — |
| Factor − Balanced | +.00164 [−.00049,.00398] | **+.02270 [.00238,.05577]** | **+.00429 [.00092,.00824]** | **+.00302 [.00022,.00558]** |

相同有效权重半径时，raw endpoint 差异不显著，但 loss/drift-matched 轨迹和方向曲率差异仍存在。因此当前证据顺序是：**纯低维不足，动态 rank manifold 也不足，$BA$ 坐标/gauge 几何至少是机制的一部分；较大的 $\rho$ 会放大标准 Factor 的现象。**

结果：[focus_stage2b_formal_main_effects.csv](analysis/focus_stage2b_formal_main_effects.csv)、[focus_stage2b_formal_effects.csv](analysis/focus_stage2b_formal_effects.csv)、[focus_stage2b_radius_matched_effects.csv](analysis/focus_stage2b_radius_matched_effects.csv)。分析与绘图：[analyze_stage2b_parameterizations.py](analyze_stage2b_parameterizations.py)、[plot_stage2_results.py](plot_stage2_results.py)。

![Stage-2B parameterization controls](figures/stage2b_parameterization_controls.png)

![Factor minus controls](figures/stage2b_factor_minus_controls.png)

### 0.13 Stage-2C：SAM 收益由任务方向几何调节

使用公共 Task-A endpoint，在 $\{-15,+15\}$、$\{-30,+30\}$、$\{-45,+45\}$ 的两个方向上比较 Dense/Factor × SGD/SAM。每个有向 pair 20 seeds（300–319），共 480 runs；120/120 个公共 endpoint groups 通过审计。旋转角度只是操纵变量，结论使用实测的旧梯度/可达新梯度余弦和 GGN 曲率。

| direction | Factor endpoint | loss matched | drift matched | Factor−Dense endpoint interaction |
|---|---:|---:|---:|---:|
| −15→+15 | **+.00396** | +.00092 | **+.00133** | **+.01326** |
| +15→−15 | **+.00365** | **+.00404** | **+.00244** | **+.01079** |
| −30→+30 | **+.01979** | **+.02850** | **+.01691** | **+.03813** |
| +30→−30 | −.00107 | +.00662 | −.00046 | +.00703 |
| −45→+45 | +.00544 | **+.02994** | +.00244 | +.01571 |
| +45→−45 | −.03025 | −.04056 | −.02415 | −.02487 |

最关键的反例是 `+45→−45`：SAM 使方向 Hessian curvature reduction 为 **+.03261 [.00599,.05698]**、GGN cost reduction 为 **+.06974 [.02353,.11854]**，但一阶 interference reduction 为 **−.05680 [−.09027,−.02727]**，最终 endpoint benefit 反而为负。也就是说：

> **降低旧任务方向曲率不是减少遗忘的充分条件。SAM 会在一阶干扰与二阶曲率之间做轨迹 trade-off，净收益取决于任务方向。**

六个方向的描述性分析中，训练前旧梯度与可达新梯度余弦对 Factor endpoint SAM benefit 的 Pearson/Spearman 相关为 −.814/−.943；但样本只有六个有向任务对，只能作为下一步假设，不作因果证明。实测 prospective GGN curvature 与收益的相关较弱，进一步否定“曲率越大，SAM 收益必然越大”的单变量故事。

结果：[focus_stage2c_effects.csv](analysis/focus_stage2c_effects.csv)、[focus_stage2c_geometry_cells.csv](analysis/focus_stage2c_geometry_cells.csv)、[focus_stage2c_geometry_correlations.csv](analysis/focus_stage2c_geometry_correlations.csv)。代码：[analyze_stage2c_geometry.py](analyze_stage2c_geometry.py)、[plot_stage2c_geometry.py](plot_stage2c_geometry.py)。

![Task geometry and SAM benefit](figures/stage2c_task_geometry.png)

### 0.14 Stage-2D：函数保持的 Factor gauge intervention

为直接区分“有效 rank manifold”与“因子坐标度量”，在公共 $W_A$ 上施加

\[
A\to cA,\qquad B\to B/c.
\]

Task B 开始时 $B=0$，因此三种 gauge 的 $BA=0$、有效权重和 logits 完全相同，但初始 $\|A\|_F$ 按 $c$ 缩放。极端 pilot 的 $c=0.1/10$ 显示同一个 $\rho_W=.02$ 并非优化不变量：$c=.1$ 的 SAM current loss 约为 forward/reverse `1.81/2.22`，基本停止学习；这个结果只作压力测试，不进入遗忘主结论。

正式实验改用稳定且可塑性匹配的 $c\in\{0.3,1,3\}$：

| 项目 | 设置/审计 |
|---|---|
| Task B lr / momentum | `.1 / 0`，三种 gauge 相同 |
| SAM radius | 有效权重 Frobenius 半径 $\rho_W=.005$，三种 gauge 相同 |
| seeds / runs | 400–419；3 gauges × 2 schedules × 2 orders × 20 = 240 |
| 公共 endpoint | 40/40 `order × seed` groups 通过；每组 6 runs 的 hash、loss、accuracy 完全一致 |
| 实际 radius | 600 个 epoch records，最大 $|\hat\rho_W-.005|=5.55\times10^{-9}$ |
| 起始 $\|A\|_F$ | `.345 / 1.151 / 3.454`，严格体现 `.3 / 1 / 3` gauge scaling |

最终 current-task loss 的均值在 forward 为 `.450–.464`，reverse 为 `.464–.482`；同一 gauge 内 SGD/SAM 的差小于 `.004`。因此下面的差异不是由某一方法没有学会新任务造成。

| order / gauge | endpoint reduction | current-loss matched | drift matched |
|---|---:|---:|---:|
| forward / $c=.3$ | **+.01558 [.00754,.02559]** | **+.01936 [.01079,.02880]** | +.00786 [−.00140,.01907] |
| forward / $c=1$ | **+.00420 [.00283,.00567]** | **+.00547 [.00391,.00717]** | **+.00417 [.00277,.00564]** |
| forward / $c=3$ | **+.00206 [.00135,.00280]** | **+.00384 [.00211,.00598]** | **+.00255 [.00167,.00338]** |
| reverse / $c=.3$ | +.00454 [−.00660,.01651] | **+.02132 [.00506,.04250]** | −.00468 [−.01930,.00954] |
| reverse / $c=1$ | +.00089 [−.00057,.00232] | **+.00798 [.00045,.02098]** | +.00057 [−.00119,.00219] |
| reverse / $c=3$ | **+.00137 [.00056,.00233]** | **+.00847 [.00152,.02078]** | **+.00133 [.00079,.00190]** |

Forward 中 $c=.3-c=1$ 的 endpoint/loss-matched interaction 分别为 **+.01138 [.00395,.02059]** 和 **+.01389 [.00613,.02249]**；同时 path-$C$ interaction 为 **+.01504 [.00301,.02798]**，GGN-cost interaction 为 **+.04150 [.02002,.06530]**，而 path-$I$ interaction 接近 0。相反，$c=3-c=1$ 的 endpoint 与 drift interaction 分别为 **−.00214 [−.00350,−.00090]**、**−.00161 [−.00290,−.00041]**。这说明收益对函数保持的 factor gauge 显著且非单调。

Reverse 的所有 gauge-minus-$c=1$ 主 performance CI 均跨 0；虽然 $c=.3-c=1$ 的 GGN-cost interaction 仍为 **+.03992 [.01075,.07294]**，它没有稳定转化为 endpoint/drift 收益。这与 Stage-2C 一致：坐标 geometry 能改变曲率轨迹，但最终遗忘仍受任务方向和一阶项调节。

因此 Stage-2D 支持的是“Factor pullback metric 参与机制”，不是“$c=.3$ 普遍最好”。结果与复现入口：[focus_stage2d_gauge_effects.csv](analysis/focus_stage2d_gauge_effects.csv)、[focus_stage2d_gauge_cells.csv](analysis/focus_stage2d_gauge_cells.csv)、[focus_stage2d_gauge_endpoint_audit.csv](analysis/focus_stage2d_gauge_endpoint_audit.csv)、[focus_stage2d_gauge_formal.yaml](configs/focus_stage2d_gauge_formal.yaml)、[analyze_stage2d_gauge.py](analyze_stage2d_gauge.py)、[plot_stage2d_gauge.py](plot_stage2d_gauge.py)。

![Function-preserving Factor gauge sensitivity](figures/stage2d_gauge_sensitivity.png)

### 0.15 Stage-3 P0：真实路径项能否跨任务方向解释 SAM 收益

P0 不增加训练，而是对 Stage-2C 的 480 个正式 runs 做预先收缩的 existing-data test。对每个 `direction × seed × parameterization`，定义

\[
B=D_{\mathrm{SGD}}-D_{\mathrm{SAM}},\quad
\Delta I=I_{\mathrm{SGD}}-I_{\mathrm{SAM}},\quad
\Delta C=C_{\mathrm{SGD}}-C_{\mathrm{SAM}}.
\]

比较三个预测器：仅用路径长度差；路径长度加 $\Delta I$；路径长度加 $\Delta I+\Delta C$。主验证采用 leave-one-direction-out；更严格的版本对每一个样本同时排除其 task direction 和 seed，再预测这个未见 direction/seed。后者的结果为：

| 参数化 | 模型 | MAE | $R^2$ | sign accuracy |
|---|---|---:|---:|---:|
| Dense | path only | .01351 | .025 | .800 |
| Dense | path + $I$ | .01251 | .230 | .800 |
| Dense | path + $I+C$ | **.00998** | **.463** | .783 |
| Factor LoRA | path only | .02296 | −.059 | .608 |
| Factor LoRA | path + $I$ | .01899 | .338 | .833 |
| Factor LoRA | path + $I+C$ | **.01587** | **.612** | **.850** |

按六个 task directions 做 block bootstrap，加入 $C$ 后 Dense/Factor 的 MAE 额外下降分别为 `.00257 [.00062,.00382]` 和 `.00344 [.00049,.00660]`。Factor 的六个 held-out directions 中五个改善；唯一反例为 `−45→+45`，再次说明二阶项有额外总体解释力，但不是每个方向都单调有益。

Taylor residual 的方向-block mean 在 Dense/Factor 分别为 `.00359 [−.00010,.00748]` 与 `.00120 [−.00411,.00653]`，但 absolute residual 的 90th percentile 仍为 `.0326/.0373`。因此 $I+C$ 是有增量预测力的机制近似，不是逐样本完整解释；高阶余项不能删除。

P0 还避免只在单个 endpoint 做 current-loss matching：对每个有向任务对，先取所有 Dense/Factor、SGD/SAM 和 20 seeds 共同覆盖的 current-loss support，再积分得到 normalized damage AUC。Factor SAM 的 AUC reduction 在五个方向显著为正，在 `+45→−45` 跨 0；Factor-minus-Dense interaction 在六个方向全部显著为正：

| direction | Factor AUC reduction | Factor − Dense interaction |
|---|---:|---:|
| −15→+15 | +.00090 [.00037,.00151] | +.00463 [.00301,.00636] |
| −30→+30 | +.00808 [.00631,.00986] | +.01819 [.01348,.02343] |
| −45→+45 | +.00623 [.00270,.00989] | +.01227 [.00796,.01698] |
| +15→−15 | +.00098 [.00073,.00127] | +.00853 [.00487,.01301] |
| +30→−30 | +.00185 [.00057,.00309] | +.00681 [.00489,.00893] |
| +45→−45 | +.00040 [−.00384,.00397] | +.00770 [.00304,.01244] |

P0 的结论是：**真实轨迹的一阶干扰与方向曲率，比更新量本身更能跨任务方向解释遗忘收益；$C$ 在控制 path length 和 $I$ 后仍有增量信息。** 这仍是六个有向任务对、每 run 一个固定 128-sample diagnostic batch 上的预测证据，不是 $C$ 的单独因果效应；多 diagnostic batches 的重复性仍待补充。P1 才使用函数保持 intervention 改变因子坐标几何。

结果与代码：[analysis/stage3_p0](analysis/stage3_p0)、[analyze_stage3_p0_path_safety.py](analyze_stage3_p0_path_safety.py)、[plot_stage3_p0_p1.py](plot_stage3_p0_p1.py)。

![P0 pathwise safety prediction and damage AUC](figures/stage3_p0_path_safety.png)

### 0.16 Stage-3 P1：成熟 Factor 状态上的一般 gauge 因果分叉

Stage-2D 从 $B=0$ 的奇异初始化 chart 分叉，只能说明初始化缩放敏感。P1 先在 Task B 上用共享 SGD warmup 训练 1 epoch，使 $A_*,B_*\neq0$，再从**同一个成熟状态**分叉。一般 gauge 写成

\[
B'=BS,\qquad A'=S^{-1}A,
\]

所以 $B'A'=BA$。对于任意可逆 $S$，有效权重、logits 和 tangent range 都不变；但有效权重中的 pullback operator 变为

\[
\mathcal M_S[G]
=s^2\left(
BSS^TB^TG
+GA^TS^{-T}S^{-1}A
\right).
\]

若 $S=Q$ 为正交矩阵，则 $\mathcal M_Q=\mathcal M_I$，SGD/SAM 应保持数值等价；非正交 $S$ 保持函数与可达集合，却改变坐标度量。这给出比 Balanced LoRA 更干净的因果负对照。
P1 的 scalar $c$ 定义为 $B\to cB,A\to A/c$，与 Stage-2D 把 $c$ 写在 $A$ 一侧的记号互为倒数；比较数值时必须先统一约定。

| P1 项目 | 设置 |
|---|---|
| 公共前缀 | Dense Task A 训练 5 epochs；在精确 $W_A$ 上重参数化为 Factor LoRA；共享 Task-B SGD warmup 1 epoch |
| 有向任务对 | 正收益候选 `−30→+30`；机制反例 `+45→−45` |
| 成熟 gauges | Identity；scalar $c=.5,2$；Haar orthogonal；determinant-one anisotropic $\operatorname{cond}(S)=2,4$ |
| 后续训练 | 从同一成熟 state 继续 Task B 4 epochs；SGD 或 SAM；lr `.1`、momentum `0` |
| SAM 公平性 | 二分搜索匹配有效权重扰动 $\|W(\theta+\epsilon)-W(\theta)\|_F=\rho_W$ |
| Pilot | seeds 500–504；$\rho_W\in\{.005,.02\}$；180 branches |
| Formal | seeds 600–619；锁定 $\rho_W=.005$；480 branches |
| 轨迹 | 每 epoch 保存有效权重路径；每 step 累计实际路径长度并精确分解 linear 与 bilinear factor update |

每次分叉强制审计：$BA$ 与有效权重误差、最大 logit error、tangent projector distance；每一步审计

\[
\delta W_k
=s(B_k\delta A_k+\delta B_kA_k)
+s\delta B_k\delta A_k.
\]

主指标为 old-loss damage、current loss、net effective drift、累计 step path length、pathwise $I/C$/GGN/Taylor residual；机制指标为 $\operatorname{cond}(J)$、$\cos(g_A,\mathcal M_Sg_B)$、$-g_A^T\mathcal M_Sg_B$ 与实际 bilinear/linear ratio。除 endpoint matching 外，还在所有 gauges/methods 的全局共同 current-loss、net-drift 和累计-path support 上计算 normalized damage AUC。

Pilot 选参只看 plasticity/drift，不看遗忘收益：$\rho=.005/.02$ 的 pooled current-loss penalty 分别为 `−.00121/−.00465`，relative drift change 为 `+.00031/+.00130`，两者均稳定。因此按预定保守规则选择更小的 `.005` 进入正式实验。Pilot 的最大有效权重、logit、tangent-projector 和 step-closure 误差分别为 $4.62\times10^{-7}$、$3.81\times10^{-6}$、$6.66\times10^{-6}$、$1.02\times10^{-5}$。

#### P1 正式结果：一般 factor chart 确实改变轨迹，但静态 pullback 不足以预测终点

正式 480/480 branches 与 40/40 shared prefixes 全部完成。最大有效权重误差、logit error、tangent-projector distance、step decomposition closure error 和 SAM radius error 分别为 $5.12\times10^{-7}$、$5.72\times10^{-6}$、$1.83\times10^{-5}$、$1.10\times10^{-5}$、$1.13\times10^{-8}$。Identity 与 orthogonal gauge 的 endpoint old-damage 绝对差仅为 forward/reverse 约 $2.2\times10^{-7}/4.1\times10^{-7}$；训练 4 epochs 后仍保持数值等价，严格通过负对照。

下表为 `SGD damage − SAM damage` 的 20-seed paired mean `[95% CI]`，正值表示 SAM 更安全。P1 的 scalar 记号为 $B\to cB,A\to A/c$。

| gauge | forward endpoint | forward loss matched | forward drift matched | reverse endpoint | reverse loss matched | reverse drift matched |
|---|---:|---:|---:|---:|---:|---:|
| Identity | **+.00330 [.00245,.00414]** | **+.00457 [.00205,.00751]** | **+.00359 [.00254,.00462]** | **+.00420 [.00192,.00710]** | −.00130 [−.01341,.00791] | **+.00519 [.00288,.00800]** |
| Orthogonal | **+.00330 [.00245,.00414]** | **+.00457 [.00205,.00751]** | **+.00359 [.00254,.00462]** | **+.00420 [.00192,.00710]** | −.00130 [−.01341,.00791] | **+.00519 [.00288,.00800]** |
| Scalar .5 | **+.00485 [.00313,.00694]** | **+.02515 [.00783,.05222]** | **+.00521 [.00346,.00720]** | **+.00862 [.00062,.02043]** | **+.03038 [.00056,.07994]** | +.00219 [−.00232,.00713] |
| Scalar 2 | +.00035 [−.00108,.00191] | −.00076 [−.00253,.00074] | +.00054 [−.00083,.00212] | +.00116 [−.00110,.00379] | **−.00211 [−.00427,−.00022]** | +.00221 [−.00006,.00490] |
| Anisotropic 2 | **+.00345 [.00241,.00451]** | −.00060 [−.01454,.00897] | **+.00373 [.00250,.00497]** | **+.00249 [.00093,.00414]** | −.00823 [−.02892,.00563] | **+.00324 [.00187,.00469]** |
| Anisotropic 4 | **+.00291 [.00158,.00428]** | −.01005 [−.03628,.00483] | **+.00320 [.00181,.00464]** | +.00186 [−.00061,.00411] | −.00160 [−.00960,.00436] | **+.00271 [.00018,.00516]** |

非正交 gauge 不只改变 SAM。以 SGD 为例，forward 的 Scalar-.5/Anisotropic-4 相对 Identity 使旧任务 damage 增加 `+.1171 [.0676,.1660]`/`+.0547 [.0181,.0899]`；reverse Scalar-.5 增加 `+.1122 [.0326,.1938]`。这些条件具有相同 $W$、logits 和 tangent range，所以**一般 factor-coordinate dynamics 对实际有效权重轨迹具有因果作用**。

但“哪个 gauge 放大 SAM benefit”不是单调规律。相对 Identity，Scalar-2 的 endpoint SAM-benefit interaction 在 forward/reverse 分别为 **−.00295 [−.00405,−.00179]**、**−.00304 [−.00669,−.00011]**；reverse Anisotropic-2 也为 **−.00171 [−.00394,−.00014]**。Scalar-.5 虽有较大的 mean benefit，interaction CI 在两个方向仍跨 0。结果确认 gauge sensitivity，却不支持“condition number 越大，SAM 越有效”。

#### P1 的 $I/C$ 机制分流

不同 chart 不仅改变 damage 大小，还交换一阶与二阶贡献。表中均为 `SGD − SAM` reduction：

| order / gauge | endpoint benefit | $\Delta I$ | $\Delta C$ |
|---|---:|---:|---:|
| forward / Identity | **+.00330** | **+.00275 [.00142,.00410]** | −.00012 [−.00149,.00111] |
| forward / Scalar .5 | **+.00485** | **−.00397 [−.00684,−.00125]** | **+.00648 [.00383,.00922]** |
| forward / Scalar 2 | +.00035 | +.00224 [−.00003,.00437] | **−.00247 [−.00431,−.00060]** |
| reverse / Identity | **+.00420** | +.00211 [−.00146,.00589] | **+.00336 [.00022,.00693]** |
| reverse / Scalar .5 | **+.00862** | **−.02447 [−.04529,−.00745]** | **+.04210 [.00620,.10392]** |
| reverse / Scalar 2 | +.00116 | +.00296 [.00012,.00611] | −.00227 [−.00500,.00065] |

因此 Identity-forward 的收益主要由一阶 interference 改善承载，而 Scalar-.5 的收益是“更坏的一阶项被更大的 curvature reduction 抵消”。这与 P0 和 Stage-2C 的核心约束一致：不能把 SAM 收益直接等同于 $\Delta C>0$。

全局 common-support AUC 进一步收缩 estimand。Identity 在 current-loss support 上的 reduction 在 forward/reverse 仅为 `+.00039 [−.00018,.00098]` / `+.00031 [−.00063,.00125]`，但在 net-drift support 上为 **+.00152 [.00126,.00177]** / **+.00086 [.00030,.00143]**，在累计 step-path support 上为 **+.00598 [.00543,.00654]** / **+.00754 [.00628,.00882]**。这说明保护信号沿相同运动预算存在，但不是整段 current-loss progress 上都均匀出现。Scalar-2 的 current-loss AUC 在两个方向均显著为负，提供了另一个 chart-sensitive 反例。

最后，240 个真实首 batch 审计验证了因子更新公式：加入 bilinear correction 后的最大相对误差为 $5.87\times10^{-6}$，预测/实际一阶干扰相关为 `.99999999999998`；bilinear 项的 median step-norm fraction 为 `.00817`，虽小但不能在所有样本中删除。可是 leave-one-seed-out 对最终 gauge damage 的预测显示，静态 start pullback 的 $R^2$ 在 forward/reverse 为 `−.022/−.369`，Jacobian condition 为 `−.025/−.009`，实际全轨迹 bilinear ratio 也仅为 `.057/.008`。对第一 epoch，condition/bilinear 有一定解释力，但仍不稳定跨方向。

所以 P1 最严格的结论是：

> **函数和 tangent range 保持不变时，非正交 Factor gauge 会因果性地改变 SGD/SAM 的有效权重轨迹、$I/C$ 分解与遗忘；正交 gauge 完全不改变结果。但一个静态 $\mathcal M_{A,B}$、condition number 或单个 bilinear scalar 都不足以预测最终轨迹。当前机制应称为 time-varying factor-chart dynamics，而不能缩写成“pullback metric 已经单独解释 SAM 收益”。**

![P1 mature-gauge endpoint and matched effects](figures/stage3_p1_gauge_effects.png)

![P1 common-support current-loss AUC](figures/stage3_p1_global_support_auc.png)

![P1 static pullback versus realized gauge damage](figures/stage3_p1_pullback_prediction.png)

配置、结果与实现：[stage3_p1_mature_gauge_pilot.yaml](configs/stage3_p1_mature_gauge_pilot.yaml)、[stage3_p1_mature_gauge_formal.yaml](configs/stage3_p1_mature_gauge_formal.yaml)、[analysis/stage3_p1_mature_gauge_formal](analysis/stage3_p1_mature_gauge_formal)、[run_stage3_p1_mature_gauge.py](run_stage3_p1_mature_gauge.py)、[small_cl/gauge.py](small_cl/gauge.py)、[analyze_stage3_p1_mature_gauge.py](analyze_stage3_p1_mature_gauge.py)、[analyze_stage3_p1_first_step.py](analyze_stage3_p1_first_step.py)。

### 0.17 Stage-3 P2 预注册：动态 factor chart 与多批次稳健性

P1 已证明 chart intervention 会改变轨迹，但静态起点 pullback 不能预测终点。P2 不再增加 rank 或 GAM 因子，而是检验一个更窄的命题：**沿训练轨迹共同演化的 pullback/Jacobian 几何，是否比起点单次测量更能预测 gauge 引起的旧任务 damage；P1 的效应是否能跨独立 diagnostic batches 重复。**

| P2 项目 | 预注册设置 |
|---|---|
| 公共训练协议 | 与 P1 相同：Dense Task A 5 epochs，Factor Task-B warmup 1 epoch，继续 Task B 4 epochs |
| 任务方向 | `−30→+30` 与 `+45→−45` |
| Gauges | Identity、orthogonal、scalar `.5/2`、anisotropic `4` |
| 优化器 | SGD、SAM `rho_W=.005`；lr `.1`，momentum `0` |
| Pilot seeds | 700–704；只用于运行完整性、符号复现与方差估计 |
| Formal seeds | 与 P0/P1 不重合的 720–739；只有 pilot 通过停止规则才启动 |
| 诊断重复 | 每个任务固定抽取 4 个互不重叠的 128-sample test batches；batch 是 seed 内重复测量，不扩充统计样本量 |
| 时间分辨率 | continuation 的 epoch 0–4 记录 $A_k,B_k$ norms、$\mathcal M_k$ pullback、Jacobian operator norm/condition；epoch 0–3 对齐四段真实更新 |
| 真实路径 | 每个 old diagnostic batch 分别精确计算四段 $I_k,C_k,R_k$ 和 GGN cost |

预测对象是同 seed/order/method 内，相对 Identity 的 gauge damage。留一 seed 交叉验证依次比较：constant、static-start pullback、static pullback + start condition、time-averaged dynamic pullback、dynamic pullback + condition trajectory、再加入实际 bilinear ratio 的 post-hoc 模型，以及 $I+C$ oracle。static/dynamic 的 condition-augmented 模型成对报告，避免把增加一个特征误认为时间分辨率的收益；不能把 post-hoc bilinear 或 oracle 模型称为训练前预测器。

预先固定三条判据：

1. **批次稳健性**：预指定 Scalar-2 的 SAM-benefit interaction 和强 SGD gauge-damage 效应，至少 3/4 batches 与 seed-mean 同号；正式 CI 以每 seed 的四批次均值 bootstrap，不能把 80 个 batch 行当 80 个 seeds。
2. **动态增量**：dynamic-pullback 的 leave-one-seed-out R²/MAE 必须在两个任务方向都优于 static-start，才支持“time-varying pullback 有预测增量”；只在一个方向成立则报告任务依赖，不提升为一般机制。
3. **负对照与停止规则**：orthogonal 与 Identity 的轨迹差必须停留在数值误差。若独立 seeds 上 P1 的关键符号不能在 3/4 batches 重复，停止 formal；若 exact $I+C$ 仍不能明显优于 pullback 模型，则下一步改为 step-level old-geometry 记录，而不是进入五任务扩展。

实现复用 [run_stage3_p1_mature_gauge.py](run_stage3_p1_mature_gauge.py) 的共享前缀/分叉逻辑，并通过 `dynamic_diagnostics: true` 开启 P2 文件；[small_cl/diagnostics.py](small_cl/diagnostics.py) 用 Kronecker-sum 谱精确计算 factor Jacobian condition，避免逐时间点显式构造 $1024\times256$ Jacobian。配置与分析代码：[stage3_p2_dynamic_chart_pilot.yaml](configs/stage3_p2_dynamic_chart_pilot.yaml)、[stage3_p2_dynamic_chart_formal.yaml](configs/stage3_p2_dynamic_chart_formal.yaml)、[analyze_stage3_p2_dynamic_chart.py](analyze_stage3_p2_dynamic_chart.py)。

#### P2 pilot 结果：多批次复现 chart sensitivity，但动态 pullback 假设未通过

Pilot 的 100/100 branches 全部完成，产生 400 个 run-batch 行与 2,000 个动态时间点。Taylor 恒等式最大 closure error 为 $1.47\times10^{-7}$；orthogonal 相对 Identity 的最大真实 loss-change 差与动态 pullback 差分别为 $2.15\times10^{-6}$、$6.23\times10^{-6}$，负对照通过。

Forward 中 P1 的关键 gauge 现象在所有批次复现：Scalar-2 的 SAM-benefit interaction 为 **−.00576 [−.00621,−.00542]**，Scalar-.5 为 **+.00164 [.00033,.00259]**，两者均 4/4 batches 同号且各 batch CI 排除 0。Reverse 的 Scalar-2 interaction 为 `−.00010 [−.00414,.00634]`，只有 2/4 batches 同号；因此它没有独立复现 P1 reverse 的边界效应。SGD gauge damage 仍显示强 chart sensitivity，例如 reverse Anisotropic-4 为 **+.3144 [.1564,.5203]**，4/4 batches 同号。

但是主预测假设失败。下表是对 gauge damage 的 leave-one-seed-out $R^2$；除固定 $I+C$ 外，其余模型在训练 seeds 上拟合、在完整 held-out seed 的四个 gauges 上测试。$I+C$ 使用理论固定系数 1，不做拟合。

| order / method | static pullback | dynamic pullback | static + condition | dynamic + condition | fixed $I+C$ |
|---|---:|---:|---:|---:|---:|
| forward / SGD | −.244 | −.404 | −.840 | +.006 | +.783 |
| forward / SAM | −.253 | −.390 | −.853 | −.006 | +.790 |
| reverse / SGD | −.149 | −1.051 | −.135 | −1.106 | −1.708 |
| reverse / SAM | −.097 | −1.393 | −.143 | −1.438 | −1.982 |

所以 time-averaged dynamic pullback 在两个方向都没有优于 static start，且 reverse 明显恶化；按预注册规则，**不启动 720–739 的 P2 formal，也不进入五任务扩展**。Forward 的 $I+C$ 有效而 reverse 失败，进一步定位到 reverse 的 epoch-level Taylor residual：四段更新仍太长，固定 $I+C$ 的 MAE 为 `.206/.210`（SGD/SAM），甚至大于 gauge damage 的平均绝对值 `.169/.167`。

这组否定结果把故事再次收缩为：

> **Factor chart 对轨迹有因果作用，但 clean full-batch pullback 的时间平均不是遗忘的充分预测统计量；任务方向不仅调节效应大小，还调节局部二阶展开是否足以描述轨迹。**

![P2 dynamic chart diagnostics](figures/stage3_p2_dynamic_chart.png)

结果与绘图代码：[analysis/stage3_p2_dynamic_chart_pilot](analysis/stage3_p2_dynamic_chart_pilot)、[plot_stage3_p2_dynamic_chart.py](plot_stage3_p2_dynamic_chart.py)。

#### P2b：细粒度 Taylor 与真实 minibatch pullback 审计

P2b 是 P2 停止规则触发后的定位实验，不是新一轮大网格。它保留两个任务方向、Identity/Scalar-.5/Scalar-2、SGD/SAM `.005`，使用独立 seeds 705–707；每 10 个 minibatch 保存一次有效权重，把 4 个 epoch 段细化为约 32 段。相同间隔还在真实训练 batch 上计算 clean gradient pullback，并在更新后比较

\[
I_{\rm actual}=g_{old}^T\delta W,
\qquad
I_{\rm clean}= -\eta g_{old}^T\mathcal M_{A,B}[g_{batch}]
 +g_{old}^T\delta W_{bilinear}.
\]

SGD 下二者应只差数值误差；SAM 下差值量化 perturbed-gradient correction 对一步干扰的贡献。主诊断是 coarse/fine 路径起终点 loss change 必须完全相同，以及 fine Taylor residual 是否尤其在 reverse 显著收缩。配置与分析：[stage3_p2b_step_taylor_pilot.yaml](configs/stage3_p2b_step_taylor_pilot.yaml)、[analyze_stage3_p2b_step_taylor.py](analyze_stage3_p2b_step_taylor.py)。

P2b 的 36/36 branches 完成，coarse/fine 的 endpoint loss-change 最大差为 0；1,152 个 sampled steps 中，SGD 的 clean-pullback + bilinear 最大绝对误差仅 $2.65\times10^{-7}$，run-level 相对 MAE 约 $10^{-6}$、$R^2=1$。SAM 的**一阶干扰预测**相对 MAE 为约 `1.2%–2.3%`，$R^2>.9995$。因此局部公式本身正确，并且 clean-gradient 预测与 SAM 实际一阶干扰很接近；但这个比例不是 $\|\delta W_{SAM}-\delta W_{clean}\|/\|\delta W_{clean}\|$，不能据此声称完整 SAM 更新只改变 `1%–2.3%`。

把 4 段改为约 32 段后，mean absolute Taylor residual 如下：

| order | Identity | Scalar .5 | Scalar 2 |
|---|---:|---:|---:|
| forward / SGD | .0398 → .0035 | .0303 → .0160 | .0470 → .0122 |
| forward / SAM | .0394 → .0036 | .0295 → .0162 | .0477 → .0124 |
| reverse / SGD | .1194 → .0089 | .2369 → .1568 | .0702 → .0352 |
| reverse / SAM | .1081 → .0084 | .2536 → .1519 | .0701 → .0397 |

Identity 的 reverse 失败主要是分段过粗；但非正交 gauges 的余项仍大。为避免 10-step chord 抵消真实振荡，P2c 在相同 reverse seeds 上保存全部 316 个 optimizer steps。真实 per-step path 下，Identity residual 仍收缩至 `.014/.017`（SGD/SAM），但 Scalar-.5 为 `1.134/1.139`，Scalar-2 为 `.166/.137`。这不是 Taylor identity 或 pullback 实现错误，而是路径本身改变：平均累计 step path 为 Identity `48.3`、Scalar-.5 `108.2`、Scalar-2 `71.5`，平均最大单步 norm 为 `.48/1.07/.89`。10-step chord 曾隐藏大量往返运动；非正交 chart 下单个 optimizer step 已不属于足够局部的区域。

因此 P2b/P2c 支持的新机制分解是：

1. clean minibatch pullback 准确描述 SGD 的**局部一步方向**；
2. SAM 相对 clean-gradient 预测的一阶干扰残差约为 `1.2%–2.3%`；完整 effective-weight correction 的范数和方向角此前没有直接测量，必须由 P3 补充；
3. 静态或 epoch-mean pullback 预测终点失败，主要不是公式误差，而是 gauge 改变了实际 step scale、振荡路径及随路径变化的 old-task geometry；
4. 必须先匹配累计 effective step path，才能判断 gauge 是否还通过方向而非纯运动预算改变遗忘。

![P2b ten-step Taylor diagnostics](figures/stage3_p2b_step_taylor.png)

![P2c every-step Taylor diagnostics](figures/stage3_p2c_every_step_taylor.png)

P2d 因而冻结由 P2 seeds 700–704 得到、跨任务方向和 SGD/SAM 都稳定的 path-ratio 校准：Identity/Orthogonal lr `.1`，Scalar-.5 `.046`，Scalar-2 `.062`，Anisotropic-4 `.063`；在独立 seeds 708–712 上验证累计 path 是否接近 Identity，并重新估计 gauge damage 与 SAM-benefit interaction。该校准是近似 path matching，不是逐步强制归一化；必须同时报告 path ratio、current loss 与 net drift。配置与分析：[stage3_p2c_every_step_taylor.yaml](configs/stage3_p2c_every_step_taylor.yaml)、[stage3_p2d_path_matched_gauge_pilot.yaml](configs/stage3_p2d_path_matched_gauge_pilot.yaml)、[analyze_stage3_p2d_path_matched_gauge.py](analyze_stage3_p2d_path_matched_gauge.py)。

P2d 的 100/100 branches 与 10/10 prefixes 全部完成。orthogonal 的最大 path-ratio error 和 old-damage difference 分别为 $9.46\times10^{-8}$、$7.17\times10^{-7}$。独立 cohort 中的 SGD path ratio 为：

| order | Scalar .5 | Scalar 2 | Anisotropic 4 |
|---|---:|---:|---:|
| forward | 1.139 [1.055,1.251] | .947 [.876,1.006] | .931 [.787,1.096] |
| reverse | 1.126 [1.062,1.194] | .846 [.748,.951] | 1.051 [.847,1.254] |

校准不是精确逐步归一化，但已把原来的 `1.57–2.24` 倍压至约 `.85–1.14`。此时 raw gauge damage 全部变为负值，但这是低 lr 带来的 lower current progress，不能称为更少遗忘。最关键的 current-loss-matched SGD gauge damage 如下，所有非正交 CI 均跨 0：

| order | Scalar .5 | Scalar 2 | Anisotropic 4 |
|---|---:|---:|---:|
| forward | −.0160 [−.1727,.1132] | −.0249 [−.1332,.0637] | +.0286 [−.0633,.1308] |
| reverse | +.0349 [−.1477,.2175] | −.1307 [−.4193,.0543] | −.0000 [−.2856,.2645] |

所以 P1 的巨大 raw SGD gauge-damage effect 在同时控制有效路径尺度和新任务进度后没有复现。更准确地说，结果**强烈提示** chart 首先改变隐式 preconditioner 的尺度和塑性速度；目前没有证据表明相同 current-task progress 下，非正交 chart 本身系统性增加遗忘，但 5-seed 宽 CI 跨 0 不能作为等价证明。

SAM-benefit interaction 则依赖 matching estimand：

| order / gauge | endpoint | step-path matched | current-loss matched |
|---|---:|---:|---:|
| forward / Scalar .5 | **−.00182** | −.00052 | **−.00974** |
| forward / Scalar 2 | **−.00341** | −.00041 | **−.01067** |
| forward / Aniso. 4 | **−.00098** | +.00052 | −.00741 |
| reverse / Scalar .5 | **−.00503** | **+.00831** | +.00226 |
| reverse / Scalar 2 | −.00496 | −.00193 | **−.00587** |
| reverse / Aniso. 4 | −.00289 | **+.00402** | −.00205 |

粗体表示 5-seed bootstrap CI 排除 0。Forward 的非正交 interaction 在 step-path matching 后全部跨 0，但在 current-loss matching 下 Scalar-.5/2 显著为负；reverse Scalar-.5/Aniso.-4 在相同 step path 下反而显著为正。这不是矛盾，而是说明 SAM 可以用不同的路径效率达到相同运动预算或相同新任务进度，二者回答不同问题。

P2d 后最严格的新结论是：

> **P2c/P2d 强烈提示 Factor gauge 的巨大 raw forgetting effect 有相当一部分来自隐式有效步幅和 plasticity speed；但尚未用等价检验排除独立方向效应。SAM 的相对轨迹效应仍会随 chart、任务方向与 matching estimand 改变，而完整 effective-weight SAM correction 的幅度和单位运动安全性需要 P3 直接测量。**

当前仍是 5-seed calibration pilot。下一步若继续机制确认，应使用逐步 effective-step normalization，并在更密的 current-loss checkpoints 上做 20-seed 验证；在此之前不进入五任务外推。

![P2d path-matched gauge diagnostics](figures/stage3_p2d_path_matched_gauge.png)

结果与绘图代码：[analysis/stage3_p2d_path_matched_gauge_pilot](analysis/stage3_p2d_path_matched_gauge_pilot)、[plot_stage3_p2d_path_matched_gauge.py](plot_stage3_p2d_path_matched_gauge.py)。

### 0.18 Stage-3 P3：逐步有效权重归一化与共同状态反事实

P3 直接回答 P2d 尚未解决的问题：**当每个 optimizer step 的有效权重范数严格相同时，SAM 是否仍在 Factor LoRA 中产生更安全的单位更新方向？** 令

\[
\delta W_k=q_ku_k,\qquad q_k=\|\delta W_k\|_F,\qquad \|u_k\|_F=1,
\]

则

\[
I_k=q_k\,g_{old,k}^Tu_k,
\qquad
C_k=\frac12q_k^2u_k^TH_{old,k}u_k.
\]

P3 将步幅 $q_k$、单位方向的一阶安全性 $\bar I_k=g_{old,k}^Tu_k$ 和单位方向曲率 $\bar C_k=u_k^TH_{old,k}u_k$ 分开记录，并增加

\[
N_{eff}=\frac{(\sum_kq_k)^2}{\sum_kq_k^2},
\]

以检查相同累计路径是否仍由少数大步主导。

#### P3 设计

| 项目 | 预注册设置 |
|---|---|
| 公共状态 | 共享 Dense Task-A endpoint；Task B 共享 SGD warmup 1 epoch，保证 $A,B\neq0$ |
| Pilot directions | `−30→+30`、`+45→−45`；`+30→−30` 只在 formal 加入 |
| Gauges | Pilot：Identity、Orthogonal、Scalar `.5/2`；formal 再加入 Anisotropic `4` |
| 方法 | SGD、SAM，$\rho_W=.005$，momentum/weight decay 为 0 |
| 更新模式 | raw；逐步 exact-norm normalized |
| 目标步幅 | 从独立 P2b Identity-SGD seeds 705–707 的每步中位数冻结；扫描 `.25/.5/1.0×` |
| Pilot seeds | 800–804；2 directions × 4 gauges × 2 methods × 4 modes × 5 seeds = 320 branches |
| Progress 记录 | 每 5 steps 在固定完整 evaluation set 上记录 current/old loss，不再只做 epoch 插值 |
| 方向诊断 | 每 20 steps、两个非重叠 old-task batches 上精确计算 $I,C,R,\bar I,\bar C$ |
| 共同状态反事实 | 沿 Identity-SGD normalized 参考轨迹，对同一状态、同一 minibatch、相同 $q_k$ 同时构造各 gauge 的 SGD/SAM 候选更新，不提交分支更新 |
| 主 estimand | 共同 current-loss support 上的 progress AUC：相同新任务进度下的旧任务损伤差 |
| 次 estimand | 共同累计 path support 上的 path AUC：相同运动预算下的旧任务损伤差 |
| 描述性指标 | endpoint benefit、gauge damage、gauge × SAM interaction |

因子候选步为 $p_A,p_B$ 时，P3 通过一维求根选择最小正 $\alpha_k$：

\[
\left\|s\left[\alpha_kBp_A+\alpha_kp_BA+\alpha_k^2p_Bp_A\right]\right\|_F
=q_k^{target}.
\]

由于二次项会使缩放略微旋转有效更新，代码同时审计 raw/normalized $\delta W$ cosine、求根 $\alpha$、目标范数误差和 exact factor-step closure。SAM 不再使用 P2b 的干扰残差代替 update correction，而是直接报告

\[
r_k^{SAM}=
\frac{\|\delta W_k^{SAM}-\delta W_k^{clean}\|_F}
{\|\delta W_k^{clean}\|_F+\epsilon},
\qquad
1-\cos(\delta W_k^{SAM},\delta W_k^{clean}).
\]

#### 不看遗忘收益的 scale 选择规则

在读取 P3 retention benefit 前已冻结：选择在两个方向均通过全部规则的**最大** scale。规则为 target relative error $\le10^{-5}$、candidate closure $\le5\times10^{-5}$、$\alpha_{max}\le16$、raw/normalized cosine $\ge.995$、共同状态 Taylor residual median/p90 $\le.01/.03$、Identity SGD/SAM 的 current-loss common-support width $\ge.05$。若没有 scale 通过，不启动 formal；不能根据遗忘方向重新选 scale。

#### Pilot 完整性与 scale 选择

CPU smoke 已完成 8/8 branches；正式 pilot 完成 320/320 branches、10/10 共享前缀、320/320 step-geometry 文件和 30/30 counterfactual 文件，共 7,680 个共同状态反事实单元。自动完整性审计为：

| 审计量 | 结果 |
|---|---:|
| 最大 normalized target error | $1.98\times10^{-7}$ |
| 最大 candidate closure error | $2.75\times10^{-5}$ |
| 最大 Orthogonal endpoint 差 | $1.22\times10^{-6}$ |
| 最小 raw/resolved cosine | $0.99938$ |
| 通过全部冻结规则的最大 scale | **1.0** |

三个 scale 在两个方向均通过规则；因此按预注册策略选择最大值 `q=1.0`，不是按 retention benefit 选择。该条件把 SGD/SAM 和全部 gauges 的每步 $q_k$、累计路径、$\sum q_k^2$、最大步幅与 $N_{eff}$ 同时固定；例如 `−30→+30` 的 path 为 `28.88409`、$\sum q_k^2=2.77490$、$N_{eff}=300.66$，`+45→−45` 分别为 `47.08830`、`7.45772`、`297.32`。

#### Pilot 主要结果

Identity Factor 在选定 `q=1.0` 下的配对 SAM benefit（正值表示 SAM 的旧任务损伤更小）为：

| 方向 | Endpoint | Progress AUC（主） | Path AUC（次） |
|---|---:|---:|---:|
| `−30→+30` | $.00745\;[.00551,.00909]$ | **$.00136\;[.00056,.00215]$** | $.00688\;[.00565,.00761]$ |
| `+45→−45` | $.00722\;[.00539,.00915]$ | $.00067\;[-.00091,.00244]$ | $.00769\;[.00698,.00868]$ |

因此 endpoint/path 都显示 SAM benefit，但主 estimand 只在 forward 方向稳定；reverse 的五个 seed 中两个为负。**SAM 提高单位运动效率**与**SAM 改善相同学习进度下的 retention–plasticity 权衡**不是同一个结论。

逐步归一化把 non-orthogonal raw progress-gauge damage 明显压低，但未全部消除：Scalar-.5 在 forward/reverse 从 `.165/.167` 降到 `.066/.026`；Scalar-2 从 `.031/.179` 变为 `-.003/.104`。与此同时，`q=1` 的 progress-AUC gauge×SAM interaction 没有任何正向放大证据：forward Scalar-.5 为 $-.00113\;[-.00160,-.00076]$，Scalar-2 为 $-.00128\;[-.00326,.00070]$；reverse 两个区间均跨 0。Orthogonal 在所有条件保持数值等价。结论是：

1. raw gauge effect 的大部分确实来自隐式步幅/塑性混杂；
2. exact step matching 后仍有 chart-dependent retention–plasticity effect，不能说“纯方向效应等价于零”；
3. 剩余 effect 没有表现为 non-orthogonal chart 放大 SAM，forward Scalar-.5 反而显著削弱 SAM benefit。

共同状态、同 batch、同 $q_k$ 的反事实进一步定位了 SAM correction。Identity 下，SAM 的 exact old-direction safety 在 forward/reverse 分别为 $7.43\times10^{-5}\;[-2.06\times10^{-5},1.58\times10^{-4}]$ 和 $4.31\times10^{-5}\;[-2.56\times10^{-5},1.19\times10^{-4}]$，均未排除 0；但 new-task cost 均为正。SAM 同时降低单位一阶干扰（约 `.00125–.00133`）并增加单位方向曲率（curvature-reduction 为约 `−.0113` 至 `−.0125`），两者抵消后没有形成稳定的净单步保护。直接测得的 effective-weight SAM correction 比例为 raw 候选约 `1.6%–2.0%`、归一化后约 `0.95%–1.24%`；这才是完整 correction 的测量，不再误用 P2b 的预测误差。

因此 P3 pilot 支持的新机制是：

> **SAM 在 Factor LoRA 中不是简单地逐步选择更低旧任务曲率的方向。它施加约 1% 的有效权重方向修正，降低一阶冲突却提高方向曲率，并通过整条非局部轨迹在特定任务方向上改善 retention–plasticity 权衡。Factor chart 既改变运动尺度，也改变该折衷，但目前没有 LoRA/gauge 普遍放大 SAM 的证据。**

#### P3 formal（已冻结，等待空闲 GPU）

Pilot 只用于数值尺度选择，不能作为等价证明。Formal 使用新 seeds 820–839，加入 `+30→−30` 与 Anisotropic-4，仅比较 raw 与选定 `q=1`：`3 directions × 5 gauges × 2 methods × 2 modes × 20 seeds = 1,200 branches`。主指标仍是 progress AUC；path AUC 为次指标，endpoint 仅描述。正式等价检验在运行前冻结 progress-AUC interaction 的 SESOI 为 $\pm0.001$，采用 90% CI/TOST 判定；该界限约为独立 P1 formal 中最小稳健 non-orthogonal current-loss interaction 的一半。七个 worker 仅在对应 GPU 显存占用低于 1 GB 后启动，不与当前外部作业争用。

实现、配置与分析入口：

- [small_cl/normalized_steps.py](small_cl/normalized_steps.py)；
- [prepare_stage3_p3_target_profiles.py](prepare_stage3_p3_target_profiles.py) 与 [stage3_p3_target_profiles.json](configs/stage3_p3_target_profiles.json)；
- [stage3_p3_step_normalized_pilot.yaml](configs/stage3_p3_step_normalized_pilot.yaml)；
- [stage3_p3_step_normalized_formal.yaml](configs/stage3_p3_step_normalized_formal.yaml)；
- [run_stage3_p3_step_normalized.py](run_stage3_p3_step_normalized.py)；
- [analyze_stage3_p3_step_normalized.py](analyze_stage3_p3_step_normalized.py)；
- [plot_stage3_p3_step_normalized.py](plot_stage3_p3_step_normalized.py)；
- [P3 pilot figure](figures/stage3_p3_step_normalized.png) 与 [analysis tables](analysis/stage3_p3_step_normalized_pilot/integrity.json)。

## 1. 收缩后的论文故事

核心现象不是“LoRA 一定放大 SAM”，而是：在某些任务方向和比较基准下，Factor LoRA 中的 SAM 即使经过 current-loss、effective-drift 或 exact step-path matching，仍会改变旧任务损伤；同样的轨迹现象没有被固定低维空间或直接 rank-manifold 更新稳定复现。P3 已确认 P1 中巨大的 raw gauge forgetting 不能直接当作方向机制，因为精确控制每步运动预算后它会大幅缩小甚至改变 endpoint 符号；但 chart-dependent progress effect 仍然存在。共同状态反事实显示，SAM 的约 1% 有效权重修正降低一阶冲突、同时提高方向曲率，净单步安全性不稳定，最终收益来自任务条件性的整条轨迹，而不是单一 flatness scalar。

Factor LoRA 写成

\[
W=W_A+sBA.
\]

虽然 $BA$ 的有效更新保持低秩，优化实际发生在因子坐标 $(A,B)$。若有效权重梯度为 $G=\nabla_WL$，lr 为 $\eta$ 且 momentum/weight decay 为 0，一次因子 SGD 的有效更新为

\[
\delta W
=-\eta s^2\left(BB^TG+GA^TA\right)
+\eta^2s^3GA^TB^TG.
\]

第一项是 $-\eta\mathcal M_{A,B}[G]$，第二项是同时更新两个 factors 产生的 bilinear correction。Gauge 变换 $A\to cA,B\to B/c$ 保持 $BA$ 和模型函数不变，却使 $BB^TG$ 缩放为 $c^{-2}$、$GA^TA$ 缩放为 $c^2$。因此 Factor LoRA 给有效权重空间引入了一个随轨迹变化、依赖 gauge 的 pullback metric，同时还有非线性的 bilinear correction。它不仅旋转更新方向，也改变同一 factor lr 对应的有效权重步幅。SAM 在 factor gradient 上寻找扰动，所以即使显式匹配 $\|\Delta W_{\mathrm{perturb}}\|_F$，映射回有效权重的扰动方向、实际 step path 和学习新任务的速度仍可能不同。

遗忘由实际新任务路径决定：

\[
\Delta L_{old}=\sum_k\left(I_k+C_k+R_k\right),\qquad
I_k=g_{old,k}^T\delta_k,\quad
C_k=\tfrac12\delta_k^TH_{old,k}\delta_k.
\]

由此得到当前最有证据的新故事：

> **平坦型优化在 LoRA 下的条件性收益不是由“参数更少”或“有限差分天然更准确”解释。Factor chart 会通过隐式 preconditioner 改变有效步幅、塑性速度和振荡路径；SAM 还会改变 effective-weight 更新，但在排除步幅后，这种方向修正是否提高保留–塑性效率仍需 P3 确认。**

P1 已因果确认“factor chart 会改变轨迹”；P2/P2b/P2c 进一步表明静态或 epoch-mean pullback 不能预测终点，而局部 minibatch pullback 公式本身是准确的；P2d 强烈提示巨大 raw gauge damage 受 step scale/plasticity 显著混杂，但尚未完成等价证明。因而论文不能简化成“$\mathcal M=JJ^T$ 单变量解释 SAM”。更准确的机制对象是共同演化的 $\mathcal M_k$、实际 step budget、bilinear/SAM correction、旧任务 $I/C/R$ 和 current-task progress。

Merge-reset 只改变每个任务开始时的 factor chart，不会使下一任务变成 Dense 更新，因此不会自动消除这一问题。低秩 reachable-space/safe-route 指标仍可作为后续扩展，但已不再是当前中心故事。

## 2. 三个研究问题

### RQ1：遗忘是否由真实未来更新方向解释？

将任务 (t+1) 的训练轨迹分为更新 (delta_k=W_{k+1}-W_k)。对旧任务 (i)：

\[
\Delta L_i
=\sum_k\left[
g_{i,k}^{\top}\delta_k
+\frac12\delta_k^{\top}H_{i,k}\delta_k
+R_{i,k}
\right].
\]

主文记号：

\[
I_{i,k}=g_{i,k}^{\top}\delta_k,
\qquad
C_{i,k}=\frac12\delta_k^{\top}H_{i,k}\delta_k.
\]

必须在同一批 transitions 上比较五类解释：

1. 仅更新范数 (|\delta_k|)；
2. 仅全局曲率 (lambda_{\max}(H_{i,k}))；
3. 仅一阶干扰 (I_{i,k})；
4. 范数与一阶干扰，再加入方向曲率 (C_{i,k})；
5. 完整 pathwise Taylor prediction (sum_k(I_{i,k}+C_{i,k}))。

可写入主结论的最低门槛是：加入 (C_{i,k}) 后，在控制更新范数和 (I_{i,k}) 的模型中仍带来稳定的 out-of-seed 增量解释力；pathwise prediction 还应比 endpoint prediction 更接近真实 loss change。只观察到 (C) 与遗忘的边际相关不够。

### RQ2：收益来自低维可达集合，还是 Factor LoRA 的坐标度量？

按判别力从弱到强比较：维度匹配 Random subspace、Fixed initial/mature tangent、Projected direct-rank manifold、Balanced Factor 和 Standard Factor。所有条件共享精确相同的 $W_A$，并匹配 current-task loss、effective drift 与 effective-weight SAM radius。

进一步使用函数保持的 gauge intervention：

\[
(A,B)\mapsto(cA,B/c),\qquad c\in\{0.3,1,3\}.
\]

若只由有效 rank manifold 决定，函数起点、lr 和 $\rho_W$ 相同时，结果应近似 gauge-invariant；若 Factor pullback metric 是机制的一部分，则 SAM 的轨迹、$I/C$ 和遗忘收益会随 $c$ 系统变化。极端 $c\in\{0.1,10\}$ 只作稳定性压力测试，不能把欠拟合产生的低遗忘作为收益。

P1 已把这个检验推进到成熟非零 factors 和一般 $S$：orthogonal $S$ 完全等价，非正交 $S$ 显著改变轨迹，因而排除了“只是有效函数或 tangent range 改变”。P2–P2c 表明 static/epoch-mean pullback 的跨 seed 预测失败，而真实 minibatch 的局部 pullback closure 成立。P2d 又显示，在近似控制 step path 并匹配 current loss 后，非正交 gauge 的巨大 SGD damage 不再显著。因此 RQ2 的当前答案是：**Factor chart 确实因果参与优化，但其最大 raw 效应主要通过隐式有效步幅和 plasticity speed 实现；相同学习进度下剩余的方向性 SAM interaction 是条件性的，不能由静态 metric scalar 或纯 path length 单独解释。**

### RQ3：何种任务几何使曲率修正转化为更少遗忘？

任务角度只用于产生不同方向；每个有向任务对必须实测：旧梯度与可达新梯度 cosine/alignment、prospective Hessian/GGN curvature、realized path 的 $I/C$/GGN cost，以及 endpoint/loss-matched/drift-matched damage。

主假设不是“曲率越大收益越大”，而是交互条件：SAM 的 curvature reduction 只有在没有付出更坏的一阶 interference 时才会减少遗忘。Stage-2C 的 `+45→−45` 已提供“$C$ 改善但遗忘恶化”的关键反例。

### 附录候选 A：LoRA 是否减少了安全更新路线？

令 (Q_r\in\mathbb R^{1024\times d_r}) 是当前有效权重可达空间的正交基，(G_i\succeq0) 是旧任务 empirical GGN，(g_{\mathrm{new}}) 是新任务梯度。定义：

\[
\mathcal P_r(\lambda)
=(Q_r^Tg_{\mathrm{new}})^T
(Q_r^TG_iQ_r+\lambda I)^{-1}
(Q_r^Tg_{\mathrm{new}}).
\]

(\mathcal P_r) 衡量“与新任务相关且对旧任务代价较低”的 safe-plasticity headroom。

为了把“安全路线数量”与“新梯度对齐”分开，本实现明确采用下面的 $\tau_r$ 操作定义。令 $U_{\mathrm{unsafe}}$ 包含 GGN 特征值大于
(eta\lambda_{\max}(G_i)) 的方向，

\[
P_{\mathrm{safe}}=I-U_{\mathrm{unsafe}}U_{\mathrm{unsafe}}^T,
\qquad
\tau_r=
\frac{\mathrm{tr}(Q_rQ_r^TP_{\mathrm{safe}})}
{\mathrm{tr}(P_{\mathrm{safe}})}.
\]

$\tau_r\in[0,1]$ 表示完整安全子空间中有多少维度仍可由当前参数化访问。附录同时报告

\[
\tau_r^{\mathrm{quality}}
=\frac{\mathrm{tr}(Q_rQ_r^TP_{\mathrm{safe}})}{d_r},
\]

即“可达方向中有多少比例是安全的”。阈值 (eta) 必须做敏感性分析；默认 (eta=0.01)。

真实更新在 GGN 中的单位范数暴露为

\[
\kappa_{i,k}
=\frac{\delta_k^TG_{i,k}\delta_k}{\|\delta_k\|^2}.
\]

不预设 rank 越小越好。低 rank 可能降低 $\kappa$，也可能同时降低 $\tau_r$、新梯度覆盖和 $\mathcal P_r$，因此核心预测是非单调 trade-off。

### 附录候选 B：GAM 的有限差分、修正强度与轨迹解释

竞争解释为：

1. **Approximation**：有限差分 (Hv) 更准确；
2. **Correction**：SAM/GAM 产生更强或更有效的方向曲率修正；
3. **Trajectory**：最终实际路径具有更低的历史曲率暴露。

判别方式：

- `gam_fd` 与 `gam_exact` 在关键设置中配对；
- 同时报告有限差分绝对误差和相对误差，扫描 radius；
- 记录每个 epoch 的 curvature-correction norm；
- 用 matched seed 比较 `SAM/GAM − SGD` 的 FAA、forgetting、pathwise $C$、$\kappa$、$\mathcal P_r$ 与 $\tau_r$；
- random perturbation 只作为次要 direction control。

如果有限差分误差与 optimizer gain 关系弱，而 $\mathcal P_r$、$\tau_r$、$\kappa$ 和 pathwise curvature cost 能预测收益，就应明确排除旧的“低维提高近似精度”故事。

当前正式结果来自 SAM，不能用于证明“LoRA 使有限差分 HVP 更准确”。严格检验该假设时，必须分别在 Dense/Factor 的实际训练参数空间，沿 GAM 真正使用的归一化梯度方向配对 `gam_fd` 与 `gam_exact`，并检验 FD error 是否能预测 optimizer gain。

## 3. 统一模型与参数化控制

所有实验使用同一个模型：

\[
784\rightarrow32\rightarrow32\rightarrow10,
\]

只更新中间矩阵 (W\in\mathbb R^{32\times32})。外层、分类头和全部 bias 冻结。所有条件共享同一个 base checkpoint，并从同一个有效权重 (W_0) 开始。

| 参数化 | 实现 | 研究目的 |
|---|---|---|
| Dense middle | $W=W_0+D$ | 完整 1024 维更新基线 |
| Random linear subspace | $W=W_0+Q_{\mathrm{rand}}z$ | 用与标准 LoRA 初始 tangent 相同的 $32r$ 维随机正交子空间排除纯维度效应 |
| Fixed LoRA tangent | $W=W_0+Q_{\mathrm{tan}}z$ | 固定标准初始化 $B=0$ 时的 $\{\mathrm dB A_0\}$ 空间，去掉 Jacobian 漂移与双线性项 |
| Fixed mature tangent | 在成熟 Factor tangent 上固定 240 维正交基 | 匹配完整 rank-$r$ tangent 维度但禁止后续漂移 |
| Projected direct rank | 对 $\Delta W$ 做梯度投影和 truncated-SVD retraction | 允许 rank manifold 移动但不使用 $A,B$ 因子坐标 |
| Standard factor LoRA | $W=W_0+(\alpha/r)BA$ | 同时保留动态 tangent、Jacobian 与双线性参数化 |
| Balanced factor LoRA | 同上，每步对 $BA$ 做 SVD canonical balancing | 抑制 $A\to cA,B\to B/c$ 的 gauge 自由度 |

对 random/fixed subspace，维度匹配 standard factor LoRA 的真实初始化 tangent：

\[
d_{\mathrm{init}}=32r.
\]

Standard factor LoRA 初始化为 $B=0$，所以第一个训练点的 Jacobian rank 是 $32r$，训练后通常增长到完整 rank-$r$ 流形维度 $r(64-r)$。因此 Random 与 Fixed 控制回答“相同初始维度/固定路线会怎样”，Factor LoRA 额外引入的正是动态 Jacobian 扩张与双线性参数化。这一维度变化必须逐任务报告。

Balanced LoRA 的 SVD balancing 保持 $BA$ 和模型输出不变。由于 momentum 没有唯一 gauge transform，Stage-2B 的 Task B 固定 momentum $=0$。Gauge intervention 则保持同一初始 $BA=0$ 和有效权重，仅按 $A\to cA,B\to B/c$ 改变 factor chart。

## 4. 两级任务

### 4.1 两任务 controlled teacher–student

两位 teacher 共享外层和 base middle weight，只改变 rank-(q) 的中间矩阵目标：

\[
W_i^*=W_0+sU_iV_i^T.
\]

构造 (U_1,V_1) 时，使其与 (U_0,V_0) 的 principal angle 等于指定的
(phi\in\{0^\circ,45^\circ,90^\circ\})。操纵变量为：

- teacher target rank (q)；
- learner rank (r)；
- target subspace angle (phi)。

这里显式控制的是 teacher target-update 主空间，而不是直接宣称控制了训练后曲率空间。训练起点仍必须实测梯度余弦与 GGN top-subspace overlap。两任务设置使用 step-level path，适合做 RQ1 的干净因果验证。

入口：[teacher_student.yaml](configs/teacher_student.yaml)。

### 4.2 五任务 Rotated MNIST

Rotated MNIST 用于验证真实非线性分类现象。旋转角度只作为 task similarity manipulation：

\[
\theta_t=-\Omega/2+t\Omega/(T-1),
\qquad
\Omega\in\{40^\circ,120^\circ\}.
\]

绝不把角度差直接称为“曲率重叠”。每个 transition 实际测量：

- old/new gradient cosine；
- old/new GGN top-(k) subspace overlap；
- 当前 reachable basis overlap 与 realized-update coverage。

入口：[pilot.yaml](configs/pilot.yaml)。

## 5. 优先分析即时遗忘

主分析只使用刚完成任务 (t) 到训练任务 (t+1) 的旧任务 (t)：

\[
L_t(W_{t+1})-L_t(W_t),
\qquad
R_{t,t}-R_{t+1,t}.
\]

此时旧任务更接近自己的训练驻点，二阶解释最干净。代码把这些记录单独写入 `immediate_transitions.json`。

更早任务仍保存在 `transitions.json`，但作为附录分析；此时 $g_i$ 通常不接近零，必须保留一阶干扰，不能只用 curvature 解释。默认只为即时旧任务计算完整 pathwise 分解，以控制计算量；更早任务保留 endpoint 分解。

## 6. 主次指标

### 6.1 遗忘、收益与 interaction

即时旧任务损伤定义为

\[
D_L=L_{old}^{end}-L_{old}^{start},\qquad
D_A=Acc_{old}^{start}-Acc_{old}^{end}.
\]

因此 $D$ 越小越好。所有结果表中的 `SAM reduction` 使用

\[
B_{SAM}=D_{SGD}-D_{SAM};
\]

$B_{SAM}>0$ 表示 SAM 更少遗忘。参数化 interaction 为

\[
I_{Factor-control}=B_{SAM}^{Factor}-B_{SAM}^{control}.
\]

同一 seed 内先做差，再对 seed 做 bootstrap；不能从两个独立均值的 CI 推断 interaction。

### 6.2 可塑性和漂移匹配

- **Current-loss matched**：在 SGD/SAM 两条 epoch 轨迹的重叠范围内，选择双方都能达到的最低共同 current-task loss，分别对第一次 crossing 线性插值旧任务损伤，再作 $D_{SGD}-D_{SAM}$。
- **Effective-drift matched**：令 $d=\|W-W_{task\ start}\|_F$，选择两条轨迹都达到的最大共同 $d$，按相同方式插值并作差。
- Endpoint、loss-matched、drift-matched 必须同时报告；只报告 endpoint 容易把 underfitting 或更新量不同误称为保护作用。

### 6.3 方向机制量

对新任务路径每段 $\delta_k=W_{k+1}-W_k$，在同一旧任务诊断 batch 上计算

\[
I_{i,k}=g_{i,k}^T\delta_k,\qquad
C_{i,k}=\tfrac12\delta_k^TH_{i,k}\delta_k,
\]

\[
\kappa_{i,k}=\frac{\delta_k^TG_{i,k}\delta_k}{\|\delta_k\|_2^2},\qquad
R_{i,k}=\Delta L_{i,k}-I_{i,k}-C_{i,k}.
\]

`pathwise_interference_sum`、`pathwise_directional_curvature_sum` 和 `pathwise_ggn_directional_cost_sum` 分别是沿真实轨迹的 $\sum I_k$、$\sum C_k$ 与 $\sum\tfrac12\delta_k^TG_{i,k}\delta_k$。Hessian/GGN 向量积由 autograd 精确计算；这里的“精确”指给定诊断 batch 和方向，不代表完整数据 Hessian 的闭式特征分解。

$\mathcal P_r(\lambda)$、$\tau_r$ 和 GGN subspace overlap 保留为 reachable/safe-route 扩展指标，不再承担当前主故事。性能报告保留 final average accuracy、immediate forgetting；AAA、BWT 和 average forgetting 放完整表。

下列量进入附录或机制诊断：

- finite-difference absolute/relative error；
- $H\hat g$ correction norm；
- factor balance gap 与 factor norm ratio；
- Jacobian/tangent dimension；
- bilinear perturbation ratio；
- factor-rescaling sensitivity；
- random-noise control；
- GGN top-subspace overlap、gradient cosine；
- endpoint/path length 和 Taylor residual。

## 7. 路径诊断与结果字段

训练任务 (t+1) 前保存 (W_0^{\mathrm{path}})，随后按 step 或 epoch 保存路径节点。每一段都精确计算有效权重空间中的

\[
H_{i,k}\delta_k,
\quad I_{i,k},
\quad C_{i,k},
\quad R_{i,k}.
\]

全局 Hessian 最大特征值使用相同 batch 上的 symmetric Lanczos 估计，字段为 `hessian_lambda_max`；不能在论文中写成“完整精确 eigendecomposition”。方向 HVP 本身是 autograd 精确值。

主要输出：

| 文件 | 内容 |
|---|---|
| `metrics.json` | FAA/AAA/BWT/forgetting、accuracy/loss matrix |
| `immediate_transitions.json` | 主分析的相邻任务遗忘与五个机制量 |
| `transitions.json` | 全部历史任务诊断 |
| `task_geometry.json` | tangent dimension/overlap、coverage、factor scaling、merge error |
| `training_history.json` | loss、perturbed loss、curvature correction norm |
| `shared_prefix.json` / `transform_audit.json` | P1 成熟公共状态、函数/tangent/gauge 审计 |
| `step_geometry.json` | P1 每步 linear/bilinear update、累计 path 与 closure error |
| `dynamic_geometry.json` / `pathwise_replicates.json` | P2 的 epoch-wise pullback/Jacobian 与四个非重叠 batch 的 $I/C/R$ |
| `fine_pathwise.json` / `sampled_step_alignment.json` | P2b/P2c 的细粒度 Taylor 路径和真实 minibatch 一步审计 |
| `analysis/stage3_p0/*` | direction/seed 双重留出、Taylor residual、全局 loss-AUC |
| `analysis/stage3_p1_mature_gauge_formal/*` | paired effects、gauge contrasts、global-support AUC、LOSO predictor 与 first-step audit |
| `analysis/stage3_p2_dynamic_chart_pilot/*` | 多批次稳健性、动态/静态 LOSO 与停止判定 |
| `analysis/stage3_p2b_step_taylor_pilot/*` / `stage3_p2c_every_step_taylor/*` | coarse/fine residual 与 clean-pullback 一步 closure |
| `analysis/stage3_p2d_path_matched_gauge_pilot/*` | 独立 cohort 的 path/current-loss/drift matched gauge effects |
| `analysis/rq1_predictor_comparison.csv` | RQ1 模型 R²、adjusted R² 与增量解释力 |
| `analysis/rq2_safe_routes.csv` | rank/参数化对应的 $\mathcal P_r,\tau_r,\kappa$ 等 |
| `analysis/rq3_optimizer_gains.csv` | 与相同 seed SGD 配对的 optimizer gain 与几何变化 |

## 8. 实验配置与顺序

### Stage 0：实现验证

- [smoke.yaml](configs/smoke.yaml)：download-free Rotated proxy；
- [teacher_smoke.yaml](configs/teacher_smoke.yaml)：两任务生成、step path、GGN 与 safe routes；
- 单元测试验证七种参数化同函数起点、subspace dimension、merge、balancing、五种优化器、exact HVP/GGN 与 mature-gauge invariants。

### Stage 1：现象门控与时间角色（已完成）

Stage-1A/1B 先检查 LoRA-minus-Dense optimizer interaction，再做 lr/radius robustness、current-loss matching 和 drift matching；Stage-1C 固定旧任务/新任务 optimizer 角色，Stage-1D 做 merge-reset，Stage-1E 加入固定低维控制。结果见 0.1–0.10。

### Stage 2A/2B：公共 endpoint 与参数化分解（已完成）

Stage-2A 使用共同 Dense Task-A endpoint；Stage-2B 比较 Dense、Random、Fixed initial/mature、Projected rank、Balanced 与 Standard Factor，并做同 $\rho_W$ 对照。正式配置和结果见 0.11–0.12。

### Stage 2C：任务方向几何（已完成）

对 $15^\circ/30^\circ/45^\circ$ 三种间隔的正反方向运行 480 个正式 runs，测量训练前 alignment 和训练后 $I/C$/GGN path cost。配置：[focus_stage2c_geometry_formal.yaml](configs/focus_stage2c_geometry_formal.yaml)。

### Stage 2D：函数保持 gauge intervention（已完成）

极端 $c=0.1/10$ 做稳定性压力测试；正式设置固定 $c\in\{0.3,1,3\}$、Task-B lr=.1、$\rho_W=.005$、20 seeds、正反两个方向，共 240 runs。配置：[focus_stage2d_gauge_formal.yaml](configs/focus_stage2d_gauge_formal.yaml)。

### Stage 3 P0/P1：路径预测与成熟-state gauge 因果分叉（已完成）

P0 对 Stage-2C 做 direction/seed 双重留出预测和全局 current-loss AUC；P1 在非零 $A_*,B_*$ 上运行 180-run radius pilot 与 480-run formal gauge intervention，并加入 orthogonal negative control、step-level linear/bilinear closure 和首 batch exact update audit。结果见 0.15–0.16。

### Stage 3 P2：动态 chart、分辨率与 path calibration（pilot 已完成）

P2 用四个非重叠 batches 否定了“time-averaged pullback 足以预测终点”；P2b/P2c 将失败定位到长路径与单步非局部性；P2d 在独立 seeds 上强烈提示 raw gauge forgetting 受有效步幅与 plasticity speed 显著混杂，同时保留条件性的 SAM trajectory interaction。P3 已启动逐步精确归一化与共同状态方向反事实。结果与计划见 0.17–0.18。

### 下一阶段：逐步归一化，然后再决定外部有效性

先做逐步 effective-step normalization 与 dense current-loss checkpoint matching。只有该实验能在相同运动预算和相同塑性进度下给出稳定 estimand，才复现到五任务 Rotated MNIST 和更大模型。rank 以及 [hvp_check.yaml](configs/hvp_check.yaml)/[gam_exact_key.yaml](configs/gam_exact_key.yaml) 继续作为独立附录，不与当前 SAM 机制混写。

## 9. 统计分析

### RQ1 主回归

以 seed 为独立统计单位，在 held-out seed 或 leave-one-seed-out 上比较：

\[
\Delta L
\sim \|\delta\|+I
\]

与

\[
\Delta L
\sim \|\delta\|+I+C.
\]

报告增量 (R^2)、bootstrap seed CI，并把 task transition 作为 repeated measures，而不是伪装成独立样本。`analyze_results.py` 当前生成描述性 OLS 表；正式论文统计还需增加 seed-block bootstrap 或 mixed-effects model。

### RQ2 参数化与 gauge contrasts

对每个 parameterization/gauge、task order 和 seed，先计算

\[
B(c)=D_{SGD}(c)-D_{SAM}(c),\qquad
I(c)=B(c)-B(1).
\]

同一 seed 的差值进入 paired bootstrap。参数化比较必须附上 SGD/SAM 的 endpoint current loss；若不匹配，只能使用 loss/drift-matched contrast，不能把 raw forgetting 当成机制证据。Gauge 实验还要审计同 seed/order 的 $W_A$ hash、logit error 和实际 effective perturbation norm。

### RQ3 任务几何调节

Stage-2C 的六个有向任务对各自先用 20 seeds 估计 $B$，然后在 task-pair 层面关联训练前 geometry 与收益。由于只有六个 pair，Pearson/Spearman 只标为 descriptive hypothesis generation；主证据来自逐 pair 的配对 CI，以及 $I/C$ 相反变化的反例。

### 附录统计：rank/safe routes 与 GAM-FD

未来 rank 分析同时画 $\mathcal P_r(\lambda)$、$\tau_r$、$\kappa$ 和 new-gradient reachable fraction，不预设单调方向。GAM 分析则用 finite-difference error、correction norm 和 pathwise curvature exposure 共同预测 gain；三者进入同一模型后，才讨论 approximation 与 trajectory 的额外解释力。

## 10. 当前结果状态

截至 **2026-08-09**，已完成 2,797 个真实 Rotated-MNIST runs：此前 Stage-2A–P2d 共 2,474，P3 `+30→−30` profile calibration 3，P3 pilot 320；P0 为既有 480 runs 的新分析，不重复计数，8 个 smoke branches 不计入科学 run 总数。P2 formal 因预注册停止规则未启动。MNIST 已自动下载并通过 integrity check；25/25 核心单元测试通过。

当前证据边界：

1. 公共 Task-A endpoint 后，forward Factor-LoRA 的 SAM trajectory benefit 仍存在；
2. 纯低维、固定成熟 tangent 和直接 rank manifold 均不足以复现它；
3. 成熟非零状态上的非正交 gauge 会显著改变 SGD/SAM 轨迹，而 orthogonal gauge 数值等价，因果确认 factor chart dynamics 参与机制；
4. P0 的 direction/seed 双重留出显示 $I+C$ 对收益的预测优于 path length 或 path+$I$，但 Taylor residual 仍不可忽略；
5. 静态 start pullback/condition 无法预测最终 gauge damage，不能把机制缩写成单一 $JJ^T$ 指标；
6. P2 的四批次重复确认 forward gauge interaction 的符号不是单 batch 偶然，但 time-averaged pullback 在两个任务方向都不能稳定预测终点；
7. P2b 表明真实 minibatch 上的 SGD pullback+bilinear 一步公式达到约 $10^{-6}$ 相对误差，SAM 与 clean-gradient 预测的偏差仅约 `1%–2.3%`，因此局部公式错误不是主要问题；
8. P2c 表明非正交 gauge 会把真实累计 step path 放大到 Identity 的约 `1.5–2.2×`，粗 checkpoint chord 会隐藏大量往返振荡；
9. P2d 将 path ratio 校准到约 `.85–1.14` 后，所有 current-loss-matched SGD gauge-damage CI 均跨 0；P3 又把每步 path 和二次预算精确匹配，确认 raw gauge damage 的大部分来自隐式步幅与 plasticity，但也发现剩余 progress-gauge effect 并未全部消失；
10. P3 的 Identity progress-AUC SAM benefit 只在 `−30→+30` 稳定；共同状态反事实显示 SAM 降低一阶干扰但提高方向曲率，净单步 safety CI 跨 0；尚未证明 LoRA/gauge 普遍放大 SAM，也不支持“SAM 始终选择更平坦方向”；
11. “LoRA 使有限差分 HVP 更准确”仍未被当前结果支持，且不再作为主线；P3 formal 只验证条件性 trajectory mechanism 与 effect equivalence。

下一步是完成已冻结的 P3 20-seed formal；在 formal 之前不进入五任务、rank、GAM 或 ViT。只有 progress-AUC benefit 在多个方向稳定为正，才扩展到五任务；若仍只有 forward 为正，则论文结论收缩为任务几何条件性的轨迹重塑。更大模型外部有效性、rank/safe routes 与 GAM-FD/exact-HVP 比较保持为后续独立模块。

## 11. 运行方式

```bash
cd Project1/MLP_LoRA_CL

PY=/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python

CUDA_VISIBLE_DEVICES=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
$PY run_grid.py --config configs/focus_stage2d_gauge_formal.yaml --offset 0 --limit 120 --skip-completed

CUDA_VISIBLE_DEVICES=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
$PY run_grid.py --config configs/focus_stage2d_gauge_formal.yaml --offset 120 --limit 120 --skip-completed

$PY audit_stage2a_common_endpoint.py \
  --input outputs/focus_stage2d_gauge_formal \
  --output analysis/focus_stage2d_gauge_endpoint_audit.csv
$PY analyze_stage2d_gauge.py \
  --input outputs/focus_stage2d_gauge_formal \
  --output analysis/focus_stage2d_gauge_effects.csv \
  --cells-output analysis/focus_stage2d_gauge_cells.csv
$PY plot_stage2d_gauge.py

$PY analyze_stage3_p0_path_safety.py \
  --input outputs/focus_stage2c_geometry_formal \
  --output analysis/stage3_p0

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
$PY run_stage3_p1_mature_gauge.py \
  --config configs/stage3_p1_mature_gauge_formal.yaml --skip-completed

$PY analyze_stage3_p1_mature_gauge.py \
  --input outputs/stage3_p1_mature_gauge_formal \
  --output analysis/stage3_p1_mature_gauge_formal
$PY plot_stage3_p0_p1.py \
  --p0 analysis/stage3_p0 \
  --p1 analysis/stage3_p1_mature_gauge_formal \
  --output figures

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
$PY run_stage3_p1_mature_gauge.py \
  --config configs/stage3_p2_dynamic_chart_pilot.yaml --skip-completed
$PY analyze_stage3_p2_dynamic_chart.py \
  --input outputs/stage3_p2_dynamic_chart_pilot \
  --output analysis/stage3_p2_dynamic_chart_pilot
$PY plot_stage3_p2_dynamic_chart.py \
  --analysis analysis/stage3_p2_dynamic_chart_pilot --output figures

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
$PY run_stage3_p1_mature_gauge.py \
  --config configs/stage3_p2b_step_taylor_pilot.yaml --skip-completed
$PY analyze_stage3_p2b_step_taylor.py \
  --input outputs/stage3_p2b_step_taylor_pilot \
  --output analysis/stage3_p2b_step_taylor_pilot

$PY run_stage3_p1_mature_gauge.py \
  --config configs/stage3_p2c_every_step_taylor.yaml --skip-completed
$PY analyze_stage3_p2b_step_taylor.py \
  --input outputs/stage3_p2c_every_step_taylor \
  --output analysis/stage3_p2c_every_step_taylor

$PY run_stage3_p1_mature_gauge.py \
  --config configs/stage3_p2d_path_matched_gauge_pilot.yaml --skip-completed
$PY analyze_stage3_p2d_path_matched_gauge.py \
  --input outputs/stage3_p2d_path_matched_gauge_pilot \
  --output analysis/stage3_p2d_path_matched_gauge_pilot
$PY plot_stage3_p2d_path_matched_gauge.py \
  --analysis analysis/stage3_p2d_path_matched_gauge_pilot --output figures
```

`gam_fd` 是清晰的单方向 finite-difference (H\hat g) 实现，`gam_exact` 使用 autograd HVP；二者用于机制判别，不宣称逐行复现主仓库的四次 closure GAM。与大模型结果对接时，必须额外加入主仓库 GAM 作为外部复现条件。
