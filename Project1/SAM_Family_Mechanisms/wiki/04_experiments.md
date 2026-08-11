# 04｜实验协议、CLI 与输出契约

## 1. 状态总表

| 编号 | 名称 | 状态 | 本次是否实现 |
| --- | --- | --- | --- |
| E001 | 20 维精确 Hessian 二次算子实验 | **current scope** | 是，本次唯一实现范围 |
| E002 | Two Moons 一层隐藏层非凸轨迹实验 | **planned** | 否 |
| E003 | 小型 FashionMNIST 端点验证 | **planned** | 否 |

`planned` 只表示设计已记录，不表示入口、配置、产物或结果已经存在。

## 2. E001：20 维二次算子实验

### 2.1 目的

E001 用一个 Hessian 已知且在全空间恒定的模型，隔离四个问题：

1. SAM 有限差分是否正确提取 $H\hat g$；
2. GAM probe increment 的 $H^2$ 型响应与 final regularizer 的最低阶 $H$ 型响应是否被正确区分；
3. MS-SAM 与 Lookbehind 在不同路径预算下如何放大 Hessian 谱；
4. Lookbehind 在 matched-SAM 后还剩多少路径新信息。

E001 不训练到端点，也没有 basin 选择或泛化问题。

### 2.2 默认构造

```yaml
dimension: 20
lambda_min: 0.1
lambda_max: 10.0
rho_scales: [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
primary_rho_scale: 1.0e-2
inner_steps: [2, 5]
path_protocols: [fixed_step, fixed_budget]
epsilon: 1.0e-15
```

实现固定使用 float64，并在 manifest 中记录 dtype；dtype 不是当前需要由用户调节的实验自由度。实际半径为 `rho_scale * ||w||`。Hessian 使用 `diag(logspace(-1, 1, dimension))`；初始化 $w_i\propto1/\lambda_i$ 后整体归一，使 $g_i=\lambda_iw_i$ 等幅。`primary_rho_scale` 用于四张主图，全部 `rho_scales` 用于 HVP 半径扫描及可配置的稳健性输出。

上述是默认协议；实际运行必须以 `manifest.json` 中的解析后参数为准。

### 2.3 方法矩阵

| 方法/对象 | $k$ | 路径协议 | 角色 |
| --- | --- | --- | --- |
| SGD | — | — | $c=0$ 参考 |
| SAM final correction | — | — | 单次 $H\hat g$ 通道 |
| GAM probe direction | — | — | Hessian 定向探测 |
| GAM probe increment | — | — | $H^2$ 型探测点梯度变化 |
| GAM final regularizer | — | — | 最低阶 $H$ 型正则项 |
| MS-SAM | 2, 5 | fixed-step, fixed-budget | 使用最远端梯度 |
| Lookbehind | 2, 5 | fixed-step, fixed-budget | 聚合路径梯度 |
| matched-SAM | 对每条 Lookbehind 记录生成 | 继承比较记录 | 有效半径对照 |

LookSAM、SAM-$k$ 与 Noise-only 不在 E001 中。

### 2.4 标准入口

从仓库根目录运行：

```bash
python Project1/SAM_Family_Mechanisms/run_quadratic.py \
  --config Project1/SAM_Family_Mechanisms/configs/quadratic.yaml \
  --output-dir Project1/SAM_Family_Mechanisms/outputs/e001_default
```

也应支持显式覆盖，便于审计不依赖配置文件的默认值：

```bash
python Project1/SAM_Family_Mechanisms/run_quadratic.py \
  --output-dir /tmp/quadratic_operator \
  --dimension 20 \
  --rho-scales 1e-4 1e-3 1e-2 1e-1 \
  --primary-rho-scale 1e-2 \
  --inner-steps 2 5 \
  --path-protocols fixed_step fixed_budget
```

CLI 覆盖值优先于 YAML；最终解析值必须进入 manifest。若保留 `--seed 0`，它只用于确定性 manifest 和未来随机扩展，不能暗示 E001 已包含随机实验。

### 2.5 输出目录契约

每次运行的 `--output-dir` 必须是自包含、可审计的产物目录：

| 文件 | 必需内容 |
| --- | --- |
| `manifest.json` | `schema_version`、解析后参数、固定 dtype、命令与运行环境、方法与对象、每种方法的实测 gradient/backward-equivalent 预算 |
| `arrays.npz` | $H$、特征值/特征向量、$w$、$g$，以及各对象的 $d/c$、扰动、路径点或路径梯度 |
| `hvp_scan.csv` | `rho_scale,rho,cos_hvp,relative_error_hvp,estimate_norm,truth_norm` |
| `spectral_gain.csv` | `key,method,object_kind,is_update,protocol,inner_steps,eigen_index,eigenvalue,correction_projection,ghat_projection,gain` |
| `method_summary.csv` | 主键 `key,method,object_kind,protocol,inner_steps`，以及修正范数、E1/E5/E10、正/负 Rayleigh、嵌套 $R^2$、$Q_0/Q_1$、路径半径、预算、gradient evaluations、matched-SAM cosine、PathNovelty |
| `metrics.json` | 与 CSV 同源的层次化机器可读汇总；未定义值为 `null` |
| `spectral_gain.png` | $G_i$ 对 $\lambda_i$ 的谱增益图，GAM 对象分开标注 |
| `top_subspace_curvature.png` | $E_q$ 与曲率暴露汇总 |
| `hp_fit.png` | $H^p\hat g$ 嵌套增量 $R^2$ |
| `inner_quality.png` | 同半径 oracle 归一化的 $Q_0/Q_1$ |

`method_summary.csv` 的方法主键是 `key,method,object_kind,protocol,inner_steps`。任何为了绘图新增的列都应保持向后兼容，并在 `schema_version` 变化时记录。

生成产物不应手工修改。重新运行前应使用新的输出目录，或明确记录覆盖行为；默认不把临时输出当作源代码提交。

### 2.6 四张图各自回答什么

1. `spectral_gain.png`：修正随特征值如何放大；只能比较同一对象语义和明确的半径协议。
2. `top_subspace_curvature.png`：能量是否集中到顶部 Hessian 子空间；曲率暴露不是越小越好。
3. `hp_fit.png`：加入 $H^2\hat g,H^3\hat g$ 后是否产生额外解释度；必须使用 QR。
4. `inner_quality.png`：候选扰动对 $R^{(0)}$ 与 $R^{(1)}$ 内层目标的相对求解质量；所有点按各自 `oracle_radius` 调用精确 oracle，多步记录的该值等于配对 `path_budget`。

matched-SAM cosine、PathNovelty、`R_end/R_path` 主要在表格和 JSON 中解释，不能只凭某一条谱曲线判断路径是否提供了新信息。

### 2.7 E001 验收顺序

1. 检查 CLI 退出成功且全部必需产物完整；
2. 检查 JSON 严格可解析且无 NaN/Infinity；
3. 检查 SAM 与 GAM 的解析恒等式；
4. 检查 fixed-step/fixed-budget 的路径长度和预算；
5. 检查 $k=1$ 退化、$Q_0/Q_1$ 上界及嵌套 $R^2$ 单调性；
6. 用相同命令重复运行，比较核心 CSV；
7. 通过以上检查后，才在 [实验日志](log.md) 登记 run ID 并解读图表。

验收阈值见 [指标页](03_metrics.md)。Wiki 不预填具体观测值。

## 3. E002：Two Moons 非凸轨迹实验（planned）

E002 的建议设置是：训练集 512、测试集 4096、数据噪声 0.15，可单独设置 10% 训练标签翻转；模型为 $2\to16\to2$ 的 tanh MLP，batch size 32，plain SGD、无 momentum、无 BatchNorm、无 Dropout。

计划比较：SGD、SAM、GAM、MS-SAM（$k=2,5$）、Lookbehind（$k=2,5$）、LookSAM（刷新间隔 5）和 SAM-5。Noise-only、Random/Shuffled/EMA 正交修正是机制对照，不应提前混入 E001。

计划在训练进度 10%、30%、60%、90%、100% 的 checkpoint 固定同一个参数点：

1. 用完整训练集计算 $g,H$ 并完整特征分解；
2. 对相同的 128 个 mini-batch 只做虚拟更新，估计 $\mu_m,\Sigma_m$；
3. 计算 $\operatorname{Tr}(H_+\Sigma_m)$、归一化 Hessian 对齐、路径 cosine 与 LookSAM 时间自相关；
4. 比较一步 Taylor 的确定性与随机二阶分解和真实一步损失变化；
5. 用 PGD 估计多半径 $R^{(0)},R^{(1)}$。

这些数值是未来实验设计参数，不是当前结果。E002 尚未实现。

## 4. E003：FashionMNIST 端点验证（planned）

E003 只用于复核 E002 中最稳定的三到四个机制结论。建议受控数据为 1000 个正确标签训练样本加 200 个随机标签样本，模型采用约 $10^4$ 参数的小型 LeNet 或 $784\to16\to10$ MLP。

计划使用 Lanczos/power iteration 的前 20 个特征对、Hutchinson Hessian trace、顶部子空间中的更新协方差、20 步 PGD sharpness，以及 5 个随机种子。端点指标包括 train/test loss、test accuracy、generalization gap、$\lambda_{\max}$、$\operatorname{Tr}(H)$、$R^{(0)}(r)$、$R^{(1)}(r)$、$\operatorname{Tr}(H_+\Sigma)$ 和 backward-equivalent compute。

E003 尚未实现，也没有任何 FashionMNIST 结果可在当前 Wiki 中引用。
