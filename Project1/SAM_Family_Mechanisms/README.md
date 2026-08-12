# SAM 家族机制实验

这个目录承载一组按“算子层面 → 轨迹层面 → 端点层面”逐级推进的受控实验，用来回答 SAM 家族方法究竟从 Hessian 中提取了什么信息，以及这些信息何时能够解释优化轨迹与最终泛化。

## 当前状态

| 编号 | 层级与数据 | 状态 | 本阶段允许回答的问题 |
| --- | --- | --- | --- |
| E001 | 20 维正定二次函数，算子层面 | **已实现，工程验收通过** | 谱滤波、Krylov 可压缩性、内层问题质量、路径半径与聚合机制 |
| E001-S | 半径/条件数/维度/$k$/初始化敏感性 | **已实现，17 项 unittest 通过** | 初始假设稳健性、严格修正强度匹配、$Q$ 半径分解 |
| E002-P | Two Moons 小型非凸 MLP，轨迹层 pilot | **GPU 5 已运行；校准门槛部分通过** | 随机更新协方差、一步 Taylor 分解、LookSAM 时间复用、非凸训练轨迹 |
| E003 | 小型 FashionMNIST 网络，端点层面 | **planned，尚未实现** | 平坦性、泛化和机制结论在较大模型上的外推 |

E002-P 已完成 2-seed 工程/统计校准，但不是正式 E002 结论：方法、Hessian、协方差 PSD 和 Taylor 门槛通过；预注册 SGD 相对降幅门槛有一个种子未通过，4/140 条协方差记录的相对 CI 半宽超过 .25。因此仍不能声称优化器泛化排名、最终性能的因果来源或 E003 外推成立。

## E001 约定入口

依赖只有 NumPy 与 Matplotlib，可先安装：

```bash
cd Project1/SAM_Family_Mechanisms
python -m pip install -r requirements.txt
```

标准配置文件与入口为：

```bash
cd Project1/SAM_Family_Mechanisms
python run_quadratic.py \
  --config configs/quadratic.yaml \
  --output-dir outputs/e001_default
```

所有实验自由参数都允许通过 CLI 覆盖；固定的 float64/epsilon 不变量由配置校验保护。完整示例和产物契约见 [实验协议](wiki/04_experiments.md)。标准 E001 与 E001-S 敏感性运行均已完成；数值解读和假设审计见 [E001 结果与敏感性](wiki/06_e001_results_and_sensitivity.md)。这些结果仍只属于 PSD 二次族。

## E002-P GPU 5 入口

E002 使用单独的 PyTorch 依赖文件。绑定物理 GPU 5 后，进程内部设备必须写作 `cuda:0`：

```bash
cd Project1/SAM_Family_Mechanisms
python -m pip install -r requirements-e002.txt

CUDA_VISIBLE_DEVICES=5 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python run_e002_pilot.py \
  --config configs/e002_pilot.yaml \
  --output-dir outputs/e002_gpu5_pilot
```

本次正式运行已在 RTX 6000 Ada 的物理 GPU 5 上完成，设备 UUID、启动/结束占用、峰值显存、配置和代码指纹均写入 `manifest.json`。入口也支持 `--quick --device cpu` 做 smoke；CPU quick 不能冒充 GPU 正式运行。协议、产物和实际结果见 [E002 GPU 5 pilot](wiki/07_e002_gpu5_pilot.md)；便于版本管理的小型产物见 [E002-P 精简结果摘要](summaries/e002_gpu5_pilot.md)。

## Wiki 导航

- [总览与导航](WIKI.md)
- [研究范围与研究问题](wiki/01_scope_and_rqs.md)
- [方法与统一比较对象](wiki/02_methods_and_objects.md)
- [指标、精确 oracle 与判定规则](wiki/03_metrics.md)
- [E001/E002 实验协议及 E003 路线图](wiki/04_experiments.md)
- [公平性、复现与结论边界](wiki/05_fairness_and_reproduction.md)
- [E001 结果、敏感性与假设审计](wiki/06_e001_results_and_sensitivity.md)
- [E002 GPU 5 pilot 结果与门槛](wiki/07_e002_gpu5_pilot.md)
- [实验日志](wiki/log.md)

## 一条必须遵守的解释规则

GAM 的三个对象不能混写：

1. probe direction：$u_{\mathrm{GAM}}=H\hat g/(\lVert H\hat g\rVert+\epsilon)$；
2. probe increment：$\nabla L(w+\rho u_{\mathrm{GAM}})-g=\rho H u_{\mathrm{GAM}}$，在二次模型中是 $H^2$ 型；
3. final regularizer：$h_{\mathrm{GAM}}=\rho H(w^{\mathrm{adv}})\hat g(w^{\mathrm{adv}})$，最低阶仍是 $H$ 型。

因此，即使 probe increment 呈现 $H^2$ 谱响应，也不能把 GAM 的最终总更新称为 $H^2g$。
