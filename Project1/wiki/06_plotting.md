# 绘图代码

[返回首页](README.md)

## 1. 统一入口

```bash
MPLCONFIGDIR=/tmp/project1-mpl \
python Project1/wiki/scripts/plot_project1_results.py --figure all
```

默认输出到 `Project1/wiki/figures/`，每张图同时生成 PNG 与 PDF。

可单独选择：

```bash
python Project1/wiki/scripts/plot_project1_results.py --figure support_direction
python Project1/wiki/scripts/plot_project1_results.py --figure trajectory
python Project1/wiki/scripts/plot_project1_results.py --figure cross_domain
python Project1/wiki/scripts/plot_project1_results.py --figure scope
python Project1/wiki/scripts/plot_project1_results.py --figure nlp
```

指定输出格式/目录：

```bash
python Project1/wiki/scripts/plot_project1_results.py \
  --figure all \
  --format png \
  --output-dir /tmp/project1-figures
```

## 2. 输入数据

脚本只读取 `Project1/wiki/data/*.csv` 的审计快照，不依赖 pandas：

| 文件 | 图 |
|---|---|
| `support_direction.csv` | Exp E，SAM/Random 在相同 support 上的 FAA/BWT |
| `trajectory.csv` | Exp F，forked trajectory 的 FAA/prefix forgetting |
| `vision_cross_domain_aaa.csv` | 五个细粒度数据集 + Het5 的 AAA 小 multiples |
| `scope_effect.csv` | Full/LoRA scope strength sweep |
| `nlp_t5_faa.csv`、`nlp_llama_aaa.csv` | SAM/GAM 相对 baseline 的语言侧增益 |

数据来源和舍入方式见 [`data/README.md`](data/README.md)。若论文数字更新，应先更新 CSV 与来源说明，再重新出图。

## 3. 生成图

| 文件 | 内容 |
|---|---|
| `support_direction.{png,pdf}` | 五种 support 的 SAM vs Random；FAA 与 BWT 双面板 |
| `forked_trajectory.{png,pdf}` | Exp F 各 schedule 的 FAA 与 prefix forgetting |
| `cross_domain_aaa.{png,pdf}` | 6 datasets × 5 methods × 4 optimizers |
| `scope_effect.{png,pdf}` | Full/LoRA scope 随 strength 的相对 accuracy |
| `nlp_gain.{png,pdf}` | T5 与 Llama 的 SAM/GAM gain |

## 4. 结果图预览

![Exp E support-direction](figures/support_direction.png)

![Exp F forked trajectory](figures/forked_trajectory.png)

![Cross-domain AAA](figures/cross_domain_aaa.png)

![Scope effect](figures/scope_effect.png)

![NLP gain](figures/nlp_gain.png)

## 5. 图表约定

- performance/transfer 越高越好；forgetting 越低越好。
- BWT 保留负号，不转换成绝对值。
- baseline 用灰色或零线表示；SAM/AS(0) 与 GAM/AS(1) 使用固定颜色。
- 不在图中混用 FAA 和 AAA；标题和轴标签明确写指标。
- 单 seed 图不绘制伪造的 error bar。
- 缺失 GAM 用空值跳过，不按 0 绘制。

## 6. 原实验自带绘图代码

统一脚本用于 Wiki 快照；需要从原始 matrix 重画 trajectory 时，应优先使用：

```text
config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction/plot_exp_E_support_direction.py
config_exps_paper1_PAC/exp_F_imagenetr_r16_t20_forked_taskwise/plot_forked_taskwise_curves.py
config_exps_paper1_PAC/exp1_showPertuabtion/draw_rho.py
config_exps_paper1_PAC/exp1_showPertuabtion/draw_rho2.py
evaluation_CL_mechanism/visualize.py
```

原脚本负责从 raw logs/metrics 生成实验级曲线；Wiki 脚本负责跨实验统一风格和汇总图，两者职责不同。
