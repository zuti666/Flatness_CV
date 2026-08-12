# 03｜指标、精确 Oracle 与判定规则

## 1. 记号与记录单位

令 Hessian 特征分解为

\[
H=V\Lambda V^\top=\sum_i\lambda_i v_iv_i^\top.
\]

涉及“顶部”子空间时，特征对按 $\lambda_i$ 从大到小排序。每一行方法记录至少由以下键唯一确定：

```text
(key, method, object_kind, protocol, inner_steps)
```

单步对象的 `protocol` 与 `inner_steps` 为空；matched-SAM 使用 `method=matched_sam`，并保留所配对 Lookbehind 的 `protocol/inner_steps`。GAM 固定为三行：`gam/final_regularizer`、`gam_probe_direction/probe_direction`、`gam_probe_increment/probe_increment`。未定义指标输出 `null`，不能输出 NaN、无穷或虚构的零。

每行用 `object_kind` 解释其分析向量 $x$。`is_update` 表示该行是否对应实际外层下降方向，`is_correction` 表示 $x$ 能否解释为 $d-g$；`correction_norm` 只对 correction 定义，而 `object_norm` 对三个 GAM 对象和所有普通 correction 都定义。`associated_perturbation_radius` 只说明该对象所属 trace 的扰动尺度，不自动表示该行具有 $Q_0/Q_1$。

## 2. SAM 的 HVP 近似检查

解析 HVP 为

\[
h_{\mathrm{true}}=H\hat g.
\]

SAM 的有限差分估计为

\[
h_{\mathrm{SAM}}
=\frac{\nabla L(w+\rho\hat g)-\nabla L(w)}{\rho}.
\]

报告

\[
\mathrm{CosHVP}
=\frac{h_{\mathrm{SAM}}^\top h_{\mathrm{true}}}
{\lVert h_{\mathrm{SAM}}\rVert\lVert h_{\mathrm{true}}\rVert+\epsilon},
\]

\[
\mathrm{RelErrHVP}
=\frac{\lVert h_{\mathrm{SAM}}-h_{\mathrm{true}}\rVert}
{\lVert h_{\mathrm{true}}\rVert+\epsilon}.
\]

默认扫描

\[
\rho=s\lVert w\rVert,
\qquad
s\in\{10^{-4},10^{-3},10^{-2},10^{-1}\}.
\]

二次模型中有限差分恒等于解析 HVP；扫描的作用是捕获符号、半径缩放、归一化或数组实现错误。到 E002 后，半径扫描才用于判断局部二阶近似在哪些半径失效。

## 3. Hessian 谱增益

对任意分析对象 $x$，定义

\[
G_i(x)
=\frac{|v_i^\top x|}
{|v_i^\top\hat g|+\epsilon}.
\]

绘制 $\log G_i$ 对 $\log\lambda_i$，同时在 CSV 中保留未取对数的投影和增益。理论参照是：

- SAM final correction：$G_i\propto\lambda_i$；
- GAM probe direction：归一化只改变整体尺度，因此 $G_i\propto\lambda_i$；
- GAM probe increment：$G_i\propto\lambda_i^2$；
- GAM final regularizer：最低阶应回到 $G_i\propto\lambda_i$，不能沿用 probe 标签；
- MS-SAM 和 Lookbehind：不预设固定幂次，由曲线与嵌套拟合判断。

比例关系是二次模型下的理论预期，不是尚未运行便成立的经验结果。

`spectral_gain.csv` 用 `object_projection` 保存所有对象的 $v_i^\top x$。只有 `is_correction=true` 的行才同时填写 `correction_projection`；两个 GAM probe 对象的该列为 `null`。因此 probe direction/increment 不会再被字段名称误称为 update correction。

E001 的特征值和各坐标投影均为正对照，绝对值不会暴露符号问题。E002 面对不定 Hessian 时必须补 signed transfer；对 $|v_i^\top\hat g|$ 接近零的模态使用预注册 mask 或谱带聚合，不能把分母爆炸解释为巨大滤波增益。

## 4. 有序 Krylov 子空间的嵌套拟合

依次构造

\[
b_1=H\hat g,\qquad b_2=H^2\hat g,\qquad b_3=H^3\hat g.
\]

这些基向量可能高度共线，必须先对设计矩阵 $[b_1,b_2,b_3]$ 做 QR 正交化，再做无截距的嵌套投影。对目标对象 $x$，定义

\[
R_p^2
=1-
\frac{\lVert x-\Pi_{\operatorname{span}(b_1,\ldots,b_p)}x\rVert^2}
{\lVert x\rVert^2+\epsilon},
\qquad p=1,2,3.
\]

输出：

\[
R_1^2,\qquad
\Delta R_2^2=R_2^2-R_1^2,\qquad
\Delta R_3^2=R_3^2-R_2^2.
\]

嵌套投影的 $R^2$ 应单调不减；若数值上出现明显下降，优先判定为实现或病态数值问题。对于 $x=c=0$ 的 SGD，拟合没有解释意义，应输出 `null`。

该指标回答“对象在按 $H$ 生成的 Krylov 子空间中是否可压缩”，不唯一识别物理 Hessian 阶数。QR 只改善设计矩阵数值条件；$\Delta R_2^2$ 依赖基向量加入顺序，也不等于 $H^2\hat g$ 的多项式系数。E002 应同时报告设计矩阵条件数，并用解析式、held-out eigenmode 或 signed transfer 辅助解释。

## 5. 顶部子空间能量与曲率暴露

令 $V_q=[v_1,\ldots,v_q]$ 包含最大的 $q$ 个特征值，对 $q\in\{1,5,10\}$ 和分析对象 $x$ 定义

\[
E_q(x)=\frac{\lVert V_q^\top x\rVert^2}
{\lVert x\rVert^2+\epsilon}.
\]

并报告正、负曲率分量和正曲率 Rayleigh quotient：

\[
H_+=\sum_{\lambda_i>0}\lambda_i v_iv_i^\top,
\qquad
H_-=\sum_{\lambda_i<0}\lambda_i v_iv_i^\top,
\]

\[
\kappa_+(x)=\frac{x^\top H_+x}{\lVert x\rVert^2+\epsilon},
\qquad
\kappa_-(x)=\frac{x^\top(-H_-)x}{\lVert x\rVert^2+\epsilon}.
\]

E001 的 $H$ 正定，所以 $\kappa_-=0$；保留字段是为避免 E002 中用完整不定 Hessian 互相抵消正负曲率。

## 6. 零阶与一阶 sharpness

在半径 $r$ 的欧氏球内定义

\[
R_r^{(0)}(w)
=\max_{\lVert\delta\rVert\le r}
\bigl[L(w+\delta)-L(w)\bigr],
\]

\[
R_r^{(1)}(w)
=r\max_{\lVert\delta\rVert\le r}
\lVert\nabla L(w+\delta)\rVert.
\]

SAM、MS-SAM 与 Lookbehind 的路径最初针对零阶损失上升；GAM 的 probe 针对邻域梯度范数。不能只用同一个最终 $\lambda_{\max}$ 替代两类目标。

## 7. E001 的精确 $Q_0/Q_1$ Oracle

只对具有关联内层候选的记录计算质量，并使用该记录的 `oracle_radius` 作为共同球半径 $r$；对多步记录，它等于配对的 `path_budget`。候选扰动约定为：

- SGD update：零扰动，只作为 $Q_0=Q_1=0$ 的参考行；
- SAM：$\delta_{\mathrm{SAM}}$；
- GAM probe direction：使用 $\delta_{\mathrm{probe}}=\rho u_{\mathrm{GAM}}$，这是唯一承载 GAM $Q_0/Q_1$ 的行；
- GAM final regularizer：不把最终修正冒充内层扰动，$Q_0,Q_1$ 为 `null`；
- GAM probe increment：该行分析梯度增量，$Q_0,Q_1$ 为 `null`；
- MS-SAM 与 Lookbehind：共享路径端点 $z_k-w$；
- matched-SAM：使用其 $\rho_{\mathrm{eff}}\hat g$ 扰动，但用配对 Lookbehind 的路径预算调用 oracle；

路径端点必须满足 $\lVert z_k-w\rVert\le r$。fixed-step 的 $r=k\rho$，fixed-budget 的 $r=\rho$。

### 7.1 零阶 oracle

二次模型中

\[
L(w+\delta)-L(w)=g^\top\delta+\frac12\delta^\top H\delta.
\]

全局最大解位于球面。对角坐标中可由 secular equation 求得：

\[
\delta_{0,i}^\star=\frac{g_i}{\nu-\lambda_i},
\qquad
\sum_i\frac{g_i^2}{(\nu-\lambda_i)^2}=r^2,
\qquad
\nu>\lambda_{\max}.
\]

定义

\[
Q_0^{(m)}
=\frac{L(w+\delta_m)-L(w)}
{L(w+\delta_0^\star)-L(w)+\epsilon}.
\]

### 7.2 一阶 oracle 与增量口径

最大化 $\lVert g+H\delta\rVert$ 等价于最大化其平方。全局最大解满足

\[
\delta_{1,i}^\star
=\frac{\lambda_i g_i}{\nu-\lambda_i^2},
\qquad
\sum_i\frac{\lambda_i^2g_i^2}{(\nu-\lambda_i^2)^2}=r^2,
\qquad
\nu>\lambda_{\max}^2.
\]

当前 schema 为兼容原实验蓝图仍使用字段名 `q1`，其实际定义是归一化的梯度范数增量：

\[
Q_1^{(m)}
=\frac{\lVert\nabla L(w+\delta_m)\rVert-\lVert g\rVert}
{\lVert\nabla L(w+\delta_1^\star)\rVert-\lVert g\rVert+\epsilon}.
\]

原始一阶 sharpness 定义中的共同半径因子 $r$ 在同半径比值中抵消，因此不重复写入 $Q_1$ 的分子和分母。

更准确的语义标签是 $\Delta Q_1$，而不是原始目标值比。接近驻点或极小半径时，其增量分母可能病态；E002 必须同时报告 raw objective ratio、增量比和 absolute regret。

### 7.3 候选方向与半径利用率

若候选没有用满注册 oracle 球，raw $Q$ 会同时惩罚方向和半径。E001-S 因此另外报告：

1. `raw_q*`：使用注册的 path/oracle budget；
2. `boundary_q*`：保持候选方向，将其径向投影到同一 oracle 球面；
3. `own_radius_q*`：把 oracle 半径改为候选自身范数。

这三者只用于分解混杂，不能选择其中最有利的一项替代预注册主指标。MS 与 Lookbehind 共享端点，其 $Q$ 应视为一条 inner-path 证据；外层聚合另用方向与一步损失评价。

在精确 oracle 与可行候选下，$Q_0,Q_1\le1+\text{tol}$。若明显超过上界，应检查候选半径、oracle 求根和归一化是否使用了同一个预算，不能把超界值解释成方法“优于 oracle”。

## 8. 路径与 matched-SAM 指标

每条多步路径报告

\[
R_{\mathrm{end}}=\lVert z_k-w\rVert,
\qquad
R_{\mathrm{path}}=\sum_{i=1}^k\lVert z_i-z_{i-1}\rVert,
\]

以及协议定义的 `path_budget` 与实际用于 oracle 的 `oracle_radius`。Lookbehind 与半径

\[
\rho_{\mathrm{eff}}=\frac{k+1}{2}\rho_{\mathrm{step}}
\]

的 matched-SAM 比较时，报告修正 cosine：

\[
\operatorname{cos}
\bigl(c_{\mathrm{LB}},c_{\mathrm{matched\text{-}SAM}}\bigr),
\]

对任意非零对象，通用的 H1 residual 定义为

\[
\mathrm{H1Residual}
=\frac{
\min_a\lVert c_{\mathrm{LB}}-aH\hat g\rVert
}{\lVert c_{\mathrm{LB}}\rVert+\epsilon}.
\]

只有 MS-SAM/Lookbehind 等真实路径对象才把同一数值另存为 `path_novelty`；GAM 等非路径对象的该字段为 `null`，只保留 `h1_residual`。对正 cosine 的 LB，H1 residual 与 correction cosine 存在解析冗余，因此不能把两者算作独立证据。

若 cosine 很高且 H1 residual 很小，只能支持“有效半径放大加路径聚合”的解释，不能声称获得了本质不同的高阶曲率信息。还必须报告

\[
\rho_{\mathrm{fit}}
=\frac{c_{\mathrm{LB}}^\top H\hat g}{\lVert H\hat g\rVert^2},
\quad
\frac{\rho_{\mathrm{fit}}}{\rho_{\mathrm{eff}}},
\quad
\frac{\lVert c_{\mathrm{LB}}\rVert}{\lVert c_{\mathrm{matched}}\rVert},
\quad
\frac{\lVert c_{\mathrm{LB}}-c_{\mathrm{matched}}\rVert}{\lVert c_{\mathrm{LB}}\rVert}.
\]

E001 还从保存的路径梯度导出两个确定性诊断：

\[
\mathrm{PathMisalign}
=1-\frac{2}{k(k-1)}\sum_{i<j}\cos(g_i,g_j),
\]

\[
D_{\mathrm{last\text{-}avg}}
=1-\cos\left(g_k,\frac1k\sum_i g_i\right).
\]

它们进入 `method_summary.csv` 的 `path_misalign` 与 `last_average_difference` 字段。跨 mini-batch 的最后梯度/平均梯度方差需要 E002，E001 不能报告该随机方差。

## 9. E001 实现验收不变量

以下是测试目标，不是实验结果：

- SAM 的 $c_{\mathrm{SAM}}=\rho H\hat g$ 相对数值误差小于 $10^{-12}$；
- GAM 的 $\Delta g_{\mathrm{probe}}=\rho H u_{\mathrm{GAM}}$ 相对数值误差小于 $10^{-12}$；
- GAM 三行语义及其 $Q_0/Q_1$ 空值/非空值位置符合契约；
- fixed-step 的路径预算为 $k\rho$，fixed-budget 为 $\rho$，且实测路径总长与协议一致；
- Lookbehind 在 $k=1$ 时与 SAM 方向一致；
- 所有可定义的 $Q_0,Q_1\le1+\text{tol}$；
- 嵌套 Krylov $R^2$ 在数值容差内单调不减；
- JSON 不含 NaN/Infinity，重复运行的核心 CSV 一致。
- 配置拒绝 NaN/Infinity、非整数 `inner_steps`、字符串伪布尔值和重复的扫描/协议条目。

只有通过这些不变量后，图表才可用于机制解释。
