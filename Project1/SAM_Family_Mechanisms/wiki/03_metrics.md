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

其中单步对象在语义上没有多步路径；实现可用 `single_step` 与 `inner_steps=1` 显式表示。GAM 的 `probe_increment` 必须与表示 `final_regularizer` 的最终 `update_correction` 分行。未定义指标输出 `null`，不能输出 NaN、无穷或虚构的零。

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

对任意修正对象 $c$，定义

\[
G_i(c)
=\frac{|v_i^\top c|}
{|v_i^\top\hat g|+\epsilon}.
\]

绘制 $\log G_i$ 对 $\log\lambda_i$，同时在 CSV 中保留未取对数的投影和增益。理论参照是：

- SAM final correction：$G_i\propto\lambda_i$；
- GAM probe increment：$G_i\propto\lambda_i^2$；
- GAM final regularizer：最低阶应回到 $G_i\propto\lambda_i$，不能沿用 probe 标签；
- MS-SAM 和 Lookbehind：不预设固定幂次，由曲线与嵌套拟合判断。

比例关系是二次模型下的理论预期，不是尚未运行便成立的经验结果。

## 4. $H^p\hat g$ 的嵌套拟合

依次构造

\[
x_1=H\hat g,\qquad x_2=H^2\hat g,\qquad x_3=H^3\hat g.
\]

这些向量可能高度共线，必须先对设计矩阵 $[x_1,x_2,x_3]$ 做 QR 正交化，再做无截距的嵌套投影。对目标修正 $c$，定义

\[
R_p^2
=1-
\frac{\lVert c-\Pi_{\operatorname{span}(x_1,\ldots,x_p)}c\rVert^2}
{\lVert c\rVert^2+\epsilon},
\qquad p=1,2,3.
\]

输出：

\[
R_1^2,\qquad
\Delta R_2^2=R_2^2-R_1^2,\qquad
\Delta R_3^2=R_3^2-R_2^2.
\]

嵌套投影的 $R^2$ 应单调不减；若数值上出现明显下降，优先判定为实现或病态数值问题。对于 $c=0$ 的 SGD，拟合没有解释意义，应输出 `null`。

## 5. 顶部子空间能量与曲率暴露

令 $V_q=[v_1,\ldots,v_q]$ 包含最大的 $q$ 个特征值，对 $q\in\{1,5,10\}$ 定义

\[
E_q(c)=\frac{\lVert V_q^\top c\rVert^2}
{\lVert c\rVert^2+\epsilon}.
\]

并报告正、负曲率分量和正曲率 Rayleigh quotient：

\[
H_+=\sum_{\lambda_i>0}\lambda_i v_iv_i^\top,
\qquad
H_-=\sum_{\lambda_i<0}\lambda_i v_iv_i^\top,
\]

\[
\kappa_+(c)=\frac{c^\top H_+c}{\lVert c\rVert^2+\epsilon},
\qquad
\kappa_-(c)=\frac{c^\top(-H_-)c}{\lVert c\rVert^2+\epsilon}.
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

对每条记录，使用该记录的 `oracle_radius` 作为共同球半径 $r$；对多步记录，它等于配对的 `path_budget`。候选扰动约定为：

- SAM：$\delta_{\mathrm{SAM}}$；
- GAM 的最终 update summary：使用该方法产生 final regularizer 前的 $\delta_{\mathrm{probe}}$；
- MS-SAM 与 Lookbehind：共享路径端点 $z_k-w$；
- matched-SAM：使用其 $\rho_{\mathrm{eff}}\hat g$ 扰动，但用配对 Lookbehind 的路径预算调用 oracle；
- 单独的 `probe_increment` 诊断行可将 $Q_0,Q_1$ 置为 `null`，避免与 GAM update summary 重复。

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

### 7.2 一阶 oracle

最大化 $\lVert g+H\delta\rVert$ 等价于最大化其平方。全局最大解满足

\[
\delta_{1,i}^\star
=\frac{\lambda_i g_i}{\nu-\lambda_i^2},
\qquad
\sum_i\frac{\lambda_i^2g_i^2}{(\nu-\lambda_i^2)^2}=r^2,
\qquad
\nu>\lambda_{\max}^2.
\]

定义

\[
Q_1^{(m)}
=\frac{\lVert\nabla L(w+\delta_m)\rVert-\lVert g\rVert}
{\lVert\nabla L(w+\delta_1^\star)\rVert-\lVert g\rVert+\epsilon}.
\]

原始一阶 sharpness 定义中的共同半径因子 $r$ 在同半径比值中抵消，因此不重复写入 $Q_1$ 的分子和分母。

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

并定义路径新信息残差

\[
\mathrm{PathNovelty}
=\frac{
\min_a\lVert c_{\mathrm{LB}}-aH\hat g\rVert
}{\lVert c_{\mathrm{LB}}\rVert+\epsilon}.
\]

若 cosine 很高且 PathNovelty 很小，只能支持“有效半径放大加路径聚合”的解释，不能声称获得了本质不同的高阶曲率信息。

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
- fixed-step 的路径预算为 $k\rho$，fixed-budget 为 $\rho$，且实测路径总长与协议一致；
- Lookbehind 在 $k=1$ 时与 SAM 方向一致；
- 所有可定义的 $Q_0,Q_1\le1+\text{tol}$；
- 嵌套 $R^2$ 在数值容差内单调不减；
- JSON 不含 NaN/Infinity，重复运行的核心 CSV 一致。

只有通过这些不变量后，图表才可用于机制解释。
