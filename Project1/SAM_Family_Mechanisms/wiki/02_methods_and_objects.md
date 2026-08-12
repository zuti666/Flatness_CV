# 02｜方法与统一比较对象

## 1. E001 的精确二次问题

定义

\[
L(w)=\frac12w^\top H w,\qquad
H=\operatorname{diag}(\lambda_1,\ldots,\lambda_d),\qquad d=20,
\]

其中

\[
\lambda_i=10^{-1+2(i-1)/(d-1)},\qquad i=1,\ldots,d.
\]

默认先取 $w_i\propto1/\lambda_i$，再对整个向量统一归一化。于是

\[
g=Hw
\]

在每个 Hessian 特征方向上具有相同幅度，谱增益不会被初始梯度天然偏向顶部特征方向所遮蔽。全程使用 float64，并定义

\[
\hat g=\frac{g}{\lVert g\rVert+\epsilon}.
\]

这个等梯度起点是刻意激发全部谱模态的 `diagnostic fixture`，不是典型初始化的概率模型；E001-S 另外报告 random-$w$/random-$g$ ensemble。这个问题中 $H$ 恒定且正定，特征向量是坐标轴。正曲率与负曲率字段仍保留在输出 schema 中，以便 E002 复用；E001 的负曲率量应是零或不适用，而不是伪造非零值。

## 2. 统一对象：$g,c,d$

对每种方法 $m$，必须保存：

\[
g=\nabla L(w),\qquad
d^{(m)}=\text{实际用于下降的方向},\qquad
c^{(m)}=d^{(m)}-g.
\]

如果某个中间对象不是最终下降方向，例如 GAM probe direction 或 probe increment，必须使用单独的 `object_kind` 标签，不能塞进 final correction。为统一谱分析，记每行实际分析的向量为 $x$：普通更新行取 $x=c$，GAM 的两个 probe 行分别取 $x=u_{\mathrm{GAM}}$ 与 $x=\Delta g_{\mathrm{probe}}$。SGD 参考为

\[
d^{(\mathrm{SGD})}=g,\qquad c^{(\mathrm{SGD})}=0.
\]

SGD 的 `correction_projection` 是零参考；为避免把“无修正”混作一条可拟合曲线，`gain` 以及以 $\lVert x\rVert$ 为分母的顶部能量、Rayleigh 和拟合指标输出 `null`，而不是 NaN。非 correction 对象使用 `object_norm`，其 `correction_norm` 为 `null`。

## 3. SAM：单次方向性 Hessian 修正

令基础半径 $\rho=s\lVert w\rVert$，其中 $s$ 是 `rho_scale`。SAM 使用

\[
\delta_{\mathrm{SAM}}=\rho\hat g,
\]

并在扰动点取下降方向：

\[
d_{\mathrm{SAM}}
=\nabla L(w+\delta_{\mathrm{SAM}})
=g+\rho H\hat g.
\]

因此二次模型中

\[
c_{\mathrm{SAM}}=\rho H\hat g,
\qquad
\frac{\nabla L(w+\rho\hat g)-\nabla L(w)}{\rho}=H\hat g
\]

是解析恒等式。数值扫描用于验证实现，不应把该恒等式误称为新发现。

## 4. GAM：三个对象必须严格分开

### 4.1 Probe direction

\[
u_{\mathrm{GAM}}
=\frac{H\hat g}{\lVert H\hat g\rVert+\epsilon}.
\]

这是 GAM 用 Hessian 定向探测邻域的方向，本身不是最终正则项。

汇总表中它是 `key=gam_probe_direction, object_kind=probe_direction`。实际内层候选是 $\delta_{\mathrm{probe}}=\rho u_{\mathrm{GAM}}$，因此 GAM 的 $Q_0/Q_1$ 只放在这一行；方向向量的 `object_norm` 与候选扰动半径由不同字段记录。

### 4.2 Probe increment：$H^2$ 型

令

\[
\delta_{\mathrm{probe}}=\rho u_{\mathrm{GAM}}.
\]

在本二次模型中，探测点梯度变化为

\[
\Delta g_{\mathrm{probe}}
=\nabla L(w+\delta_{\mathrm{probe}})-g
=\rho H u_{\mathrm{GAM}}
=\rho\frac{H^2\hat g}{\lVert H\hat g\rVert+\epsilon}.
\]

只有这个 `probe_increment` 对象具有明确的 $H^2$ 型谱响应。

汇总表中它是 `key=gam_probe_increment, object_kind=probe_increment`。该行分析梯度增量本身，不把它再次当作内层候选，所以 $Q_0/Q_1$ 为 `null`。

### 4.3 Final regularizer：最低阶 $H$ 型

在探测点 $w^{\mathrm{adv}}=w+\delta_{\mathrm{probe}}$ 上定义

\[
\hat g_{\mathrm{adv}}
=\frac{\nabla L(w^{\mathrm{adv}})}
{\lVert\nabla L(w^{\mathrm{adv}})\rVert+\epsilon},
\]

最终一阶平坦性正则项为

\[
h_{\mathrm{GAM}}
=\rho H(w^{\mathrm{adv}})\hat g_{\mathrm{adv}}
=\rho H\hat g_{\mathrm{adv}}
\]

（最后一步利用了 E001 中 Hessian 恒定）。其小半径最低阶项仍是 $\rho H\hat g$。E001 是同一 loss、exact-HVP、正则权重 1 的理想化特例，最终方向约定为 $d_{\mathrm{GAM}}=g+h_{\mathrm{GAM}}$。汇总表中 `key=gam, object_kind=final_regularizer` 明确表示这条最终修正；它不承担内层候选质量，所以 $Q_0/Q_1$ 为 `null`。`arrays.npz` 也分别保存 `probe_direction`、`probe_increment` 和 `final_regularizer`。E002 必须显式记录实际正则权重、oracle loss 和 batch 语义。

**禁止的解释：** 由 probe increment 的 $H^2$ 性质推导“GAM 最终总更新等价于 $H^2g$”。

## 5. Multistep-SAM：最远端梯度

设 $z_0=w$，每一步使用当前梯度归一化上升：

\[
z_i=z_{i-1}+\rho_{\mathrm{step}}
\frac{\nabla L(z_{i-1})}
{\lVert\nabla L(z_{i-1})\rVert+\epsilon},
\qquad
g_i=\nabla L(z_i),
\quad i=1,\ldots,k.
\]

MS-SAM 使用最远端梯度：

\[
d_{\mathrm{MS}}=g_k,
\qquad c_{\mathrm{MS}}=g_k-g.
\]

E001 使用 $k\in\{2,5\}$。不能预设它严格等于某个固定 $H^p\hat g$；谱曲线和嵌套 Krylov 拟合只能描述响应与可压缩性，不能唯一识别物理阶数。

## 6. Lookbehind：路径梯度聚合

Lookbehind 与 MS-SAM 共享同一条上升路径，但使用中间梯度平均：

\[
d_{\mathrm{LB}}=\frac1k\sum_{i=1}^k g_i,
\qquad
c_{\mathrm{LB}}=d_{\mathrm{LB}}-g.
\]

E001 只研究

\[
\texttt{path\_mean\_surrogate}=\frac1k\sum_i g_i.
\]

它不是包含 inner ascent/descent、slow-weight interpolation、学习率、动量和 weight decay 状态的完整 Lookbehind optimizer，也不能简单标成 faithful algorithm 的 $\alpha=1$。E002 必须将该 surrogate 与真实 slow-weight delta 分行，并显式记录 interpolation 系数和实际 forward/backward/data-access 成本。

当 $k=1$ 时，路径定义应退化为 SAM。这一退化关系是实现验收项。

## 7. 两个路径半径协议

令配置中的基础半径为 $\rho$。每条多步记录都必须写出 `protocol`、`rho_step`、`inner_steps`、`endpoint_radius`、`path_radius`、`path_budget` 和用于 $Q_0/Q_1$ 的 `oracle_radius`。其中 `endpoint_radius/path_radius/path_budget` 分别对应下文的 $R_{\mathrm{end}}/R_{\mathrm{path}}/\text{radius\_budget}$。

### 7.1 fixed-step：固定每步半径

\[
\rho_{\mathrm{step}}=\rho,
\qquad
R_{\mathrm{path}}=\sum_{i=1}^k\lVert z_i-z_{i-1}\rVert\approx k\rho,
\qquad
\text{radius\_budget}=k\rho.
\]

它保留原始“每步同强度”的多步设置，但增加 $k$ 同时增加总路径预算。

### 7.2 fixed-budget：固定总路径预算

\[
\rho_{\mathrm{step}}=\frac{\rho}{k},
\qquad
R_{\mathrm{path}}\approx\rho,
\qquad
\text{radius\_budget}=\rho.
\]

它控制总路径长度，从而大幅减弱“更大总扰动”的混杂，但并未完全隔离路径采样点数：Lookbehind 的一阶有效半径 $(k+1)\rho/(2k)$ 仍随 $k$ 变化。端点半径

\[
R_{\mathrm{end}}=\lVert z_k-w\rVert
\]

通常不超过路径总长，必须实测而不能用预算替代。

### 7.3 fixed-effective-radius：E001-S 的补充诊断

为固定 Lookbehind 的一阶 matched-SAM 半径，E001-S 额外设置

\[
\rho_{\mathrm{step}}=\frac{2\rho_{\mathrm{eff}}}{k+1}.
\]

该协议只用于敏感性因果控制，不替代 fixed-step/fixed-budget。它允许在 matched-SAM 完全相同的情况下观察增加路径分辨率还留下多少方向残差。

## 8. Matched-SAM：匹配 Lookbehind 的一阶有效半径

局部一阶展开给出

\[
g_i\approx g+i\rho_{\mathrm{step}}H\hat g,
\]

所以

\[
d_{\mathrm{LB}}
\approx g+\frac{k+1}{2}\rho_{\mathrm{step}}H\hat g.
\]

对应的 matched-SAM 半径为

\[
\rho_{\mathrm{eff}}
=\frac{k+1}{2}\rho_{\mathrm{step}}.
\]

因此：

- fixed-step 下，$\rho_{\mathrm{eff}}=(k+1)\rho/2$；
- fixed-budget 下，$\rho_{\mathrm{eff}}=(k+1)\rho/(2k)$。

比较时不能只报告 cosine，还要同时给出修正范数比、$\rho_{\mathrm{fit}}/\rho_{\mathrm{eff}}$、向量相对误差和 H1 residual。cosine 会忽略幅度失配；默认 k=5 fixed-step 虽有 correction cosine .9949，修正范数比仍为 1.221、向量相对误差为 .203。E001-S 另用二分求解原生半径实现严格 $\lVert c\rVert/\lVert g\rVert$ 匹配。
