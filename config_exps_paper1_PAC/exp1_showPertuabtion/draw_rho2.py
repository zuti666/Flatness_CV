import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

rng = np.random.default_rng(7)

# -----------------------------
# Three-center "triple-junction" terrain (smooth min of 3 quadratic wells)
# -----------------------------
centers = [
    (-2.0,  0.0, 1.2, 0.9, 0.00),
    ( 1.2,  1.7, 1.1, 0.9, 0.35),
    ( 1.2, -1.7, 1.1, 0.9, 0.70),
]
tau = 0.38

def loss_grad_hess(x, y, centers, tau):
    Q = []
    G = []
    Hs = []
    for (xi, yi, sx, sy, h) in centers:
        dx = x - xi
        dy = y - yi
        qi = (dx*dx)/(sx*sx) + (dy*dy)/(sy*sy) + h
        gi = np.array([2.0*dx/(sx*sx), 2.0*dy/(sy*sy)])
        Hi = np.array([[2.0/(sx*sx), 0.0],
                       [0.0, 2.0/(sy*sy)]])
        Q.append(qi)
        G.append(gi)
        Hs.append(Hi)

    Q = np.array(Q)
    G = np.stack(G, axis=0)        # (m,2)
    Hs = np.stack(Hs, axis=0)      # (m,2,2)

    w = np.exp(-Q / tau)
    S = np.sum(w) + 1e-12
    w = w / S

    L = -tau * np.log(S)
    g = np.sum(w[:, None] * G, axis=0)

    # Hessian of smooth-min:
    # H = sum_i w_i H_i + (1/tau)( g g^T - sum_i w_i g_i g_i^T )
    GG = np.einsum("mi,mj->mij", G, G)  # (m,2,2)
    H = np.sum(w[:, None, None] * Hs, axis=0) + (1.0/tau) * (
        np.outer(g, g) - np.sum(w[:, None, None] * GG, axis=0)
    )
    return L, g, H

def loss_only(x, y):
    L, _, _ = loss_grad_hess(x, y, centers, tau)
    return L

# -----------------------------
# Random theta near the triple junction (not exactly on it)
# -----------------------------
# theta = np.array([rng.uniform(-0.30, 0.55), rng.uniform(-0.25, 0.45)])
theta = np.array([rng.uniform(-0.30, 0.55), rng.uniform(-0.25, 0.45)])
theta = theta + np.array([-0.01, -0.05])
rho = 0.30

_, g, H = loss_grad_hess(theta[0], theta[1], centers, tau)

# AS0: gradient direction (on circle), choose sign by higher loss (ascent side)
g_dir = g / (np.linalg.norm(g) + 1e-12)
g_plus = theta + rho * g_dir
g_minus = theta - rho * g_dir
as0 = g_minus if loss_only(g_minus[0], g_minus[1]) > loss_only(g_plus[0], g_plus[1]) else g_plus
# GD: fixed step along positive gradient (inside the circle, ascent)
gd = theta + 1 * rho * g_dir

# AS1: max curvature direction (largest eigenvector), choose sign by higher loss (ascent side)
eigvals, eigvecs = np.linalg.eigh(H)
v_max = eigvecs[:, np.argmax(eigvals)]
p_plus = theta + rho * v_max
p_minus = theta - rho * v_max
if loss_only(p_minus[0], p_minus[1]) > loss_only(p_plus[0], p_plus[1]):
    v_max = -v_max
as1 = theta + rho * v_max

# RS: random point inside the circle
r_rs = 0.65 * rho * np.sqrt(rng.uniform(0, 1))
ang_rs = rng.uniform(0, 2 * np.pi)
rs = theta + np.array([r_rs * np.cos(ang_rs), r_rs * np.sin(ang_rs)])
rs = rs + np.array([0.10, -0.10])

# Gradients at AS0/AS1/RS for dashed direction hints
_, g_as0, _ = loss_grad_hess(as0[0], as0[1], centers, tau)
_, g_as1, _ = loss_grad_hess(as1[0], as1[1], centers, tau)
_, g_rs, _ = loss_grad_hess(rs[0], rs[1], centers, tau)
g_as0_dir = g_as0 / (np.linalg.norm(g_as0) + 1e-12)
g_as1_dir = g_as1 / (np.linalg.norm(g_as1) + 1e-12)
g_rs_dir = g_rs / (np.linalg.norm(g_rs) + 1e-12)
dash_len = 0.6 * rho
as0_dash_end = as0 - dash_len * g_as0_dir
as1_dash_end = as1 - dash_len * g_as1_dir
as0_dash_end_from_theta = theta - dash_len * g_as0_dir
as1_dash_end_from_theta = theta - dash_len * g_as1_dir
rs_dash_end = rs - dash_len * g_rs_dir
rs_dash_end_from_theta = theta - dash_len * g_rs_dir
# sum_vec = (-dash_len * g_dir) + (-dash_len * g_as1_dir)
# sum_end_from_theta = theta + sum_vec

# -----------------------------
# 3D plot: surface + points + arrows
# -----------------------------
def draw_rho2_3d(
    *,
    theta_point=None,
    gd_point=None,
    as0_point=None,
    as1_point=None,
    rs_point=None,
    xlim=(-3.0, 3.0),
    ylim=(-2.4, 2.4),
    grid_size=(160, 140),
    elev=28,
    azim=-55,
    show=True,
    save=True,
    out_prefix="exp1_showPertuabtion/triple_junction_random_theta_three_points_3d",
    dpi=220,
):
    """3D surface with points and arrows from theta to GD/AS0/AS1/RS."""
    if theta_point is None:
        theta_point = theta
    if gd_point is None:
        gd_point = gd
    if as0_point is None:
        as0_point = as0
    if as1_point is None:
        as1_point = as1
    if rs_point is None:
        rs_point = rs

    nx, ny = int(grid_size[0]), int(grid_size[1])
    x = np.linspace(float(xlim[0]), float(xlim[1]), nx)
    y = np.linspace(float(ylim[0]), float(ylim[1]), ny)
    X, Y = np.meshgrid(x, y)

    Qs = []
    for (xi, yi, sx, sy, h) in centers:
        Qs.append(((X - xi) ** 2) / (sx ** 2) + ((Y - yi) ** 2) / (sy ** 2) + h)
    Qs = np.stack(Qs, axis=0)
    W = np.exp(-Qs / tau)
    S = np.sum(W, axis=0) + 1e-12
    Z = -tau * np.log(S)

    def _point_xyz(p):
        return np.array([p[0], p[1], loss_only(p[0], p[1])], dtype=float)

    def _arrow(ax, start, end, color):
        d = end - start
        ax.quiver(
            start[0], start[1], start[2],
            d[0], d[1], d[2],
            color=color, linewidth=1.2, arrow_length_ratio=0.12
        )

    theta_xyz = _point_xyz(theta_point)
    gd_xyz = _point_xyz(gd_point)
    as0_xyz = _point_xyz(as0_point)
    as1_xyz = _point_xyz(as1_point)
    rs_xyz = _point_xyz(rs_point)

    fig = plt.figure(figsize=(9.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.85)

    ax.scatter([theta_xyz[0]], [theta_xyz[1]], [theta_xyz[2]], s=40, color="black", label="theta")
    ax.scatter([gd_xyz[0]], [gd_xyz[1]], [gd_xyz[2]], s=40, color="blue", label="GD")
    ax.scatter([as0_xyz[0]], [as0_xyz[1]], [as0_xyz[2]], s=40, color="green", label="AS0")
    ax.scatter([as1_xyz[0]], [as1_xyz[1]], [as1_xyz[2]], s=40, color="red", label="AS1")
    ax.scatter([rs_xyz[0]], [rs_xyz[1]], [rs_xyz[2]], s=40, color="orange", label="RS")

    _arrow(ax, theta_xyz, gd_xyz, "blue")
    _arrow(ax, theta_xyz, as0_xyz, "green")
    _arrow(ax, theta_xyz, as1_xyz, "red")
    _arrow(ax, theta_xyz, rs_xyz, "orange")

    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.legend()

    if save and out_prefix:
        fig.savefig(out_prefix + ".png", dpi=dpi, bbox_inches="tight")
        fig.savefig(out_prefix + ".pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

# -----------------------------
# Plot: keep contour design, add direction arrows
# -----------------------------
x = np.linspace(-3.0, 3.0, 720)
y = np.linspace(-2.4, 2.4, 600)
X, Y = np.meshgrid(x, y)

# Vectorized smooth-min loss on grid
Qs = []
for (xi, yi, sx, sy, h) in centers:
    Qs.append(((X - xi) ** 2) / (sx ** 2) + ((Y - yi) ** 2) / (sy ** 2) + h)
Qs = np.stack(Qs, axis=0)
W = np.exp(-Qs / tau)
S = np.sum(W, axis=0) + 1e-12
Z = -tau * np.log(S)

fig, ax = plt.subplots(figsize=(9.2, 4.7))

levels = np.linspace(np.quantile(Z, 0.05), np.quantile(Z, 0.85), 10)
ax.contour(X, Y, Z, levels=levels, colors="#B0B0B0", linewidths=1.0)

ax.add_patch(Circle(theta, rho, fill=False, linewidth=1.0, color="black", zorder=5))
ax.scatter([theta[0]], [theta[1]], s=25, color="black", zorder=7)
radius_dir = np.array([1.0, -1.0])
radius_dir = radius_dir / (np.linalg.norm(radius_dir) + 1e-12)
radius_end = theta + rho * radius_dir
# ax.annotate(
#     "",
#     xy=(radius_end[0], radius_end[1]),
#     xytext=(theta[0], theta[1]),
#     arrowprops=dict(arrowstyle="<->", color="#808080", lw=0.8, shrinkA=3, shrinkB=3),
#     zorder=6,
# )
ax.annotate(
    "",
    xy=(gd[0], gd[1]),
    xytext=(theta[0], theta[1]),
    arrowprops=dict(arrowstyle="->", color="blue", lw=0.8, shrinkA=2, shrinkB=2),
    zorder=6,
)
# gd_dash_end = theta - 0.6 * rho * g_dir
# ax.annotate(
#     "",
#     xy=(gd_dash_end[0], gd_dash_end[1]),
#     xytext=(theta[0], theta[1]),
#     arrowprops=dict(arrowstyle="->", color="black", lw=0.8, linestyle="--"),
#     zorder=6,
# )
# ax.annotate(
#     "",
#     xy=(sum_end_from_theta[0], sum_end_from_theta[1]),
#     xytext=(theta[0], theta[1]),
#     arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
#     zorder=6,
# )
ax.annotate(
    "",
    xy=(as0[0], as0[1]),
    xytext=(theta[0], theta[1]),
    arrowprops=dict(arrowstyle="->", color="green", lw=0.8, shrinkA=2, shrinkB=2),
    zorder=6,
)
ax.annotate(
    "",
    xy=(as1[0], as1[1]),
    xytext=(theta[0], theta[1]),
    arrowprops=dict(arrowstyle="->", color="red", lw=0.8, shrinkA=2, shrinkB=2),
    zorder=6,
)
ax.annotate(
    "",
    xy=(rs[0], rs[1]),
    xytext=(theta[0], theta[1]),
    arrowprops=dict(arrowstyle="->", color="orange", lw=0.8, shrinkA=2, shrinkB=2),
    zorder=6,
)
# ax.annotate(
#     "",
#     xy=(as0_dash_end[0], as0_dash_end[1]),
#     xytext=(as0[0], as0[1]),
#     arrowprops=dict(arrowstyle="->", color="green", lw=0.8, linestyle="--"),
#     zorder=6,
# )
# ax.annotate(
#     "",
#     xy=(as1_dash_end[0], as1_dash_end[1]),
#     xytext=(as1[0], as1[1]),
#     arrowprops=dict(arrowstyle="->", color="red", lw=0.8, linestyle="--"),
#     zorder=6,
# )
# ax.annotate(
#     "",
#     xy=(rs_dash_end[0], rs_dash_end[1]),
#     xytext=(rs[0], rs[1]),
#     arrowprops=dict(arrowstyle="->", color="orange", lw=0.8, linestyle="--"),
#     zorder=6,
# )
# ax.annotate(
#     "",
#     xy=(as0_dash_end_from_theta[0], as0_dash_end_from_theta[1]),
#     xytext=(theta[0], theta[1]),
#     arrowprops=dict(arrowstyle="->", color="green", lw=0.8, linestyle="--"),
#     zorder=6,
# )
# ax.annotate(
#     "",
#     xy=(as1_dash_end_from_theta[0], as1_dash_end_from_theta[1]),
#     xytext=(theta[0], theta[1]),
#     arrowprops=dict(arrowstyle="->", color="red", lw=0.8, linestyle="--"),
#     zorder=6,
# )
# ax.annotate(
#     "",
#     xy=(rs_dash_end_from_theta[0], rs_dash_end_from_theta[1]),
#     xytext=(theta[0], theta[1]),
#     arrowprops=dict(arrowstyle="->", color="orange", lw=0.8, linestyle="--"),
#     zorder=6,
# )
ax.scatter([rs[0]], [rs[1]], s=25, color="orange", zorder=7)
ax.scatter([gd[0]], [gd[1]], s=25, color="blue", zorder=8)
ax.scatter([as0[0]], [as0[1]], s=25, color="green", zorder=8)
ax.scatter([as1[0]], [as1[1]], s=25, color="red", zorder=9)

ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-3.0, 3.0)
ax.set_ylim(-2.2, 2.2)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

plt.show()
plt.savefig("exp1_showPertuabtion/triple_junction_random_theta_three_points.png", dpi=220, bbox_inches="tight")
plt.savefig("exp1_showPertuabtion/triple_junction_random_theta_three_points.pdf", bbox_inches="tight")
# Zoomed square view centered at theta with half-width = 2 * rho
zoom_half = 2.5 * rho
ax.set_xlim(theta[0] - zoom_half, theta[0] + zoom_half)
ax.set_ylim(theta[1] - zoom_half, theta[1] + zoom_half)
plt.savefig("exp1_showPertuabtion/triple_junction_random_theta_three_points_zoom.png", dpi=220, bbox_inches="tight")
plt.savefig("exp1_showPertuabtion/triple_junction_random_theta_three_points_zoom.pdf", bbox_inches="tight")

draw_rho2_3d(show=True)