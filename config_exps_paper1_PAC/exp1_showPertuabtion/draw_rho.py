import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

rng = np.random.default_rng(7)

# -----------------------------
# Terrain with a visible secondary local minimum ("secondary center")
# -----------------------------
def loss_fn(x, y):
    # Mild global bowl to provide large-scale contours
    a, b, c = 0.18, 0.10, 0.03
    base = a * x**2 + b * y**2 + c * x * y

    # Secondary local well (negative Gaussian): creates a local minimum near theta
    amp = -2.4
    xc, yc = 1.55, 1.10
    sx, sy = 0.40, 0.28   # larger -> less dense contours around the basin
    dx = (x - xc) / sx
    dy = (y - yc) / sy
    well = amp * np.exp(-0.5 * (dx**2 + dy**2))

    # Gentle outer ridge to outline the basin without overly tightening contours
    amp2 = 0.55
    sx2, sy2 = 0.90, 0.70
    dx2 = (x - xc) / sx2
    dy2 = (y - yc) / sy2
    ridge = amp2 * np.exp(-0.5 * (dx2**2 + dy2**2))

    return base + well + ridge

def grad_loss(x, y):
    a, b, c = 0.18, 0.10, 0.03
    gx_base = 2 * a * x + c * y
    gy_base = 2 * b * y + c * x

    amp = -2.4
    xc, yc = 1.55, 1.10
    sx, sy = 0.40, 0.28
    dx = (x - xc) / sx
    dy = (y - yc) / sy
    e = np.exp(-0.5 * (dx**2 + dy**2))
    gx_well = amp * e * (-(x - xc) / (sx**2))
    gy_well = amp * e * (-(y - yc) / (sy**2))

    amp2 = 0.55
    sx2, sy2 = 0.90, 0.70
    dx2 = (x - xc) / sx2
    dy2 = (y - yc) / sy2
    e2 = np.exp(-0.5 * (dx2**2 + dy2**2))
    gx_ridge = amp2 * e2 * (-(x - xc) / (sx2**2))
    gy_ridge = amp2 * e2 * (-(y - yc) / (sy2**2))

    return np.array([gx_base + gx_well + gx_ridge, gy_base + gy_well + gy_ridge])

def hess_gauss(x, y, amp, xc, yc, sx, sy):
    e = np.exp(-0.5 * (((x - xc) / sx) ** 2 + ((y - yc) / sy) ** 2))
    dxx = amp * e * (((x - xc) ** 2) / (sx ** 4) - 1.0 / (sx ** 2))
    dyy = amp * e * (((y - yc) ** 2) / (sy ** 4) - 1.0 / (sy ** 2))
    dxy = amp * e * (((x - xc) * (y - yc)) / ((sx ** 2) * (sy ** 2)))
    return np.array([[dxx, dxy], [dxy, dyy]])

def hess_loss(x, y):
    a, b, c = 0.18, 0.10, 0.03
    H_base = np.array([[2 * a, c],
                       [c, 2 * b]])

    xc, yc = 1.55, 1.10
    H_well = hess_gauss(x, y, -2.4, xc, yc, 0.40, 0.28)
    H_ridge = hess_gauss(x, y,  0.55, xc, yc, 0.90, 0.70)

    return H_base + H_well + H_ridge

# -----------------------------
# Theta: close to (but not at) the secondary center
# -----------------------------
secondary_center = np.array([1.95, 0.60])
theta = secondary_center + np.array([-0.28, -0.22])  # near but not on the center
rho = 0.32  # smaller radius as requested

# AS0: max gradient direction
g = grad_loss(theta[0], theta[1])
g_dir = g / (np.linalg.norm(g) + 1e-12)
as0 = theta - rho * g_dir

# AS1: max curvature direction (largest Hessian eigenvector), choose sign by higher loss increase
H = hess_loss(theta[0], theta[1])
eigvals, eigvecs = np.linalg.eigh(H)
v_max = eigvecs[:, np.argmax(eigvals)]
p_plus = theta + rho * v_max
p_minus = theta - rho * v_max
# choose sign by lower loss (descent side)
if loss_fn(p_minus[0], p_minus[1]) < loss_fn(p_plus[0], p_plus[1]):
    v_max = -v_max
as1 = theta + rho * v_max

# RS: inside circle, closer to the center than the boundary
r_rs = 0.62 * rho * np.sqrt(rng.uniform(0, 1))
ang_rs = rng.uniform(0, 2 * np.pi)
rs = theta - np.array([r_rs * np.cos(ang_rs), r_rs * np.sin(ang_rs)])

# -----------------------------
# Plot (no text), show full scene
# -----------------------------
x = np.linspace(-3.2, 3.6, 800)
y = np.linspace(-2.6, 2.6, 700)
X, Y = np.meshgrid(x, y)
Z = loss_fn(X, Y)

fig, ax = plt.subplots(figsize=(8.4, 6.6))

# fewer levels -> less dense near the basin
levels = np.linspace(np.quantile(Z, 0.05), np.quantile(Z, 0.80), 11)
ax.contour(X, Y, Z, levels=levels, colors="black", linestyles="--", linewidths=1.3)

ax.add_patch(Circle(theta, rho, fill=False, linewidth=1.5, color="black", zorder=5))
ax.scatter([theta[0]], [theta[1]], s=65, color="black", zorder=7)
ax.scatter([rs[0]], [rs[1]], s=60, color="orange", zorder=8)
ax.scatter([as0[0]], [as0[1]], s=60, color="green", zorder=9)
ax.scatter([as1[0]], [as1[1]], s=60, color="red", zorder=10)

ax.set_xticks(np.arange(-3.0, 3.5, 1.0))
ax.set_yticks(np.arange(-2.0, 2.5, 1.0))
ax.tick_params(axis="both", which="major", length=4, width=1.0, labelsize=10)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-3.0, 3.4)
ax.set_ylim(-2.4, 2.4)

plt.show()
plt.savefig("exp1_showPertuabtion/visual.pdf")
