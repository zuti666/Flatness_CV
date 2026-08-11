"""
visualize.py
------------
Generate the mechanistic figures from saved results.

Figure A  –  Plasticity–Stability scatter  (from existing cl_metrics.json)
Figure B  –  L_old / L_new per task       (from cl_mechanism JSON)
Figure C  –  S_old / S_new / S_all per task
Figure D  –  method-specific tr(H_old · Σ) and cos(ε, u₁)
Figure E  –  core theorem chain: ratio / ΔL_old / drift

All figures are saved as PDF + PNG.
Existing CL results (cl_metrics.json) are read directly — no retraining.
New mechanism metrics are read from cl_mechanism JSON files.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# matplotlib is optional at import; we check inside each function
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    logger.warning("[visualize] matplotlib not available; no figures will be generated")


# ── colour / style constants ──────────────────────────────────────────────────
METHOD_STYLE: Dict[str, Dict] = {
    "finetune": {"color": "#555555", "marker": "s", "label": "Fine-tune"},
    "ogd":      {"color": "#1f77b4", "marker": "^", "label": "OGD"},
    "gaussian": {"color": "#ff7f0e", "marker": "D", "label": "OGD+Gaussian"},
    "sam":      {"color": "#d62728", "marker": "v", "label": "OGD+SAM"},
    "ours":     {"color": "#2ca02c", "marker": "o", "label": "Ours (Fisher)"},
}
DEFAULT_STYLE = {"color": "#9467bd", "marker": "P", "label": "Unknown"}

def _style(method_key: str) -> Dict:
    k = method_key.lower()
    for key in METHOD_STYLE:
        if key in k:
            return METHOD_STYLE[key]
    return {**DEFAULT_STYLE, "label": method_key}


def _savefig(fig, path_noext: str) -> None:
    for ext in (".pdf", ".png"):
        fpath = path_noext + ext
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        logger.info("[visualize] saved → %s", fpath)


def _row_geometry(row: Dict[str, Any]) -> str:
    return str(row.get("noise_geometry", "none")).lower()


def _method_tr_hsigma_value(row: Dict[str, Any]) -> float:
    geom = _row_geometry(row)
    mc_key = f"mc_tr_h_sigma_{geom}"
    an_key = f"analytic_tr_h_sigma_{geom}"
    if mc_key in row:
        return row.get(mc_key, np.nan)
    if an_key in row:
        return row.get(an_key, np.nan)
    if geom in {"none", "off", "disabled", "zero"}:
        return 0.0
    return np.nan


def _method_alignment_value(row: Dict[str, Any], eig_idx: int = 0) -> float:
    geom = _row_geometry(row)
    key = f"cos_{geom}_u{eig_idx}"
    if key in row:
        return row.get(key, np.nan)
    if geom in {"none", "off", "disabled", "zero"}:
        return 0.0
    return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# Figure A  –  Plasticity–Stability scatter
# ══════════════════════════════════════════════════════════════════════════════

def _load_cl_metrics_matrix(json_path: str, section: str = "cnn") -> Optional[np.ndarray]:
    """Load the final T×T accuracy matrix from a cl_metrics.json file."""
    with open(json_path, "r") as f:
        J = json.load(f)
    if section not in J or "matrices" not in J[section]:
        return None
    mats  = J[section]["matrices"]
    tkeys = sorted([k for k in mats if re.match(r"t\d+$", k)], key=lambda x: int(x[1:]))
    if not tkeys:
        return None
    T = len(tkeys)
    R = np.full((T, T), np.nan)
    for i, tk in enumerate(tkeys):
        row = np.asarray(mats[tk], dtype=float)
        if row.ndim == 2:
            row = row[-1]
        L = min(T, row.shape[0])
        R[i, :L] = row[:L]
    return R


def _plasticity_stability(R: np.ndarray) -> Tuple[float, float]:
    """
    plasticity : mean of diagonal  (accuracy when first learned)
    stability  : BWT = mean(last_row[0:T-1] - diag[0:T-1])
    """
    T = R.shape[0]
    diag = np.array([R[t, t] for t in range(T) if not np.isnan(R[t, t])])
    last = R[T - 1, :]

    plasticity = float(np.nanmean(diag))
    # BWT: compare last-row vs diagonal for all but the final task
    bwt_vals = [
        last[j] - R[j, j]
        for j in range(T - 1)
        if not np.isnan(R[j, j]) and not np.isnan(last[j])
    ]
    stability = float(np.mean(bwt_vals)) if bwt_vals else float("nan")
    return plasticity, stability


def figure_A_plasticity_stability(
    cl_metrics_files: Dict[str, List[str]],
    out_dir: str,
    section: str = "cnn",
) -> None:
    """
    cl_metrics_files : {method_key: [path1, path2, ...]}  (one path per seed)
    """
    if not _MPL_OK:
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for method_key, paths in cl_metrics_files.items():
        st = _style(method_key)
        pts_x, pts_y = [], []
        for path in paths:
            try:
                R = _load_cl_metrics_matrix(path, section)
                if R is None:
                    continue
                p, s = _plasticity_stability(R)
                pts_x.append(p)
                pts_y.append(s)
            except Exception as exc:
                logger.warning("[figure_A] %s: %s", path, exc)

        if not pts_x:
            continue

        ax.scatter(
            pts_x, pts_y,
            color=st["color"], marker=st["marker"],
            s=80, zorder=3, label=st["label"],
        )
        # mean cross-hair
        mx, my = np.mean(pts_x), np.mean(pts_y)
        ax.scatter([mx], [my], color=st["color"], marker="+",
                   s=200, linewidths=2, zorder=4)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Plasticity  (mean diagonal accuracy, %)", fontsize=12)
    ax.set_ylabel("Stability  (BWT, %)", fontsize=12)
    ax.set_title("Plasticity–Stability Decomposition", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "figA_plasticity_stability"))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure B  –  L_old / L_new per task
# ══════════════════════════════════════════════════════════════════════════════

def figure_B_loss_trajectory(
    per_method_task_results: Dict[str, List[Dict[str, Any]]],
    out_dir: str,
) -> None:
    """
    per_method_task_results : {method_key: [task_dict_t0, task_dict_t1, ...]}
    """
    if not _MPL_OK:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for method_key, task_list in per_method_task_results.items():
        st = _style(method_key)
        tasks = [d["task"] for d in task_list]
        l_old = [d.get("L_old", np.nan) for d in task_list]
        l_new = [d.get("L_new", np.nan) for d in task_list]

        axes[0].plot(tasks, l_old, color=st["color"], marker=st["marker"],
                     linewidth=1.5, markersize=5, label=st["label"])
        axes[1].plot(tasks, l_new, color=st["color"], marker=st["marker"],
                     linewidth=1.5, markersize=5, label=st["label"])

    for ax, title in zip(axes, ["$L_{\\mathrm{old}}$ (old-task loss)", "$L_{\\mathrm{new}}$ (new-task loss)"]):
        ax.set_xlabel("Task index $t$", fontsize=11)
        ax.set_ylabel("Cross-entropy loss", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle("Loss Trajectory per Task Boundary", fontsize=13)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "figB_loss_trajectory"))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure C  –  S_old / S_new / S_all per task
# ══════════════════════════════════════════════════════════════════════════════

def figure_C_sharpness_old_new(
    per_method_task_results: Dict[str, List[Dict[str, Any]]],
    out_dir: str,
) -> None:
    if not _MPL_OK:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for method_key, task_list in per_method_task_results.items():
        st = _style(method_key)
        tasks = [d["task"] for d in task_list]
        s_old = [d.get("S_old", np.nan) for d in task_list]
        s_new = [d.get("S_new", np.nan) for d in task_list]
        s_all = [d.get("S_all", np.nan) for d in task_list]

        axes[0].semilogy(tasks, [max(v, 1e-12) for v in s_old],
                         color=st["color"], marker=st["marker"],
                         linewidth=1.5, markersize=5, label=st["label"])
        axes[1].semilogy(tasks, [max(v, 1e-12) for v in s_new],
                         color=st["color"], marker=st["marker"],
                         linewidth=1.5, markersize=5, label=st["label"])
        axes[2].semilogy(tasks, [max(v, 1e-12) for v in s_all],
                         color=st["color"], marker=st["marker"],
                         linewidth=1.5, markersize=5, label=st["label"])

    for ax, title in zip(
        axes,
        ["$S_{\\mathrm{old}}=\\lambda_{\\max}(H_{\\mathrm{old}})$",
         "$S_{\\mathrm{new}}=\\lambda_{\\max}(H_{\\mathrm{new}})$",
         "$S_{\\mathrm{all}}=\\lambda_{\\max}(H_{\\mathrm{all}})$"],
    ):
        ax.set_xlabel("Task index $t$", fontsize=11)
        ax.set_ylabel("λ_max  (log scale)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)

    plt.suptitle("Old / New / Global Curvature (λ_max)", fontsize=13)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "figC_sharpness_old_new"))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure D  –  tr(H_old · Σ) and cos(ε, u₁)
# ══════════════════════════════════════════════════════════════════════════════

def figure_D_tr_hsigma_and_alignment(
    per_method_task_results: Dict[str, List[Dict[str, Any]]],
    out_dir: str,
) -> None:
    """
    Left  : method-specific tr(H_old · Σ_noise) per task
    Right : method-specific |cos(ε, u₁^old)| bar chart  (mean across tasks)
    """
    if not _MPL_OK:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── left: tr(H_old · Σ) trajectory ──────────────────────────────────────
    for method_key, task_list in per_method_task_results.items():
        st = _style(method_key)
        tasks = [d["task"] for d in task_list if d["task"] >= 1]
        vals = [_method_tr_hsigma_value(d) for d in task_list if d["task"] >= 1]

        axes[0].semilogy(tasks, [max(v, 1e-30) for v in vals],
                         color=st["color"], marker=st["marker"],
                         linewidth=1.5, markersize=5, label=st["label"])

    axes[0].set_xlabel("Task index $t$", fontsize=11)
    axes[0].set_ylabel("$E[\\epsilon^\\top H_{\\mathrm{old}} \\epsilon]$  (log)", fontsize=11)
    axes[0].set_title("$\\mathrm{tr}(H_{\\mathrm{old}} \\Sigma_{\\mathrm{noise}})$", fontsize=12)
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(fontsize=7)

    # ── right: cos(ε, u₁) bar chart ──────────────────────────────────────────
    method_keys  = list(per_method_task_results.keys())
    cos_f_means  = []
    cos_g_means  = []
    bar_labels   = []

    for method_key in method_keys:
        task_list = per_method_task_results[method_key]
        method_vals = [_method_alignment_value(d, eig_idx=0) for d in task_list if d.get("task", -1) >= 1]
        gaussian_vals = [d.get("cos_gaussian_u0", np.nan) for d in task_list if d.get("task", -1) >= 1]
        method_valid = [v for v in method_vals if not np.isnan(v)]
        gaussian_valid = [v for v in gaussian_vals if not np.isnan(v)]
        cos_f_means.append(np.mean(method_valid) if method_valid else 0.0)
        cos_g_means.append(np.mean(gaussian_valid) if gaussian_valid else 0.0)
        bar_labels.append(_style(method_key)["label"])

    x = np.arange(len(method_keys))
    w = 0.35
    bars_f = axes[1].bar(x - w / 2, cos_f_means, w,
                         label="Method geometry", color="#2ca02c", alpha=0.8)
    bars_g = axes[1].bar(x + w / 2, cos_g_means, w,
                         label="Gaussian counterfactual", color="#ff7f0e", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bar_labels, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel(r"$|\cos(\epsilon, u_1^{\mathrm{old}})|$  (mean)", fontsize=11)
    axes[1].set_title("Noise–Curvature Alignment", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Core Mechanistic Evidence", fontsize=13)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "figD_tr_hsigma_alignment"))
    plt.close(fig)


def figure_E_core_mechanism(
    per_method_task_results: Dict[str, List[Dict[str, Any]]],
    out_dir: str,
) -> None:
    """Directly visualize the theorem chain with the three key proxies."""
    if not _MPL_OK:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=False)

    for method_key, task_list in per_method_task_results.items():
        st = _style(method_key)
        rows = [d for d in task_list if d.get("task", -1) >= 1]
        tasks = [d["task"] for d in rows]
        ratio = [d.get("analytic_tr_ratio_gauss_fisher", np.nan) for d in rows]
        delta_old = [d.get("delta_L_old", np.nan) for d in rows]
        drift_vq = [d.get("drift_cos_vq", np.nan) for d in rows]

        axes[0].plot(tasks, ratio, color=st["color"], marker=st["marker"],
                     linewidth=1.5, markersize=5, label=st["label"])
        axes[1].plot(tasks, delta_old, color=st["color"], marker=st["marker"],
                     linewidth=1.5, markersize=5, label=st["label"])
        axes[2].plot(tasks, drift_vq, color=st["color"], marker=st["marker"],
                     linewidth=1.5, markersize=5, label=st["label"])

    axes[0].set_xlabel("Task index $t$", fontsize=11)
    axes[0].set_ylabel(r"$\mathrm{tr}(H_{\mathrm{old}}\Sigma_{\mathrm{gauss}})\ /\ \mathrm{tr}(H_{\mathrm{old}}\Sigma_{\mathrm{fisher}})$", fontsize=10)
    axes[0].set_title("Diffusion Ratio", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Task index $t$", fontsize=11)
    axes[1].set_ylabel(r"$\Delta L_{\mathrm{old}}$", fontsize=11)
    axes[1].set_title("Forgetting Increment", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].set_xlabel("Task index $t$", fontsize=11)
    axes[2].set_ylabel(r"$\cos(\nabla L_{\mathrm{old}}, q)$", fontsize=11)
    axes[2].set_title("Drift Proxy", fontsize=12)
    axes[2].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    plt.suptitle("Theorem-Chain Mechanism Proxies", fontsize=13)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "figE_core_mechanism"))
    plt.close(fig)


def _render_lossland_npz(npz_path: str, out_prefix: str, title: str) -> None:
    if not _MPL_OK or not os.path.exists(npz_path):
        return

    data = np.load(npz_path)
    if {"x", "loss"}.issubset(set(data.files)) and "y" not in data.files:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        x = data["x"]
        loss = data["loss"]
        ax.plot(x, loss, color="#1f77b4", linewidth=1.8)
        ax.set_xlabel("Direction coefficient", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _savefig(fig, out_prefix)
        plt.close(fig)
        return

    if {"x", "y", "loss"}.issubset(set(data.files)):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        x = data["x"]
        y = data["y"]
        z = data["loss"]
        im = ax.contourf(x, y, z.T, levels=20, cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=0.9)
        ax.set_xlabel("Direction 1", fontsize=11)
        ax.set_ylabel("Direction 2", fontsize=11)
        ax.set_title(title, fontsize=12)
        plt.tight_layout()
        _savefig(fig, out_prefix)
        plt.close(fig)


def render_lossland_artifacts(
    per_method_task_results: Dict[str, List[Dict[str, Any]]],
    out_dir: str,
) -> None:
    """Render any saved 1D/2D loss-landscape npz artifacts into figures."""
    if not _MPL_OK:
        return

    for method_key, task_list in per_method_task_results.items():
        method_dir = os.path.join(out_dir, "lossland", method_key)
        for row in task_list:
            task = int(row.get("task", -1))
            for split in ("old", "new", "all"):
                for dim in ("1d", "2d"):
                    key = f"lossland_{split}_{dim}_file"
                    path = row.get(key)
                    if not path or not os.path.exists(path):
                        continue
                    title = f"{_style(method_key)['label']} | task {task} | {split} | {dim.upper()}"
                    out_prefix = os.path.join(method_dir, f"task{task:02d}_{split}_{dim}")
                    _render_lossland_npz(path, out_prefix, title)


# ══════════════════════════════════════════════════════════════════════════════
# Convenience runner  –  generate all four figures from saved JSONs
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_figures(
    mechanism_roots: Dict[str, str],       # {method_key: path_to_cl_mechanism_dir}
    cl_metrics_files: Optional[Dict[str, List[str]]],  # for Figure A
    out_dir: str,
    section: str = "cnn",
) -> None:
    """
    Parameters
    ----------
    mechanism_roots : {method_key: directory containing cl_mechanism_t??.json}
    cl_metrics_files: {method_key: [path_to_cl_metrics.json, ...]}  (for Fig A)
    out_dir         : output directory for figures
    """
    from evaluation_CL_mechanism.io_utils import load_task_results

    # Load per-method task results
    per_method: Dict[str, List[Dict[str, Any]]] = {}
    for method_key, root in mechanism_roots.items():
        task_results = load_task_results(root)
        if task_results:
            per_method[method_key] = task_results
        else:
            logger.warning("[visualize] no task JSONs found in %s", root)

    os.makedirs(out_dir, exist_ok=True)

    # Figure A (uses cl_metrics.json, no mechanism data needed)
    if cl_metrics_files:
        logger.info("[visualize] generating Figure A ...")
        figure_A_plasticity_stability(cl_metrics_files, out_dir, section)

    if not per_method:
        logger.warning("[visualize] no mechanism data available; skipping Figs B/C/D")
        return

    logger.info("[visualize] generating Figure B ...")
    figure_B_loss_trajectory(per_method, out_dir)

    logger.info("[visualize] generating Figure C ...")
    figure_C_sharpness_old_new(per_method, out_dir)

    logger.info("[visualize] generating Figure D ...")
    figure_D_tr_hsigma_and_alignment(per_method, out_dir)

    logger.info("[visualize] generating Figure E ...")
    figure_E_core_mechanism(per_method, out_dir)

    logger.info("[visualize] rendering loss landscape artifacts (if any) ...")
    render_lossland_artifacts(per_method, out_dir)

    logger.info("[visualize] all figures saved to %s", out_dir)
