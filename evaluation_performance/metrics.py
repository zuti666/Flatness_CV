
import os
import json
import numpy as np
import pandas as pd

__all__ = [
    "compute_step_metrics",        # backward-compatible API (old name)
    "compute_per_step_metrics",    # new, explicit API
    "compute_sequence_metrics",    # aggregate metrics (now backward-compatible)
    "save_metrics_and_vectors",
]

# ---------------------- utilities ----------------------
def _safe_mean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float('nan')
    return float(np.nanmean(x))

# ---------------------- per-step metrics ----------------------
def compute_per_step_metrics(R_time_by_task: np.ndarray):
    """
    New explicit API.

    Args
    ----
    R_time_by_task : ndarray of shape (T, T)
        Row t is after training task t (0-indexed), column i is accuracy on task i.
        Upper triangle (i > t) may be NaN.

    Returns
    -------
    dict with arrays (length T unless specified):
      - CA: current accuracy per step (diag)
      - BWT_t: backward transfer per step (NaN at t=0)
      - FG_t: forward generalization per step (NaN at t=T-1 or if future scores missing)
      - prefix_mean_t: mean over seen tasks at step t (for AAA)
    """
    R = np.array(R_time_by_task, dtype=float)
    T = R.shape[0]

    # CA_t = R[t, t]
    CA = np.diag(R)

    # BWT_t for t > 0: average over i < t of (R[t, i] - R[i, i])
    BWT_t = np.full(T, np.nan, dtype=float)
    diag = np.diag(R)
    for t in range(1, T):
        deltas = R[t, :t] - diag[:t]
        BWT_t[t] = _safe_mean(deltas)

    # FG_t for t < T-1: average over j > t of R[t, j]
    FG_t = np.full(T, np.nan, dtype=float)
    for t in range(T-1):
        FG_t[t] = _safe_mean(R[t, t+1:])

    # prefix means for AAA
    prefix_mean_t = np.array([_safe_mean(R[t, :t+1]) for t in range(T)], dtype=float)

    return dict(CA=CA, BWT_t=BWT_t, FG_t=FG_t, prefix_mean_t=prefix_mean_t)

def compute_step_metrics(R: np.ndarray):
    """
    Backward-compatible wrapper matching the *previous* project API.

    Parameters
    ----------
    R : np.ndarray of shape (T, T)
        Accuracy matrix where R[t, j] records the accuracy on task j after
        finishing training on tasks 1..t. Unavailable entries as NaN.

    Returns
    -------
    dict
        - 'CA'  : current accuracy per step (CA[t] = R[t, t])
        - 'BWT' : backward transfer per step (mean of R[t,i] - R[i,i] over i < t)
        - 'FG'  : forward generalization per step (mean of R[t,j] over j > t)
    """
    per = compute_per_step_metrics(R)
    # Rename keys to the legacy ones
    return {"CA": per["CA"], "BWT": per["BWT_t"], "FG": per["FG_t"]}

# ---------------------- sequence (aggregate) metrics ----------------------
def compute_sequence_metrics(R_time_by_task: np.ndarray):
    """
    Aggregate ACA, ABWT, AFG, FAA(=Acc), AAA, BWT_final/Forget and
    return per-step arrays. This function is backward-compatible with the
    earlier project's `compute_sequence_metrics` (it also exposes top-level
    keys 'CA', 'BWT', 'FG' in addition to a 'vectors' subdict).

    Notes
    -----
    - NaN values are ignored in means.
    - If your pipeline previously accessed 'CA'/'BWT'/'FG' at top-level,
      that continues to work.
    """
    R = np.array(R_time_by_task, dtype=float)
    T = R.shape[0]

    # per-step (new API)
    per = compute_per_step_metrics(R)
    CA = per["CA"]
    BWT_t = per["BWT_t"]
    FG_t = per["FG_t"]
    prefix_mean_t = per["prefix_mean_t"]

    # aggregates
    ACA = _safe_mean(CA)
    ABWT = _safe_mean(BWT_t[1:]) if T > 1 else float('nan')
    # FWT as uniform average over the strict upper triangle (i<j)
    if T > 1:
        upper_mask = np.triu(np.ones_like(R, dtype=bool), k=1)
        try:
            FWT = float(np.nanmean(R[upper_mask]))
        except Exception:
            FWT = float('nan')
        # AFG kept for backward-compatibility: average of FG_t across steps
        AFG = _safe_mean(FG_t[:-1])
    else:
        FWT = float('nan')
        AFG = float('nan')
    FAA = _safe_mean(R[T-1, :T])     # final-row mean over seen tasks
    Acc = FAA                        # naming equivalence
    # AAA as uniform average over the lower triangle (j<=i)
    lower_mask = np.tril(np.ones_like(R, dtype=bool), k=0)
    try:
        AAA = float(np.nanmean(R[lower_mask]))
    except Exception:
        AAA = float('nan')

    # final-time BWT per task and average
    if T > 1:
        BWT_final_per_task = R[T-1, :T-1] - np.diag(R)[:T-1]
        BWT_final_avg = _safe_mean(BWT_final_per_task)
        # forgetting per task and average (exclude last task by convention)
        with np.errstate(all='ignore'):
            max_per_task = np.nanmax(R[:, :T-1], axis=0) if R.shape[1] > 1 else np.array([])
        Forget_per_task = max_per_task - R[T-1, :T-1] if max_per_task.size else np.array([])
        Forget_avg = _safe_mean(Forget_per_task) if Forget_per_task.size else float('nan')
    else:
        BWT_final_per_task = np.array([])
        BWT_final_avg = float('nan')
        Forget_per_task = np.array([])
        Forget_avg = float('nan')

    # Backward-compatible top-level keys + richer 'vectors'
    return dict(
        ACA=float(ACA),
        ABWT=float(ABWT),
        AFG=float(AFG),
        FWT=float(FWT),
        FAA=float(FAA),
        Acc=float(Acc),
        AAA=float(AAA),
        BWT_final_avg=float(BWT_final_avg),
        Forget_avg=float(Forget_avg),

        # ----- legacy per-step at top-level (for old code) -----
        CA=CA,
        BWT=BWT_t,
        FG=FG_t,

        # ----- structured vectors (for new code) -----
        vectors=dict(
            CA=CA.tolist(),
            BWT_t=np.nan_to_num(BWT_t, nan=np.nan).tolist(),
            FG_t=np.nan_to_num(FG_t, nan=np.nan).tolist(),
            prefix_mean_t=prefix_mean_t.tolist(),
            BWT_final_per_task=BWT_final_per_task.tolist(),
            Forget_per_task=Forget_per_task.tolist(),
        )
    )

# ---------------------- persistence helpers ----------------------
def save_metrics_and_vectors(
    R_time_by_task: np.ndarray,
    out_dir: str,
    out_stub: str,
    tag: str,
    step_T: int = None,
    logger=None,
    *,
    write_csv: bool = False,
    write_json: bool = True,
):
    """
    Compute metrics from R_time_by_task and persist:
      - JSON summary with aggregates and vectors
      - CSV files for main vectors (CA, BWT_t, FG_t, prefix_mean_t, BWT_final_per_task, Forget_per_task)
    """
    os.makedirs(out_dir, exist_ok=True)
    M = compute_sequence_metrics(R_time_by_task)

    T = R_time_by_task.shape[0]
    t_suffix = f"_t{(T-1 if step_T is None else step_T):02d}"
    base = os.path.join(out_dir, f"{out_stub}{t_suffix}_{tag}")

    # ---------- JSON-safe copy (关键修复) ----------
    def _to_jsonable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, dict):
            return {k: _to_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_jsonable(v) for v in o]
        return o

    M_json = _to_jsonable(M)
    # ----------------------------------------------

    # JSON summary (optional)
    json_path = base + "_metrics.json"
    if write_json:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(M_json, f, indent=2)

    # CSV vectors (optional)
    if write_csv:
        vecs = M["vectors"]
        for name, arr in vecs.items():
            arr_np = np.array(arr, dtype=float)
            csv_path = base + f"_{name}.csv"
            pd.DataFrame({name: arr_np}).to_csv(csv_path, index=False)

    # Optional logging
    if logger is not None:
        logger.info("[%s] FAA/Acc=%.4f, AAA=%.4f, ACA=%.4f, ABWT=%.4f, AFG=%.4f, BWT_final_avg=%.4f, Forget_avg=%.4f",
                    tag, M["FAA"], M["AAA"], M["ACA"], M["ABWT"], M["AFG"], M["BWT_final_avg"], M["Forget_avg"])

    return dict(json=(json_path if write_json else None), vectors_dir=out_dir)
