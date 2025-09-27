import numpy as np


def compute_step_metrics(R: np.ndarray):
    """Compute per-task metrics (CA, BWT, FG) from the accuracy matrix ``R``.

    Parameters
    ----------
    R : np.ndarray of shape (T, T)
        Accuracy matrix where ``R[t, j]`` records the accuracy on task ``j`` after
        finishing training on tasks ``1..t``. Unavailable entries should be stored
        as ``NaN``.

    Returns
    -------
    dict
        A dictionary containing per-step arrays for:
        - ``CA`` (current accuracy) with entries ``CA[t] = R[t, t]``;
        - ``BWT`` (backward transfer) computed as the mean of
          ``R[t, i] - R[i, i]`` over ``i < t``;
        - ``FG`` (forward generalization) computed as the mean of ``R[t, j]`` for
          ``j > t``.
    """

    T = R.shape[0]
    CA = np.full(T, np.nan, dtype=float)
    BWT = np.full(T, np.nan, dtype=float)
    FG = np.full(T, np.nan, dtype=float)

    diag = np.diag(R)
    CA[: len(diag)] = diag  # CA_t = R_{t,t}

    for t in range(1, T):
        past = R[t, :t]
        past_ref = diag[:t]
        valid = ~np.isnan(past) & ~np.isnan(past_ref)
        if np.any(valid):
            # BWT_t = (1/(t-1)) sum_{i=1}^{t-1} (R_{t,i} - R_{i,i})
            BWT[t] = np.mean(past[valid] - past_ref[valid])

    for t in range(T - 1):
        future = R[t, t + 1 :]
        valid = ~np.isnan(future)
        if np.any(valid):
            # FG_t = (1/(T-t)) sum_{j=t+1}^{T} R_{t,j}
            FG[t] = np.mean(future[valid])

    return {"CA": CA, "BWT": BWT, "FG": FG}


def compute_sequence_metrics(R: np.ndarray):
    """Aggregate ACA, ABWT, AFG and FAA from the accuracy matrix ``R``.

    The implementation follows the definitions from the provided CL formulation.
    NaN values are ignored, matching standard continual-learning reporting
    practices. The per-step arrays produced by :func:`compute_step_metrics` are
    also returned for downstream analysis.
    """

    step = compute_step_metrics(R)
    CA = step["CA"]
    BWT = step["BWT"]
    FG = step["FG"]

    metrics = {
        # ACA = (1/T) sum_t R_{t,t}
        "ACA": np.nanmean(CA),
        # ABWT = (1/(T-1)) sum_{t=2}^{T} BWT_t
        "ABWT": np.nanmean(BWT[1:]) if len(BWT) > 1 else np.nan,
        # AFG = (1/(T-1)) sum_{t=1}^{T-1} FG_t
        "AFG": (
            np.nanmean(FG[:-1])
            if len(FG) > 1 and np.any(~np.isnan(FG[:-1]))
            else np.nan
        ),
        # FAA = (1/T) sum_j R_{T,j}
        "FAA": np.nanmean(R[-1, :]),
        "CA": CA,
        "BWT": BWT,
        "FG": FG,
    }

    return metrics
