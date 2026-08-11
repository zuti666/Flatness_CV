from __future__ import annotations

from typing import Sequence

import numpy as np


def continual_metrics(matrix: Sequence[Sequence[float]]) -> dict[str, float]:
    """Compute standard metrics from a lower-triangular accuracy matrix.

    R[t][i] is test accuracy on task i after finishing training task t.
    Values may be proportions or percentages; outputs preserve the same scale.
    """
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("accuracy matrix must be square")
    task_count = array.shape[0]
    final = array[task_count - 1, :task_count]
    aaa_by_step = [float(np.nanmean(array[t, : t + 1])) for t in range(task_count)]
    bwt_terms = [array[-1, i] - array[i, i] for i in range(task_count - 1)]
    forgetting_terms = [
        np.nanmax(array[i:, i]) - array[-1, i] for i in range(task_count - 1)
    ]
    return {
        "final_average_accuracy": float(np.nanmean(final)),
        "average_anytime_accuracy": float(np.mean(aaa_by_step)),
        "backward_transfer": float(np.mean(bwt_terms)) if bwt_terms else 0.0,
        "average_forgetting": float(np.mean(forgetting_terms)) if forgetting_terms else 0.0,
        "aaa_by_step": aaa_by_step,
    }
