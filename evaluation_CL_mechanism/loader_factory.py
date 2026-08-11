"""
loader_factory.py
-----------------
Build old / new / all / per-task data loaders from a DataManager instance.

Three canonical views at task boundary t (after learning task t):

  old_loader(t)    : test data for tasks 0 .. t-1  (stability signal)
  new_loader(t)    : test data for task t only      (plasticity signal)
  all_loader(t)    : test data for tasks 0 .. t     (global signal)
  ref_loader(t, i) : test data for a specific old task i  (per-task CKA)

All loaders use the test split and test-mode transforms (no augmentation)
to ensure reproducible, augmentation-free sharpness/loss measurements.

Uses data_manager.get_task_class_range() and get_dataset() which already
exist in utils/data_manager.py — no new code in DataManager.
"""
from __future__ import annotations

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import Optional

logger = logging.getLogger(__name__)


def _make_loader(data_manager, class_indices: np.ndarray, args: dict) -> DataLoader:
    """Create a DataLoader over a specific set of class indices."""
    bs          = int(args.get("cl_eval_batch_size", 64))
    num_workers = int(args.get("cl_eval_num_workers", 0))
    dataset = data_manager.get_dataset(
        class_indices, source="test", mode="test"
    )
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_old_loader(
    data_manager,
    cur_task: int,
    args: dict,
) -> Optional[DataLoader]:
    """
    Tasks 0 .. cur_task-1.  Returns None if cur_task == 0 (no old tasks yet).

    Samples are drawn uniformly across all old tasks to avoid early-task
    under-representation when #tasks is large.
    """
    if cur_task == 0:
        return None

    all_old_classes = []
    for t in range(cur_task):
        start, end = data_manager.get_task_class_range(t)
        all_old_classes.extend(range(start, end))

    if not all_old_classes:
        return None

    logger.debug(
        "[loader_factory] old_loader: task 0..%d  (#classes=%d)",
        cur_task - 1, len(all_old_classes),
    )
    return _make_loader(data_manager, np.array(all_old_classes), args)


def make_new_loader(
    data_manager,
    cur_task: int,
    args: dict,
) -> DataLoader:
    """Task cur_task only."""
    start, end = data_manager.get_task_class_range(cur_task)
    logger.debug(
        "[loader_factory] new_loader: task %d  classes [%d, %d)",
        cur_task, start, end,
    )
    return _make_loader(data_manager, np.arange(start, end), args)


def make_all_loader(
    data_manager,
    cur_task: int,
    args: dict,
) -> DataLoader:
    """Tasks 0 .. cur_task (inclusive)."""
    start_0, _ = data_manager.get_task_class_range(0)
    _, end_t   = data_manager.get_task_class_range(cur_task)
    logger.debug(
        "[loader_factory] all_loader: task 0..%d  classes [%d, %d)",
        cur_task, start_0, end_t,
    )
    return _make_loader(data_manager, np.arange(start_0, end_t), args)


def make_ref_loader(
    data_manager,
    ref_task: int,
    args: dict,
) -> DataLoader:
    """Single reference task (for per-task CKA / prototype drift)."""
    start, end = data_manager.get_task_class_range(ref_task)
    return _make_loader(data_manager, np.arange(start, end), args)
