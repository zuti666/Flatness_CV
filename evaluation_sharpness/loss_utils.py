from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


_GLOBAL_MAX_EXAMPLES_PER_BATCH: Optional[int] = None


def _get_max_examples_per_batch() -> Optional[int]:
    return _GLOBAL_MAX_EXAMPLES_PER_BATCH


def _set_max_examples_per_batch(value: Optional[int]) -> None:
    global _GLOBAL_MAX_EXAMPLES_PER_BATCH
    _GLOBAL_MAX_EXAMPLES_PER_BATCH = value


def _truncate_first_dim(obj: Any, limit: int):
    if isinstance(obj, torch.Tensor):
        if obj.dim() > 0 and obj.size(0) > limit:
            return obj[:limit]
        return obj
    if isinstance(obj, np.ndarray):
        if obj.ndim > 0 and obj.shape[0] > limit:
            return obj[:limit]
        return obj
    if isinstance(obj, list):
        if len(obj) > limit:
            return obj[:limit]
        return obj
    if isinstance(obj, tuple):
        if len(obj) > limit:
            return obj[:limit]
        return obj
    return obj


def _maybe_truncate_batch(inputs, targets):
    limit = _GLOBAL_MAX_EXAMPLES_PER_BATCH
    if limit is None or limit <= 0:
        return inputs, targets
    inputs = _truncate_first_dim(inputs, limit)
    targets = _truncate_first_dim(targets, limit)
    return inputs, targets


def _unwrap_batch(batch):
    """Convert a (idx, inputs, targets) batch into tensors only."""
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            _, inputs, targets = batch
            inputs, targets = _maybe_truncate_batch(inputs, targets)
            return inputs, targets
        if len(batch) == 2:
            inputs, targets = batch
            inputs, targets = _maybe_truncate_batch(inputs, targets)
            return inputs, targets
    raise ValueError("Unexpected batch format")


@contextmanager
def _limit_batch_examples(limit: Optional[int]):
    prev = _GLOBAL_MAX_EXAMPLES_PER_BATCH
    _set_max_examples_per_batch(limit)
    try:
        yield
    finally:
        _set_max_examples_per_batch(prev)


def _extract_logits(model: nn.Module, inputs: torch.Tensor):
    """Run model forward and return logits tensor."""
    outputs = model(inputs)
    if isinstance(outputs, dict):
        return outputs.get("logits", outputs)
    return outputs


def _forward_logits_full(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Best-effort to obtain logits over all seen classes."""
    logits = _extract_logits(model, inputs)
    try:
        need_classes = (int(targets.max().item()) + 1) if (targets is not None) else None
    except Exception:
        need_classes = None
    if need_classes is not None and logits.size(-1) < need_classes:
        if hasattr(model, "interface") and callable(getattr(model, "interface")):
            try:
                logits_full = model.interface(inputs)
                if isinstance(logits_full, torch.Tensor) and logits_full.size(-1) >= need_classes:
                    return logits_full
            except Exception:
                pass
    return logits


def _compute_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    known_classes: Optional[int] = None,
) -> float:
    """Average cross-entropy loss on loader."""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs, targets = _unwrap_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = _forward_logits_full(model, inputs, targets)
            if known_classes is not None and known_classes > 0:
                loss = criterion(logits[:, known_classes:], targets - known_classes)
            else:
                loss = criterion(logits, targets)

            total_loss += loss.item()
            total_samples += targets.size(0)

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    if total_samples == 0:
        return 0.0
    return total_loss / total_samples
