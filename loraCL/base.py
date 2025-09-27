import logging
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from models.base import BaseLearner


class LoraBaseLearner(BaseLearner):
    """Base helper for LoRA learners with DataParallel/DDP awareness."""

    def __init__(self, args):
        super().__init__(args)
        self._ddp_enabled = False
        self._is_main_process = True
        self._dp_device_ids: list[int] = []
        self._data_parallel_enabled = False

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------
    def _refresh_distributed_context(self) -> None:
        """Refresh cached distributed flags based on current torch.distributed state."""
        self._ddp_enabled = dist.is_available() and dist.is_initialized()
        self._is_main_process = (not self._ddp_enabled) or dist.get_rank() == 0

        device_ids: list[int] = []
        for device in self._multiple_gpus:
            if isinstance(device, torch.device):
                if device.type == "cuda" and device.index is not None:
                    device_ids.append(device.index)
            elif isinstance(device, int):
                device_ids.append(device)
        self._dp_device_ids = device_ids
        self._data_parallel_enabled = (len(self._dp_device_ids) > 1) and (not self._ddp_enabled)

    # ------------------------------------------------------------------
    def _unwrap_network(self) -> nn.Module:
        """Return the underlying nn.Module, removing any parallel wrappers."""
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network

    def _apply_parallel_wrapper(self) -> None:
        """Apply DataParallel/DDP wrapping according to current context."""
        if self._ddp_enabled:
            if isinstance(self._network, nn.parallel.DistributedDataParallel):
                return
            device = self._device
            device_index: Optional[int] = device.index if device.type == "cuda" else None
            self._network = nn.parallel.DistributedDataParallel(
                self._network,
                device_ids=[device_index] if device_index is not None else None,
                output_device=device_index,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        elif self._data_parallel_enabled:
            if isinstance(self._network, nn.DataParallel):
                return
            if self._dp_device_ids:
                self._network = nn.DataParallel(self._network, device_ids=self._dp_device_ids)
        else:
            if isinstance(self._network, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
                self._network = self._network.module

    def _prepare_network(self) -> None:
        """Move the underlying network to device and wrap if needed."""
        base_network = self._unwrap_network()
        base_network.to(self._device)
        self._network = base_network
        self._apply_parallel_wrapper()

    # ------------------------------------------------------------------
    def _build_sampler(self, dataset, shuffle: bool) -> Optional[DistributedSampler]:
        if not self._ddp_enabled:
            return None
        return DistributedSampler(dataset, shuffle=shuffle)

    @staticmethod
    def _set_epoch(loader, epoch: int) -> None:
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self._is_main_process:
            logging.info(message)

    def _maybe_barrier(self) -> None:
        if self._ddp_enabled:
            dist.barrier()

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def restore_task_snapshot(self, data_manager, task_idx: int) -> None:
        """Restore model state for a particular task for offline evaluation."""
        task_start, task_end = data_manager.get_task_class_range(task_idx)

        self._cur_task = task_idx + 1
        self._total_classes = task_end
        self._known_classes = task_end

        self._refresh_distributed_context()

        self._network.update_fc(self._total_classes)
        if hasattr(self._network, "load_fc"):
            self._network.load_fc(self.args["filepath"], task_idx)

        eval_backbone = self._build_eval_backbone(task_idx)

        network = self._unwrap_network()
        network.backbone = eval_backbone
        network.backbone.to(self._device)
        self._network = network
        self._prepare_network()

        batch_size = self.args.get("batch_size", 128)
        eval_num_workers = self.args.get("eval_num_workers", 8)

        train_classes = np.arange(task_start, task_end)
        train_dataset = data_manager.get_dataset(train_classes, source="train", mode="train")
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, task_end), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
        )

    def _build_eval_backbone(self, task_idx: int) -> nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError
