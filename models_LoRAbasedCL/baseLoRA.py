import logging
from typing import Optional
import numpy as np
from torch import optim
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

# from models.base import BaseLearner
from models.baseLearner import BaseLearner


class LoraBaseLearner(BaseLearner):
    """Base helper for LoRA learners with DataParallel/DDP awareness."""

    def __init__(self, args):
        super().__init__(args)
        self._ddp_enabled = False
        self._is_main_process = True
        self._dp_device_ids: list[int] = []
        self._data_parallel_enabled = False
        self._rank = self.args.get("lora_rank", 10)


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
            shuffle=True,
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

        # Compute NME class means over all seen classes for offline evaluation
        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as _nme_exc:  # pylint: disable=broad-except
            logging.exception("[LoRA][restore] Failed to compute class means for NME: %s", _nme_exc)

    def _build_eval_backbone(self, task_idx: int) -> nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError
    
    # ------------------------------------------------------------------
    # Prototype/NME helpers
    # ------------------------------------------------------------------
    def compute_all_seen_class_means(self, data_manager) -> None:
        """Compute per-class prototypes (means) for all seen classes.

        - Uses training split for all seen classes [0, _total_classes)
        - Applies test transforms (mode="test") to avoid heavy augmentation
        - Normalizes features and prototypes (cosine-friendly)
        """
        if getattr(self, "_total_classes", 0) <= 0:
            return

        nb_classes = int(self._total_classes)
        feat_dim = int(self.feature_dim)
        class_means = np.zeros((nb_classes, feat_dim), dtype=np.float64)

        bs = int(self.args.get("eval_batch_size", self.args.get("batch_size", 128)))
        nw = int(self.args.get("eval_num_workers", 8))

        for c in range(nb_classes):
            try:
                dataset_c = data_manager.get_dataset(
                    np.arange(c, c + 1), source="train", mode="test"
                )
                loader_c = DataLoader(dataset_c, batch_size=bs, shuffle=False, num_workers=nw)
                vectors, _ = self._extract_vectors(loader_c)  # np.ndarray [N, D]
                if vectors.size == 0:
                    continue
                # L2-normalize features then mean; normalize prototype as well
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + 1e-12)).T
                mu = np.mean(vectors, axis=0)
                norm = np.linalg.norm(mu) + 1e-12
                class_means[c, :] = (mu / norm).astype(np.float64)
            except Exception as _exc:  # pylint: disable=broad-except
                logging.exception("[LoRA][NME] Failed to compute mean for class %d: %s", c, _exc)

        # Attach for BaseLearner.eval_task() to use
        self._class_means = class_means

    
