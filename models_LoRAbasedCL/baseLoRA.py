import logging
import os
from typing import Optional
import numpy as np
from torch import optim
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

# from models.base import BaseLearner
from models_CL.baseLearner import BaseLearner
from optimer_PerturabtionType.util import generate_pertubation
from backbone.lora import _LoRA_qkv_timm_train


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
    # FlatLoRA helpers
    # ------------------------------------------------------------------
    def _init_flatlora_state(self, args) -> None:
        self._flatlora_rho = float(args.get("flatlora_rho", args.get("rho", 0.05)))
        self._flatlora_noise_type = str(args.get("flatlora_noise_type", "flatLoRA_Gauss"))
        self._flatlora_use_cosine_schedule = bool(args.get("flatlora_use_cosine_schedule", True))
        self._flatlora_total_steps = 1
        self._flatlora_step_idx = 0
        self._flatlora_warned_no_modules = False

    def _is_flatlora_optimizer(self) -> bool:
        return getattr(self, "_optimizer_type", "").lower() in {
            "flatlora",
            "flatlora_full",
            "faltlora",
            "faltlora_full",
        }

    def _reset_flatlora_schedule(self, total_steps: int) -> None:
        self._flatlora_total_steps = max(int(total_steps), 1)
        self._flatlora_step_idx = 0

    def _get_scale_value(self, scale_module) -> float:
        if hasattr(scale_module, "param"):
            param = getattr(scale_module, "param")
            if torch.is_tensor(param):
                return float(param.detach().view(-1)[0].item())
        return 1.0

    def _collect_flatlora_modules(self):
        base_model = self._unwrap_network()
        return [module for module in base_model.modules() if isinstance(module, _LoRA_qkv_timm_train)]

    def _build_effective_qkv_weight(self, module: _LoRA_qkv_timm_train) -> torch.Tensor:
        merged_weight = module.qkv.weight.detach().clone()
        device = merged_weight.device
        dtype = merged_weight.dtype

        for task_idx in range(int(module.task_id)):
            key_a = f"saved_A_{task_idx}"
            key_b = f"saved_B_{task_idx}"
            if key_a not in module.saved_A or key_b not in module.saved_B:
                continue

            saved_A = module.saved_A[key_a]
            saved_B = module.saved_B[key_b]
            layer_pairs = list(zip(saved_A, saved_B))[module.t_layer_i * 2 : module.t_layer_i * 2 + 2]
            if len(layer_pairs) < 2:
                continue

            q_entry, v_entry = layer_pairs
            A_q, B_q = q_entry
            A_v, B_v = v_entry
            hist_scale = (
                self._get_scale_value(module.scaling_factor_prev[task_idx])
                if task_idx < len(module.scaling_factor_prev)
                else 1.0
            )

            delta_q = (
                B_q.weight.detach().to(device=device, dtype=dtype)
                @ A_q.weight.detach().to(device=device, dtype=dtype)
            )
            delta_v = (
                B_v.weight.detach().to(device=device, dtype=dtype)
                @ A_v.weight.detach().to(device=device, dtype=dtype)
            )
            merged_weight[: module.dim, :].add_(hist_scale * delta_q)
            merged_weight[-module.dim :, :].add_(hist_scale * delta_v)

        cur_scale = self._get_scale_value(module.scaling_factor[0])
        delta_q_cur = (
            module.linear_b_q.weight.detach().to(device=device, dtype=dtype)
            @ module.linear_a_q.weight.detach().to(device=device, dtype=dtype)
        )
        delta_v_cur = (
            module.linear_b_v.weight.detach().to(device=device, dtype=dtype)
            @ module.linear_a_v.weight.detach().to(device=device, dtype=dtype)
        )
        merged_weight[: module.dim, :].add_(cur_scale * delta_q_cur)
        merged_weight[-module.dim :, :].add_(cur_scale * delta_v_cur)
        return merged_weight

    def _current_flatlora_std(self) -> float:
        base_rho = float(self._flatlora_rho)
        if (not self._flatlora_use_cosine_schedule) or self._flatlora_total_steps <= 1:
            return base_rho
        progress = min(max(float(self._flatlora_step_idx) / float(self._flatlora_total_steps), 0.0), 1.0)
        factor = 0.5 * (1.0 - np.cos(progress * np.pi))
        return base_rho * float(factor)

    def _apply_flatlora_noise(self, std: float):
        flatlora_modules = self._collect_flatlora_modules()
        if not flatlora_modules:
            if not self._flatlora_warned_no_modules:
                self._log("[FlatLoRA] No mergeable LoRA-qkv modules found; falling back to the clean objective.")
                self._flatlora_warned_no_modules = True
            return []

        injected = []
        with torch.no_grad():
            for module in flatlora_modules:
                effective_weight = self._build_effective_qkv_weight(module)
                noise = generate_pertubation(
                    effective_weight,
                    pertubation_mode=self._flatlora_noise_type,
                    std=float(std),
                )
                module.qkv.weight.data.add_(noise)
                injected.append((module.qkv.weight, noise))
        return injected

    def _revert_flatlora_noise(self, injected) -> None:
        if not injected:
            return
        with torch.no_grad():
            for weight, noise in injected:
                weight.data.sub_(noise)

    def _flatlora_step(self, optimizer, loss_closure):
        optimizer.zero_grad()
        std = self._current_flatlora_std()
        injected = self._apply_flatlora_noise(std) if std > 0 else []
        try:
            logits, loss = loss_closure()
            loss.backward()
        finally:
            self._revert_flatlora_noise(injected)
        optimizer.step()
        self._flatlora_step_idx += 1
        return logits.detach(), float(loss.detach().item())

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def load_task_checkpoint(self, checkpoint_dir: str, task_idx: int, data_manager) -> None:
        """Restore a SeqLoRA-style task checkpoint and continue from it.

        The saved SeqLoRA files store the trainable current A/B factors after
        each task. For forked trajectory experiments we copy those factors into
        the current trainable LoRA modules instead of rebuilding an eval-only
        backbone that would treat them as frozen historical adapters.
        """
        checkpoint_dir = os.path.abspath(os.path.expanduser(str(checkpoint_dir)))
        task_idx = int(task_idx)
        if task_idx < 0:
            raise ValueError(f"resume task_idx must be non-negative, got {task_idx}")

        _, task_end = data_manager.get_task_class_range(task_idx)
        self._cur_task = task_idx
        self._known_classes = int(task_end)
        self._total_classes = int(task_end)

        self._refresh_distributed_context()

        network = self._unwrap_network()
        network.update_fc(self._total_classes)
        if hasattr(network, "load_fc"):
            network.load_fc(checkpoint_dir, task_idx)

        backbone = getattr(network, "backbone", None)
        self._load_current_lora_factors(backbone, checkpoint_dir, task_idx)

        self._network = network
        self._prepare_network()
        self._log(
            f"[Resume] Restored task {task_idx} checkpoint from {checkpoint_dir}; "
            f"next task will be {task_idx + 1}"
        )

    def _load_current_lora_factors(self, backbone, checkpoint_dir: str, task_idx: int) -> None:
        if backbone is None:
            raise ValueError("Cannot restore LoRA checkpoint: network has no backbone")

        path_a = os.path.join(checkpoint_dir, f"lora_w_a_{task_idx}.pt")
        path_b = os.path.join(checkpoint_dir, f"lora_w_b_{task_idx}.pt")
        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            raise FileNotFoundError(
                f"LoRA checkpoint for task {task_idx} not found in {checkpoint_dir}"
            )

        saved_as = torch.load(path_a, map_location="cpu")
        saved_bs = torch.load(path_b, map_location="cpu")
        dst_as = getattr(backbone, "w_As", None)
        dst_bs = getattr(backbone, "w_Bs", None)
        if dst_as is None or dst_bs is None:
            raise ValueError("Current backbone does not expose w_As/w_Bs for resume")
        if len(dst_as) != len(saved_as) or len(dst_bs) != len(saved_bs):
            raise ValueError(
                "LoRA branch count mismatch while restoring task "
                f"{task_idx}: current=({len(dst_as)}, {len(dst_bs)}), "
                f"saved=({len(saved_as)}, {len(saved_bs)})"
            )

        for dst, src in zip(dst_as, saved_as):
            src_weight = src.weight if hasattr(src, "weight") else src
            dst.weight.data.copy_(src_weight.detach().to(dst.weight.device))
        for dst, src in zip(dst_bs, saved_bs):
            src_weight = src.weight if hasattr(src, "weight") else src
            dst.weight.data.copy_(src_weight.detach().to(dst.weight.device))

        # Keep the resumed adapter as the single trainable SeqLoRA adapter.
        # Historical saved_A/saved_B entries are eval-time artifacts for other
        # LoRA variants; using them here would double-count the cumulative A/B.
        for module in backbone.modules():
            if hasattr(module, "saved_A"):
                module.saved_A = {}
            if hasattr(module, "saved_B"):
                module.saved_B = {}
            if hasattr(module, "task_id"):
                module.task_id = 0
        if hasattr(backbone, "saved_A"):
            backbone.saved_A = {}
        if hasattr(backbone, "saved_B"):
            backbone.saved_B = {}
        if hasattr(backbone, "task_id"):
            backbone.task_id = 0

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
        self.train_loader = data_manager.build_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=eval_num_workers,
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, task_end), source="test", mode="test"
        )
        self.test_loader = data_manager.build_dataloader(
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
        nw = int(self.args.get("nme_num_workers", self.args.get("eval_num_workers", 8)))

        for c in range(nb_classes):
            try:
                dataset_c = data_manager.get_dataset(
                    np.arange(c, c + 1), source="train", mode="test"
                )
                loader_c = data_manager.build_dataloader(
                    dataset_c,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=nw,
                )
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

    
