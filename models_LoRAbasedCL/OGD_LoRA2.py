"""LoRA-based Orthogonal Gradient Descent (OGD) baseline.

Implements FO-only OGD with A/B-global gradient projection: all trainable LoRA
A/B gradients are flattened and projected together (single global vector),
instead of doing per-parameter independent projection.

Optional second-order Fisher-diag penalty path is kept for compatibility.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.lora import LoRA_ViT_timm
from models_Project2.models_Full.OGD_utils.gradients import GradientMemory
from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


def _as_bool(value, default: bool = False) -> bool:
    """Parse bool robustly so string values like 'false' won't be treated as True."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------
class Learner(LoraBaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._lora_initialized = False
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # OGD hyperparams
        # Backward-compatible: if only ogd_rank_per_param is provided in old configs,
        # reuse it as direction cap for the new global-memory implementation.
        self._ogd_max_dirs = int(args.get("ogd_max_dirs", args.get("ogd_rank_per_param", 200)))
        self._ogd_store_batches = int(args.get("ogd_store_batches", 50))
        self._ogd_eps = float(args.get("ogd_eps", 1e-8))

        # Optional second-order control
        self._ogd_second_order = _as_bool(args.get("ogd_second_order", False), default=False)
        self._ogd_so_lambda = float(args.get("ogd_so_lambda", 0.0))
        self._ogd_fisher_batches = int(args.get("ogd_fisher_batches", 30))

        # Buffers (CPU)
        self._ab_memory = GradientMemory(mode="orthonormal", max_directions=self._ogd_max_dirs)
        self._fisher: Dict[str, torch.Tensor] | None = None
        self._ref_params: Dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # LoRA backbone builder
    # ------------------------------------------------------------------
    def build_lora_backbone(self, index=True, eval_mode=False):
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        rank = int(self.args.get("lora_rank", 10))
        if rank <= 0:
            raise ValueError(f"lora_rank must be > 0, got {rank}")
        model = LoRA_ViT_timm(
            vit_model=model.eval(),
            r=rank,
            num_classes=0,
            index=index,
            increment=self.args["increment"],
            filepath=self.args.get("filepath", "./"),
            cur_task_index=0,
            learn_alpha=False,
        )
        model.out_dim = 768
        return model

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._refresh_distributed_context()

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._log(f"Learning on {self._known_classes}-{self._total_classes}")

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args.get("train_num_workers", 8),
        )

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args.get("train_num_workers", 8),
        )

        network = self._unwrap_network()
        if not self._lora_initialized:
            network.backbone = self.build_lora_backbone()
            network.backbone.to(self._device)
            self._lora_initialized = True
        self._network = network
        self._prepare_network()

        params = [p for p in self._network.parameters() if p.requires_grad]
        stage = "init" if self._cur_task == 0 else "update"
        optimizer = self._build_optimizer(params, stage=stage)
        if self._optimizer_type not in {"sgd", "adam", "adamw"}:
            raise ValueError(f"OGD_LoRA supports sgd/adam/adamw only, got {self._optimizer_type}")

        if stage == "init":
            epochs = int(self.args.get("init_epoch", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("init_milestones", self.args.get("milestones", [])),
                gamma=float(self.args.get("init_lrate_decay", self.args.get("lrate_decay", 1.0))),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )
        else:
            epochs = int(self.args.get("epochs", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )

        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        for epoch in prog_bar:
            self._network.train()
            total_loss = 0.0
            correct, total = 0, 0

            for batch in self.train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                optimizer.zero_grad()
                outputs = self._network(inputs)
                logits = outputs["logits"]

                if stage == "init":
                    loss = F.cross_entropy(logits, targets)
                    eval_logits = logits
                    eval_targets = targets
                else:
                    fake_targets = targets - self._known_classes
                    loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits = logits[:, self._known_classes :]
                    eval_targets = fake_targets

                if self._ogd_second_order and self._fisher is not None and self._ref_params is not None:
                    so_pen = self._second_order_penalty(self._unwrap_network())
                    if so_pen is not None:
                        loss = loss + so_pen

                loss.backward()
                self._project_gradients(self._unwrap_network())
                optimizer.step()

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss.detach().item())

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => Loss {total_loss/len(self.train_loader):.3f}, Train_accy {train_acc:.2f}"
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, self.test_loader)
                info += f", Test_accy {test_acc:.2f}"
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)

        # Persist LoRA params and head
        save_dir = self.args.get("filepath", "./")
        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_dir, self._cur_task)

        # Update bases and optional Fisher after finishing the task
        self._update_bases(self.train_loader)
        if self._ogd_second_order:
            self._capture_ref_params(self._unwrap_network())
            self._fisher = self._estimate_fisher(self.train_loader)

        # Rehearsal memory if configured
        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._unwrap_network()

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            self._log(f"[OGD_LoRA][NME] Failed to compute class means: {exc}")

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------
    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    @staticmethod
    def _is_lora_ab_name(name: str) -> bool:
        key = name.lower()
        return (
            "linear_a_" in key
            or "linear_b_" in key
            or "lora_new_a" in key
            or "lora_new_b" in key
            or ".w_as." in key
            or ".w_bs." in key
        )

    def _iter_lora_ab_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and self._is_lora_ab_name(name):
                yield name, p

    def _get_flat_grad_vector(self, named_params: List[Tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
        if not named_params:
            return torch.tensor([], device=self._device)
        grads = []
        for _, p in named_params:
            if p.grad is None:
                grads.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            else:
                grads.append(p.grad.view(-1))
        return torch.cat(grads) if grads else torch.tensor([], device=self._device)

    def _set_flat_grad_vector(
        self, named_params: List[Tuple[str, torch.nn.Parameter]], grad_vector: torch.Tensor
    ) -> None:
        offset = 0
        for _, p in named_params:
            numel = p.numel()
            if numel == 0:
                continue
            g = grad_vector[offset : offset + numel].view_as(p)
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.copy_(g)
            offset += numel

    def _project_gradients(self, model: nn.Module) -> None:
        if len(self._ab_memory) == 0:
            return
        named_ab_params = list(self._iter_lora_ab_trainable(model))
        if not named_ab_params:
            return
        flat_grad = self._get_flat_grad_vector(named_ab_params)
        if flat_grad.numel() == 0:
            return
        flat_grad_proj = self._ab_memory.project_orthogonal(flat_grad)
        self._set_flat_grad_vector(named_ab_params, flat_grad_proj)

    # ------------------------------------------------------------------
    # Basis / Fisher maintenance
    # ------------------------------------------------------------------
    def _update_bases(self, loader) -> None:
        model = self._unwrap_network()
        model.eval()
        named_ab_params = list(self._iter_lora_ab_trainable(model))
        if not named_ab_params:
            if self._is_main_process:
                self._log("[OGD_LoRA] No trainable LoRA A/B params found; skip direction update.")
            return

        before = len(self._ab_memory)
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= self._ogd_store_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            model.zero_grad()
            with torch.enable_grad():
                logits = model(inputs)["logits"]
                idx = torch.arange(len(targets), device=targets.device)
                anchor = logits[idx, targets].sum()
                anchor.backward()

            flat_grad = self._get_flat_grad_vector(named_ab_params).detach().cpu()
            if flat_grad.numel() > 0:
                self._ab_memory.add(flat_grad)

            if len(self._ab_memory) >= self._ogd_max_dirs:
                break
        if self._is_main_process:
            added = len(self._ab_memory) - before
            self._log(
                f"[OGD_LoRA] Stored {added} A/B global directions "
                f"(total {len(self._ab_memory)}, cap {self._ogd_max_dirs})"
            )

    def _capture_ref_params(self, model: nn.Module) -> None:
        self._ref_params = {name: p.detach().cpu().clone() for name, p in self._iter_trainable(model)}

    def _estimate_fisher(self, loader) -> Dict[str, torch.Tensor]:
        model = self._unwrap_network()
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        total = 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= self._ogd_fisher_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            model.zero_grad()
            with torch.enable_grad():
                logits = model(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                loss.backward()

            for name, p in self._iter_trainable(model):
                if p.grad is None:
                    continue
                g2 = (p.grad.detach().cpu() ** 2)
                fisher[name] = g2 if name not in fisher else fisher[name] + g2
            total += 1

        if total > 0:
            for name in fisher:
                fisher[name] = fisher[name] / float(total)
        if self._is_main_process:
            self._log(f"[OGD_LoRA] Fisher estimated over {total} batches")
        return fisher

    def _second_order_penalty(self, model: nn.Module):
        if self._fisher is None or self._ref_params is None:
            return None
        pen = None
        for name, p in self._iter_trainable(model):
            ref = self._ref_params.get(name)
            fisher = self._fisher.get(name)
            if ref is None or fisher is None:
                continue
            if ref.numel() != p.numel():
                ref = self._align_tensor_to_param(ref, p)
                self._ref_params[name] = ref
            if fisher.numel() != p.numel():
                fisher = self._align_tensor_to_param(fisher, p)
                self._fisher[name] = fisher
            diff = p - ref.to(device=p.device, dtype=p.dtype)
            fisher_d = fisher.to(device=p.device, dtype=p.dtype)
            term = (fisher_d * diff.pow(2)).sum()
            pen = term if pen is None else pen + term
        if pen is None:
            return None
        return 0.5 * self._ogd_so_lambda * pen

    def _align_tensor_to_param(self, tensor: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        flat = tensor.detach().reshape(-1).cpu()
        target_numel = param.numel()
        if flat.numel() == target_numel:
            return flat.reshape_as(param).clone()

        out = torch.zeros(target_numel, dtype=flat.dtype, device="cpu")
        n = min(target_numel, flat.numel())
        out[:n] = flat[:n]
        return out.reshape_as(param)
