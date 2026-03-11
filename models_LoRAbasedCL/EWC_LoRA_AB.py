"""Elastic Weight Consolidation (EWC) for LoRA-only training.

Diagonal Fisher over trainable LoRA parameters; backbone frozen.
Online accumulation with decay gamma, penalty weight lambda.

Args (all optional):
    ewc_lambda        : weight for quadratic penalty (default 20.0)
    ewc_gamma         : Fisher decay for online EWC (default 1.0)
    ewc_max_batches   : cap batches for Fisher estimate (default 100)
    ewc_eps           : numerical floor for Fisher entries (default 1e-5)

This mirrors the full-parameter EWC but restricts to requires_grad params
(LoRA A/B and classifier head if trainable).
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.lora import LoRA_ViT_timm
from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(LoraBaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._lora_initialized = False
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # EWC hyperparameters
        self._ewc_lambda = float(args.get("ewc_lambda", 20.0))
        self._ewc_gamma = float(args.get("ewc_gamma", 1.0))
        self._ewc_max_batches = int(args.get("ewc_max_batches", 100))
        self._ewc_eps = float(args.get("ewc_eps", 1e-5))

        # Buffers (CPU)
        self._fisher: Dict[str, torch.Tensor] | None = None
        self._checkpoint: Dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------#
    # LoRA backbone builder                                             #
    # ------------------------------------------------------------------#
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

    # ------------------------------------------------------------------#
    # Lifecycle                                                         #
    # ------------------------------------------------------------------#
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
        self._align_ewc_buffers(self._unwrap_network())

        params = [p for p in self._network.parameters() if p.requires_grad]
        stage = "init" if self._cur_task == 0 else "update"
        optimizer = self._build_optimizer(params, stage=stage)
        if self._optimizer_type not in {"sgd", "adam", "adamw"}:
            raise ValueError(f"EWC_LoRA supports sgd/adam/adamw only, got {self._optimizer_type}")

        if stage == "init":
            epochs = int(self.args.get("init_epoch", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
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
                    loss_task = F.cross_entropy(logits, targets)
                    eval_logits = logits
                    eval_targets = targets
                else:
                    fake_targets = targets - self._known_classes
                    loss_task = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits = logits[:, self._known_classes :]
                    eval_targets = fake_targets

                loss = loss_task + self._ewc_penalty(self._unwrap_network())
                loss.backward()
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

        # Save LoRA + head
        save_dir = self.args.get("filepath", "./")
        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_dir, self._cur_task)

        # After-task Fisher + checkpoint
        self._checkpoint = {name: p.detach().cpu().clone() for name, p in self._iter_trainable(self._unwrap_network())}
        self._fisher = self._compute_fisher(self.train_loader, self._unwrap_network())

        # Rehearsal if any
        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._unwrap_network()

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            self._log(f"[EWC_LoRA][NME] Failed to compute class means: {exc}")

    # ------------------------------------------------------------------#
    # Helpers                                                           #
    # ------------------------------------------------------------------#
    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    def _match_tensor(self, stored: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if stored.shape == param.shape:
            return stored
        target = param.detach().cpu()
        new = torch.zeros_like(target)
        if stored.ndim != target.ndim:
            return new
        slices = tuple(slice(0, min(a, b)) for a, b in zip(stored.shape, target.shape))
        new[slices] = stored[slices]
        return new

    def _align_ewc_buffers(self, model: nn.Module):
        if self._fisher is not None:
            for name, p in model.named_parameters():
                if name in self._fisher:
                    self._fisher[name] = self._match_tensor(self._fisher[name], p)
        if self._checkpoint is not None:
            for name, p in model.named_parameters():
                if name in self._checkpoint:
                    self._checkpoint[name] = self._match_tensor(self._checkpoint[name], p)

    def _ewc_penalty(self, model: nn.Module):
        if self._fisher is None or self._checkpoint is None:
            return torch.tensor(0.0, device=self._device)
        pen = None
        for name, p in self._iter_trainable(model):
            ref = self._checkpoint.get(name)
            fisher = self._fisher.get(name)
            if ref is None or fisher is None:
                continue
            diff = p - ref.to(device=p.device, dtype=p.dtype)
            fisher_d = fisher.to(device=p.device, dtype=p.dtype)
            term = (fisher_d * diff.pow(2)).sum()
            pen = term if pen is None else pen + term
        if pen is None:
            return torch.tensor(0.0, device=self._device)
        return 0.5 * self._ewc_lambda * pen

    def _compute_fisher(self, loader, model: nn.Module) -> Dict[str, torch.Tensor]:
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        total = 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= self._ewc_max_batches:
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
                fisher[name] = (fisher[name] / float(total)).clamp_min(self._ewc_eps)

        if self._fisher is not None:
            for name, new_f in fisher.items():
                old = self._fisher.get(name)
                if old is None:
                    self._fisher[name] = new_f
                else:
                    self._fisher[name] = self._ewc_gamma * old + new_f
            for name, old in self._fisher.items():
                if name not in fisher:
                    self._fisher[name] = self._ewc_gamma * old
        else:
            self._fisher = fisher

        if self._is_main_process:
            self._log(f"[EWC_LoRA] Fisher estimated over {total} batches")
        return self._fisher
