"""Synaptic Intelligence (SI) for LoRA-only training.

Tracks path-integral importance (small_omega) over a task and updates
big_omega at task end:
    big_omega += small_omega / ((theta - checkpoint)^2 + xi)
Penalty during training: c * big_omega * (theta - checkpoint)^2.
Only LoRA (requires_grad) parameters are regularized.
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

        # SI hyperparameters
        self._si_c = float(args.get("si_c", 1.0))
        self._si_xi = float(args.get("si_xi", 0.1))

        # buffers (CPU)
        self._big_omega: Dict[str, torch.Tensor] | None = None
        self._small_omega: Dict[str, torch.Tensor] = {}
        self._checkpoint: Dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------#
    # Backbone builder                                                  #
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

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------#
    # Training loop                                                     #
    # ------------------------------------------------------------------#
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

        # init SI buffers for this task
        self._checkpoint = {n: p.detach().cpu().clone() for n, p in self._iter_trainable(self._unwrap_network())}
        self._small_omega = {n: torch.zeros_like(p, device="cpu") for n, p in self._iter_trainable(self._unwrap_network())}
        if self._big_omega is None:
            self._big_omega = {n: torch.zeros_like(p, device="cpu") for n, p in self._iter_trainable(self._unwrap_network())}

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

                pre_params = {n: p.detach().cpu().clone() for n, p in self._iter_trainable(self._unwrap_network())}

                optimizer.zero_grad()
                outputs = self._network(inputs)
                logits = outputs["logits"]

                if stage == "init":
                    loss_task = F.cross_entropy(logits, targets)
                    eval_logits, eval_targets = logits, targets
                else:
                    fake_targets = targets - self._known_classes
                    loss_task = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets

                loss_task.backward()

                # add SI penalty grads if not first task
                if self._big_omega is not None and self._checkpoint is not None:
                    for name, p in self._iter_trainable(self._unwrap_network()):
                        if p.grad is None:
                            continue
                        ref = self._checkpoint.get(name)
                        omega = self._big_omega.get(name)
                        if ref is None or omega is None:
                            continue
                        p.grad.data.add_(self._si_c * 2.0 * omega.to(p.device, p.dtype) * (p - ref.to(p.device, p.dtype)))

                grad_copy = {n: (p.grad.detach().cpu().clone() if p.grad is not None else None) for n, p in self._iter_trainable(self._unwrap_network())}

                optimizer.step()

                with torch.no_grad():
                    for name, p in self._iter_trainable(self._unwrap_network()):
                        g = grad_copy.get(name)
                        if g is None:
                            continue
                        delta = pre_params[name] - p.detach().cpu()
                        self._small_omega[name].add_(g * delta)

                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss_task.detach().item())

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

        # end-of-task update big_omega
        assert self._big_omega is not None and self._checkpoint is not None
        for name, p in self._iter_trainable(self._unwrap_network()):
            ref = self._checkpoint.get(name)
            if ref is None:
                continue
            denom = (p.detach().cpu() - ref).pow(2) + self._si_xi
            self._big_omega[name].add_(self._small_omega[name] / denom)

        # refresh checkpoint for next task
        self._checkpoint = {n: p.detach().cpu().clone() for n, p in self._iter_trainable(self._unwrap_network())}
        self._small_omega = {n: torch.zeros_like(p, device="cpu") for n, p in self._iter_trainable(self._unwrap_network())}

    # ------------------------------------------------------------------#
    # Helpers                                                           #
    # ------------------------------------------------------------------#
    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

