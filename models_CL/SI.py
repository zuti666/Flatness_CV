"""Synaptic Intelligence (SI) baseline for full-parameter training.

Implements path-integral accumulation (small_omega) during a task and
updates importance weights (big_omega) at task end:
    big_omega += small_omega / ((theta - checkpoint)^2 + xi)
Penalty during training uses big_omega to regularize deviation from
previous checkpoint.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_CL.baseLearner import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

        self._si_c = float(args.get("si_c", 1.0))
        self._si_xi = float(args.get("si_xi", 0.1))

        self._big_omega: Dict[str, torch.Tensor] | None = None
        self._small_omega: Dict[str, torch.Tensor] = {}
        self._checkpoint: Dict[str, torch.Tensor] | None = None

        self._optimizer_type = args.get("optimizer_type", "sgd").lower()
        if self._optimizer_type in {"ngd", "naturalgradient"}:
            self._ngd_damping = float(args.get("ngd_damping", 1e-3))
            self._ngd_ema_decay = float(args.get("ngd_ema_decay", 0.95))
            self._ngd_eps = float(args.get("ngd_eps", 1e-8))

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # Expand stored importance to new classifier shapes when task grows
        self._align_buffers_to_model(self._network)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args.get("train_num_workers", 8),
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args.get("train_num_workers", 8),
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[SI] Failed to compute class means: %s", exc)

    def _train(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage="init" if self._cur_task == 0 else "update")
        epochs = int(self.args.get("init_epoch" if self._cur_task == 0 else "epochs", 1))
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("init_milestones" if self._cur_task == 0 else "milestones", []),
            gamma=float(self.args.get("init_lrate_decay" if self._cur_task == 0 else "lrate_decay", 1.0)),
            T_max=epochs,
            eta_min=self.args.get("min_lr", 0.0),
        )

        # init buffers at task start
        self._checkpoint = {n: p.detach().cpu().clone() for n, p in self._iter_trainable(model)}
        self._small_omega = {n: torch.zeros_like(p, device="cpu") for n, p in self._iter_trainable(model)}
        if self._big_omega is None:
            self._big_omega = {n: torch.zeros_like(p, device="cpu") for n, p in self._iter_trainable(model)}

        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            model.train()
            total_loss = 0.0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # snapshot before update
                pre_params = {n: p.detach().cpu().clone() for n, p in self._iter_trainable(model)}

                optimizer.zero_grad()
                outputs = model(inputs)
                logits = outputs["logits"]

                if self._cur_task == 0:
                    loss_task = F.cross_entropy(logits, targets)
                    eval_logits, eval_targets = logits, targets
                else:
                    fake_targets = targets - self._known_classes
                    loss_task = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets

                # 1) backprop task loss to get pure task gradients
                loss_task.backward()

                # store task-only gradients for path integral accumulation (SI definition)
                grad_copy = {n: (p.grad.detach().cpu().clone() if p.grad is not None else None) for n, p in self._iter_trainable(model)}

                # 2) add SI surrogate gradients (do not contaminate grad_copy)
                if self._big_omega is not None and self._checkpoint is not None:
                    for name, p in self._iter_trainable(model):
                        if p.grad is None:
                            continue
                        ref = self._checkpoint.get(name)
                        omega = self._big_omega.get(name)
                        if ref is None or omega is None:
                            continue
                        p.grad.data.add_(self._si_c * 2.0 * omega.to(p.device, p.dtype) * (p - ref.to(p.device, p.dtype)))

                optimizer.step()

                # accumulate small_omega
                with torch.no_grad():
                    for name, p in self._iter_trainable(model):
                        g = grad_copy.get(name)
                        if g is None:
                            continue
                        delta = pre_params[name] - p.detach().cpu()
                        self._small_omega[name].add_(g * delta)

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss_task.detach().item())

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => Loss {total_loss/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            if (epoch % 5 == 4):
                test_acc = self._compute_accuracy(model, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            prog_bar.set_description(info)
        logger.info(info)

        # end-of-task big_omega update
        assert self._big_omega is not None and self._checkpoint is not None
        for name, p in self._iter_trainable(model):
            ref = self._checkpoint.get(name)
            if ref is None:
                continue
            denom = (p.detach().cpu() - ref).pow(2) + self._si_xi
            self._big_omega[name].add_(self._small_omega[name] / denom)

        # refresh checkpoint for next task
        self._checkpoint = {n: p.detach().cpu().clone() for n, p in self._iter_trainable(model)}
        self._small_omega = {n: torch.zeros_like(p, device="cpu") for n, p in self._iter_trainable(model)}

    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network

    def _match_tensor(self, stored: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if stored.shape == param.shape:
            return stored
        new = torch.zeros_like(param.detach().cpu())
        slices = tuple(slice(0, min(a, b)) for a, b in zip(stored.shape, param.shape))
        new[slices] = stored[slices]
        return new

    def _align_buffers_to_model(self, model: nn.Module):
        """Resize stored big_omega to match params (e.g., classifier growth)."""
        if self._big_omega is not None:
            for name, p in model.named_parameters():
                if name in self._big_omega:
                    self._big_omega[name] = self._match_tensor(self._big_omega[name], p)
