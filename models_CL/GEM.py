"""GEM: Gradient Episodic Memory (multi-constraint projection, Lopez-Paz & Ranzato).

Stores one reference gradient per past task. At each step, solve the QP:
    min 0.5||g~ - g||^2  s.t.  <g~, g_k> >= 0 for all past tasks k
We approximate the dual solution with projected gradient descent (small k).
"""

from __future__ import annotations

import logging
from typing import Dict, List

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


def _flatten_grads(params) -> torch.Tensor:
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    return torch.cat(grads) if grads else torch.tensor([])


def _overwrite_grads(params, new_flat: torch.Tensor):
    offset = 0
    for p in params:
        if p.grad is None:
            continue
        numel = p.numel()
        p.grad.copy_(new_flat[offset : offset + numel].view_as(p))
        offset += numel


def _project_multi(g: torch.Tensor, memories: torch.Tensor, iters: int = 50, lr: float = 1.0):
    """Approximate dual PGD for QP: min 0.5||g + A^T v||^2 s.t. v>=0, with A=memories."""
    if memories.numel() == 0:
        return g
    # memories: [K, P], g: [P]
    A = memories
    Gram = (A @ A.t())  # [K,K]
    b = A @ g * -1  # [K]
    v = torch.zeros_like(b)
    for _ in range(iters):
        grad = Gram @ v + b
        v = torch.clamp(v - lr * grad, min=0.0)
    g_proj = g + A.t() @ v
    return g_proj


def _align_vector_dim(vec: torch.Tensor, target_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    flat = vec.reshape(-1)
    if flat.numel() == target_dim:
        return flat.to(device=device, dtype=dtype)
    out = torch.zeros(target_dim, device=device, dtype=dtype)
    n = min(int(flat.numel()), int(target_dim))
    if n > 0:
        out[:n] = flat[:n].to(device=device, dtype=dtype)
    return out


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        self._gem_mem_batch = int(args.get("gem_mem_batch", 64))
        self._gem_grad_batches = int(args.get("gem_grad_batches", 1))
        self._gem_pgditers = int(args.get("gem_pgditers", 50))
        self._gem_pgdlr = float(args.get("gem_pgdlr", 1.0))

        # stored gradients per past task (CPU tensors)
        self._task_grads: List[torch.Tensor] = []

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

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

        # store gradient snapshot for this task
        g_task = self._compute_task_gradient(self.train_loader, self._unwrap_network())
        if g_task is not None:
            self._task_grads.append(g_task.cpu())

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[GEM] Failed to compute class means: %s", exc)

    def _train(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)

        stage = "init" if self._cur_task == 0 else "update"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage=stage)

        epochs = int(self.args.get("init_epoch" if self._cur_task == 0 else "epochs", 1))
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("init_milestones" if self._cur_task == 0 else "milestones", []),
            gamma=float(self.args.get("init_lrate_decay" if self._cur_task == 0 else "lrate_decay", 1.0)),
            T_max=epochs,
            eta_min=self.args.get("min_lr", 0.0),
        )

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

                optimizer.zero_grad()
                outputs = model(inputs)
                logits = outputs["logits"]

                if stage == "init":
                    loss = F.cross_entropy(logits, targets)
                    eval_logits, eval_targets = logits, targets
                else:
                    fake_targets = targets - self._known_classes
                    loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets

                loss.backward()

                if self._cur_task > 0 and self._task_grads:
                    cur_grad = _flatten_grads(params)
                    memories = torch.stack(
                        [
                            _align_vector_dim(g, cur_grad.numel(), cur_grad.device, cur_grad.dtype)
                            for g in self._task_grads
                        ],
                        dim=0,
                    )  # [K, P]
                    dotprod = memories @ cur_grad
                    if (dotprod < 0).any():
                        g_proj = _project_multi(cur_grad, memories, iters=self._gem_pgditers, lr=self._gem_pgdlr)
                        _overwrite_grads(params, g_proj)

                optimizer.step()

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss.detach().item())

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => Loss {total_loss/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            if (epoch % 5 == 4):
                test_acc = self._compute_accuracy(model, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            prog_bar.set_description(info)
        logging.info(info)

    def _compute_task_gradient(self, loader, model: nn.Module):
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        total = 0
        g_accum = None
        for b_idx, batch in enumerate(loader):
            if b_idx >= self._gem_grad_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            model.zero_grad()
            logits = model(inputs)["logits"]
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            g = _flatten_grads(params).cpu()
            g_accum = g if g_accum is None else g_accum + g
            total += 1
        if g_accum is None or total == 0:
            return None
        return g_accum / float(total)

    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
