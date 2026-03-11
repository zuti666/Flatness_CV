"""A-GEM: Averaged Gradient Episodic Memory (single-constraint projection).

Simplified GEM variant: use one reference gradient from a random memory
mini-batch; if current gradient conflicts, project to half-space.
"""

from __future__ import annotations

import logging
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


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        self._gem_gamma = float(args.get("gem_gamma", 0.5))
        self._gem_mem_batch = int(args.get("gem_mem_batch", 64))

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

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[A-GEM] Failed to compute class means: %s", exc)

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

                # reference gradient from memory
                mem_grad = None
                mem_available = False
                mem = self._get_memory()
                if mem is not None and len(mem[0]) > 0 and self._cur_task > 0:
                    mem_available = True
                    mem_data, mem_targets = mem
                    sel = np.random.choice(len(mem_data), size=min(self._gem_mem_batch, len(mem_data)), replace=False)
                    mem_inputs = torch.tensor(mem_data[sel]).to(self._device)
                    mem_t = torch.tensor(mem_targets[sel]).to(self._device)
                    optimizer.zero_grad()
                    mem_logits = model(mem_inputs)["logits"]
                    mem_loss = F.cross_entropy(mem_logits, mem_t)
                    mem_loss.backward()
                    mem_grad = _flatten_grads(params)

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

                if mem_available and mem_grad is not None and mem_grad.numel() > 0:
                    cur_grad = _flatten_grads(params)
                    dot = torch.dot(cur_grad, mem_grad)
                    if dot < -self._gem_gamma:
                        proj = cur_grad - ((dot + self._gem_gamma) / (mem_grad.norm() ** 2 + 1e-12)) * mem_grad
                        _overwrite_grads(params, proj)

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

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
