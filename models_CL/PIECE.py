"""PIECE-S style parameter-importance masking (approximate).

Parameter Importance-Driven Continual Learning for Foundation Models (2025).
Implements the core idea: compute per-parameter importance score
    S_i = mean_grad_i / sqrt(mean_grad_sq_i + xi)
over a few batches of the current task, then only update the top fraction
of parameters (others frozen) during this task. No rehearsal or projection.
"""

from __future__ import annotations

import logging
from typing import List

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
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        self._pi_top_frac = float(args.get("pi_top_frac", 0.001))  # 0.1%
        self._pi_xi = float(args.get("pi_xi", 1e-8))
        self._pi_batches = int(args.get("pi_batches", 10))

    def after_task(self):
        self._known_classes = self._total_classes
        # restore all params trainable for next task selection
        for p in self._unwrap_network().parameters():
            p.requires_grad = True
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

        # compute importance and freeze
        self._compute_and_mask_importance(self.train_loader)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[PIECE] Failed to compute class means: %s", exc)

    def _compute_and_mask_importance(self, loader):
        model = self._unwrap_network()
        model.to(self._device)
        params = [p for p in model.parameters() if p.requires_grad]
        # accumulators
        sum_g = [torch.zeros_like(p, device=self._device) for p in params]
        sum_g2 = [torch.zeros_like(p, device=self._device) for p in params]

        iters = 0
        for b_idx, batch in enumerate(loader):
            if b_idx >= self._pi_batches:
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
            for i, p in enumerate(params):
                if p.grad is None:
                    continue
                g = p.grad.detach()
                sum_g[i] += g
                sum_g2[i] += g * g
            iters += 1
        if iters == 0:
            return
        scores = []
        for sg, sg2 in zip(sum_g, sum_g2):
            mean_g = sg / iters
            mean_g2 = sg2 / iters
            s = mean_g / torch.sqrt(mean_g2 + self._pi_xi)
            scores.append(s.abs())

        # flatten scores
        flat_scores = torch.cat([s.view(-1) for s in scores])
        k = max(1, int(flat_scores.numel() * self._pi_top_frac))
        thresh = torch.topk(flat_scores, k, sorted=False).values.min()

        # apply mask
        idx = 0
        for p, s in zip(params, scores):
            mask = s.abs() >= thresh
            p.requires_grad = True
            if mask.numel() > 0:
                # set requires_grad False for entries below mask by detaching gradient in backward
                # easiest: clone data and register hook zeroing grad where mask False
                m = mask.detach()
                def hook_factory(m_local):
                    return lambda grad: grad * m_local
                p.register_hook(hook_factory(m))
            idx += 1

        logger.info("[PIECE] top frac %.4f => keep %d / %d params", self._pi_top_frac, k, flat_scores.numel())

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
                logits = model(inputs)["logits"]
                if self._cur_task == 0:
                    loss = F.cross_entropy(logits, targets)
                    eval_logits, eval_targets = logits, targets
                else:
                    fake_targets = targets - self._known_classes
                    loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets
                loss.backward()
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
