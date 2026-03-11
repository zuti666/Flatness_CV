"""SI_EWCon: combine Synaptic Intelligence (path-integral, first-order) with
Elastic Weight Consolidation (diagonal Fisher, second-order) for full-parameter training.

Penalty = c * SI (big_omega) + 0.5 * lambda * EWC (Fisher).

Args (all optional):
  si_c:           weight of SI surrogate (default 1.0)
  si_xi:          dampening for SI denominator (default 0.1)
  ewc_lambda:     weight of EWC penalty (default 20.0)
  ewc_gamma:      Fisher decay for online EWC (default 1.0)
  ewc_max_batches: batches to estimate Fisher (default 100)
  ewc_eps:        floor for Fisher (default 1e-5)
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

        # SI hyperparameters
        self._si_c = float(args.get("si_c", 1.0))
        self._si_xi = float(args.get("si_xi", 0.1))

        # EWC hyperparameters
        self._ewc_lambda = float(args.get("ewc_lambda", 20.0))
        self._ewc_gamma = float(args.get("ewc_gamma", 1.0))
        self._ewc_max_batches = int(args.get("ewc_max_batches", 100))
        self._ewc_eps = float(args.get("ewc_eps", 1e-5))

        self._optimizer_type = args.get("optimizer_type", "sgd").lower()
        if self._optimizer_type in {"ngd", "naturalgradient"}:
            self._ngd_damping = float(args.get("ngd_damping", 1e-3))
            self._ngd_ema_decay = float(args.get("ngd_ema_decay", 0.95))
            self._ngd_eps = float(args.get("ngd_eps", 1e-8))

        # buffers (CPU)
        self._big_omega: Dict[str, torch.Tensor] | None = None
        self._small_omega: Dict[str, torch.Tensor] = {}
        self._si_checkpoint: Dict[str, torch.Tensor] | None = None

        self._fisher: Dict[str, torch.Tensor] | None = None
        self._ewc_checkpoint: Dict[str, torch.Tensor] | None = None

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # Align stored buffers (omega, Fisher, checkpoints) to new shapes after classifier expansion
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
            logging.exception("[SI_EWCon] Failed to compute class means: %s", exc)

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

        # init SI buffers for this task
        self._si_checkpoint = {n: p.detach().cpu().clone() for n, p in self._iter_trainable(model)}
        self._small_omega = {n: torch.zeros_like(p, device="cpu") for n, p in self._iter_trainable(model)}
        if self._big_omega is None:
            self._big_omega = {n: torch.zeros_like(p, device="cpu") for n, p in self._iter_trainable(model)}

        # EWC checkpoint for this task (used during training) is last task params
        if self._ewc_checkpoint is None:
            self._ewc_checkpoint = {n: p.detach().cpu().clone() for n, p in self._iter_trainable(model)}

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

                pre_params = {n: p.detach().cpu().clone() for n, p in self._iter_trainable(model)}

                optimizer.zero_grad()
                outputs = model(inputs)
                logits = outputs["logits"]

                if stage == "init":
                    loss_task = F.cross_entropy(logits, targets)
                    eval_logits, eval_targets = logits, targets
                else:
                    fake_targets = targets - self._known_classes
                    loss_task = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets
                # task loss + EWC penalty (second-order) -> backprop to get task + EWC grads
                loss = loss_task + self._ewc_penalty(model)
                loss.backward()

                # capture task+EWC gradients BEFORE adding SI surrogate (pure path-integral requirement)
                grad_copy = {n: (p.grad.detach().cpu().clone() if p.grad is not None else None) for n, p in self._iter_trainable(model)}

                # add SI penalty grads (first-order surrogate) after copying grads
                if self._big_omega is not None and self._si_checkpoint is not None:
                    for name, p in self._iter_trainable(model):
                        if p.grad is None:
                            continue
                        ref = self._si_checkpoint.get(name)
                        omega = self._big_omega.get(name)
                        if ref is None or omega is None:
                            continue
                        p.grad.data.add_(self._si_c * 2.0 * omega.to(p.device, p.dtype) * (p - ref.to(p.device, p.dtype)))
                optimizer.step()

                with torch.no_grad():
                    for name, p in self._iter_trainable(model):
                        g = grad_copy.get(name)
                        if g is None:
                            continue
                        delta = pre_params[name] - p.detach().cpu()
                        self._small_omega[name].add_(g * delta)

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

        # end-of-task SI big_omega update
        assert self._big_omega is not None and self._si_checkpoint is not None
        for name, p in self._iter_trainable(model):
            ref = self._si_checkpoint.get(name)
            if ref is None:
                continue
            denom = (p.detach().cpu() - ref).pow(2) + self._si_xi
            self._big_omega[name].add_(self._small_omega[name] / denom)

        # refresh checkpoints for next task
        self._si_checkpoint = {n: p.detach().cpu().clone() for n, p in self._iter_trainable(model)}
        self._ewc_checkpoint = {n: p.detach().cpu().clone() for n, p in self._iter_trainable(model)}
        self._small_omega = {n: torch.zeros_like(p, device="cpu") for n, p in self._iter_trainable(model)}

        # update Fisher after task
        self._fisher = self._compute_fisher(train_loader, model)

    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    def _ewc_penalty(self, model: nn.Module):
        if self._fisher is None or self._ewc_checkpoint is None:
            return torch.tensor(0.0, device=self._device)
        pen = None
        for name, p in self._iter_trainable(model):
            ref = self._ewc_checkpoint.get(name)
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
                logp = F.log_softmax(logits, dim=1)
                nll = -F.nll_loss(logp, targets, reduction="none")
                exp_cp = torch.mean(torch.exp(nll.detach()))
                loss = nll.mean()
                loss.backward()

            for name, p in self._iter_trainable(model):
                if p.grad is None:
                    continue
                g2 = (p.grad.detach().cpu() ** 2) * exp_cp.cpu()
                fisher[name] = g2 if name not in fisher else fisher[name] + g2
            total += 1

        if total > 0:
            for name in fisher:
                fisher[name] = (fisher[name] / float(total)).clamp_min(self._ewc_eps)

        if self._fisher is not None:
            for name, new_f in fisher.items():
                old = self._fisher.get(name)
                self._fisher[name] = new_f if old is None else self._ewc_gamma * old + new_f
            for name, old in self._fisher.items():
                if name not in fisher:
                    self._fisher[name] = self._ewc_gamma * old
        else:
            self._fisher = fisher

        logger.info("[SI_EWCon] Fisher estimated over %d batches", total)
        return self._fisher

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
        # resize big_omega
        if self._big_omega is not None:
            for name, p in model.named_parameters():
                if name in self._big_omega:
                    self._big_omega[name] = self._match_tensor(self._big_omega[name], p)
        # resize Fisher and EWC checkpoint
        if self._fisher is not None:
            for name, p in model.named_parameters():
                if name in self._fisher:
                    self._fisher[name] = self._match_tensor(self._fisher[name], p)
        if self._ewc_checkpoint is not None:
            for name, p in model.named_parameters():
                if name in self._ewc_checkpoint:
                    self._ewc_checkpoint[name] = self._match_tensor(self._ewc_checkpoint[name], p)
