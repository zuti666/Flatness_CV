"""SI_EWC: combine Synaptic Intelligence (path-integral, first-order) with
Elastic Weight Consolidation (diagonal Fisher, second-order) for full-parameter training.

Penalty = c * SI (big_omega) + 0.5 * lambda * EWC (Fisher).

Args (all optional):
  si_c:           weight of SI surrogate (default 1.0)
  si_xi:          dampening for SI denominator (default 0.1)
  ewc_lambda:     weight of EWC penalty (default 20.0)
  ewc_max_batches: batches to estimate Fisher (default 100)
  fisher_batch_size: optional cap of samples for Fisher estimation
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

from models_CL.EWC_utils.fisher import DiagonalFisherEstimator, FisherEstimator
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

        # EWC buffers (CPU): accumulated Fisher and previous-task optima
        use_vmap = bool(args.get("ewc_fisher_use_vmap", False))
        self.fisher_estimator: FisherEstimator = DiagonalFisherEstimator(use_vmap=use_vmap)
        if "fisher_batch_size" in args:
            fisher_batch_size = args.get("fisher_batch_size")
        elif "ewc_fisher_batch_size" in args:
            fisher_batch_size = args.get("ewc_fisher_batch_size")
        else:
            fisher_batch_size = int(args.get("ewc_max_batches", 100)) * int(args.get("batch_size", 1))
        self._fisher_batch_size = None if fisher_batch_size is None else int(fisher_batch_size)
        self._fisher_dict: Dict[str, torch.Tensor] = {}
        self._optimal_params: Dict[str, torch.Tensor] = {}

    def after_task(self):
        model = self._unwrap_network()

        if hasattr(self, "train_loader") and self.train_loader is not None:
            logger.info("Computing Fisher information for SI_EWC after task %d...", self._cur_task)
            task_fisher = self._estimate_task_fisher(model, self.train_loader)
            for name, param in self._iter_trainable(model):
                f_task = task_fisher.get(name)
                if f_task is not None:
                    if name in self._fisher_dict:
                        self._fisher_dict[name] += f_task
                    else:
                        self._fisher_dict[name] = f_task.clone()
                self._optimal_params[name] = param.detach().cpu().clone()
            logger.info("SI_EWC: Fisher information computed and accumulated for task %d", self._cur_task)

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
            logging.exception("[SI_EWC] Failed to compute class means: %s", exc)

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
        self._small_omega = {n: torch.zeros_like(p, device="cpu") for n, p in self._iter_trainable(model)}

    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    def _ewc_penalty(self, model: nn.Module):
        if not self._fisher_dict or not self._optimal_params:
            return torch.tensor(0.0, device=self._device)
        pen = None
        for name, p in self._iter_trainable(model):
            ref = self._optimal_params.get(name)
            fisher = self._fisher_dict.get(name)
            if ref is None or fisher is None:
                continue
            diff = p - ref.to(device=p.device, dtype=p.dtype)
            fisher_d = fisher.to(device=p.device, dtype=p.dtype)
            term = (fisher_d * diff.pow(2)).sum()
            pen = term if pen is None else pen + term
        if pen is None:
            return torch.tensor(0.0, device=self._device)
        return 0.5 * self._ewc_lambda * pen

    def _estimate_task_fisher(self, model: nn.Module, loader: DataLoader) -> Dict[str, torch.Tensor]:
        criterion = nn.CrossEntropyLoss()
        fisher_flat = self.fisher_estimator.estimate(
            model=model,
            dataloader=loader,
            criterion=criterion,
            device=self._device,
            batch_size=self._fisher_batch_size,
        )
        fisher_flat = fisher_flat.detach().cpu()

        task_fisher: Dict[str, torch.Tensor] = {}
        idx = 0
        total_numel = fisher_flat.numel()
        for name, param in model.named_parameters():
            numel = param.numel()
            if idx + numel > total_numel:
                raise RuntimeError(
                    f"Fisher flat vector too short when parsing '{name}': "
                    f"need {idx + numel}, got {total_numel}"
                )
            fisher_block = fisher_flat[idx : idx + numel].view_as(param).clone()
            if param.requires_grad:
                task_fisher[name] = fisher_block
            idx += numel

        if idx != total_numel:
            logger.warning(
                "[SI_EWC] Fisher flat vector has trailing values: parsed=%d, total=%d", idx, total_numel
            )
        return task_fisher

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
        # resize accumulated EWC Fisher and stored EWC optima
        if self._fisher_dict:
            for name, p in model.named_parameters():
                if name in self._fisher_dict:
                    self._fisher_dict[name] = self._match_tensor(self._fisher_dict[name], p)
        if self._optimal_params:
            for name, p in model.named_parameters():
                if name in self._optimal_params:
                    self._optimal_params[name] = self._match_tensor(self._optimal_params[name], p)
