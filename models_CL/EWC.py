"""Elastic Weight Consolidation (EWC) baseline (full-parameter variant).

Implements original EWC with diagonal Fisher approximation:
    L = L_t + (lambda/2) * sum_i F_i * (theta_i - theta_i*)^2

Unlike online EWC, Fisher is accumulated by plain summation across tasks:
    F <- F + F_task

Key args (all optional, defaults chosen to be safe):
    ewc_lambda        : float, weight of the quadratic penalty (default 20.0)
    ewc_max_batches   : int, cap batches used to estimate Fisher (default 100)
    ewc_eps           : float, numerical floor for Fisher (default 1e-5)

Only trainable parameters are protected. Fisher and checkpoints are
stored on CPU to minimize GPU memory pressure.
"""

from __future__ import annotations

import logging
from typing import Dict
from types import SimpleNamespace

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

        # EWC hyperparameters
        self._ewc_lambda = float(args.get("ewc_lambda", 20.0))
        self._ewc_max_batches = int(args.get("ewc_max_batches", 100))
        self._ewc_eps = float(args.get("ewc_eps", 1e-5))

        self._optimizer_type = args.get("optimizer_type", "sgd").lower()
        if self._optimizer_type == "sam":
            self._sam_rho = float(args.get("sam_rho", 0.05))
            self._sam_adaptive = bool(args.get("sam_adaptive", False))
        elif self._optimizer_type == "cflat":
            self._cflat_rho = float(args.get("cflat_rho", 0.2))
            self._cflat_lambda = float(args.get("cflat_lambda", 0.2))
            self._cflat_adaptive = bool(args.get("cflat_adaptive", False))
            self._cflat_perturb_eps = float(args.get("cflat_perturb_eps", 1e-12))
            self._cflat_grad_reduce = args.get("cflat_grad_reduce", "mean")
        elif self._optimizer_type == "gam":
            self._gam_adaptive = bool(args.get("gam_adaptive", False))
            self._gam_grad_reduce = args.get("gam_grad_reduce", "mean")
            self._gam_perturb_eps = float(args.get("gam_perturb_eps", 1e-12))
            self._gam_grad_rho = float(args.get("gam_grad_rho", args.get("grad_rho", 0.2)))
            self._gam_grad_norm_rho = float(
                args.get("gam_grad_norm_rho", args.get("grad_norm_rho", 0.2))
            )
            self._gam_beta1 = float(args.get("gam_grad_beta_1", args.get("grad_beta_1", 1.0)))
            self._gam_beta2 = float(args.get("gam_grad_beta_2", args.get("grad_beta_2", 1.0)))
            self._gam_beta3 = float(args.get("gam_grad_beta_3", args.get("grad_beta_3", 1.0)))
            self._gam_gamma = float(args.get("gam_grad_gamma", args.get("grad_gamma", 0.1)))
            self._gam_args = SimpleNamespace(
                grad_beta_1=self._gam_beta1,
                grad_beta_2=self._gam_beta2,
                grad_beta_3=self._gam_beta3,
                grad_gamma=self._gam_gamma,
                grad_rho=self._gam_grad_rho,
                grad_norm_rho=self._gam_grad_norm_rho,
                adaptive=self._gam_adaptive,
                perturb_eps=self._gam_perturb_eps,
                grad_reduce=str(self._gam_grad_reduce),
            )
        elif self._optimizer_type in {"ngd", "naturalgradient"}:
            self._ngd_damping = float(args.get("ngd_damping", 1e-3))
            self._ngd_ema_decay = float(args.get("ngd_ema_decay", 0.95))
            self._ngd_eps = float(args.get("ngd_eps", 1e-8))

        # Fisher estimator backend (from models_CL/EWC_utils/fisher.py)
        use_vmap = bool(args.get("ewc_fisher_use_vmap", False))
        self.fisher_estimator: FisherEstimator = DiagonalFisherEstimator(use_vmap=use_vmap)
        if "fisher_batch_size" in args:
            fisher_batch_size = args.get("fisher_batch_size")
        elif "ewc_fisher_batch_size" in args:
            fisher_batch_size = args.get("ewc_fisher_batch_size")
        else:
            # Backward-compatible fallback: roughly cap Fisher data usage.
            fisher_batch_size = int(args.get("ewc_max_batches", 100)) * int(args.get("batch_size", 1))
        self._fisher_batch_size = None if fisher_batch_size is None else int(fisher_batch_size)

        # Buffers (CPU): accumulated Fisher and task-optimal params
        self._fisher_dict: Dict[str, torch.Tensor] = {}
        self._optimal_params: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def after_task(self):
        model = self._unwrap_network()

        if hasattr(self, "train_loader") and self.train_loader is not None:
            logger.info("Computing Fisher information for EWC after task %d...", self._cur_task)

            task_fisher = self._estimate_task_fisher(model, self.train_loader)
            for name, param in self._iter_trainable(model):
                f_task = task_fisher.get(name)
                if f_task is not None:
                    if name in self._fisher_dict:
                        self._fisher_dict[name] += f_task
                    else:
                        self._fisher_dict[name] = f_task.clone()
                self._optimal_params[name] = param.detach().cpu().clone()

            logger.info("EWC: Fisher information computed and accumulated for task %d", self._cur_task)

        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # If classifier expanded, align stored Fisher/checkpoints to new shapes
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
            logging.exception("[EWC] Failed to compute class means: %s", exc)

    # ------------------------------------------------------------------
    def _train(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)

        stage = "init" if self._cur_task == 0 else "update"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage=stage)

        if stage == "init":
            epochs = int(self.args.get("init_epoch", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("init_milestones", []),
                gamma=float(self.args.get("init_lrate_decay", 1.0)),
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

                if self._optimizer_type == "cflat":
                    def closure():
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        logits = outputs["logits"]
                        loss_task = F.cross_entropy(logits, targets)
                        loss = loss_task + self._ewc_penalty(model)
                        loss.backward()
                        return outputs, [loss]

                    _, loss_list = optimizer.step(closure=closure)
                    loss_value = torch.stack([loss_term.detach() for loss_term in loss_list]).sum()
                    with torch.no_grad():
                        logits = model(inputs)["logits"]
                    eval_logits = logits
                    eval_targets = targets
                    total_loss += float(loss_value.item())
                elif self._optimizer_type == "gam":
                    def closure():
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        logits = outputs["logits"]
                        loss_task = F.cross_entropy(logits, targets)
                        loss = loss_task + self._ewc_penalty(model)
                        loss_value = loss.detach()
                        loss.backward()
                        return outputs, loss_value

                    outputs, loss_value = optimizer.step(closure=closure)
                    logits = outputs["logits"].detach()
                    eval_logits = logits
                    eval_targets = targets
                    total_loss += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                else:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    logits = outputs["logits"]
                    loss_task = F.cross_entropy(logits, targets)
                    loss = loss_task + self._ewc_penalty(model)

                    if self._optimizer_type == "sam":
                        loss.backward()
                        optimizer.first_step(zero_grad=True)

                        outputs = model(inputs)
                        logits = outputs["logits"]
                        second_task = F.cross_entropy(logits, targets)
                        second_loss = second_task + self._ewc_penalty(model)
                        second_loss.backward()
                        optimizer.second_step(zero_grad=True)
                        loss_value = second_loss.detach()
                    else:
                        loss.backward()
                        optimizer.step()
                        loss_value = loss.detach()

                    eval_logits = logits
                    eval_targets = targets
                    total_loss += float(loss_value.item())

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => Loss {total_loss/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            if (epoch % 5 == 4):
                test_acc = self._compute_accuracy(model, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            prog_bar.set_description(info)
        logging.info(info)

    # ------------------------------------------------------------------
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

    def _match_tensor(self, stored: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if stored.shape == param.shape:
            return stored
        new = torch.zeros_like(param.detach().cpu())
        slices = tuple(slice(0, min(a, b)) for a, b in zip(stored.shape, param.shape))
        new[slices] = stored[slices]
        return new

    def _align_buffers_to_model(self, model: nn.Module):
        """Ensure Fisher/checkpoint tensors match current param shapes (e.g., after fc expansion)."""
        if self._fisher_dict:
            for name, p in model.named_parameters():
                if name in self._fisher_dict:
                    self._fisher_dict[name] = self._match_tensor(self._fisher_dict[name], p)
        if self._optimal_params:
            for name, p in model.named_parameters():
                if name in self._optimal_params:
                    self._optimal_params[name] = self._match_tensor(self._optimal_params[name], p)

    def _estimate_task_fisher(self, model: nn.Module, loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Estimate diagonal Fisher for current task via pluggable estimator."""
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
                "[EWC] Fisher flat vector has trailing values: parsed=%d, total=%d", idx, total_numel
            )
        return task_fisher

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
