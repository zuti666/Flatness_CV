"""SI_EWC_slowfast: full-parameter SI + EWC with slow/fast interaction.

Fast layer:
  - Standard SI (path integral) + EWC (diag Fisher).
  - Additional constrictors on *current-task drift*:
      * SO penalty: Fisher_old diag on delta (θ - θ_checkpoint).
      * FO penalty: gradient projection energy onto slow dictionary atoms.

Slow layer:
  - Maintains a small dictionary of normalized task deltas (atoms).
  - Slow merge selects atoms by utility (reconstruction of past deltas)
    minus risk (Fisher_old diag energy), with redundancy pruning.
  - Fisher_old is refreshed each task from data (diag grad^2).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

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


def _flatten(params: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in params]) if params else torch.tensor([], device=params[0].device if params else "cpu")


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

        # fast constriction
        self._lambda_so = float(args.get("slowfast_lambda_so", 0.0))
        self._lambda_fo = float(args.get("slowfast_lambda_fo", 0.0))

        # slow dictionary
        self._dict_size = int(args.get("slowfast_dict_size", 32))
        self._merge_sim_thresh = float(args.get("slowfast_merge_sim_thresh", 0.95))
        self._risk_lambda = float(args.get("slowfast_risk_lambda", 0.0))

        # buffers (CPU)
        self._big_omega: Dict[str, torch.Tensor] | None = None
        self._small_omega: Dict[str, torch.Tensor] = {}
        self._si_checkpoint: Dict[str, torch.Tensor] | None = None

        self._fisher: Dict[str, torch.Tensor] | None = None  # EWC current
        self._fisher_old_flat: Optional[torch.Tensor] = None  # diag Fisher for SO/slow (cpu flat)
        self._ewc_checkpoint: Dict[str, torch.Tensor] | None = None

        self._checkpoint_flat: Optional[torch.Tensor] = None
        self._dict: List[torch.Tensor] = []  # atoms (flat, cpu)
        self._task_deltas: List[torch.Tensor] = []

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

        # start-of-task checkpoint (flat)
        self._checkpoint_flat = _flatten([p.detach().cpu() for p in self._unwrap_network().parameters() if p.requires_grad])

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # slow-layer update: capture task delta and merge dict
        task_delta = _flatten([p.detach().cpu() for p in self._unwrap_network().parameters() if p.requires_grad]) - self._checkpoint_flat
        self._task_deltas.append(task_delta)
        self._slow_merge(task_delta)

        # update Fisher_old (flat diag) for next task
        _, fisher_task_flat = self._compute_fisher_flat(self.train_loader, self._unwrap_network())
        if fisher_task_flat is not None:
            if self._fisher_old_flat is None:
                self._fisher_old_flat = fisher_task_flat.cpu()
            else:
                self._fisher_old_flat = self._ewc_gamma * self._fisher_old_flat + fisher_task_flat.cpu()

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[SI_EWC_slowfast] Failed to compute class means: %s", exc)

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

                loss = loss_task + self._ewc_penalty(model)

                # fast constriction terms
                if self._lambda_so > 0 and self._fisher_old_flat is not None:
                    loss = loss + 0.5 * self._lambda_so * self._so_penalty(params)
                if self._lambda_fo > 0 and self._dict:
                    loss = loss + self._lambda_fo * self._fo_penalty(params)

                loss.backward()

                # add SI penalty grads (first-order surrogate)
                if self._big_omega is not None and self._si_checkpoint is not None:
                    for name, p in self._iter_trainable(model):
                        if p.grad is None:
                            continue
                        ref = self._si_checkpoint.get(name)
                        omega = self._big_omega.get(name)
                        if ref is None or omega is None:
                            continue
                        p.grad.data.add_(self._si_c * 2.0 * omega.to(p.device, p.dtype) * (p - ref.to(p.device, p.dtype)))

                grad_copy = {n: (p.grad.detach().cpu().clone() if p.grad is not None else None) for n, p in self._iter_trainable(model)}

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
                    total_loss += float(loss_task.detach().item())

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

        # update Fisher for EWC
        self._fisher = self._compute_fisher(train_loader, model)

    # ---------------- fast constrictors ---------------- #
    def _so_penalty(self, params: List[torch.Tensor]) -> torch.Tensor:
        flat = _flatten([p for p in params])
        ref = self._checkpoint_flat.to(self._device) if self._checkpoint_flat is not None else None
        if ref is None or self._fisher_old_flat is None:
            return torch.tensor(0.0, device=self._device)
        delta = flat - ref.to(self._device)
        fisher = self._fisher_old_flat.to(self._device)
        return torch.dot(fisher, delta.pow(2))

    def _fo_penalty(self, params: List[torch.Tensor]) -> torch.Tensor:
        flat_g = _flatten([p.grad for p in params if p.grad is not None])
        if flat_g.numel() == 0 or not self._dict:
            return torch.tensor(0.0, device=self._device)
        pen = 0.0
        for a in self._dict:
            a_d = a.to(flat_g.device)
            pen += torch.dot(flat_g, a_d) ** 2
        return pen

    # ---------------- slow merge ---------------- #
    def _slow_merge(self, task_delta: torch.Tensor):
        cand: List[torch.Tensor] = []
        cand.extend(self._dict)
        if task_delta.norm() > 0:
            cand.append(task_delta / task_delta.norm())
        if not cand:
            return

        def utility(atom: torch.Tensor) -> float:
            if not self._task_deltas:
                return 0.0
            acc = 0.0
            for d in self._task_deltas:
                acc += float(torch.dot(atom, d) ** 2)
            return acc

        def risk(atom: torch.Tensor) -> float:
            if self._fisher_old_flat is None:
                return 0.0
            f = self._fisher_old_flat.to(atom.device)
            return float((f * atom.pow(2)).sum())

        uniq: List[torch.Tensor] = []
        for a in cand:
            redundant = False
            for b in uniq:
                cos = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))
                if abs(cos) >= self._merge_sim_thresh:
                    redundant = True
                    break
            if not redundant:
                uniq.append(a)

        scored: List[Tuple[float, torch.Tensor]] = []
        for a in uniq:
            u = utility(a)
            r = risk(a)
            scored.append((u - self._risk_lambda * r, a))
        scored.sort(key=lambda x: x[0], reverse=True)

        self._dict = [a.detach().cpu() for _, a in scored[: self._dict_size]]

    # ---------------- EWC & SI helpers ---------------- #
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
                self._fisher[name] = new_f if old is None else self._ewc_gamma * old + new_f
            for name, old in list(self._fisher.items()):
                if name not in fisher:
                    self._fisher[name] = self._ewc_gamma * old
        else:
            self._fisher = fisher

        logger.info("[SI_EWC_slowfast] Fisher estimated over %d batches", total)
        return self._fisher

    @torch.no_grad()
    def _compute_fisher_flat(self, loader, model: nn.Module) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        fisher = None
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
            logits = model(inputs)["logits"]
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            flat_g = _flatten([p.grad for p in params]).detach().cpu()
            fisher = flat_g * flat_g if fisher is None else fisher + flat_g * flat_g
            total += 1
        fisher = fisher / float(total) if (fisher is not None and total > 0) else None
        return None, fisher

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
