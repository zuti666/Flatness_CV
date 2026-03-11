"""FOPNG_slowfast: Full-parameter slow/fast split with risk-aware slow merge.

Fast layer  = FOPNG (projected natural gradient with diag Fisher) + constrictors:
    - FO penalty: discourage updates along slow dictionary atoms (old sensitive directions)
    - SO penalty: Fisher-diagonal trust-region style on current task drift

Slow layer  = dictionary of atoms built from task weight deltas, merged by
              utility - lambda_risk * Fisher_diag_risk, with redundancy pruning.

Notes:
 - This is full-parameter (no LoRA). Slow atoms live in flattened param space.
 - Fisher_diag is estimated on-the-fly (diag of grad^2) reusing task data.
 - Dictionary size is small (e.g., 32) to keep overhead low.
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


def _assign_flat(params: List[torch.Tensor], flat: torch.Tensor):
    offset = 0
    for p in params:
        numel = p.numel()
        p.grad = flat[offset : offset + numel].view_as(p).clone()
        offset += numel


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # FOPNG hyperparams
        self._fopng_store_batches = int(args.get("fopng_store_batches", 50))
        self._fopng_eta = float(args.get("fopng_eta", 1.0))
        self._fopng_eps = float(args.get("fopng_eps", 1e-8))
        self._fopng_old_decay = float(args.get("fopng_old_decay", 1.0))
        self._fopng_new_beta = float(args.get("fopng_new_beta", 0.9))

        # Slow-fast constrictor
        self._lambda_so = float(args.get("slowfast_lambda_so", 0.0))
        self._lambda_fo = float(args.get("slowfast_lambda_fo", 0.0))
        self._dict_size = int(args.get("slowfast_dict_size", 32))
        self._merge_sim_thresh = float(args.get("slowfast_merge_sim_thresh", 0.95))
        self._risk_lambda = float(args.get("slowfast_risk_lambda", 0.0))

        # stored stats
        self._old_grads: List[torch.Tensor] = []  # columns for FOPNG
        self._fisher_old: Optional[torch.Tensor] = None  # diag fisher (cpu)
        self._fisher_new: Optional[torch.Tensor] = None  # diag fisher (cpu)

        # slow dictionary of atoms (cpu) and per-task deltas (for utility)
        self._dict: List[torch.Tensor] = []  # each [D]
        self._task_deltas: List[torch.Tensor] = []

        self._checkpoint_flat: Optional[torch.Tensor] = None  # start-of-task params

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logger.info("Learning on %d-%d", self._known_classes, self._total_classes)

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

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # record start-of-task params
        self._checkpoint_flat = _flatten([p.detach().cpu() for p in self._unwrap_network().parameters() if p.requires_grad])

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # slow-layer: capture task delta and update dict + fisher_old
        task_delta = _flatten([p.detach().cpu() for p in self._unwrap_network().parameters() if p.requires_grad]) - self._checkpoint_flat
        self._task_deltas.append(task_delta)
        self._slow_merge(task_delta)

        # update old Fisher using current task data
        _, fisher_task = self._compute_task_stats(self.train_loader, self._unwrap_network())
        if fisher_task is not None:
            if self._fisher_old is None:
                self._fisher_old = fisher_task.cpu()
            else:
                self._fisher_old = self._fopng_old_decay * self._fisher_old + (1 - self._fopng_old_decay) * fisher_task.cpu()

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("[FOPNG_slowfast] Failed to compute class means: %s", exc)

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

                loss_task = F.cross_entropy(logits, targets) if stage == "init" else F.cross_entropy(
                    logits[:, self._known_classes :], targets - self._known_classes
                )

                # constrictors (fast penalties)
                loss = loss_task
                if self._lambda_so > 0 and self._fisher_old is not None:
                    loss = loss + 0.5 * self._lambda_so * self._so_penalty(params)
                if self._lambda_fo > 0 and self._dict:
                    loss = loss + self._lambda_fo * self._fo_penalty(params)

                loss.backward()

                # update F_new online (diag)
                self._update_fisher_new([p.grad for p in params])

                # FOPNG natural gradient + projection
                if self._old_grads and self._fisher_old is not None and self._fisher_new is not None:
                    self._apply_fopng(params)

                optimizer.step()

                with torch.no_grad():
                    eval_logits = logits if stage == "init" else logits[:, self._known_classes :]
                    eval_targets = targets if stage == "init" else targets - self._known_classes
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

    # ---------------- slow merge ---------------- #
    def _slow_merge(self, task_delta: torch.Tensor):
        # build candidate atoms: existing dict + new delta normalized
        cand: List[torch.Tensor] = []
        cand.extend(self._dict)
        if task_delta.norm() > 0:
            cand.append(task_delta / task_delta.norm())
        if not cand:
            return

        # utility: reconstruction energy over stored task deltas
        def utility(atom: torch.Tensor) -> float:
            if not self._task_deltas:
                return 0.0
            acc = 0.0
            for d in self._task_deltas:
                proj = torch.dot(atom, d) ** 2
                acc += float(proj)
            return acc

        # risk: Fisher-diag weighted energy
        def risk(atom: torch.Tensor) -> float:
            if self._fisher_old is None:
                return 0.0
            f = self._fisher_old.to(atom.device)
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
            score = u - self._risk_lambda * r
            scored.append((score, a))
        scored.sort(key=lambda x: x[0], reverse=True)

        self._dict = [a.detach().cpu() for _, a in scored[: self._dict_size]]

    # ---------------- fast penalties ---------------- #
    def _so_penalty(self, params: List[torch.Tensor]) -> torch.Tensor:
        flat = _flatten([p for p in params])
        ref = self._checkpoint_flat.to(self._device) if self._checkpoint_flat is not None else None
        if ref is None or self._fisher_old is None:
            return torch.tensor(0.0, device=self._device)
        delta = flat - ref.to(self._device)
        fisher = self._fisher_old.to(self._device)
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

    # ---------------- FOPNG core ---------------- #
    def _apply_fopng(self, params):
        flat_g = _flatten([p.grad for p in params]).to(self._device)
        if flat_g.numel() == 0:
            return
        f_old = self._fisher_old.to(self._device)
        f_new = self._fisher_new.to(self._device)
        G = torch.stack([g.to(self._device) for g in self._old_grads], dim=1) if self._old_grads else None
        if G is None or G.numel() == 0:
            return
        A = f_old.unsqueeze(1) * G
        temp = A / (f_new.unsqueeze(1) + self._fopng_eps)
        M = torch.matmul(G.t(), f_old.unsqueeze(1) * temp)
        M = M + torch.eye(M.size(0), device=self._device) * self._fopng_eps
        b = torch.matmul(G.t(), f_old * flat_g)
        alpha = torch.linalg.solve(M, b)
        Pg = flat_g - torch.matmul(A, alpha)
        g_nat = Pg / (f_new + self._fopng_eps)
        denom = torch.sqrt(torch.sum(Pg * g_nat) + self._fopng_eps)
        g_final = self._fopng_eta * g_nat / denom
        _assign_flat(params, g_final)

    def _update_fisher_new(self, grads):
        flat = _flatten([g for g in grads if g is not None]).detach().cpu()
        if flat.numel() == 0:
            return
        if self._fisher_new is None:
            self._fisher_new = flat.pow(2)
        else:
            beta = self._fopng_new_beta
            self._fisher_new = beta * self._fisher_new + (1 - beta) * flat.pow(2)

    @torch.no_grad()
    def _compute_task_stats(self, loader, model: nn.Module):
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        grads: List[torch.Tensor] = []
        fisher = None
        total = 0
        for b_idx, batch in enumerate(loader):
            if b_idx >= self._fopng_store_batches:
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
            if flat_g.numel() == 0:
                continue
            grads.append(flat_g)
            fisher = flat_g * flat_g if fisher is None else fisher + flat_g * flat_g
            total += 1
        fisher = fisher / float(total) if (fisher is not None and total > 0) else None
        return grads, fisher

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
