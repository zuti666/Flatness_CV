"""EWCon + weight-normalised Gaussian RWP perturbation (ablation baseline).

Core idea
---------
EWCon provides **soft protection** of old-task important weights via diagonal
Fisher penalty applied to the current-task loss:

    L = L_task + (lambda/2) * sum_i  F_i * (theta_i - theta_i*)^2

On top of this we add **flatness-seeking exploration** via RWP:
  1. Sample weight-normalised Gaussian perturbation  delta
  2. Compute gradient of (L_task + L_ewc) at  theta + delta
  3. Restore to theta and apply that gradient with the base optimiser.

This is the Gaussian-noise ablation for EWCon_RWP_Fisher.
Its role in the paper is to isolate the contribution of Fisher structure in
the noise distribution (Gaussian here vs diagonal-Fisher-inverse-sqrt there).

Args (all optional)
-------------------
ewc_lambda       : float, EWC penalty weight                (default 20.0)
ewc_gamma        : float, Fisher online decay               (default 1.0)
ewc_max_batches  : int,   Fisher estimation cap             (default 100)
ewc_eps          : float, Fisher numerical floor            (default 1e-5)
rwp_std          : float, perturbation scale                (default 7e-3)
rwp_start_task   : int,   first task to apply noise (0 = always)  (default 0)
optimizer_type   : str,   "sgd" only supported              (default "sgd")
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

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

        # EWC
        self._ewc_lambda = float(args.get("ewc_lambda", 20.0))
        self._ewc_gamma  = float(args.get("ewc_gamma", 1.0))
        self._ewc_max_batches = int(args.get("ewc_max_batches", 100))
        self._ewc_eps    = float(args.get("ewc_eps", 1e-5))

        # RWP Gaussian
        self._rwp_std        = float(args.get("rwp_std", 7e-3))
        self._rwp_start_task = int(args.get("rwp_start_task", 0))

        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # Buffers (CPU)
        self._fisher:     Dict[str, torch.Tensor] | None = None
        self._checkpoint: Dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

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
        except Exception as exc:
            logging.exception("[EWCon_RWP_GAUSS] Failed to compute class means: %s", exc)

    # ------------------------------------------------------------------
    def _train(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)

        stage = "init" if self._cur_task == 0 else "update"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage=stage)

        epochs = int(self.args.get("init_epoch" if stage == "init" else "epochs", 1))
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("init_milestones" if stage == "init" else "milestones", []),
            gamma=float(self.args.get("init_lrate_decay" if stage == "init" else "lrate_decay", 1.0)),
            T_max=epochs,
            eta_min=self.args.get("min_lr", 0.0),
        )

        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            model.train()
            total_loss, correct, total = 0.0, 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                optimizer.zero_grad()

                # ── RWP: perturb → forward+backward at θ+δ → restore ──
                perturbations = self._apply_gaussian_rwp(params)
                try:
                    outputs = model(inputs)
                    logits  = outputs["logits"]
                    if stage == "init":
                        loss = F.cross_entropy(logits, targets)
                        eval_logits, eval_targets = logits, targets
                    else:
                        fake_targets = targets - self._known_classes
                        loss = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                        eval_logits, eval_targets = logits[:, self._known_classes:], fake_targets
                    # EWC penalty evaluated at perturbed θ+δ
                    loss = loss + self._ewc_penalty(model)
                    loss.backward()
                finally:
                    self._restore_perturbation(perturbations)
                # ──────────────────────────────────────────────────────

                optimizer.step()

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total   += len(eval_targets)
                    total_loss += float(loss.detach().item())

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => "
                f"Loss {total_loss/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            )
            if epoch % 5 == 4:
                test_acc = self._compute_accuracy(model, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            prog_bar.set_description(info)
        logging.info(info)

        # After-task: save Fisher + checkpoint
        self._checkpoint = {
            name: p.detach().cpu().clone()
            for name, p in self._iter_trainable(model)
        }
        self._fisher = self._compute_fisher(train_loader, model)

    # ------------------------------------------------------------------
    # Perturbation helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_gaussian_rwp(
        self, params: List[torch.nn.Parameter]
    ) -> List[Tuple[torch.nn.Parameter, torch.Tensor]]:
        """Sample weight-normalised Gaussian delta and add to each param."""
        if self._cur_task < self._rwp_start_task or self._rwp_std <= 0.0:
            return []
        applied: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        seen: set = set()
        for p in params:
            if p is None or not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            delta = self._weight_normalised_noise(p, self._rwp_std)
            p.add_(delta)
            applied.append((p, delta))
        return applied

    @torch.no_grad()
    def _weight_normalised_noise(
        self, param: torch.nn.Parameter, std: float
    ) -> torch.Tensor:
        """delta_i ~ N(0, std^2 * ||row_i||^2)  (weight-normalised Gaussian)."""
        if param.dim() > 1:
            shape = tuple(param.shape)
            row_norms = param.detach().reshape(shape[0], -1).norm(dim=1, keepdim=True)
            row_norms = row_norms.reshape(shape[0], *([1] * (param.dim() - 1)))
            return torch.randn_like(param) * (std * row_norms)
        scale = std * (param.detach().norm().item() + 1e-16)
        return torch.randn_like(param) * scale

    @torch.no_grad()
    def _restore_perturbation(
        self, perturbations: List[Tuple[torch.nn.Parameter, torch.Tensor]]
    ) -> None:
        for p, delta in perturbations:
            p.sub_(delta)

    # ------------------------------------------------------------------
    # EWC helpers  (identical to EWCon.py)
    # ------------------------------------------------------------------
    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    def _ewc_penalty(self, model: nn.Module) -> torch.Tensor:
        if self._fisher is None or self._checkpoint is None:
            return torch.tensor(0.0, device=self._device)
        pen = None
        for name, p in self._iter_trainable(model):
            ref    = self._checkpoint.get(name)
            fisher = self._fisher.get(name)
            if ref is None or fisher is None:
                continue
            diff  = p - ref.to(device=p.device, dtype=p.dtype)
            f_dev = fisher.to(device=p.device, dtype=p.dtype)
            term  = (f_dev * diff.pow(2)).sum()
            pen   = term if pen is None else pen + term
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

    def _align_buffers_to_model(self, model: nn.Module) -> None:
        if self._fisher is not None:
            for name, p in model.named_parameters():
                if name in self._fisher:
                    self._fisher[name] = self._match_tensor(self._fisher[name], p)
        if self._checkpoint is not None:
            for name, p in model.named_parameters():
                if name in self._checkpoint:
                    self._checkpoint[name] = self._match_tensor(self._checkpoint[name], p)

    def _compute_fisher(
        self, loader, model: nn.Module
    ) -> Dict[str, torch.Tensor]:
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
                logp   = F.log_softmax(logits, dim=1)
                nll    = -F.nll_loss(logp, targets, reduction="none")
                exp_cp = torch.mean(torch.exp(nll.detach()))
                nll.mean().backward()

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
                self._fisher[name] = (
                    new_f if old is None else self._ewc_gamma * old + new_f
                )
            for name, old in self._fisher.items():
                if name not in fisher:
                    self._fisher[name] = self._ewc_gamma * old
        else:
            self._fisher = fisher

        logger.info("[EWCon_RWP_GAUSS] Fisher estimated over %d batches", total)
        return self._fisher

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
