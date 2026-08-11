"""FLAD-FD (finite-difference FLAD) in BaseLearner framework.

Core idea:
- Keep FLAD outer update form: g_final = g0 + gamma * g1.
- Replace all HVP-dependent parts with finite-difference proxies:
  1) Use w_plus = w + rho_probe * normalize(g_hat) to approximate grad-norm direction.
  2) Use w2 = w + delta1 and w3 = w2 + rho_sharp * normalize(g2) to approximate first-order sharpness term.

Unlike early draft versions, this implementation can run without a dedicated `flad_rho_fd`.
By default it reuses GAM-style radii (`gam_grad_norm_rho`, `gam_grad_rho`) as the two FD steps.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_CL.baseLearner import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


def _flatten(grads: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads]) if grads else torch.tensor([])


@torch.no_grad()
def _add_vector(params: List[torch.nn.Parameter], vec: torch.Tensor) -> None:
    idx = 0
    for p in params:
        n = p.numel()
        p.add_(vec[idx : idx + n].view_as(p))
        idx += n


@torch.no_grad()
def _set_flat_grad(params: List[torch.nn.Parameter], grad_vec: torch.Tensor) -> None:
    idx = 0
    for p in params:
        n = p.numel()
        g = grad_vec[idx : idx + n].view_as(p)
        if p.grad is None:
            p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)
        p.grad.copy_(g)
        idx += n


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = str(args.get("optimizer_type", "sgd")).lower()

        self._flad_rho = float(args.get("flad_rho", 0.5))
        self._flad_gamma = float(args.get("flad_gamma", 0.1))
        self._flad_lambda0 = float(args.get("flad_lambda0", 0.9))
        self._flad_lambda1 = float(args.get("flad_lambda1", 0.9))
        self._flad_sigma = float(args.get("flad_sigma", 0.3))
        self._flad_c = float(args.get("flad_c", 1e-6))

        # FD step sizes:
        # - If legacy `flad_rho_fd` is set, keep backward compatibility (same step for both places).
        # - Otherwise borrow GAM semantics: grad-norm FD step uses gam_grad_norm_rho,
        #   sharpness FD step uses gam_grad_rho.
        legacy_rho_fd = args.get("flad_rho_fd", None)
        if legacy_rho_fd is not None:
            fd_rho = float(legacy_rho_fd)
            self._fd_rho_probe = fd_rho
            self._fd_rho_sharp = fd_rho
        else:
            self._fd_rho_probe = float(
                args.get(
                    "flad_fd_rho_probe",
                    args.get("gam_grad_norm_rho", args.get("grad_norm_rho", self._flad_rho)),
                )
            )
            self._fd_rho_sharp = float(
                args.get(
                    "flad_fd_rho_sharp",
                    args.get("gam_grad_rho", args.get("grad_rho", self._flad_rho)),
                )
            )

        self.m: Optional[torch.Tensor] = None
        self.n: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
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

        # Match FLAD task-level behavior: reset EMA states every task.
        self.m, self.n = None, None
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[FLAD-FD] Failed to compute class means: %s", exc)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def _stage_loss(self, logits: torch.Tensor, targets: torch.Tensor, stage: str):
        if stage == "init":
            loss = F.cross_entropy(logits, targets)
            eval_logits, eval_targets = logits, targets
        else:
            fake_targets = targets - self._known_classes
            eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets
            loss = F.cross_entropy(eval_logits, eval_targets)
        return loss, eval_logits, eval_targets

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

                rho_probe = max(float(self._fd_rho_probe), 1e-12)
                rho_sharp = max(float(self._fd_rho_sharp), 1e-12)

                # 1) Gradient at w.
                logits_w = model(inputs)["logits"]
                loss_w, eval_logits, eval_targets = self._stage_loss(logits_w, targets, stage=stage)
                g_hat_list = autograd.grad(loss_w, params, retain_graph=False, create_graph=False)
                g_hat = _flatten(g_hat_list).detach()

                # 2) Update m and compute delta0.
                if self.m is None:
                    self.m = torch.zeros_like(g_hat, device=self._device)
                self.m = self._flad_lambda0 * self.m + (1 - self._flad_lambda0) * g_hat
                diff0 = g_hat - self._flad_sigma * self.m
                delta0 = self._flad_rho * diff0 / (diff0.norm() + self._flad_c)

                # 3) g0 = grad at w + delta0.
                with torch.no_grad():
                    _add_vector(params, delta0)
                logits_delta0 = model(inputs)["logits"]
                loss_delta0, _, _ = self._stage_loss(logits_delta0, targets, stage=stage)
                g0_list = autograd.grad(loss_delta0, params, retain_graph=False, create_graph=False)
                g0 = _flatten(g0_list).detach()
                with torch.no_grad():
                    _add_vector(params, -delta0)

                # 4) Finite-difference proxy of grad-norm direction at w.
                step_plus = rho_probe * g_hat / (g_hat.norm() + self._flad_c)
                with torch.no_grad():
                    _add_vector(params, step_plus)
                logits_plus = model(inputs)["logits"]
                loss_plus, _, _ = self._stage_loss(logits_plus, targets, stage=stage)
                g_plus_list = autograd.grad(loss_plus, params, retain_graph=False, create_graph=False)
                g_plus = _flatten(g_plus_list).detach()
                with torch.no_grad():
                    _add_vector(params, -step_plus)
                d_hat = (g_plus - g_hat) / rho_probe

                # 5) Update n and build delta1 via FD proxy.
                if self.n is None:
                    self.n = torch.zeros_like(d_hat, device=self._device)
                self.n = self._flad_lambda1 * self.n + (1 - self._flad_lambda1) * d_hat
                diff1 = d_hat - self._flad_sigma * self.n
                delta1 = self._flad_rho * diff1 / (diff1.norm() + self._flad_c)

                # 6) g2 at w2 = w + delta1.
                with torch.no_grad():
                    _add_vector(params, delta1)
                logits_w2 = model(inputs)["logits"]
                loss_w2, _, _ = self._stage_loss(logits_w2, targets, stage=stage)
                g2_list = autograd.grad(loss_w2, params, retain_graph=False, create_graph=False)
                g2 = _flatten(g2_list).detach()

                # 7) g3 at w3 = w2 + rho_sharp * normalize(g2); then g1_fd=(g3-g2)/rho_sharp.
                step3 = rho_sharp * g2 / (g2.norm() + self._flad_c)
                with torch.no_grad():
                    _add_vector(params, step3)
                logits_w3 = model(inputs)["logits"]
                loss_w3, _, _ = self._stage_loss(logits_w3, targets, stage=stage)
                g3_list = autograd.grad(loss_w3, params, retain_graph=False, create_graph=False)
                g3 = _flatten(g3_list).detach()
                g1_fd = (g3 - g2) / rho_sharp

                # Back to original weights before optimizer step.
                with torch.no_grad():
                    _add_vector(params, -step3)
                    _add_vector(params, -delta1)

                g_final = g0 + self._flad_gamma * g1_fd
                optimizer.zero_grad(set_to_none=True)
                _set_flat_grad(params, g_final)
                optimizer.step()

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss_w.detach().item())

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => "
                f"Loss {total_loss/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            )
            if (epoch % 5 == 4):
                test_acc = self._compute_accuracy(model, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            prog_bar.set_description(info)
        logging.info(info)

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
