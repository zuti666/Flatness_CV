"""FOPNG: Fisher-Orthogonal Projected Natural Gradient.

Collector/Fisher-estimator-based implementation aligned with original method logic:
- memory mode is 'raw' (columns as stored directions)
- first task (or empty memory) uses regular SGD/Adam training
- otherwise: estimate F_new, use stored F_old and G to compute natural projected update
- after each task: update F_old and collect directions via collector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_CL.FOPNG_utils.fisher import DiagonalFisherEstimator, FisherEstimator
from models_CL.FOPNG_utils.gradients import (
    AVECollector,
    GradientCollector,
    GradientMemory,
    get_grad_vector,
)
from models_CL.baseLearner import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class _LogitModel(nn.Module):
    """Adapter for collectors: always return logits tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, task_id=None):
        if task_id is None:
            out = self.model(x)
        else:
            out = self.model(x, task_id=task_id)
        if isinstance(out, dict):
            return out["logits"]
        return out


@dataclass
class _XYLoader:
    """Adapt loader that may yield (idx,x,y) into (x,y)."""

    loader: DataLoader

    def __iter__(self):
        for batch in self.loader:
            if len(batch) == 3:
                _, x, y = batch
            else:
                x, y = batch
            yield x, y

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset


def apply_update(model: nn.Module, update: torch.Tensor):
    """In-place parameter update from flattened update vector (same order as model.parameters())."""
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            step = update[idx : idx + n].view_as(p)
            p.add_(step)
            idx += n


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # FOPNG core components
        use_vmap = bool(args.get("fopng_fisher_use_vmap", False))
        self.fisher_estimator: FisherEstimator = DiagonalFisherEstimator(use_vmap=use_vmap)
        self.collector: GradientCollector = AVECollector()
        self.memory = GradientMemory(mode="raw", max_directions=int(args.get("fopng_max_directions", 2000)))
        self.F_old: Optional[torch.Tensor] = None
        self.is_diagonal = isinstance(self.fisher_estimator, DiagonalFisherEstimator)

        # hyperparameters
        self.lambda_reg = float(args.get("fopng_lambda_reg", 1e-3))
        self._grads_per_task = int(args.get("grads_per_task", args.get("fopng_grads_per_task", 200)))
        self._fisher_batch_size = args.get("fisher_batch_size", args.get("fopng_fisher_batch_size", None))
        self._new_fisher_weight = float(args.get("fopng_new_fisher_weight", 0.5))

        self.A_inv: Optional[torch.Tensor] = None
        self.A: Optional[torch.Tensor] = None
        self.global_batch_idx = 0

    # ------------------------------------------------------------------ #
    def after_task(self):
        model = self._unwrap_network()
        criterion = nn.CrossEntropyLoss()

        # Update F_old with current task Fisher
        if hasattr(self, "train_loader") and self.train_loader is not None:
            fisher_loader = _XYLoader(self.train_loader)
            F_current = self.fisher_estimator.estimate(
                model=_LogitModel(model),
                dataloader=fisher_loader,
                criterion=criterion,
                device=self._device,
                batch_size=self._fisher_batch_size,
            ).detach().cpu()

            if self.F_old is None:
                self.F_old = F_current
            else:
                w = float(self._new_fisher_weight)
                self.F_old = (1.0 - w) * self.F_old + w * F_current

            # Collect gradient directions for projection memory
            logger.info("Collecting FOPNG directions from task %d...", self._cur_task)
            self.collector.collect(
                self.memory,
                _LogitModel(model),
                fisher_loader,
                self._grads_per_task,
                self._device,
                multihead=False,
                task_id=None,
            )
            logger.info("[FOPNG] collected directions total: %d", len(self.memory))

        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._align_state_to_model(self._unwrap_network())
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
            logging.exception("[FOPNG] Failed to compute class means: %s", exc)

    # ------------------------------------------------------------------ #
    # core FOPNG math
    # ------------------------------------------------------------------ #
    def _compute_update_prep(self, F_new: torch.Tensor, F_old: torch.Tensor, G: torch.Tensor, device):
        lam = self.lambda_reg
        if self.is_diagonal:
            F_new_inv_diag = 1.0 / (F_new + lam)
            F_old_diag = F_old.view(-1, 1)
            F_old_G = F_old_diag * G
            weighted_G = F_old_diag * (F_new_inv_diag.view(-1, 1) * F_old_G)
            A = G.T @ weighted_G + lam * torch.eye(G.size(1), device=device)
            self.A_inv = torch.pinverse(A)
            self.A = A
            return
        raise NotImplementedError("Precomputation for full Fisher not implemented.")

    def _compute_update(
        self,
        gradient: torch.Tensor,
        F_new: torch.Tensor,
        F_old: torch.Tensor,
        G: torch.Tensor,
        device,
        lr: float,
        return_intermediate: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        lam = self.lambda_reg
        stats: Dict[str, float] = {}

        if self.is_diagonal:
            F_old_sqrt = torch.sqrt(F_old + 1e-10)
            g_fisher = F_old_sqrt * gradient

            F_new_inv_diag = 1.0 / (F_new + lam)

            F_old_g = F_old * gradient
            G_T_F_old_g = G.T @ F_old_g
            A_inv_G_T_F_old_g = self.A_inv @ G_T_F_old_g
            correction = (G @ A_inv_G_T_F_old_g).view(-1) * F_old.squeeze()
            P_g = gradient - correction

            P_g_fisher = F_old_sqrt * P_g

            F_new_inv_P_g = P_g * F_new_inv_diag
            denom = torch.sqrt((P_g * F_new_inv_P_g).sum() + 1e-8)
            v_star = -lr * F_new_inv_P_g / (denom + 1e-8)

            if return_intermediate:
                raw_norm = gradient.norm().item()
                fisher_norm = g_fisher.norm().item()
                correction_norm = correction.norm().item()
                P_g_eucl_norm = P_g.norm().item()
                P_g_fisher_norm = P_g_fisher.norm().item()
                v_star_norm = v_star.norm().item()
                diff_eucl = (gradient - P_g).norm().item()
                diff_fisher = (g_fisher - P_g_fisher).norm().item()
                stats.update(
                    {
                        "raw_grad_norm": raw_norm,
                        "fisher_grad_norm": fisher_norm,
                        "correction_norm": correction_norm,
                        "projected_grad_eucl_norm": P_g_eucl_norm,
                        "projected_grad_fisher_norm": P_g_fisher_norm,
                        "update_norm": v_star_norm,
                        "projection_relative_change": diff_eucl / (raw_norm + 1e-10),
                        "fisher_projection_relative_change": diff_fisher / (fisher_norm + 1e-10),
                        "correction_to_raw_ratio": correction_norm / (raw_norm + 1e-10),
                        "update_to_raw_ratio": v_star_norm / (raw_norm + 1e-10),
                        "projected_to_raw_ratio_eucl": P_g_eucl_norm / (raw_norm + 1e-10),
                        "projected_to_raw_ratio_fisher": P_g_fisher_norm / (fisher_norm + 1e-10),
                    }
                )
        else:
            raise NotImplementedError("Full Fisher branch is not enabled in this framework variant.")

        if return_intermediate:
            return v_star, stats
        return v_star

    # ------------------------------------------------------------------ #
    def _train(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)

        stage = "init" if self._cur_task == 0 else "update"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage=stage)

        # Optional first-task optimizer override (same intention as original method API)
        if self._cur_task == 0:
            first_task_lr = self.args.get("first_task_lr", None)
            if first_task_lr is None:
                first_task_lr = optimizer.param_groups[0].get("lr", self.args.get("lrate", 0.1))
            use_adam = bool(self.args.get("use_adam", False))
            use_sgd = bool(self.args.get("use_sgd", False))
            if use_adam:
                optimizer = torch.optim.Adam(model.parameters(), lr=float(first_task_lr))
            elif use_sgd:
                optimizer = torch.optim.SGD(model.parameters(), lr=float(first_task_lr))

        epochs = int(self.args.get("init_epoch" if self._cur_task == 0 else "epochs", 1))
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("init_milestones" if self._cur_task == 0 else "milestones", []),
            gamma=float(self.args.get("init_lrate_decay" if self._cur_task == 0 else "lrate_decay", 1.0)),
            T_max=epochs,
            eta_min=self.args.get("min_lr", 0.0),
        )

        criterion = nn.CrossEntropyLoss()

        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            # If first task or no stored directions, use regular training this epoch.
            G = self.memory.get_matrix()
            use_regular = (self._cur_task == 0) or (G is None)

            if not use_regular:
                fisher_loader = _XYLoader(train_loader)
                F_new = self.fisher_estimator.estimate(
                    model=_LogitModel(model),
                    dataloader=fisher_loader,
                    criterion=criterion,
                    device=self._device,
                    batch_size=self._fisher_batch_size,
                ).detach().to(self._device)
                if self.F_old is None:
                    self.F_old = F_new.detach().cpu().clone()
                F_old = self.F_old.to(self._device)
                G = G.to(self._device)
                self._compute_update_prep(F_new, F_old, G, self._device)

                batch_stats = {
                    "raw_grad_norm": [],
                    "fisher_grad_norm": [],
                    "correction_norm": [],
                    "projected_grad_eucl_norm": [],
                    "projected_grad_fisher_norm": [],
                    "update_norm": [],
                    "projection_relative_change": [],
                    "fisher_projection_relative_change": [],
                    "correction_to_raw_ratio": [],
                    "update_to_raw_ratio": [],
                    "projected_to_raw_ratio_eucl": [],
                    "projected_to_raw_ratio_fisher": [],
                }
            else:
                F_new = None
                F_old = None
                batch_stats = {"raw_grad_norm": []}

            iterator = tqdm(train_loader, desc=None, leave=False)
            for batch in iterator:
                if len(batch) == 3:
                    _, x, y = batch
                else:
                    x, y = batch
                x = x.to(self._device)
                y = y.to(self._device)

                if use_regular:
                    optimizer.zero_grad()
                    output = model(x)["logits"]
                    loss = criterion(output, y)
                    loss.backward()

                    grad_vec = get_grad_vector(model)
                    batch_stats["raw_grad_norm"].append(grad_vec.norm().item())

                    optimizer.step()
                else:
                    output = model(x)["logits"]
                    loss = criterion(output, y)
                    model.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_vec = get_grad_vector(model)
                    lr_now = optimizer.param_groups[0].get("lr", self.args.get("lrate", 0.1))
                    update, stats = self._compute_update(
                        gradient=grad_vec,
                        F_new=F_new,
                        F_old=F_old,
                        G=G,
                        device=self._device,
                        lr=float(lr_now),
                        return_intermediate=True,
                    )
                    apply_update(model, update)
                    for key, values in batch_stats.items():
                        if key in stats:
                            values.append(stats[key])

                # monotonic batch index log (lightweight)
                logger.info(
                    "[FOPNG][batch] global_batch_idx=%d task=%d loss=%.6f",
                    self.global_batch_idx,
                    self._cur_task,
                    float(loss.item()),
                )
                self.global_batch_idx += 1

                total_loss += loss.item() * x.size(0)
                preds = output.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += x.size(0)

            scheduler.step()
            train_acc = total_correct / max(1, total_samples)
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => "
                f"Loss {total_loss/max(1,total_samples):.3f}, Train_accy {100.0*train_acc:.2f}"
            )
            if (epoch % 5 == 4):
                test_acc = self._compute_accuracy(model, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            prog_bar.set_description(info)

            # epoch metrics (means/std)
            metrics = {"task_id": self._cur_task, "epoch": epoch + 1, "num_directions": 0 if G is None else int(G.size(1))}
            for k, vals in batch_stats.items():
                if len(vals) > 0:
                    metrics[f"{k}_mean"] = float(np.mean(vals))
                    metrics[f"{k}_std"] = float(np.std(vals))
            logger.info("[FOPNG][epoch_metrics] %s", metrics)

        logging.info(info)

    # ------------------------------------------------------------------ #
    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network

    def _align_state_to_model(self, model: nn.Module):
        """Align stored Fisher and memory vectors to current parameter dimensionality."""
        target_dim = sum(p.numel() for p in model.parameters())

        def _match(v: torch.Tensor) -> torch.Tensor:
            if v.numel() == target_dim:
                return v
            out = torch.zeros(target_dim, dtype=v.dtype, device=v.device)
            n = min(target_dim, v.numel())
            out[:n] = v[:n]
            return out

        if self.F_old is not None:
            self.F_old = _match(self.F_old.cpu())

        if len(self.memory.vectors) > 0:
            self.memory.vectors = [_match(v.cpu()) for v in self.memory.vectors]
