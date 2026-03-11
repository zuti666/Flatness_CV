"""Orthogonal Gradient Descent (collector-based variant).

Core logic preserved from the collector-style design:
- Directions are collected in `after_task()` via external collector.
- Gradient type is decided by collector (e.g., GTL / AVE), not hardcoded here.
- Training projects gradients with `memory.project_orthogonal(...)`.
- Task loss uses unified cross-entropy on full logits: CE(logits, y).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_CL.OGD_utils.gradients import (
    AVECollector,
    GTLCollector,
    GradientCollector,
    GradientMemory,
)
from models_CL.baseLearner import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class _LogitModel(nn.Module):
    """Adapter so collector always sees a tensor of logits."""

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
    """Iterate a loader and always yield (x, y), dropping optional sample index."""

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


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()
        self._ogd_eps = float(args.get("ogd_eps", 1e-10))
        self._collector_name = str(args.get("ogd_collector", "gtl")).lower()

        max_dirs = int(args.get("ogd_max_dirs", args.get("max_directions", 2000)))
        self.memory = GradientMemory(mode="orthonormal", max_directions=max_dirs)
        self.collector = self._build_collector(args)
        self._grads_per_task = int(args.get("grads_per_task", args.get("ogd_grads_per_task", 200)))

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def after_task(self):
        # Collector-based OGD: collect task directions AFTER task training.
        if hasattr(self, "train_loader") and self.train_loader is not None:
            logger.info("Collecting OGD directions from task %d...", self._cur_task)
            collect_loader = _XYLoader(self.train_loader)
            collect_model = _LogitModel(self._unwrap_network())
            if self._collector_name == "gtl":
                self._collect_gtl_batch(
                    self.memory,
                    collect_model,
                    collect_loader,
                    self._grads_per_task,
                    self._device,
                )
            elif self._collector_name == "ave":
                self._collect_ave_sample(
                    self.memory,
                    collect_model,
                    collect_loader,
                    self._grads_per_task,
                    self._device,
                )
            else:
                self.collector.collect(
                    self.memory,
                    collect_model,
                    collect_loader,
                    self._grads_per_task,
                    self._device,
                    multihead=False,
                    task_id=None,
                )
            logger.info("[OGD_collector] total directions in memory: %d", len(self.memory))

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
            logging.exception("[OGD_collector] Failed to compute class means: %s", exc)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
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

            raw_grad_norms = []
            proj_grad_norms = []
            proj_to_raw_ratios = []
            projection_relative_changes = []

            num_directions = len(self.memory)

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                optimizer.zero_grad()
                outputs = model(inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs

                if stage == "init":
                    loss = F.cross_entropy(logits, targets)
                    eval_logits, eval_targets = logits, targets
                else:
                    fake_targets = targets - self._known_classes
                    loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets
                loss.backward()

                g = self._get_grad_vector_requires_grad(model)
                raw_norm = g.norm().item()
                raw_grad_norms.append(raw_norm)

                if num_directions > 0:
                    g_tilde = self.memory.project_orthogonal(g)
                    proj_norm = g_tilde.norm().item()
                    proj_grad_norms.append(proj_norm)
                    proj_to_raw_ratios.append(proj_norm / (raw_norm + 1e-8))
                    diff_norm = (g - g_tilde).norm().item()
                    projection_relative_changes.append(diff_norm / (raw_norm + 1e-10))
                    self._set_grad_vector_requires_grad(model, g_tilde)
                else:
                    proj_grad_norms.append(raw_norm)
                    proj_to_raw_ratios.append(1.0)
                    projection_relative_changes.append(0.0)

                optimizer.step()

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss.detach().item())

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

            # lightweight metric log by epoch
            logger.info(
                (
                    "[OGD_collector][Task %d][Epoch %d] dirs=%d "
                    "raw=%.4e proj=%.4e ratio=%.4f rel_change=%.4f"
                ),
                self._cur_task,
                epoch + 1,
                num_directions,
                float(np.mean(raw_grad_norms)) if raw_grad_norms else 0.0,
                float(np.mean(proj_grad_norms)) if proj_grad_norms else 0.0,
                float(np.mean(proj_to_raw_ratios)) if proj_to_raw_ratios else 1.0,
                float(np.mean(projection_relative_changes)) if projection_relative_changes else 0.0,
            )

        logging.info(info)

    # ------------------------------------------------------------------ #
    @staticmethod
    @torch.no_grad()
    def _collect_gtl_batch(
        memory: GradientMemory,
        model: nn.Module,
        dataloader: DataLoader,
        num_directions: int,
        device: str,
    ) -> None:
        """Batch-GTL: each mini-batch contributes one direction via sum(gt_logit)."""
        model.eval()
        collected = 0
        iterator = tqdm(dataloader, desc="Collecting GTL gradients (batch)", leave=False)

        for x, y in iterator:
            if collected >= num_directions:
                break

            x = x.to(device)
            y = y.to(device)
            model.zero_grad()
            with torch.enable_grad():
                logits = model(x)
                idx = torch.arange(y.size(0), device=y.device)
                gt_sum = logits[idx, y].sum()
                gt_sum.backward()
            grad_vec = Learner._get_grad_vector_requires_grad(model).detach().cpu()
            memory.add(grad_vec)
            collected += 1

        logger.info(
            "[OGD_collector] Collected %d GTL-batch directions (target %d, total %d)",
            collected,
            num_directions,
            len(memory),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    @torch.no_grad()
    def _collect_ave_sample(
        memory: GradientMemory,
        model: nn.Module,
        dataloader: DataLoader,
        num_directions: int,
        device: str,
    ) -> None:
        """Sample-wise AVE collection with requires_grad-only vectors."""
        model.eval()
        collected = 0
        iterator = tqdm(dataloader, desc="Collecting AVE gradients", leave=False)

        for x, _y in iterator:
            if collected >= num_directions:
                break
            x = x.to(device)
            for i in range(x.size(0)):
                if collected >= num_directions:
                    break
                model.zero_grad()
                with torch.enable_grad():
                    logits = model(x[i : i + 1])
                    avg_logit = logits.mean()
                    avg_logit.backward()
                grad_vec = Learner._get_grad_vector_requires_grad(model).detach().cpu()
                memory.add(grad_vec)
                collected += 1

        logger.info(
            "[OGD_collector] Collected %d AVE directions (target %d, total %d)",
            collected,
            num_directions,
            len(memory),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_grad_vector_requires_grad(model: nn.Module) -> torch.Tensor:
        """Concatenate gradients only for trainable parameters."""
        grads = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                grads.append(torch.zeros_like(p.data).view(-1))
            else:
                grads.append(p.grad.view(-1))
        if grads:
            return torch.cat(grads)
        return torch.tensor([], device=next(model.parameters()).device)

    @staticmethod
    def _set_grad_vector_requires_grad(model: nn.Module, grad_vector: torch.Tensor) -> None:
        """Write a flat grad vector back only to trainable parameters."""
        idx = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            numel = p.data.numel()
            g = grad_vector[idx : idx + numel].view_as(p.data)
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
            p.grad.copy_(g)
            idx += numel

    # ------------------------------------------------------------------ #
    def _build_collector(self, args) -> GradientCollector:
        name = str(args.get("ogd_collector", "gtl")).lower()
        if name == "ave":
            return AVECollector()
        return GTLCollector()

    @staticmethod
    def _gram_schmidt(vectors, eps: float = 1e-10):
        ortho = []
        for v in vectors:
            w = v.clone()
            for u in ortho:
                w -= torch.dot(u, w) * u
            norm = w.norm()
            if norm > eps:
                ortho.append(w / norm)
        return ortho

    def _memory_dim_mismatch(self, target_dim: int) -> bool:
        return any(v.numel() != target_dim for v in self.memory.vectors)

    def _align_memory_to_model(self, model: nn.Module) -> None:
        target_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._align_memory_to_dim(target_dim)

    def _align_memory_to_dim(self, target_dim: int) -> None:
        if len(self.memory.vectors) == 0:
            return

        changed = False
        aligned = []
        for v in self.memory.vectors:
            flat = v.detach().clone().reshape(-1)
            if flat.numel() == target_dim:
                aligned.append(flat)
                continue

            changed = True
            out = torch.zeros(target_dim, dtype=flat.dtype, device=flat.device)
            n = min(target_dim, flat.numel())
            out[:n] = flat[:n]
            aligned.append(out)

        if not changed:
            return

        if self.memory.mode == "orthonormal":
            self.memory.vectors = self._gram_schmidt(aligned, eps=self._ogd_eps)
        else:
            self.memory.vectors = aligned

        if len(self.memory.vectors) > self.memory.max_directions:
            self.memory.vectors = self.memory.vectors[: self.memory.max_directions]

        logger.info(
            "[OGD_collector] aligned %d directions to dim=%d",
            len(self.memory.vectors),
            target_dim,
        )

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
