"""OGD + isotropic Gaussian noise injection.

Design goals for controlled comparison:
- Keep OGD gradient projection exactly the same as `models_CL/OGD.py`.
- Add only isotropic Gaussian parameter noise after optimizer step.
- No EWC/Fisher term and no projected-noise basis.
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


def gram_schmidt(vectors: List[torch.Tensor], eps: float = 1e-10) -> List[torch.Tensor]:
    ortho: List[torch.Tensor] = []
    for v in vectors:
        w = v.clone()
        for u in ortho:
            w -= torch.dot(u, v) * u
        norm = w.norm()
        if norm > eps:
            ortho.append(w / norm)
    return ortho


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # OGD hyperparams
        self._ogd_max_dirs = int(args.get("ogd_max_dirs", 200))
        self._ogd_store_batches = int(args.get("ogd_store_batches", 50))
        self._ogd_eps = float(args.get("ogd_eps", 1e-10))

        # Isotropic Gaussian noise hyperparams
        self._noise_std = float(args.get("noise_std", 0.01))
        self._noise_start_task = int(args.get("noise_start_task", 1))
        self._noise_only_with_grad = self._parse_bool(args.get("noise_only_with_grad", True))

        # global basis: list of orthonormal vectors (CPU)
        self._directions: List[torch.Tensor] = []

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._align_directions_to_model(self._unwrap_network())
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

        # update OGD basis after finishing current task
        self._store_directions(self.train_loader, self._unwrap_network())

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[OGD_GAUSS] Failed to compute class means: %s", exc)

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
                outputs = model(inputs)
                logits = outputs["logits"]

                if stage == "init":
                    loss = F.cross_entropy(logits, targets)
                    eval_logits, eval_targets = logits, targets
                else:
                    fake_targets = targets - self._known_classes
                    loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets

                loss.backward()

                # OGD projection: full gradient to orthogonal complement.
                if self._directions:
                    flat_grad = self._flatten_current_grads(params)
                    flat_grad = self._project(flat_grad)
                    self._assign_flat_grad(params, flat_grad)

                optimizer.step()
                self._inject_gaussian_noise(model, optimizer)

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

    @staticmethod
    def _parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @torch.no_grad()
    def _inject_gaussian_noise(self, model: nn.Module, optimizer) -> None:
        # Keep task-0 comparable to plain OGD unless user changes noise_start_task.
        if self._cur_task < self._noise_start_task:
            return
        if self._noise_std <= 0.0:
            return

        params_with_grad = {id(p) for p in model.parameters() if p.grad is not None}
        sigma = float(self._noise_std)

        for group in optimizer.param_groups:
            lr = float(group.get("lr", 0.0))
            if lr <= 0.0:
                continue
            scale = (lr ** 0.5) * sigma
            for p in group.get("params", []):
                if p is None:
                    continue
                if self._noise_only_with_grad and id(p) not in params_with_grad:
                    continue
                p.add_(scale * torch.randn_like(p))

    @torch.no_grad()
    def _store_directions(self, loader, model: nn.Module):
        model.eval()
        collected: List[torch.Tensor] = []
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= self._ogd_store_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            model.zero_grad()
            with torch.enable_grad():
                logits = model(inputs)["logits"]
                idx = torch.arange(len(targets), device=targets.device)
                # Same as OGD: model gradients from correct-class logits.
                anchor = logits[idx, targets].sum()
                anchor.backward()
            flat = self._flatten_current_grads([p for p in model.parameters() if p.requires_grad]).detach().cpu()
            if flat.numel() > 0:
                collected.append(flat)

        if not collected:
            return

        new_dirs = gram_schmidt(collected, eps=self._ogd_eps)
        merged = gram_schmidt(self._directions + new_dirs, eps=self._ogd_eps)
        if len(merged) > self._ogd_max_dirs:
            merged = merged[: self._ogd_max_dirs]
        self._directions = merged
        logger.info("[OGD_GAUSS] Stored %d directions (cap %d)", len(self._directions), self._ogd_max_dirs)

    def _flatten_current_grads(self, params: List[torch.nn.Parameter]) -> torch.Tensor:
        flats = []
        for p in params:
            if p.grad is not None:
                flats.append(p.grad.view(-1))
        return torch.cat(flats) if flats else torch.tensor([], device=self._device)

    def _assign_flat_grad(self, params: List[torch.nn.Parameter], flat: torch.Tensor) -> None:
        offset = 0
        for p in params:
            if p.grad is None:
                continue
            numel = p.numel()
            p.grad.copy_(flat[offset : offset + numel].view_as(p))
            offset += numel

    def _project(self, g: torch.Tensor) -> torch.Tensor:
        if not self._directions or g.numel() == 0:
            return g
        g_proj = g.clone()
        for s in self._directions:
            if g_proj.numel() == s.numel():
                s_use = s
                if s_use.device != g_proj.device or s_use.dtype != g_proj.dtype:
                    s_use = s_use.to(device=g_proj.device, dtype=g_proj.dtype)
                dot = torch.dot(g_proj, s_use)
                g_proj = g_proj - dot * s_use
            else:
                n = min(g_proj.numel(), s.numel())
                if n == 0:
                    continue
                s_part = s[:n]
                if s_part.device != g_proj.device or s_part.dtype != g_proj.dtype:
                    s_part = s_part.to(device=g_proj.device, dtype=g_proj.dtype)
                dot = torch.dot(g_proj[:n], s_part)
                g_proj[:n] = g_proj[:n] - dot * s_part
        return g_proj

    def _align_directions_to_model(self, model: nn.Module) -> None:
        target_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._align_directions_to_dim(target_dim)

    def _align_directions_to_dim(self, target_dim: int) -> None:
        if not self._directions:
            return

        changed = False
        aligned: List[torch.Tensor] = []
        for v in self._directions:
            flat = v.detach().clone().reshape(-1).cpu()
            if flat.numel() == target_dim:
                aligned.append(flat)
                continue

            changed = True
            out = torch.zeros(target_dim, dtype=flat.dtype, device="cpu")
            n = min(target_dim, flat.numel())
            out[:n] = flat[:n]
            aligned.append(out)

        if not changed:
            return

        self._directions = gram_schmidt(aligned, eps=self._ogd_eps)
        if len(self._directions) > self._ogd_max_dirs:
            self._directions = self._directions[: self._ogd_max_dirs]
        logger.info("[OGD_GAUSS] Aligned %d directions to dim=%d", len(self._directions), target_dim)

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
