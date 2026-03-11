"""SAM_OGD: SAM optimizer with OGD-style historical direction filtering.

OGD part is aligned with models_CL/OGD.py:
- Maintain a global orthonormal direction basis (flattened full gradient).
- Collect directions from correct-class logit gradients after each task.
- Align historical directions when trainable-parameter dimension changes.
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
from optimer_PerturabtionType.optimer_sam_filiter import (
    SAM_OGD as SAM_OGD_Optimizer,
    disable_running_stats,
    enable_running_stats,
)
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
        self._optimizer_type = str(args.get("optimizer_type", "sgd")).lower()

        self._ogd_max_dirs = int(
            args.get("sam_ogd_max_directions", args.get("ogd_max_dirs", args.get("max_directions", 2000)))
        )
        self._ogd_store_batches = int(
            args.get(
                "sam_ogd_store_batches",
                args.get("ogd_store_batches", args.get("grads_per_task", args.get("sam_ogd_grads_per_task", 50))),
            )
        )
        self._ogd_eps = float(args.get("ogd_eps", 1e-10))
        self._directions: List[torch.Tensor] = []

        self._sam_rho = float(args.get("sam_rho", 0.05))
        self._sam_adaptive = bool(args.get("sam_adaptive", False))
        self._sam_disable_bn_stats = bool(args.get("sam_disable_bn_stats", True))

        self._first_order_filter_strength = float(
            args.get("first_order_filter_strength", args.get("sam_first_order_filter_strength", 1.0))
        )
        self._second_order_filter_strength = float(
            args.get("second_order_filter_strength", args.get("sam_second_order_filter_strength", 1.0))
        )
        self._perturb_filter_strength = float(
            args.get(
                "sam_ogd_perturb_filter_strength",
                args.get(
                    "sam_ogd_perturabtion_filter_strength",
                    args.get("perturb_filter_strength", self._first_order_filter_strength),
                ),
            )
        )
        self._result_filter_strength = float(
            args.get(
                "sam_ogd_result_filter_strength",
                args.get("result_filter_strength", self._first_order_filter_strength),
            )
        )
        self._projection_eps = float(args.get("projection_eps", args.get("sam_projection_eps", 1e-12)))

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

        # Align with OGD.py: collect first-order directions after current task training.
        self._store_directions(self.train_loader, self._unwrap_network())

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[SAM_OGD] Failed to compute class means: %s", exc)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def _train(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)

        stage = "init" if self._cur_task == 0 else "update"
        params = [p for p in model.parameters() if p.requires_grad]
        base_optimizer = self._build_base_optimizer(params, stage=stage)

        sam_optimizer = SAM_OGD_Optimizer(
            params,
            base_optimizer=base_optimizer,
            rho=self._sam_rho,
            adaptive=self._sam_adaptive,
            first_order_filter_strength=self._first_order_filter_strength,
            second_order_filter_strength=self._second_order_filter_strength,
            perturb_filter_strength=self._perturb_filter_strength,
            result_filter_strength=self._result_filter_strength,
            projection_eps=self._projection_eps,
        )

        epochs = int(self.args.get("init_epoch" if self._cur_task == 0 else "epochs", 1))
        scheduler = self.build_scheduler(
            sam_optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("init_milestones" if self._cur_task == 0 else "milestones", []),
            gamma=float(self.args.get("init_lrate_decay" if self._cur_task == 0 else "lrate_decay", 1.0)),
            T_max=epochs,
            eta_min=self.args.get("min_lr", 0.0),
        )

        history_information = self._first_order_information()
        if history_information is not None:
            sam_optimizer.set_history_information(
                first_order_information=history_information,
                second_order_information=history_information,
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

                cache = {}

                def closure():
                    sam_optimizer.zero_grad()
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

                    # Keep first forward/backward result for metrics.
                    if "loss" not in cache:
                        cache["loss"] = loss.detach()
                        cache["eval_logits"] = eval_logits.detach()
                        cache["eval_targets"] = eval_targets.detach()
                    return loss

                # First backward pass at w.
                if self._sam_disable_bn_stats:
                    enable_running_stats(model)
                closure()

                # SAM step: perturb -> second backward -> restore/update.
                if self._sam_disable_bn_stats:
                    disable_running_stats(model)
                sam_optimizer.step(
                    closure=closure,
                )
                if self._sam_disable_bn_stats:
                    enable_running_stats(model)

                batch_loss = float(cache["loss"].item())
                with torch.no_grad():
                    _, preds = torch.max(cache["eval_logits"], dim=1)
                    correct += preds.eq(cache["eval_targets"]).cpu().sum()
                    total += len(cache["eval_targets"])
                    total_loss += batch_loss

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

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _build_base_optimizer(self, params, stage: str):
        lr, momentum, weight_decay = self._resolve_optimizer_hyper(stage)
        if stage == "init":
            first_task_lr = self.args.get("first_task_lr", None)
            if first_task_lr is not None:
                lr = float(first_task_lr)
            if bool(self.args.get("use_adam", False)):
                return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
            if bool(self.args.get("use_sgd", False)):
                return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

        base_name = str(
            self.args.get("sam_base_optimizer", self.args.get("optimizer", self.args.get("optimizer_type", "sgd")))
        ).lower()
        if base_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if base_name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def _first_order_information(self):
        if not self._directions:
            return None
        # Avoid an extra full-memory clone here; optimizer canonicalization
        # will materialize its own detached copy.
        return self._directions

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
        logger.info("[SAM_OGD] Stored %d directions (cap %d)", len(self._directions), self._ogd_max_dirs)

    def _flatten_current_grads(self, params: List[torch.nn.Parameter]) -> torch.Tensor:
        flats = []
        for p in params:
            if p.grad is not None:
                flats.append(p.grad.view(-1))
        return torch.cat(flats) if flats else torch.tensor([], device=self._device)

    def _align_directions_to_model(self, model: nn.Module):
        target_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._align_directions_to_dim(target_dim)

    def _align_directions_to_dim(self, target_dim: int):
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
        logger.info("[SAM_OGD] Aligned %d directions to dim=%d", len(self._directions), target_dim)

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
