"""GAM + OGD/Fisher historical filtering integrated with BaseLearner.

Core logic preserved from the method sketch:
- First-order history: OGD-style global orthonormal direction basis.
- Second-order history: EWCon-style online diagonal Fisher accumulation.
- Optimizer: GAM filter optimizer consumes both histories during perturb/projection.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_CL.baseLearner import BaseLearner
from optimer_PerturabtionType.optimer_gam_filiter import GAM_Filiter 
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
            args.get("gam_ogd_fisher_max_directions", args.get("ogd_max_dirs", args.get("max_directions", 2000)))
        )
        self._ogd_store_batches = int(
            args.get(
                "gam_ogd_fisher_store_batches",
                args.get(
                    "ogd_store_batches",
                    args.get("grads_per_task", args.get("gam_ogd_fisher_grads_per_task", 50)),
                ),
            )
        )
        self._ogd_eps = float(args.get("ogd_eps", 1e-10))
        self._directions: List[torch.Tensor] = []

        # Fisher (EWCon-style online diagonal fisher)
        self._fisher_gamma = float(args.get("gam_ogd_fisher_gamma", args.get("ewc_gamma", 1.0)))
        self._fisher_max_batches = int(args.get("gam_ogd_fisher_max_batches", args.get("ewc_max_batches", 100)))
        self._fisher_eps = float(args.get("gam_ogd_fisher_eps", args.get("ewc_eps", 1e-5)))
        self._fisher: Dict[str, torch.Tensor] | None = None

        self.max_second_order_history = int(
            args.get("gam_ogd_fisher_max_second_order_history", args.get("max_second_order_history", 100))
        )
        self.second_order_history: List[torch.Tensor] = []
        self.second_order_accum: Optional[torch.Tensor] = None

        self._gam_grad_rho = float(args.get("gam_grad_rho", args.get("sam_rho", 0.05)))
        self._gam_grad_norm_rho = float(args.get("gam_grad_norm_rho", args.get("sam_rho", 0.05)))
        self._gam_adaptive = bool(args.get("gam_adaptive", False))
        self._gam_perturb_eps = float(args.get("gam_perturb_eps", 1e-12))
        self._gam_grad_reduce = str(args.get("gam_grad_reduce", "mean"))
        
        
        self._first_order_filter_strength = float(
            args.get("first_order_filter_strength", args.get("gam_first_order_filter_strength", 1.0))
        )
        self._second_order_filter_strength = float(
            args.get("second_order_filter_strength", args.get("gam_second_order_filter_strength", 1.0))
        )
        self._perturb1_filter_strength = float(
            args.get(
                "gam_ogd_fisher_perturb1_filter_strength",
                args.get("perturb1_filter_strength", self._first_order_filter_strength),
            )
        )
        self._perturb2_filter_strength = float(
            args.get(
                "gam_ogd_fisher_perturb2_filter_strength",
                args.get("perturb2_filter_strength", self._second_order_filter_strength),
            )
        )
        self._final_gradient_filter_strength = float(
            args.get(
                "gam_ogd_fisher_final_gradient_filter_strength",
                args.get("final_gradient_filter_strength", self._first_order_filter_strength),
            )
        )

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def after_task(self):
        model = self._unwrap_network()

        if hasattr(self, "train_loader") and self.train_loader is not None:
            logger.info("Estimating GAM-OGD-Fisher second-order information for task %d...", self._cur_task)
            self._fisher = self._compute_fisher(self.train_loader, model)
            fisher_vec = self._fisher_to_vector(model)
            if fisher_vec is not None:
                self.second_order_history.append(fisher_vec)
                if len(self.second_order_history) > self.max_second_order_history:
                    self.second_order_history = self.second_order_history[-self.max_second_order_history :]
                self.second_order_accum = fisher_vec.clone()

            logger.info(
                "[GAM_OGD_Fisher] task=%d first_order_dirs=%d second_order_hist=%d fisher_dim=%d",
                self._cur_task,
                len(self._directions),
                len(self.second_order_history),
                0 if fisher_vec is None else int(fisher_vec.numel()),
            )

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

        # Align with OGD.py: collect first-order directions after current task training.
        self._store_directions(self.train_loader, self._unwrap_network())

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[GAM_OGD_Fisher] Failed to compute class means: %s", exc)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def _train(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)

        stage = "init" if self._cur_task == 0 else "update"
        params = [p for p in model.parameters() if p.requires_grad]
        base_optimizer = self._build_base_optimizer(params, stage=stage)

        gam_optimizer = GAM_Filiter(
            params,
            base_optimizer=base_optimizer,
            model=model,
            grad_rho=self._gam_grad_rho,
            grad_norm_rho=self._gam_grad_norm_rho,
            adaptive=self._gam_adaptive,
            perturb_eps=self._gam_perturb_eps,
            args=self._build_gam_args(),
            grad_reduce=self._gam_grad_reduce,
            first_order_filter_strength=self._first_order_filter_strength,
            second_order_filter_strength=self._second_order_filter_strength,
            perturb1_filter_strength=self._perturb1_filter_strength,
            perturb2_filter_strength=self._perturb2_filter_strength,
            final_gradient_filter_strength=self._final_gradient_filter_strength,
        )

        epochs = int(self.args.get("init_epoch" if self._cur_task == 0 else "epochs", 1))
        scheduler = self.build_scheduler(
            gam_optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("init_milestones" if self._cur_task == 0 else "milestones", []),
            gamma=float(self.args.get("init_lrate_decay" if self._cur_task == 0 else "lrate_decay", 1.0)),
            T_max=epochs,
            eta_min=self.args.get("min_lr", 0.0),
        )

        criterion = nn.CrossEntropyLoss()
        first_order_information = self._first_order_information()
        second_order_information = self._second_order_information()

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

                def closure():
                    gam_optimizer.zero_grad()
                    outputs = model(inputs)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    if stage == "init":
                        loss = criterion(logits, targets)
                        eval_logits, eval_targets = logits, targets
                    else:
                        fake_targets = targets - self._known_classes
                        eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets
                        loss = criterion(eval_logits, eval_targets)
                    loss.backward()
                    return {"eval_logits": eval_logits.detach(), "eval_targets": eval_targets.detach()}, loss.detach()

                payload, loss_value = gam_optimizer.step(
                    closure=closure,
                    first_order_information=first_order_information,
                    second_order_information=second_order_information,
                )

                batch_loss = loss_value.item() if torch.is_tensor(loss_value) else float(loss_value)
                with torch.no_grad():
                    eval_logits = payload["eval_logits"]
                    eval_targets = payload["eval_targets"]
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(batch_loss)

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
            self.args.get("gam_base_optimizer", self.args.get("optimizer", self.args.get("optimizer_type", "sgd")))
        ).lower()
        if base_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if base_name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def _build_gam_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            grad_beta_1=float(self.args.get("gam_grad_beta_1", self.args.get("grad_beta_1", 1.0))),
            grad_beta_2=float(self.args.get("gam_grad_beta_2", self.args.get("grad_beta_2", 1.0))),
            grad_beta_3=float(self.args.get("gam_grad_beta_3", self.args.get("grad_beta_3", 1.0))),
            grad_gamma=float(self.args.get("gam_grad_gamma", self.args.get("grad_gamma", 0.0))),
        )

    def _first_order_information(self) -> Optional[List[torch.Tensor]]:
        if not self._directions:
            return None
        return [v.detach().clone() for v in self._directions]

    def _second_order_information(self) -> Optional[List[torch.Tensor]]:
        # EWCon-style online Fisher: use the current consolidated fisher as
        # second-order information source.
        fisher_vec = self._fisher_to_vector(self._unwrap_network())
        if fisher_vec is not None and fisher_vec.numel() > 0:
            return [fisher_vec]

        infos: List[torch.Tensor] = []
        if self.second_order_history:
            infos.extend(v.detach().clone() for v in self.second_order_history)
        if self.second_order_accum is not None:
            infos.append(self.second_order_accum.detach().clone())
        return infos or None

    def _align_state_to_model(self, model: nn.Module):
        target_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)

        def _match(v: torch.Tensor) -> torch.Tensor:
            if v.numel() == target_dim:
                return v.detach().clone().cpu()
            out = torch.zeros(target_dim, dtype=v.dtype, device="cpu")
            n = min(target_dim, v.numel())
            out[:n] = v[:n].detach().cpu()
            return out

        if self._directions:
            aligned = [_match(v) for v in self._directions]
            self._directions = gram_schmidt(aligned, eps=self._ogd_eps)
            if len(self._directions) > self._ogd_max_dirs:
                self._directions = self._directions[: self._ogd_max_dirs]
        if self._fisher is not None:
            for name, p in model.named_parameters():
                if name in self._fisher:
                    self._fisher[name] = self._match_tensor(self._fisher[name], p)
        if self.second_order_history:
            self.second_order_history = [_match(v) for v in self.second_order_history]
        if self.second_order_accum is not None:
            self.second_order_accum = _match(self.second_order_accum)

    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    def _match_tensor(self, stored: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if stored.shape == param.shape:
            return stored
        new = torch.zeros_like(param.detach().cpu())
        slices = tuple(slice(0, min(a, b)) for a, b in zip(stored.shape, param.shape))
        new[slices] = stored[slices]
        return new

    def _fisher_to_vector(self, model: nn.Module) -> Optional[torch.Tensor]:
        if self._fisher is None:
            return None
        parts: List[torch.Tensor] = []
        for name, p in self._iter_trainable(model):
            f = self._fisher.get(name)
            if f is None:
                parts.append(torch.zeros(p.numel(), dtype=p.detach().cpu().dtype, device="cpu"))
                continue
            if f.shape != p.shape:
                f = self._match_tensor(f, p)
                self._fisher[name] = f
            parts.append(f.detach().reshape(-1).cpu())
        if not parts:
            return None
        return torch.cat(parts)

    def _compute_fisher(self, loader, model: nn.Module) -> Dict[str, torch.Tensor]:
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        total = 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= self._fisher_max_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            model.zero_grad()
            with torch.enable_grad():
                logits = model(inputs)["logits"]
                logp = F.log_softmax(logits, dim=1)
                nll = -F.nll_loss(logp, targets, reduction="none")
                exp_cp = torch.mean(torch.exp(nll.detach()))
                loss = nll.mean()
                loss.backward()

            for name, p in self._iter_trainable(model):
                if p.grad is None:
                    continue
                g2 = (p.grad.detach().cpu() ** 2) * exp_cp.cpu()
                fisher[name] = g2 if name not in fisher else fisher[name] + g2
            total += 1

        if total > 0:
            for name in fisher:
                fisher[name] = (fisher[name] / float(total)).clamp_min(self._fisher_eps)

        if self._fisher is not None:
            updated: Dict[str, torch.Tensor] = {}
            for name, new_f in fisher.items():
                old = self._fisher.get(name)
                if old is None:
                    updated[name] = new_f
                else:
                    old_aligned = self._match_tensor(old, new_f)
                    updated[name] = self._fisher_gamma * old_aligned + new_f
            for name, old in self._fisher.items():
                if name not in updated:
                    updated[name] = self._fisher_gamma * old
            self._fisher = updated
        else:
            self._fisher = fisher

        logger.info("[GAM_OGD_Fisher] Fisher estimated over %d batches", total)
        return self._fisher

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
        logger.info("[GAM_OGD_Fisher] Stored %d directions (cap %d)", len(self._directions), self._ogd_max_dirs)

    def _flatten_current_grads(self, params: List[torch.nn.Parameter]) -> torch.Tensor:
        flats = []
        for p in params:
            if p.grad is not None:
                flats.append(p.grad.view(-1))
        return torch.cat(flats) if flats else torch.tensor([], device=self._device)

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
