from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models_Project2.models_Full.OGD_utils.gradients import GradientMemory
from models_LoRAbasedCL.seqlora import Learner as SeqLoRALearner
from optimer_PerturabtionType.util import (
    disable_running_stats,
    enable_running_stats,
    generate_pertubation,
)
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(SeqLoRALearner):
    """SeqLoRA + OGD over the fixed LoRA A/B parameter space."""

    def __init__(self, args):
        super().__init__(args)
        self._ogd_max_dirs = int(args.get("ogd_max_dirs", args.get("ogd_rank_per_param", 20)))
        self._ogd_store_batches = int(args.get("ogd_store_batches", 4))
        self._ab_memory = GradientMemory(mode="orthonormal", max_directions=self._ogd_max_dirs)

    @staticmethod
    def _is_lora_ab_name(name: str) -> bool:
        key = name.lower()
        return (
            "linear_a_" in key
            or "linear_b_" in key
            or "lora_new_a" in key
            or "lora_new_b" in key
            or ".w_as." in key
            or ".w_bs." in key
        )

    def _iter_lora_ab_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and self._is_lora_ab_name(name):
                yield name, p

    def _get_flat_grad_vector(self, named_params: List[Tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
        if not named_params:
            return torch.tensor([], device=self._device)
        grads = []
        for _, p in named_params:
            if p.grad is None:
                grads.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            else:
                grads.append(p.grad.view(-1))
        return torch.cat(grads) if grads else torch.tensor([], device=self._device)

    def _set_flat_grad_vector(
        self, named_params: List[Tuple[str, torch.nn.Parameter]], grad_vector: torch.Tensor
    ) -> None:
        offset = 0
        for _, p in named_params:
            numel = p.numel()
            if numel == 0:
                continue
            g = grad_vector[offset : offset + numel].view_as(p)
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.copy_(g)
            offset += numel

    def _apply_gradient_constraint(self, model: nn.Module) -> None:
        if len(self._ab_memory) == 0:
            return
        named_ab_params = list(self._iter_lora_ab_trainable(model))
        if not named_ab_params:
            return
        flat_grad = self._get_flat_grad_vector(named_ab_params)
        if flat_grad.numel() == 0:
            return
        flat_grad_proj = self._ab_memory.project_orthogonal(flat_grad)
        self._set_flat_grad_vector(named_ab_params, flat_grad_proj)

    def incremental_train(self, data_manager):
        super().incremental_train(data_manager)
        self._update_bases(self.train_loader)

    def _task_loss(self, logits: torch.Tensor, targets: torch.Tensor, is_init: bool):
        if is_init:
            return F.cross_entropy(logits, targets), logits, targets
        fake_targets = targets - self._known_classes
        return (
            F.cross_entropy(logits[:, self._known_classes :], fake_targets),
            logits[:, self._known_classes :],
            fake_targets,
        )

    def _run_constraint_train(self, train_loader, test_loader, optimizer, scheduler, *, is_init: bool):
        if self._optimizer_type in {"cflat", "gam"}:
            raise NotImplementedError(
                f"{self.__class__.__name__} currently supports sgd/adam/adamw/sam/rwp/arwp, got {self._optimizer_type}."
            )

        epochs = int(self.args["init_epoch"] if is_init else self.args["epochs"])
        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                base_model = self._unwrap_network()

                if self._optimizer_type == "rwp":
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = (
                                float(scheduler.get_last_lr()[0])
                                if hasattr(scheduler, "get_last_lr")
                                else float(optimizer.param_groups[0]["lr"])
                            )
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"])
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr))
                        scale = (cur_lr / base_lr) if base_lr > 0 else 1.0
                        rwp_std = float(self._rwp_std) * float(scale)
                    else:
                        rwp_std = float(self._rwp_std)

                    enable_running_stats(self._network)
                    optimizer.zero_grad()
                    logits_clean = self._network(inputs)["logits"]
                    loss_clean, eval_logits, eval_targets = self._task_loss(logits_clean, targets, is_init)
                    loss_clean.backward()
                    g0 = {}
                    for name, p in base_model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            g0[name] = p.grad.detach().clone()

                    disable_running_stats(self._network)
                    noise_dict = {}
                    with torch.no_grad():
                        std_for_noise = float(self.args.get("noise_std", rwp_std))
                        for name, p in base_model.named_parameters():
                            if self._rwp_range == "lora" and (not p.requires_grad or p.numel() == 0):
                                continue
                            fisher_param = None
                            if hasattr(self, "_rwp_fisher") and name in self._rwp_fisher:
                                fisher_param = self._rwp_fisher[name]
                            noise = generate_pertubation(
                                p,
                                pertubation_mode=self.rwp_noise_type,
                                std=std_for_noise,
                                fisher_param=fisher_param,
                                fisher_scaler=float(self._rwp_eta),
                            )
                            p.data.add_(noise)
                            noise_dict[name] = noise

                    optimizer.zero_grad()
                    logits_noisy = self._network(inputs)["logits"]
                    loss_noisy, _, _ = self._task_loss(logits_noisy, targets, is_init)
                    loss_noisy.backward()

                    if hasattr(self, "_rwp_fisher"):
                        with torch.no_grad():
                            for name, p in base_model.named_parameters():
                                if not p.requires_grad or p.grad is None:
                                    continue
                                g2 = p.grad.detach() ** 2
                                if name not in self._rwp_fisher:
                                    self._rwp_fisher[name] = g2
                                else:
                                    self._rwp_fisher[name] = float(self._rwp_beta) * self._rwp_fisher[name] + g2

                    with torch.no_grad():
                        for name, p in base_model.named_parameters():
                            if name in noise_dict:
                                p.data.sub_(noise_dict[name])

                    lam = float(self._rwp_lambda)
                    for name, p in base_model.named_parameters():
                        if not p.requires_grad or p.grad is None:
                            continue
                        g1 = p.grad.detach()
                        g0_n = g0.get(name, torch.zeros_like(g1))
                        p.grad.data.copy_(lam * g1 + (1.0 - lam) * g0_n)

                    self._apply_gradient_constraint(base_model)
                    optimizer.step()
                    logits = eval_logits.detach()
                    losses += lam * float(loss_noisy.detach().item()) + (1.0 - lam) * float(loss_clean.detach().item())
                elif self._optimizer_type == "arwp":
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = (
                                float(scheduler.get_last_lr()[0])
                                if hasattr(scheduler, "get_last_lr")
                                else float(optimizer.param_groups[0]["lr"])
                            )
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"])
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr))
                        scale = (cur_lr / base_lr) if base_lr > 0 else 1.0
                        optimizer.std = float(self._rwp_std) * scale

                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits_in = outputs["logits"]
                        loss_in, _, _ = self._task_loss(logits_in, targets, is_init)
                        loss_value = loss_in.detach()
                        loss_in.backward()
                        self._apply_gradient_constraint(base_model)
                        return outputs, loss_value

                    outputs, loss_value = optimizer.step(closure=closure)
                    logits = outputs["logits"].detach() if is_init else outputs["logits"][:, self._known_classes :].detach()
                    eval_targets = targets if is_init else targets - self._known_classes
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                else:
                    optimizer.zero_grad()
                    outputs = self._network(inputs)
                    raw_logits = outputs["logits"]
                    loss, eval_logits, eval_targets = self._task_loss(raw_logits, targets, is_init)

                    if self._optimizer_type == "sam":
                        loss.backward()
                        self._apply_gradient_constraint(base_model)
                        optimizer.first_step(zero_grad=True)
                        raw_logits = self._network(inputs)["logits"]
                        second_loss, eval_logits, eval_targets = self._task_loss(raw_logits, targets, is_init)
                        second_loss.backward()
                        self._apply_gradient_constraint(base_model)
                        optimizer.second_step(zero_grad=True)
                        losses += float(second_loss.detach().item())
                    else:
                        loss.backward()
                        self._apply_gradient_constraint(base_model)
                        optimizer.step()
                        losses += float(loss.detach().item())
                    logits = eval_logits.detach()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(eval_targets.expand_as(preds)).cpu().sum()
                total += len(eval_targets)

            scheduler.step()
            train_acc = 0.0 if total == 0 else tensor2numpy(correct) * 100 / total
            train_acc = round(float(train_acc), 2)
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / len(train_loader):.3f}, Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}"
                )
            elif self._is_main_process:
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / len(train_loader):.3f}, Train_accy {train_acc:.2f}"
                )
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        self._run_constraint_train(train_loader, test_loader, optimizer, scheduler, is_init=True)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        self._run_constraint_train(train_loader, test_loader, optimizer, scheduler, is_init=False)

    def _update_bases(self, loader) -> None:
        model = self._unwrap_network()
        model.eval()
        named_ab_params = list(self._iter_lora_ab_trainable(model))
        if not named_ab_params:
            if self._is_main_process:
                self._log("[SeqLoRA_OGD] No trainable LoRA A/B params found; skip direction update.")
            return

        before = len(self._ab_memory)
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= self._ogd_store_batches:
                break
            _, inputs, targets = batch
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            model.zero_grad()
            with torch.enable_grad():
                logits = model(inputs)["logits"]
                # Use CE loss (consistent with training) instead of per-sample score sum
                F.cross_entropy(logits, targets).backward()

            flat_grad = self._get_flat_grad_vector(named_ab_params).detach().cpu()
            if flat_grad.numel() > 0:
                self._ab_memory.add(flat_grad)
            if len(self._ab_memory) >= self._ogd_max_dirs:
                break

        if self._is_main_process:
            added = len(self._ab_memory) - before
            self._log(
                f"[SeqLoRA_OGD] Stored {added} LoRA A/B directions "
                f"(total {len(self._ab_memory)}, cap {self._ogd_max_dirs})"
            )
