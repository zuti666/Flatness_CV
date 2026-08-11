from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

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
    """SeqLoRA + OGD in delta_W space instead of raw A/B parameter space."""

    def __init__(self, args):
        super().__init__(args)
        self._ogd_max_dirs = int(args.get("ogd_max_dirs", args.get("ogd_rank_per_param", 20)))
        self._ogd_store_batches = int(args.get("ogd_store_batches", 4))
        self._delta_memory = GradientMemory(mode="orthonormal", max_directions=self._ogd_max_dirs)

    def _task_loss(self, logits: torch.Tensor, targets: torch.Tensor, is_init: bool):
        if is_init:
            return F.cross_entropy(logits, targets), logits, targets
        fake_targets = targets - self._known_classes
        sliced_logits = logits[:, self._known_classes :]
        return F.cross_entropy(sliced_logits, fake_targets), sliced_logits, fake_targets

    @staticmethod
    def _is_delta_module(module: nn.Module) -> bool:
        return all(
            hasattr(module, attr)
            for attr in ("linear_a_q", "linear_b_q", "linear_a_v", "linear_b_v")
        )

    def _iter_delta_modules(self, model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
        for name, module in model.named_modules():
            if self._is_delta_module(module):
                yield name, module

    def _set_delta_hook_state(self, model: nn.Module, enabled: bool) -> None:
        for _, module in self._iter_delta_modules(model):
            if hasattr(module, "_register_delta_hook"):
                module._register_delta_hook = enabled

    def _clear_delta_hook_grads(self, model: nn.Module) -> None:
        for _, module in self._iter_delta_modules(model):
            if hasattr(module, "delta_w_q_new_grad"):
                module.delta_w_q_new_grad = None
            if hasattr(module, "delta_w_v_new_grad"):
                module.delta_w_v_new_grad = None

    def _flat_delta_grad(self, model: nn.Module) -> torch.Tensor:
        flats = []
        for _, module in self._iter_delta_modules(model):
            for key in ("q", "v"):
                grad = getattr(module, f"delta_w_{key}_new_grad", None)
                if grad is None:
                    linear_b = getattr(module, f"linear_b_{key}")
                    linear_a = getattr(module, f"linear_a_{key}")
                    zeros = torch.zeros(
                        linear_b.weight.shape[0],
                        linear_a.weight.shape[1],
                        device=self._device,
                        dtype=linear_b.weight.dtype,
                    )
                    flats.append(zeros.reshape(-1))
                else:
                    flats.append(grad.reshape(-1))
        if not flats:
            return torch.tensor([], device=self._device)
        return torch.cat(flats)

    def _assign_ab_grads_from_delta(self, model: nn.Module, flat_delta_grad: torch.Tensor) -> None:
        offset = 0
        for _, module in self._iter_delta_modules(model):
            for key in ("q", "v"):
                linear_a = getattr(module, f"linear_a_{key}")
                linear_b = getattr(module, f"linear_b_{key}")
                a_w = linear_a.weight
                b_w = linear_b.weight
                numel = b_w.shape[0] * a_w.shape[1]
                g_delta = flat_delta_grad[offset : offset + numel].view(b_w.shape[0], a_w.shape[1])
                offset += numel

                grad_b = g_delta @ a_w.detach().t()
                grad_a = b_w.detach().t() @ g_delta

                if linear_b.weight.grad is None:
                    linear_b.weight.grad = torch.zeros_like(b_w)
                if linear_a.weight.grad is None:
                    linear_a.weight.grad = torch.zeros_like(a_w)
                linear_b.weight.grad.copy_(grad_b)
                linear_a.weight.grad.copy_(grad_a)

    def _project_delta_grad(self, flat_delta_grad: torch.Tensor) -> torch.Tensor:
        if flat_delta_grad.numel() == 0 or len(self._delta_memory) == 0:
            return flat_delta_grad
        return self._delta_memory.project_orthogonal(flat_delta_grad)

    def _apply_delta_constraint_from_current_hooks(self, model: nn.Module) -> None:
        flat_delta = self._flat_delta_grad(model)
        if flat_delta.numel() == 0:
            return
        flat_delta = self._project_delta_grad(flat_delta)
        self._assign_ab_grads_from_delta(model, flat_delta)

    def incremental_train(self, data_manager):
        super().incremental_train(data_manager)
        self._update_bases(self.train_loader)

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
            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
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

                    self._set_delta_hook_state(base_model, True)
                    try:
                        enable_running_stats(self._network)
                        optimizer.zero_grad()
                        self._clear_delta_hook_grads(base_model)
                        logits_clean = self._network(inputs)["logits"]
                        loss_clean, eval_logits, eval_targets = self._task_loss(logits_clean, targets, is_init)
                        loss_clean.backward()
                        delta_g0 = self._flat_delta_grad(base_model).detach().clone()
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
                        self._clear_delta_hook_grads(base_model)
                        logits_noisy = self._network(inputs)["logits"]
                        loss_noisy, _, _ = self._task_loss(logits_noisy, targets, is_init)
                        loss_noisy.backward()
                        delta_g1 = self._flat_delta_grad(base_model).detach().clone()

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

                        delta_mix = lam * delta_g1 + (1.0 - lam) * delta_g0
                        delta_proj = self._project_delta_grad(delta_mix)
                        self._assign_ab_grads_from_delta(base_model, delta_proj)
                        optimizer.step()
                        logits = eval_logits.detach()
                        losses += lam * float(loss_noisy.detach().item()) + (1.0 - lam) * float(loss_clean.detach().item())
                    finally:
                        self._set_delta_hook_state(base_model, False)
                        self._clear_delta_hook_grads(base_model)
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
                        self._set_delta_hook_state(base_model, True)
                        try:
                            optimizer.zero_grad()
                            self._clear_delta_hook_grads(base_model)
                            outputs = self._network(inputs)
                            logits_in = outputs["logits"]
                            loss_in, _, _ = self._task_loss(logits_in, targets, is_init)
                            loss_value = loss_in.detach()
                            loss_in.backward()
                            self._apply_delta_constraint_from_current_hooks(base_model)
                            return outputs, loss_value
                        finally:
                            self._set_delta_hook_state(base_model, False)
                            self._clear_delta_hook_grads(base_model)

                    outputs, loss_value = optimizer.step(closure=closure)
                    logits = outputs["logits"].detach() if is_init else outputs["logits"][:, self._known_classes :].detach()
                    eval_targets = targets if is_init else targets - self._known_classes
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                else:
                    self._set_delta_hook_state(base_model, True)
                    try:
                        optimizer.zero_grad()
                        self._clear_delta_hook_grads(base_model)
                        outputs = self._network(inputs)
                        raw_logits = outputs["logits"]
                        loss, eval_logits, eval_targets = self._task_loss(raw_logits, targets, is_init)

                        if self._optimizer_type == "sam":
                            loss.backward()
                            self._apply_delta_constraint_from_current_hooks(base_model)
                            optimizer.first_step(zero_grad=True)

                            self._clear_delta_hook_grads(base_model)
                            raw_logits = self._network(inputs)["logits"]
                            second_loss, eval_logits, eval_targets = self._task_loss(raw_logits, targets, is_init)
                            second_loss.backward()
                            self._apply_delta_constraint_from_current_hooks(base_model)
                            optimizer.second_step(zero_grad=True)
                            losses += float(second_loss.detach().item())
                        else:
                            loss.backward()
                            self._apply_delta_constraint_from_current_hooks(base_model)
                            optimizer.step()
                            losses += float(loss.detach().item())
                        logits = eval_logits.detach()
                    finally:
                        self._set_delta_hook_state(base_model, False)
                        self._clear_delta_hook_grads(base_model)

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
        before = len(self._delta_memory)
        is_init = self._cur_task == 0

        self._set_delta_hook_state(model, True)
        try:
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= self._ogd_store_batches or len(self._delta_memory) >= self._ogd_max_dirs:
                    break
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                model.zero_grad(set_to_none=True)
                self._clear_delta_hook_grads(model)
                with torch.enable_grad():
                    logits = model(inputs)["logits"]
                    loss, _, _ = self._task_loss(logits, targets, is_init)
                    loss.backward()
                flat_delta = self._flat_delta_grad(model).detach().cpu()
                if flat_delta.numel() > 0:
                    self._delta_memory.add(flat_delta)
        finally:
            self._set_delta_hook_state(model, False)
            self._clear_delta_hook_grads(model)

        if self._is_main_process:
            added = len(self._delta_memory) - before
            self._log(
                f"[SeqLoRA_OGD_new] Stored {added} delta_W directions "
                f"(total {len(self._delta_memory)}, cap {self._ogd_max_dirs})"
            )
