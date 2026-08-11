from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models_LoRAbasedCL.seqlora import Learner as SeqLoRALearner
from optimer_PerturabtionType.util import (
    disable_running_stats,
    enable_running_stats,
    generate_pertubation,
)
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(SeqLoRALearner):
    """SeqLoRA host with the legacy EWCLoRA core logic.

    Relative to ``SeqLoRA_EWC_yaoyue`` this variant:
    1. uses omega * delta_W^2 instead of F * (theta - theta*)^2;
    2. keeps no parameter anchor/checkpoint;
    3. estimates Fisher/omega in delta_W space via hook-captured grads.

    The outer training flow still follows SeqLoRA: one persistent LoRA branch
    is trained sequentially across tasks.
    """

    def __init__(self, args):
        super().__init__(args)
        self._ewc_lambda = float(args.get("ewc_lambda", args.get("lambda", 20.0)))
        self._ewc_gamma = float(args.get("ewc_gamma", args.get("gamma", 1.0)))
        fisher_max = args.get("ewc_max_batches", 100)
        self._ewc_max_batches = None if fisher_max is None else int(fisher_max)
        self._omega_w: Dict[str, torch.Tensor] = {}

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

    def _iter_delta_terms(self, model: nn.Module):
        for name, module in self._iter_delta_modules(model):
            delta_q = module.linear_b_q.weight @ module.linear_a_q.weight
            delta_v = module.linear_b_v.weight @ module.linear_a_v.weight
            yield f"{name}.q", delta_q
            yield f"{name}.v", delta_v

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

    def _iter_hook_grads(self, model: nn.Module):
        for name, module in self._iter_delta_modules(model):
            grad_q = getattr(module, "delta_w_q_new_grad", None)
            grad_v = getattr(module, "delta_w_v_new_grad", None)
            if grad_q is not None:
                yield f"{name}.q", grad_q
            if grad_v is not None:
                yield f"{name}.v", grad_v

    def _legacy_delta_penalty(self, model: nn.Module) -> torch.Tensor:
        if not self._omega_w:
            return torch.tensor(0.0, device=self._device)

        penalty = None
        for key, delta_w in self._iter_delta_terms(model):
            omega = self._omega_w.get(key)
            if omega is None:
                continue
            omega_t = omega.to(device=delta_w.device, dtype=delta_w.dtype)
            term = (omega_t * delta_w.pow(2)).sum()
            penalty = term if penalty is None else penalty + term

        if penalty is None:
            return torch.tensor(0.0, device=self._device)
        return 0.5 * self._ewc_lambda * penalty

    def _accumulate_omega(
        self, old: Dict[str, torch.Tensor], new: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not old:
            return new
        out: Dict[str, torch.Tensor] = {}
        keys = set(old.keys()) | set(new.keys())
        for key in keys:
            if key in old and key in new:
                out[key] = old[key] * self._ewc_gamma + new[key]
            elif key in old:
                out[key] = old[key] * self._ewc_gamma
            else:
                out[key] = new[key]
        return out

    def after_task(self):
        base_model = self._unwrap_network()
        new_omega = self._compute_delta_fisher(self.train_loader, base_model)
        self._omega_w = self._accumulate_omega(self._omega_w, new_omega)
        if self._is_main_process:
            self._log(f"[SeqLoRA_EWC_new] Updated omega for {len(new_omega)} delta_W tensors")
        super().after_task()

    def _run_penalty_train(self, train_loader, test_loader, optimizer, scheduler, *, is_init: bool):
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

                def build_loss(logits_tensor: torch.Tensor):
                    task_loss, eval_logits, eval_targets = self._task_loss(logits_tensor, targets, is_init)
                    return task_loss + self._legacy_delta_penalty(base_model), eval_logits, eval_targets

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
                    loss_clean, eval_logits, eval_targets = build_loss(logits_clean)
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
                    loss_noisy, _, _ = build_loss(logits_noisy)
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
                        loss_in, _, _ = build_loss(logits_in)
                        loss_value = loss_in.detach()
                        loss_in.backward()
                        return outputs, loss_value

                    outputs, loss_value = optimizer.step(closure=closure)
                    logits = outputs["logits"].detach() if is_init else outputs["logits"][:, self._known_classes :].detach()
                    eval_targets = targets if is_init else targets - self._known_classes
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                else:
                    optimizer.zero_grad()
                    outputs = self._network(inputs)
                    raw_logits = outputs["logits"]
                    loss, eval_logits, eval_targets = build_loss(raw_logits)

                    if self._optimizer_type == "sam":
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        raw_logits = self._network(inputs)["logits"]
                        second_loss, eval_logits, eval_targets = build_loss(raw_logits)
                        second_loss.backward()
                        optimizer.second_step(zero_grad=True)
                        losses += float(second_loss.detach().item())
                    else:
                        loss.backward()
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
        self._run_penalty_train(train_loader, test_loader, optimizer, scheduler, is_init=True)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        self._run_penalty_train(train_loader, test_loader, optimizer, scheduler, is_init=False)

    def _compute_delta_fisher(self, loader, model: nn.Module) -> Dict[str, torch.Tensor]:
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        total = 0
        is_init = self._cur_task == 0

        self._set_delta_hook_state(model, True)
        try:
            for batch_idx, batch in enumerate(loader):
                if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
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

                bs = int(inputs.size(0))
                total += bs
                for key, grad in self._iter_hook_grads(model):
                    g2 = (grad.detach().cpu() ** 2) * bs
                    fisher[key] = g2 if key not in fisher else fisher[key] + g2
        finally:
            self._set_delta_hook_state(model, False)
            self._clear_delta_hook_grads(model)

        if total > 0:
            for key in fisher:
                fisher[key] = fisher[key] / float(total)
        return fisher
