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

logger = logging.getLogger(__name__)


class Learner(SeqLoRALearner):
    """SeqLoRA-based EWCLoRA.

    This version stays fully on top of the current SeqLoRA lifecycle:

    - reuse SeqLoRA's persistent LoRA backbone and task lifecycle;
    - snapshot the task-start effective LoRA update in delta-W space;
    - regularize only the current task-induced delta change with past-task
      Fisher accumulated in the same delta-W space.

    The resulting penalty is:

        0.5 * lambda * sum_k (F_past[k] + eta) * (delta_k - delta_ref_k)^2

    where ``delta_ref_k`` is the task-start checkpoint for task ``t``.
    """

    _UNSUPPORTED_OPTIMIZERS = {
        "flatlora",
        "flatlora_full",
        "faltlora",
        "faltlora_full",
        "mergegam",
        "mergegam_lora",
    }

    def __init__(self, args):
        super().__init__(args)

        if self._optimizer_type in self._UNSUPPORTED_OPTIMIZERS:
            raise ValueError(
                "ewclora_youyue_fitArchitechture currently supports "
                "sgd/adam/adamw/sam/cflat/gam/rwp/arwp only, got "
                f"optimizer_type={self._optimizer_type}."
            )

        self._ewc_lambda = float(args.get("ewc_lambda", args.get("lambda", 20.0)))
        self._ewc_gamma = float(args.get("ewc_gamma", args.get("gamma", 1.0)))
        self._ewc_eta = float(args.get("ewc_eta", args.get("eta", 0.0)))

        fisher_max = args.get("ewc_max_batches", 100)
        if fisher_max is None or str(fisher_max).lower() in {"none", "all", "full"}:
            self._ewc_max_batches = None
        else:
            fisher_max = int(fisher_max)
            self._ewc_max_batches = None if fisher_max <= 0 else fisher_max

        self._fisher_past_delta: Dict[str, torch.Tensor] = {}
        self._delta_reference: Dict[str, torch.Tensor] = {}

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

    def _snapshot_task_reference(self) -> None:
        base_model = self._unwrap_network()
        self._delta_reference = {
            key: delta.detach().cpu().clone()
            for key, delta in self._iter_delta_terms(base_model)
        }

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

    def _task_loss(self, logits: torch.Tensor, targets: torch.Tensor, class_offset: int):
        if class_offset > 0:
            fake_targets = targets - class_offset
            sliced_logits = logits[:, class_offset:]
            return F.cross_entropy(sliced_logits, fake_targets), sliced_logits, fake_targets
        return F.cross_entropy(logits, targets), logits, targets

    def _past_fisher_penalty(self, model: nn.Module) -> torch.Tensor:
        if not self._fisher_past_delta or not self._delta_reference:
            return torch.tensor(0.0, device=self._device)

        penalty = None
        for key, delta_now in self._iter_delta_terms(model):
            fisher = self._fisher_past_delta.get(key)
            delta_ref = self._delta_reference.get(key)
            if fisher is None or delta_ref is None:
                continue

            delta_ref_t = delta_ref.to(device=delta_now.device, dtype=delta_now.dtype)
            fisher_t = fisher.to(device=delta_now.device, dtype=delta_now.dtype)
            if self._ewc_eta != 0.0:
                fisher_t = fisher_t + self._ewc_eta

            delta_update = delta_now - delta_ref_t
            term = (fisher_t * delta_update.pow(2)).sum()
            penalty = term if penalty is None else penalty + term

        if penalty is None:
            return torch.tensor(0.0, device=self._device)
        return 0.5 * self._ewc_lambda * penalty

    def _regularized_task_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_offset: int,
        model: nn.Module | None = None,
    ):
        task_loss, eval_logits, eval_targets = self._task_loss(logits, targets, class_offset)
        if not self._fisher_past_delta:
            return task_loss, eval_logits, eval_targets

        if model is None:
            model = self._unwrap_network()
        return task_loss + self._past_fisher_penalty(model), eval_logits, eval_targets

    def _accumulate_fisher(
        self, old: Dict[str, torch.Tensor], new: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not old:
            return new

        out: Dict[str, torch.Tensor] = {}
        for key in set(old.keys()) | set(new.keys()):
            if key in old and key in new:
                out[key] = old[key] * self._ewc_gamma + new[key]
            elif key in old:
                out[key] = old[key] * self._ewc_gamma
            else:
                out[key] = new[key]
        return out

    def after_task(self):
        base_model = self._unwrap_network()
        new_fisher = self._compute_delta_fisher(
            loader=self.train_loader,
            model=base_model,
            class_offset=int(self._known_classes),
        )
        self._fisher_past_delta = self._accumulate_fisher(self._fisher_past_delta, new_fisher)
        self._delta_reference = {}

        if self._is_main_process:
            self._log(
                "[EWCLoRA-Seq] Updated past Fisher for "
                f"{len(new_fisher)} delta tensors after task {self._cur_task}."
            )
        super().after_task()

    def _run_regularized_train(
        self,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        *,
        epochs: int,
        class_offset: int,
    ) -> None:
        self._snapshot_task_reference()
        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        info = ""

        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                logits, loss_value = self._step_batch(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    inputs=inputs,
                    targets=targets,
                    class_offset=class_offset,
                )
                losses += float(loss_value)

                with torch.no_grad():
                    if class_offset > 0:
                        eval_logits = logits[:, class_offset:]
                        eval_targets = targets - class_offset
                    else:
                        eval_logits = logits
                        eval_targets = targets
                    preds = torch.argmax(eval_logits, dim=1)
                    correct += int(preds.eq(eval_targets).sum().item())
                    total += int(eval_targets.numel())

            if scheduler is not None:
                scheduler.step()

            train_acc = 0.0 if total == 0 else round(100.0 * correct / total, 2)
            if ((epoch % 5 == 4) or epoch == epochs - 1) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / max(1, len(train_loader)):.3f}, "
                    f"Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}"
                )
            elif self._is_main_process:
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / max(1, len(train_loader)):.3f}, "
                    f"Train_accy {train_acc:.2f}"
                )

            if self._is_main_process:
                prog_bar.set_description(info)

        if self._is_main_process and info:
            self._log(info)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        self._run_regularized_train(
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            epochs=int(self.args["init_epoch"]),
            class_offset=0,
        )

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        self._run_regularized_train(
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            epochs=int(self.args["epochs"]),
            class_offset=int(self._known_classes),
        )

    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset: int):
        if self._optimizer_type == "cflat":

            def closure():
                optimizer.zero_grad()
                outputs = self._network(inputs)
                loss, _, _ = self._regularized_task_loss(
                    outputs["logits"],
                    targets,
                    class_offset,
                    model=self._unwrap_network(),
                )
                loss.backward()
                return outputs, [loss]

            _, loss_list = optimizer.step(closure=closure)
            with torch.no_grad():
                logits = self._network(inputs)["logits"]
            loss_value = torch.stack([loss_term.detach() for loss_term in loss_list]).sum()
            return logits.detach(), float(loss_value.item())

        if self._optimizer_type == "gam":

            def closure():
                optimizer.zero_grad()
                outputs = self._network(inputs)
                loss, _, _ = self._regularized_task_loss(
                    outputs["logits"],
                    targets,
                    class_offset,
                    model=self._unwrap_network(),
                )
                loss_value = loss.detach()
                loss.backward()
                return outputs, loss_value

            outputs, loss_value = optimizer.step(closure=closure)
            scalar = loss_value.item() if torch.is_tensor(loss_value) else loss_value
            return outputs["logits"].detach(), float(scalar)

        if self._optimizer_type == "rwp":
            return self._rwp_step(optimizer, scheduler, inputs, targets, class_offset)

        if self._optimizer_type == "arwp":
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
                loss, _, _ = self._regularized_task_loss(
                    outputs["logits"],
                    targets,
                    class_offset,
                    model=self._unwrap_network(),
                )
                loss_value = loss.detach()
                loss.backward()
                return outputs, loss_value

            outputs, loss_value = optimizer.step(closure=closure)
            scalar = loss_value.item() if torch.is_tensor(loss_value) else loss_value
            return outputs["logits"].detach(), float(scalar)

        optimizer.zero_grad()
        logits = self._network(inputs)["logits"]
        loss, _, _ = self._regularized_task_loss(
            logits,
            targets,
            class_offset,
            model=self._unwrap_network(),
        )

        if self._optimizer_type == "sam":
            loss.backward()
            optimizer.first_step(zero_grad=True)
            logits = self._network(inputs)["logits"]
            second_loss, _, _ = self._regularized_task_loss(
                logits,
                targets,
                class_offset,
                model=self._unwrap_network(),
            )
            second_loss.backward()
            optimizer.second_step(zero_grad=True)
            return logits.detach(), float(second_loss.detach().item())

        loss.backward()
        optimizer.step()
        return logits.detach(), float(loss.detach().item())

    def _rwp_step(self, optimizer, scheduler, inputs, targets, class_offset: int):
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
            rwp_std = float(self._rwp_std) * scale
        else:
            rwp_std = float(self._rwp_std)

        enable_running_stats(self._network)
        optimizer.zero_grad()
        outputs = self._network(inputs)
        logits_clean = outputs["logits"]
        base_model = self._unwrap_network()
        loss_clean, _, _ = self._regularized_task_loss(
            logits_clean,
            targets,
            class_offset,
            model=base_model,
        )
        loss_clean.backward()

        g0 = {}
        for name, param in base_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                g0[name] = param.grad.detach().clone()

        disable_running_stats(self._network)
        noise_dict = {}
        with torch.no_grad():
            std_for_noise = float(self.args.get("noise_std", rwp_std))
            for name, param in base_model.named_parameters():
                if self._rwp_range == "lora" and (not param.requires_grad or param.numel() == 0):
                    continue
                fisher_param = None
                if hasattr(self, "_rwp_fisher") and name in self._rwp_fisher:
                    fisher_param = self._rwp_fisher[name]
                perturb = generate_pertubation(
                    param,
                    pertubation_mode=self.rwp_noise_type,
                    std=std_for_noise,
                    fisher_param=fisher_param,
                    fisher_scaler=float(self._rwp_eta),
                )
                param.data.add_(perturb)
                noise_dict[name] = perturb

        optimizer.zero_grad()
        outputs_noisy = self._network(inputs)
        logits_noisy = outputs_noisy["logits"]
        loss_noisy, _, _ = self._regularized_task_loss(
            logits_noisy,
            targets,
            class_offset,
            model=base_model,
        )
        loss_noisy.backward()

        if hasattr(self, "_rwp_fisher"):
            with torch.no_grad():
                for name, param in base_model.named_parameters():
                    if not param.requires_grad or param.grad is None:
                        continue
                    g2 = param.grad.detach() ** 2
                    if name not in self._rwp_fisher:
                        self._rwp_fisher[name] = g2
                    else:
                        self._rwp_fisher[name] = float(self._rwp_beta) * self._rwp_fisher[name] + g2

        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in noise_dict:
                    param.data.sub_(noise_dict[name])

        lam = float(self._rwp_lambda)
        for name, param in base_model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            g1 = param.grad.detach()
            g0_n = g0.get(name, torch.zeros_like(g1))
            param.grad.data.copy_(lam * g1 + (1.0 - lam) * g0_n)

        optimizer.step()
        enable_running_stats(self._network)

        mixed_loss = lam * float(loss_noisy.detach().item()) + (1.0 - lam) * float(
            loss_clean.detach().item()
        )
        return logits_clean.detach(), mixed_loss

    def _compute_delta_fisher(self, loader, model: nn.Module, class_offset: int) -> Dict[str, torch.Tensor]:
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        total = 0

        self._set_delta_hook_state(model, True)
        try:
            for batch_idx, batch in enumerate(loader):
                if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
                    break

                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                model.zero_grad(set_to_none=True)
                self._clear_delta_hook_grads(model)

                with torch.enable_grad():
                    logits = model(inputs)["logits"]
                    loss, _, _ = self._task_loss(logits, targets, class_offset)
                    loss.backward()

                batch_size = int(inputs.size(0))
                total += batch_size
                for key, grad in self._iter_hook_grads(model):
                    grad_sq = (grad.detach().cpu() ** 2) * batch_size
                    fisher[key] = grad_sq if key not in fisher else fisher[key] + grad_sq
        finally:
            self._set_delta_hook_state(model, False)
            self._clear_delta_hook_grads(model)

        if total > 0:
            for key in fisher:
                fisher[key] = fisher[key] / float(total)
        return fisher


class EWCLoRA(Learner):
    """Backward-compatible alias."""
