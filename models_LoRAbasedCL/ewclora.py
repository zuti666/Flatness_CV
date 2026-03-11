"""EWCLoRA (legacy idea) adapted to current LoRA continual-learning framework.

Core logic intentionally preserved:
1) EWC regularizer uses omega * (delta_W^2) without (theta - theta*) term.
2) Fisher is estimated from hook-captured grads of delta_w_new tensors.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.net_ewclora import Net
from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(LoraBaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.topk = 1

        self._network = Net(args)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()
        if self._optimizer_type == "sam":
            self._sam_rho = float(args.get("sam_rho", 0.05))
            self._sam_adaptive = bool(args.get("sam_adaptive", False))
        elif self._optimizer_type == "cflat":
            self._cflat_rho = float(args.get("cflat_rho", 0.2))
            self._cflat_lambda = float(args.get("cflat_lambda", 0.2))
            self._cflat_adaptive = bool(args.get("cflat_adaptive", False))
            self._cflat_perturb_eps = float(args.get("cflat_perturb_eps", 1e-12))
            self._cflat_grad_reduce = args.get("cflat_grad_reduce", "mean")
        elif self._optimizer_type == "gam":
            self._gam_adaptive = bool(args.get("gam_adaptive", False))
            self._gam_grad_reduce = args.get("gam_grad_reduce", "mean")
            self._gam_perturb_eps = float(args.get("gam_perturb_eps", 1e-12))
            self._gam_grad_rho = float(args.get("gam_grad_rho", args.get("grad_rho", 0.2)))
            self._gam_grad_norm_rho = float(
                args.get("gam_grad_norm_rho", args.get("grad_norm_rho", 0.2))
            )
            self._gam_beta1 = float(args.get("gam_grad_beta_1", args.get("grad_beta_1", 1.0)))
            self._gam_beta2 = float(args.get("gam_grad_beta_2", args.get("grad_beta_2", 1.0)))
            self._gam_beta3 = float(args.get("gam_grad_beta_3", args.get("grad_beta_3", 1.0)))
            self._gam_gamma = float(args.get("gam_grad_gamma", args.get("grad_gamma", 0.1)))
            self._gam_args = SimpleNamespace(
                grad_beta_1=self._gam_beta1,
                grad_beta_2=self._gam_beta2,
                grad_beta_3=self._gam_beta3,
                grad_gamma=self._gam_gamma,
                grad_rho=self._gam_grad_rho,
                grad_norm_rho=self._gam_grad_norm_rho,
                adaptive=self._gam_adaptive,
                perturb_eps=self._gam_perturb_eps,
                grad_reduce=str(self._gam_grad_reduce),
            )

        # Keep original hyperparameter semantics/names as much as possible.
        self._gamma = float(args.get("gamma", args.get("ewc_gamma", 1.0)))
        self._ewc_weight = float(args.get("lambda", args.get("ewc_lambda", 20.0)))
        fisher_max = args.get("ewc_max_batches", None)
        self._fisher_max_batches = None if fisher_max is None else int(fisher_max)

        # Importance matrix over LoRA delta_W terms (list aligned with module order).
        self._omega_w: List[torch.Tensor] = []
        self._count_updates = 0

    # ------------------------------------------------------------------#
    # Lifecycle                                                         #
    # ------------------------------------------------------------------#
    def incremental_train(self, data_manager):
        self._refresh_distributed_context()

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        # Net.update_fc() advances its internal task cursor.
        base_model = self._unwrap_network()
        base_model.update_fc(self._total_classes)
        self._network = base_model
        self._prepare_network()

        self._log(f"Learning on {self._known_classes}-{self._total_classes}")

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

        self._train(self.train_loader)

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            self._log(f"[EWCLoRA] Failed to compute class means: {exc}")

    def after_task(self):
        # Keep legacy flow: update Fisher/omega after each task, then fold new LoRA into accumulators.
        self._update_importance_matrix()

        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------#
    # Training                                                          #
    # ------------------------------------------------------------------#
    def _train(self, train_loader):
        model = self._unwrap_network()
        model.to(self._device)
        self._freeze_network(model)
        self._prepare_network()

        model = self._network
        stage = "init" if self._cur_task == 0 else "update"
        optimizer, scheduler, epochs = self._build_task_optimizer(model, stage=stage)

        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        for epoch in prog_bar:
            model.train()
            total_loss = 0.0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch

                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # Only train on current task classes.
                mask = (targets >= self._known_classes).nonzero(as_tuple=False).view(-1)
                if mask.numel() == 0:
                    continue
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes
                if targets.numel() == 0:
                    continue

                def _loss_from_logits(logits_tensor):
                    loss_val = F.cross_entropy(logits_tensor, targets)
                    # Core legacy EWCLoRA regularizer:
                    #   (lambda/2) * sum_i omega_i * delta_W_i^2
                    if self._count_updates != 0 and len(self._omega_w) != 0:
                        ewc_loss = self._compute_ewc_loss(model)
                        loss_val = loss_val + (self._ewc_weight / 2.0) * ewc_loss
                    return loss_val

                if self._optimizer_type == "cflat":
                    def closure():
                        optimizer.zero_grad()
                        logits_in = model(inputs, use_new=True)["logits"]
                        loss_in = _loss_from_logits(logits_in)
                        loss_in.backward()
                        return {"logits": logits_in}, [loss_in]

                    _, loss_list = optimizer.step(closure=closure)
                    with torch.no_grad():
                        logits = model(inputs, use_new=True)["logits"]
                    loss_value = torch.stack([loss_term.detach() for loss_term in loss_list]).sum()
                    total_loss += float(loss_value.item())
                elif self._optimizer_type == "gam":
                    def closure():
                        optimizer.zero_grad()
                        outputs_in = model(inputs, use_new=True)
                        logits_in = outputs_in["logits"]
                        loss_in = _loss_from_logits(logits_in)
                        loss_value_in = loss_in.detach()
                        loss_in.backward()
                        return outputs_in, loss_value_in

                    outputs, loss_value = optimizer.step(closure=closure)
                    logits = outputs["logits"].detach()
                    total_loss += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                else:
                    optimizer.zero_grad()
                    logits = model(inputs, use_new=True)["logits"]
                    loss = _loss_from_logits(logits)

                    if self._optimizer_type == "sam":
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        logits = model(inputs, use_new=True)["logits"]
                        second_loss = _loss_from_logits(logits)
                        second_loss.backward()
                        optimizer.second_step(zero_grad=True)
                        total_loss += float(second_loss.detach().item())
                    else:
                        loss.backward()
                        optimizer.step()
                        total_loss += float(loss.detach().item())

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets).cpu().sum()
                    total += int(targets.numel())

            scheduler.step()
            train_acc = 0.0 if total == 0 else np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = (
                f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                f"Loss {total_loss / max(1, len(train_loader)):.3f}, Train_accy {train_acc:.2f}"
            )
            if self._is_main_process:
                prog_bar.set_description(info)

        if self._is_main_process:
            self._log(info)

    def _build_task_optimizer(
        self, model: nn.Module, stage: str
    ) -> Tuple[torch.optim.Optimizer, object, int]:
        if stage == "init":
            epochs = int(self.args.get("init_epoch", 1))
            lr = float(self.args.get("init_lr", self.args.get("lrate", 0.1)))
            momentum = float(self.args.get("init_momentum", self.args.get("momentum", 0.0)))
            weight_decay = float(self.args.get("init_weight_decay", self.args.get("weight_decay", 0.0)))
            milestones = self.args.get("init_milestones", [])
            gamma = float(self.args.get("init_lrate_decay", 1.0))
        else:
            epochs = int(self.args.get("epochs", 1))
            lr = float(self.args.get("lrate", 0.1))
            momentum = float(self.args.get("momentum", 0.0))
            weight_decay = float(self.args.get("weight_decay", 0.0))
            milestones = self.args.get("milestones", [])
            gamma = float(self.args.get("lrate_decay", 1.0))

        fc_lr = float(self.args.get("fc_lrate", lr))
        encoder_params: List[nn.Parameter] = []
        cls_params: List[nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "classifier_pool." in name:
                cls_params.append(p)
            else:
                encoder_params.append(p)

        param_groups = []
        if len(encoder_params) > 0:
            param_groups.append({"params": encoder_params, "lr": lr, "weight_decay": weight_decay})
        if len(cls_params) > 0:
            param_groups.append({"params": cls_params, "lr": fc_lr, "weight_decay": weight_decay})

        if len(param_groups) == 0:
            raise RuntimeError("No trainable parameters found for EWCLoRA.")

        if self._optimizer_type == "adam":
            optimizer = torch.optim.Adam(param_groups)
        elif self._optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(param_groups)
        elif self._optimizer_type == "sgd":
            optimizer = torch.optim.SGD(param_groups, momentum=momentum)
        else:
            # Fallback for custom optimizers from BaseLearner; may ignore per-group fc lr.
            flat_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = self._build_optimizer(flat_params, stage=stage)

        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=milestones,
            gamma=gamma,
            T_max=epochs,
            eta_min=self.args.get("min_lr", 0.0),
        )
        return optimizer, scheduler, epochs

    def _freeze_network(self, model: nn.Module) -> None:
        target_suffix = f".{self._cur_task}"
        unfrozen_keys = [
            f"classifier_pool{target_suffix}",
            "lora_new_A_k",
            "lora_new_A_v",
            "lora_new_B_k",
            "lora_new_B_v",
        ]
        for name, param in model.named_parameters():
            param.requires_grad_(any(key in name for key in unfrozen_keys))

    def _compute_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        new_a_params = [p for p in model.parameters() if getattr(p, "_is_new_a", False)]
        new_b_params = [p for p in model.parameters() if getattr(p, "_is_new_b", False)]

        if len(new_a_params) == 0 or len(new_b_params) == 0 or len(self._omega_w) == 0:
            return torch.tensor(0.0, device=self._device)

        ewc_loss = None
        for idx, (p_a, p_b) in enumerate(zip(new_a_params, new_b_params)):
            if idx >= len(self._omega_w):
                break
            delta_w = p_b @ p_a
            omega = self._omega_w[idx].to(device=delta_w.device, dtype=delta_w.dtype)
            term = (omega * delta_w.pow(2)).sum()
            ewc_loss = term if ewc_loss is None else ewc_loss + term

        if ewc_loss is None:
            return torch.tensor(0.0, device=self._device)
        return ewc_loss

    # ------------------------------------------------------------------#
    # Fisher / omega update (legacy hook-based formulation)             #
    # ------------------------------------------------------------------#
    def _update_importance_matrix(self) -> None:
        if not hasattr(self, "train_loader") or self.train_loader is None:
            return

        if self._is_main_process:
            self._log("=== Update Importance Matrix ===")

        self._count_updates += 1
        model = self._unwrap_network()
        fisher = FisherComputer(
            model=model,
            dataloader=self.train_loader,
            target_offset=self._known_classes,
            criterion=F.cross_entropy,
            device=self._device,
            max_batches=self._fisher_max_batches,
        )
        fisher_w = fisher.compute()

        old_omega = [w.clone() for w in self._omega_w]
        self._omega_w = []
        for idx, fw in enumerate(fisher_w):
            if idx < len(old_omega):
                self._omega_w.append(self._gamma * old_omega[idx] + fw)
            else:
                self._omega_w.append(fw)

        # Merge current task LoRA into accumulated LoRA and reset "new" branch.
        if hasattr(model, "accumulate_and_reset_lora"):
            model.accumulate_and_reset_lora()

    # ------------------------------------------------------------------#
    # Evaluation helpers for Net(use_new=...) backbone                  #
    # ------------------------------------------------------------------#
    def _compute_accuracy(self, model, loader):
        net = self._unwrap_eval_model(model)
        net.eval()
        correct, total = 0, 0
        for batch in loader:
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs = inputs.to(self._device)
            with torch.no_grad():
                logits = net.interface(inputs, use_new=True)
            predicts = torch.max(logits, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        if total == 0:
            return 0.0
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        net = self._unwrap_eval_model(self._network)
        net.eval()
        y_pred, y_true = [], []
        for batch in loader:
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = net.interface(inputs, use_new=True)
            k = min(self.topk, outputs.shape[1])
            predicts = torch.topk(outputs, k=k, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _extract_vectors(self, loader):
        net = self._unwrap_eval_model(self._network)
        net.eval()
        vectors, targets = [], []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    _, _inputs, _targets = batch
                else:
                    _inputs, _targets = batch
                _targets_np = _targets.numpy()
                _vectors = tensor2numpy(net.extract_features(_inputs.to(self._device)))
                vectors.append(_vectors)
                targets.append(_targets_np)
        return np.concatenate(vectors), np.concatenate(targets)

    @staticmethod
    def _unwrap_eval_model(model: nn.Module) -> nn.Module:
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return model.module
        return model


class FisherComputer:
    def __init__(
        self,
        model: nn.Module,
        dataloader,
        target_offset: int,
        criterion,
        device=torch.device("cpu"),
        max_batches: Optional[int] = None,
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.target_offset = int(target_offset)
        self.criterion = criterion
        self.device = device
        self.max_batches = max_batches

        self.fisher_w: List[torch.Tensor] = []
        self._init_fisher_storage()

    def compute(self) -> List[torch.Tensor]:
        self.model.eval()
        num_samples = 0

        iterator = tqdm(self.dataloader, desc="Computing Fisher")
        for batch_idx, batch in enumerate(iterator):
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.model.zero_grad(set_to_none=True)
            self._clear_hook_grads()
            logits = self.model(inputs, use_new=True, register_hook=True)["logits"]
            targets = targets - self.target_offset
            loss = self.criterion(logits, targets)
            loss.backward()

            batch_size = int(inputs.size(0))
            num_samples += batch_size

            idx = 0
            for module in self.model.modules():
                if hasattr(module, "delta_w_k_new_grad"):
                    grad_k = module.delta_w_k_new_grad
                    if grad_k is not None and idx < len(self.fisher_w):
                        g2 = grad_k.detach().to(device=self.fisher_w[idx].device, dtype=self.fisher_w[idx].dtype).pow(2)
                        self.fisher_w[idx] += g2 * batch_size
                    idx += 1
                if hasattr(module, "delta_w_v_new_grad"):
                    grad_v = module.delta_w_v_new_grad
                    if grad_v is not None and idx < len(self.fisher_w):
                        g2 = grad_v.detach().to(device=self.fisher_w[idx].device, dtype=self.fisher_w[idx].dtype).pow(2)
                        self.fisher_w[idx] += g2 * batch_size
                    idx += 1

        if num_samples > 0:
            self.fisher_w = [fw / float(num_samples) for fw in self.fisher_w]
        return self.fisher_w

    def _clear_hook_grads(self) -> None:
        for module in self.model.modules():
            if hasattr(module, "delta_w_k_new_grad"):
                module.delta_w_k_new_grad = None
            if hasattr(module, "delta_w_v_new_grad"):
                module.delta_w_v_new_grad = None

    def _init_fisher_storage(self) -> None:
        for module in self.model.modules():
            if hasattr(module, "lora_new_B_k") and hasattr(module, "lora_new_A_k"):
                delta_w_k_new = module.lora_new_B_k.weight @ module.lora_new_A_k.weight
                self.fisher_w.append(torch.zeros_like(delta_w_k_new, device="cpu"))
            if hasattr(module, "lora_new_B_v") and hasattr(module, "lora_new_A_v"):
                delta_w_v_new = module.lora_new_B_v.weight @ module.lora_new_A_v.weight
                self.fisher_w.append(torch.zeros_like(delta_w_v_new, device="cpu"))


# Backward compatibility for old imports.
class EWCLoRA(Learner):
    pass
