"""Synaptic Intelligence (SI) on the full LoRA delta-W space.

Tracks path-integral importance for each LoRA delta matrix (B@A) instead of
individual A/B parameters. The workflow mirrors SI_AB_LoRA but:
  - small_omega / big_omega are stored per (layer, channel) delta_W.
  - Gradients are accumulated in delta_W space using qkv hooks.
  - SI penalty is applied on delta_W then mapped to A/B grads via chain rule.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.lora import LoRA_ViT_timm, _LoRA_qkv_timm_train
from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(LoraBaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._lora_initialized = False
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # SI hyperparameters
        self._si_c = float(args.get("si_c", 1.0))
        self._si_xi = float(args.get("si_xi", 0.1))

        # delta-W SI buffers (CPU)
        self._dw_big_omega: Dict[str, torch.Tensor] | None = None
        self._dw_small_omega: Dict[str, torch.Tensor] = {}
        self._dw_checkpoint: Dict[str, torch.Tensor] | None = None

        # hooks for capturing per-step g_{ΔW}
        self._hook_handles: list = []
        self._hook_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._lora_qkv_modules: list[_LoRA_qkv_timm_train] = []
        self._hook_lora_model = None

    # ------------------------------------------------------------------#
    # Backbone builder                                                  #
    # ------------------------------------------------------------------#
    def build_lora_backbone(self, index=True, eval_mode=False):
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        rank = int(self.args.get("lora_rank", 10))
        if rank <= 0:
            raise ValueError(f"lora_rank must be > 0, got {rank}")
        model = LoRA_ViT_timm(
            vit_model=model.eval(),
            r=rank,
            num_classes=0,
            index=index,
            increment=self.args["increment"],
            filepath=self.args.get("filepath", "./"),
            cur_task_index=0,
            learn_alpha=False,
        )
        model.out_dim = 768
        return model

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------#
    # Training loop                                                     #
    # ------------------------------------------------------------------#
    def incremental_train(self, data_manager):
        self._refresh_distributed_context()

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
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

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args.get("train_num_workers", 8),
        )

        network = self._unwrap_network()
        if not self._lora_initialized:
            network.backbone = self.build_lora_backbone()
            network.backbone.to(self._device)
            self._lora_initialized = True
        self._network = network
        self._prepare_network()

        self._ensure_lora_hooks()

        params = [p for p in self._network.parameters() if p.requires_grad]
        stage = "init" if self._cur_task == 0 else "update"
        optimizer = self._build_optimizer(params, stage=stage)

        if stage == "init":
            epochs = int(self.args.get("init_epoch", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )
        else:
            epochs = int(self.args.get("epochs", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )

        # init delta-W SI buffers for this task
        self._init_dw_buffers()

        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        for epoch in prog_bar:
            self._network.train()
            total_loss = 0.0
            correct, total = 0, 0

            for batch in self.train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                pre_dw = self._current_delta_w(cpu=True)

                optimizer.zero_grad()
                outputs = self._network(inputs)
                logits = outputs["logits"]

                if stage == "init":
                    loss_task = F.cross_entropy(logits, targets)
                    eval_logits, eval_targets = logits, targets
                else:
                    fake_targets = targets - self._known_classes
                    loss_task = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets

                loss_task.backward()

                # compute g_task over delta_W from cached hooks
                g_task_dw = self._collect_g_dw()

                # add SI penalty in delta_W space, push grads to A/B
                g_reg_dw: Dict[str, torch.Tensor] = {}
                if self._dw_big_omega is not None and self._dw_checkpoint is not None:
                    for key, A, B, _, _ in self._iter_lora_pairs():
                        dW = self._delta_w(A, B)
                        ref = self._dw_checkpoint[key].to(dW.device, dW.dtype)
                        omega = self._dw_big_omega.get(key)
                        if omega is None:
                            continue
                        omega_dev = omega.to(dW.device, dW.dtype)
                        g_reg = (2.0 * self._si_c) * omega_dev * (dW - ref)
                        g_reg_dw[key] = g_reg.detach().cpu()

                        # chain rule: grad_A = B^T g, grad_B = g A^T
                        grad_A = B.weight.t().matmul(g_reg)
                        grad_B = g_reg.matmul(A.weight.t())

                        if A.weight.grad is None:
                            A.weight.grad = grad_A
                        else:
                            A.weight.grad.add_(grad_A)
                        if B.weight.grad is None:
                            B.weight.grad = grad_B
                        else:
                            B.weight.grad.add_(grad_B)

                optimizer.step()

                with torch.no_grad():
                    post_dw = self._current_delta_w(cpu=True)
                    for key in pre_dw:
                        delta = pre_dw[key] - post_dw[key]
                        g_total = g_task_dw.get(key)
                        if g_total is None:
                            g_total = torch.zeros_like(delta)
                        if key in g_reg_dw:
                            g_total = g_total + g_reg_dw[key]
                        self._dw_small_omega[key].add_(g_total * delta)

                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss_task.detach().item())

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => Loss {total_loss/len(self.train_loader):.3f}, Train_accy {train_acc:.2f}"
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, self.test_loader)
                info += f", Test_accy {test_acc:.2f}"
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)

        # end-of-task update big_omega in delta_W space
        assert self._dw_checkpoint is not None
        if self._dw_big_omega is None:
            self._dw_big_omega = {}
        for key, A, B, _, _ in self._iter_lora_pairs():
            ref = self._dw_checkpoint.get(key)
            if ref is None:
                continue
            dW = self._delta_w(A, B).detach().cpu()
            denom = (dW - ref).pow(2) + self._si_xi
            if key not in self._dw_big_omega:
                self._dw_big_omega[key] = torch.zeros_like(dW)
            self._dw_big_omega[key].add_(self._dw_small_omega[key] / denom)

        # refresh checkpoint for next task
        self._dw_checkpoint = {k: self._delta_w(A, B).detach().cpu().clone() for k, A, B, _, _ in self._iter_lora_pairs()}
        self._dw_small_omega = {k: torch.zeros_like(v, device="cpu") for k, v in self._dw_checkpoint.items()}

    # ------------------------------------------------------------------#
    # Hooks & helpers                                                   #
    # ------------------------------------------------------------------#
    def _ensure_lora_hooks(self) -> None:
        lora_model = getattr(self._unwrap_network(), "backbone", None)
        if lora_model is None:
            return
        if self._hook_lora_model is lora_model and self._hook_handles:
            return
        self._clear_hooks()
        self._hook_lora_model = lora_model
        self._hook_cache = {}
        self._lora_qkv_modules = []

        for _, module in lora_model.named_modules():
            if isinstance(module, _LoRA_qkv_timm_train):
                mid = id(module)
                self._hook_cache[mid] = {}

                def _fwd_hook(mod, inputs, output, mid=mid):
                    self._hook_cache[mid]["x"] = inputs[0].detach()

                def _bwd_hook(mod, grad_inputs, grad_outputs, mid=mid):
                    if grad_outputs and grad_outputs[0] is not None:
                        self._hook_cache[mid]["grad_out"] = grad_outputs[0].detach()

                self._hook_handles.append(module.register_forward_hook(_fwd_hook))
                self._hook_handles.append(module.register_full_backward_hook(_bwd_hook))
                self._lora_qkv_modules.append(module)

    def _clear_hooks(self) -> None:
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []
        self._hook_cache = {}
        self._lora_qkv_modules = []
        self._hook_lora_model = None

    def _iter_lora_pairs(self) -> Iterable[Tuple[str, nn.Linear, nn.Linear, int, str]]:
        backbone = getattr(self._unwrap_network(), "backbone", None)
        if backbone is None or not hasattr(backbone, "idx_by_layer_channel"):
            return []
        for (layer_idx, channel), idx in backbone.idx_by_layer_channel.items():
            key = f"layer{layer_idx}_{channel}"
            yield key, backbone.w_As[idx], backbone.w_Bs[idx], layer_idx, channel

    def _delta_w(self, A: nn.Linear, B: nn.Linear) -> torch.Tensor:
        return B.weight @ A.weight

    def _current_delta_w(self, cpu: bool = False) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, A, B, _, _ in self._iter_lora_pairs():
            dW = self._delta_w(A, B).detach()
            out[key] = dW.cpu() if cpu else dW
        return out

    def _collect_g_dw(self) -> Dict[str, torch.Tensor]:
        g_dict: Dict[str, torch.Tensor] = {}
        for module in self._lora_qkv_modules:
            cache = self._hook_cache.get(id(module), {})
            x = cache.get("x")
            grad_out = cache.get("grad_out")
            if x is None or grad_out is None:
                continue
            dim = int(module.dim)
            x_flat = x.reshape(-1, dim)
            go = grad_out.reshape(-1, grad_out.shape[-1])

            grad_q = go[:, :dim]
            grad_v = go[:, -dim:]

            g_q = grad_q.t().matmul(x_flat)
            g_v = grad_v.t().matmul(x_flat)

            key_q = f"layer{module.t_layer_i}_q"
            key_v = f"layer{module.t_layer_i}_v"
            g_dict[key_q] = g_q.detach().cpu()
            g_dict[key_v] = g_v.detach().cpu()

        # clear cache after use to avoid stale data
        self._hook_cache = {k: {} for k in self._hook_cache}
        return g_dict

    def _init_dw_buffers(self) -> None:
        ckpt = {}
        small = {}
        for key, A, B, _, _ in self._iter_lora_pairs():
            dW = self._delta_w(A, B).detach().cpu()
            ckpt[key] = dW.clone()
            small[key] = torch.zeros_like(dW, device="cpu")
        self._dw_checkpoint = ckpt
        self._dw_small_omega = small
        if self._dw_big_omega is None:
            self._dw_big_omega = {k: torch.zeros_like(v, device="cpu") for k, v in ckpt.items()}

