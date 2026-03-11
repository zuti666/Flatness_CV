"""FO_SO_LoRA: OGD hard projection + EWC soft penalty for LoRA-only training."""

from __future__ import annotations

import logging
from typing import Dict, List

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


class _ParamBasis:
    def __init__(self, max_rank: int, eps: float = 1e-8):
        self.max_rank = int(max_rank)
        self.eps = float(eps)
        self._basis: torch.Tensor | None = None  # [k, D] on CPU

    def add(self, vec: torch.Tensor) -> None:
        v = vec.detach().float().cpu()
        if v.numel() == 0:
            return
        n = v.norm()
        if n < self.eps:
            return
        v = v / n
        if self._basis is None:
            self._basis = v.unsqueeze(0)
            return
        B = self._basis
        proj = torch.mv(B, v)
        v = v - torch.mv(B.t(), proj)
        n = v.norm()
        if n < self.eps:
            return
        v = v / n
        if B.shape[0] >= self.max_rank:
            B = torch.cat([B[1:], v.unsqueeze(0)], dim=0)
        else:
            B = torch.cat([B, v.unsqueeze(0)], dim=0)
        self._basis = B

    def project(self, grad: torch.Tensor) -> torch.Tensor:
        if self._basis is None or self._basis.numel() == 0:
            return grad
        B = self._basis.to(device=grad.device, dtype=grad.dtype)
        coeff = torch.mv(B, grad)
        return grad - torch.mv(B.t(), coeff)

    @property
    def rank(self) -> int:
        return 0 if self._basis is None else int(self._basis.shape[0])


class Learner(LoraBaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._lora_initialized = False
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # OGD hyperparams
        self._ogd_rank_per_param = int(args.get("ogd_rank_per_param", 20))
        self._ogd_store_batches = int(args.get("ogd_store_batches", 50))
        self._ogd_eps = float(args.get("ogd_eps", 1e-8))

        # EWC hyperparams
        self._ewc_lambda = float(args.get("ewc_lambda", 20.0))
        self._ewc_gamma = float(args.get("ewc_gamma", 1.0))
        self._ewc_max_batches = int(args.get("ewc_max_batches", 100))
        self._ewc_eps = float(args.get("ewc_eps", 1e-5))

        # buffers (CPU)
        self._bases: Dict[str, _ParamBasis] = {}
        self._omega_w: Dict[str, torch.Tensor] | None = None
        self._count_updates = 0

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

        params = [p for p in self._network.parameters() if p.requires_grad]
        stage = "init" if self._cur_task == 0 else "update"
        optimizer = self._build_optimizer(params, stage=stage)
        if self._optimizer_type not in {"sgd", "adam", "adamw"}:
            raise ValueError(f"EWC_LoRA supports sgd/adam/adamw only, got {self._optimizer_type}")

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

                loss = loss_task + self._ewc_penalty(self._unwrap_network())
                loss.backward()
                self._project_gradients(self._unwrap_network())
                optimizer.step()

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss.detach().item())

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

        save_dir = self.args.get("filepath", "./")
        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_dir, self._cur_task)

        fisher_cur = self._compute_fisher(self.train_loader, self._unwrap_network())
        self._omega_w = self._accumulate_omega(self._omega_w, fisher_cur)
        if len(fisher_cur) > 0:
            self._count_updates += 1
        self._update_bases(self.train_loader)

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._unwrap_network()

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            self._log(f"[FO_SO_LoRA][NME] Failed to compute class means: {exc}")

    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    def _project_gradients(self, model: nn.Module):
        for name, p in self._iter_trainable(model):
            if p.grad is None:
                continue
            g = p.grad.view(-1)
            basis = self._bases.get(name)
            if basis is None:
                continue
            g_proj = basis.project(g.detach())
            p.grad.copy_(g_proj.view_as(p))

    def _ewc_penalty(self, model: nn.Module):
        if self._omega_w is None or len(self._omega_w) == 0:
            return torch.tensor(0.0, device=self._device)

        pen = None
        for target in self._collect_lora_targets(model):
            name = target["name"]
            omega = self._omega_w.get(name)
            if omega is None:
                continue

            # EWCLoRA (youyue_github) core penalty:
            #   (lambda/2) * sum_i omega_i * delta_W_i^2
            delta_w = self._delta_from_qkv(target["module"])
            omega_d = self._match_tensor(omega, delta_w).to(device=delta_w.device, dtype=delta_w.dtype)
            term = (omega_d * delta_w.pow(2)).sum()
            pen = term if pen is None else pen + term

        if pen is None:
            return torch.tensor(0.0, device=self._device)
        return 0.5 * self._ewc_lambda * pen

    def _compute_fisher(self, loader, model: nn.Module) -> Dict[str, torch.Tensor]:
        was_training = model.training
        model.eval()
        lora_targets = self._collect_lora_targets(model)
        if len(lora_targets) == 0:
            return {}

        fisher: Dict[str, torch.Tensor] = {}
        total_samples = 0
        try:
            for target in lora_targets:
                mod = target["module"]
                if hasattr(mod, "_register_delta_hook"):
                    mod._register_delta_hook = True
                if hasattr(mod, "delta_w_q_new_grad"):
                    mod.delta_w_q_new_grad = None
                if hasattr(mod, "delta_w_v_new_grad"):
                    mod.delta_w_v_new_grad = None

            for batch_idx, batch in enumerate(loader):
                if batch_idx >= self._ewc_max_batches:
                    break
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    logits = model(inputs)["logits"]
                    if self._cur_task == 0:
                        fisher_loss = F.cross_entropy(logits, targets)
                    else:
                        fake_targets = targets - self._known_classes
                        fisher_loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    fisher_loss.backward()

                bs = int(inputs.shape[0])
                total_samples += bs
                for target in lora_targets:
                    name = target["name"]
                    mod = target["module"]
                    dim = int(mod.dim)

                    if name not in fisher:
                        fisher[name] = torch.zeros((dim * 3, dim), dtype=torch.float32, device="cpu")

                    grad_q = getattr(mod, "delta_w_q_new_grad", None)
                    if grad_q is not None:
                        g2_q = grad_q.detach().to(device="cpu", dtype=fisher[name].dtype).pow(2)
                        fisher[name][:dim] += g2_q * bs

                    grad_v = getattr(mod, "delta_w_v_new_grad", None)
                    if grad_v is not None:
                        g2_v = grad_v.detach().to(device="cpu", dtype=fisher[name].dtype).pow(2)
                        fisher[name][-dim:] += g2_v * bs

                    if hasattr(mod, "delta_w_q_new_grad"):
                        mod.delta_w_q_new_grad = None
                    if hasattr(mod, "delta_w_v_new_grad"):
                        mod.delta_w_v_new_grad = None
        finally:
            for target in lora_targets:
                mod = target["module"]
                if hasattr(mod, "_register_delta_hook"):
                    mod._register_delta_hook = False
                if hasattr(mod, "delta_w_q_new_grad"):
                    mod.delta_w_q_new_grad = None
                if hasattr(mod, "delta_w_v_new_grad"):
                    mod.delta_w_v_new_grad = None

        if total_samples > 0:
            for name in fisher:
                fisher[name] = (fisher[name] / float(total_samples)).clamp_min(self._ewc_eps)
        if was_training:
            model.train()

        if self._is_main_process:
            used_batches = min(len(loader), self._ewc_max_batches)
            self._log(f"[FO_SO_LoRA] Fisher estimated over {used_batches} batches")
        return fisher

    def _collect_lora_targets(self, model: nn.Module) -> List[Dict]:
        targets: List[Dict] = []
        for mod_name, mod in model.named_modules():
            if not isinstance(mod, _LoRA_qkv_timm_train):
                continue
            targets.append({"name": f"{mod_name}.qkv.weight", "module": mod})
        return targets

    def _delta_from_qkv(self, mod: _LoRA_qkv_timm_train) -> torch.Tensor:
        dim = mod.dim
        device = mod.linear_a_q.weight.device
        dtype = mod.linear_a_q.weight.dtype
        delta_q = mod.linear_b_q.weight @ mod.linear_a_q.weight
        delta_v = mod.linear_b_v.weight @ mod.linear_a_v.weight
        delta = torch.zeros((dim * 3, dim), device=device, dtype=dtype)
        delta[:dim] = delta_q
        delta[-dim:] = delta_v
        return delta

    def _accumulate_omega(
        self,
        old: Dict[str, torch.Tensor] | None,
        new: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if old is None:
            return {k: v.clone() for k, v in new.items()}

        out: Dict[str, torch.Tensor] = {}
        keys = set(old.keys()) | set(new.keys())
        for k in keys:
            a = old.get(k)
            b = new.get(k)
            if a is None:
                out[k] = b.clone()
            elif b is None:
                out[k] = a * self._ewc_gamma
            else:
                out[k] = self._match_tensor(a, b) * self._ewc_gamma + b
        return out

    def _match_tensor(self, stored: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_cpu = target.detach().cpu()
        if stored.shape == target_cpu.shape:
            return stored
        new = torch.zeros_like(target_cpu)
        if stored.ndim != target_cpu.ndim:
            return new
        slices = tuple(slice(0, min(a, b)) for a, b in zip(stored.shape, target_cpu.shape))
        new[slices] = stored[slices]
        return new

    def _update_bases(self, loader) -> None:
        model = self._unwrap_network()
        model.eval()
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
                loss = F.cross_entropy(logits, targets)
                loss.backward()

            for name, p in self._iter_trainable(model):
                if p.grad is None:
                    continue
                vec = p.grad.detach().view(-1).cpu()
                basis = self._bases.get(name)
                if basis is None:
                    basis = _ParamBasis(self._ogd_rank_per_param, eps=self._ogd_eps)
                    self._bases[name] = basis
                basis.add(vec)
        if self._is_main_process:
            avg_rank = 0.0 if not self._bases else sum(b.rank for b in self._bases.values()) / len(self._bases)
            self._log(f"[FO_SO_LoRA] Stored bases for {len(self._bases)} params (avg rank {avg_rank:.2f})")
