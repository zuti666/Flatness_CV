"""InfoGAM-LoRA for class-incremental learning.

Method summary
--------------
This learner extends SeqLoRA with a head-free task-information subspace:

1) Build task subspace U_t from whitened feature-label cross covariance.
2) During training, regularize feature updates to:
   - align with U_t (small residual outside U_t),
   - avoid historical subspace U_<t (small projection on U_<t).
3) Apply the same objective under SGD/SAM/GAM updates.

The implementation is designed to reuse the existing LoRA CL pipeline and
optimizer interfaces with minimal code changes.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from models_LoRAbasedCL.seqlora import Learner as SeqLoRALearner
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(SeqLoRALearner):
    """SeqLoRA + head-free subspace guided training objective."""

    def __init__(self, args):
        super().__init__(args)
        # Core InfoGAM toggles
        self._infogam_enable = bool(args.get("infogam_enable", True))
        self._infogam_lambda_align = float(args.get("infogam_lambda_align", 0.5))
        self._infogam_lambda_protect = float(args.get("infogam_lambda_protect", 0.5))
        self._infogam_topk = int(args.get("infogam_topk", max(1, int(args.get("lora_rank", 4)))))
        self._infogam_max_batches = int(args.get("infogam_max_batches", 8))
        self._infogam_tau_h = float(args.get("infogam_tau_h", 1e-4))
        self._infogam_tau_y = float(args.get("infogam_tau_y", 1e-6))
        self._infogam_eps = float(args.get("infogam_eps", 1e-12))
        self._infogam_basis_cap = int(args.get("infogam_basis_cap", 256))
        self._infogam_use_random_subspace = bool(args.get("infogam_use_random_subspace", False))
        self._infogam_save_json = bool(args.get("infogam_save_json", True))

        # Runtime state
        self._teacher_net = None
        self._U_task: Optional[torch.Tensor] = None   # [D, k] on CPU
        self._U_hist: Optional[torch.Tensor] = None   # [D, k_hist] on CPU

    # ------------------------------------------------------------------
    # Subspace estimation (head-free)
    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap_batch(batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                _, inputs, targets = batch
                return inputs, targets
            if len(batch) == 2:
                return batch
        raise ValueError("Unexpected batch format for InfoGAM-LoRA.")

    @staticmethod
    def _extract_features(outputs):
        feat = outputs["features"] if isinstance(outputs, dict) and "features" in outputs else outputs
        if feat.dim() == 3:
            feat = feat[:, 0, ...]  # CLS token
        elif feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        return feat.float()

    @torch.no_grad()
    def _snapshot_teacher(self):
        """Freeze a teacher snapshot before each task update."""
        base = self._unwrap_network()
        self._teacher_net = copy.deepcopy(base).to(self._device).eval()
        for p in self._teacher_net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _estimate_task_subspace(self, loader, known_classes: int, total_classes: int) -> Optional[torch.Tensor]:
        """Estimate U_t from whitened cross-covariance M_t = Σ_h^-1/2 C_hy Σ_y^-1/2."""
        if loader is None:
            return None

        module = self._unwrap_network()
        was_training = module.training
        module.eval()
        try:
            feat_dim = int(self.feature_dim)
            c_task = max(1, int(total_classes - known_classes))

            sum_h = torch.zeros(feat_dim, device=self._device, dtype=torch.float64)
            sum_h2 = torch.zeros(feat_dim, device=self._device, dtype=torch.float64)
            sum_y = torch.zeros(c_task, device=self._device, dtype=torch.float64)
            sum_hy = torch.zeros((feat_dim, c_task), device=self._device, dtype=torch.float64)
            n_total = 0

            max_batches = int(self._infogam_max_batches)
            for bidx, batch in enumerate(loader):
                if max_batches is not None and bidx >= max_batches:
                    break
                inputs, targets = self._unwrap_batch(batch)
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                y_local = targets - int(known_classes)
                valid = (y_local >= 0) & (y_local < c_task)
                if not valid.any():
                    continue
                inputs = inputs[valid]
                y_local = y_local[valid]

                outputs = module(inputs)
                h = self._extract_features(outputs).to(dtype=torch.float64)
                onehot = F.one_hot(y_local.long(), num_classes=c_task).to(dtype=torch.float64)

                sum_h += h.sum(dim=0)
                sum_h2 += (h * h).sum(dim=0)
                sum_y += onehot.sum(dim=0)
                sum_hy += h.transpose(0, 1) @ onehot
                n_total += int(h.size(0))

            if n_total <= 1:
                return None

            n = float(n_total)
            mu_h = sum_h / n
            p_y = sum_y / n
            c_hy = (sum_hy / n) - mu_h.unsqueeze(1) * p_y.unsqueeze(0)
            var_h = torch.clamp((sum_h2 / n) - mu_h * mu_h, min=0.0)

            inv_sqrt_h = torch.rsqrt(var_h + float(self._infogam_tau_h))
            inv_sqrt_y = torch.rsqrt(p_y + float(self._infogam_tau_y))
            m = (inv_sqrt_h.unsqueeze(1) * c_hy) * inv_sqrt_y.unsqueeze(0)  # [D, C_t]

            # U_t from SVD of M_t (equivalent to eig of G_t = M_t M_t^T)
            u, s, _vh = torch.linalg.svd(m.to(dtype=torch.float32), full_matrices=False)
            if u.numel() == 0:
                return None
            k = max(1, min(int(self._infogam_topk), int(u.shape[1])))
            u_t = u[:, :k].contiguous()  # [D, k]

            if self._infogam_use_random_subspace:
                rand = torch.randn_like(u_t)
                q, _ = torch.linalg.qr(rand, mode="reduced")
                u_t = q[:, :k].contiguous()

            return u_t.detach().cpu()
        finally:
            module.train(was_training)

    @staticmethod
    def _merge_orth_basis(prev: Optional[torch.Tensor], cur: Optional[torch.Tensor], cap: int) -> Optional[torch.Tensor]:
        if cur is None:
            return prev
        merged = cur if prev is None else torch.cat([prev, cur], dim=1)
        q, _ = torch.linalg.qr(merged, mode="reduced")
        cap = max(1, min(int(cap), int(q.shape[1])))
        return q[:, :cap].contiguous()

    # ------------------------------------------------------------------
    # Loss terms
    # ------------------------------------------------------------------
    def _class_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self._known_classes > 0:
            fake_targets = targets - self._known_classes
            return F.cross_entropy(logits[:, self._known_classes :], fake_targets)
        return F.cross_entropy(logits, targets)

    def _composite_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Return total loss and detached diagnostics."""
        outputs = self._network(inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        h_new = self._extract_features(outputs)

        loss_task = self._class_loss(logits, targets)
        loss_align = torch.zeros((), device=logits.device, dtype=logits.dtype)
        loss_protect = torch.zeros((), device=logits.device, dtype=logits.dtype)

        if self._infogam_enable and self._teacher_net is not None and self._U_task is not None:
            with torch.no_grad():
                out_old = self._teacher_net(inputs)
                h_old = self._extract_features(out_old).to(device=h_new.device, dtype=h_new.dtype)
            delta = h_new - h_old
            denom = delta.pow(2).sum(dim=1).mean() + float(self._infogam_eps)

            u_t = self._U_task.to(device=delta.device, dtype=delta.dtype)
            proj_t = delta @ u_t
            recon_t = proj_t @ u_t.transpose(0, 1)
            residual = delta - recon_t
            loss_align = residual.pow(2).sum(dim=1).mean() / denom

            if self._U_hist is not None:
                u_h = self._U_hist.to(device=delta.device, dtype=delta.dtype)
                proj_h = delta @ u_h
                loss_protect = proj_h.pow(2).sum(dim=1).mean() / denom

        total_loss = (
            loss_task
            + float(self._infogam_lambda_align) * loss_align
            + float(self._infogam_lambda_protect) * loss_protect
        )
        diag = {
            "task": float(loss_task.detach().item()),
            "align": float(loss_align.detach().item()),
            "protect": float(loss_protect.detach().item()),
            "total": float(total_loss.detach().item()),
        }
        return total_loss, logits, diag

    # ------------------------------------------------------------------
    # Train loops
    # ------------------------------------------------------------------
    def _run_info_epochs(self, train_loader, test_loader, optimizer, scheduler, epochs: int):
        prog_bar = tqdm(range(int(epochs)), disable=not self._is_main_process)
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            losses_task = 0.0
            losses_align = 0.0
            losses_protect = 0.0
            correct, total = 0, 0

            for _i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                if self._optimizer_type == "gam":
                    def closure():
                        optimizer.zero_grad()
                        loss, logits, diag = self._composite_loss(inputs, targets)
                        loss.backward()
                        return {"logits": logits.detach()}, loss.detach()

                    outputs, loss_value = optimizer.step(closure=closure)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    with torch.no_grad():
                        _loss, _logits, diag = self._composite_loss(inputs, targets)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    losses_task += float(diag["task"])
                    losses_align += float(diag["align"])
                    losses_protect += float(diag["protect"])

                elif self._optimizer_type == "sam":
                    optimizer.zero_grad()
                    loss, logits, diag = self._composite_loss(inputs, targets)
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    second_loss, logits2, diag2 = self._composite_loss(inputs, targets)
                    second_loss.backward()
                    optimizer.second_step(zero_grad=True)
                    logits = logits2.detach()
                    losses += float(second_loss.detach().item())
                    losses_task += float(diag2["task"])
                    losses_align += float(diag2["align"])
                    losses_protect += float(diag2["protect"])

                else:  # sgd/adam/others
                    optimizer.zero_grad()
                    loss, logits, diag = self._composite_loss(inputs, targets)
                    loss.backward()
                    optimizer.step()
                    logits = logits.detach()
                    losses += float(loss.detach().item())
                    losses_task += float(diag["task"])
                    losses_align += float(diag["align"])
                    losses_protect += float(diag["protect"])

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / len(train_loader):.3f} "
                    f"(task {losses_task / len(train_loader):.3f}, "
                    f"align {losses_align / len(train_loader):.3f}, "
                    f"protect {losses_protect / len(train_loader):.3f}), "
                    f"Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}"
                )
            elif self._is_main_process:
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / len(train_loader):.3f} "
                    f"(task {losses_task / len(train_loader):.3f}, "
                    f"align {losses_align / len(train_loader):.3f}, "
                    f"protect {losses_protect / len(train_loader):.3f}), "
                    f"Train_accy {train_acc:.2f}"
                )
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)

    def _save_infogam_stats(self, task_id: int):
        if not self._infogam_save_json:
            return
        save_dir = self.args.get("filepath", "./")
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "infogam_stats.json")
        data = []
        if os.path.exists(out):
            try:
                with open(out, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append(
            {
                "task": int(task_id),
                "u_task_dim": 0 if self._U_task is None else int(self._U_task.shape[1]),
                "u_hist_dim": 0 if self._U_hist is None else int(self._U_hist.shape[1]),
            }
        )
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _train(self, train_loader, test_loader):
        """Inject InfoGAM setup then reuse SeqLoRA save behavior."""
        network = self._unwrap_network()

        if self._cur_task == 0:
            if not self._lora_initialized:
                network.backbone = self.build_lora_backbone()
                network.backbone.to(self._device)
                self._lora_initialized = True
            self._network = network
            self._prepare_network()
            stage = "init"
            epochs = int(self.args["init_epoch"])
        else:
            self._network = network
            self._prepare_network()
            stage = "update"
            epochs = int(self.args["epochs"])

        # Prepare teacher + U_t before each task training
        if self._infogam_enable:
            self._snapshot_teacher()
            self._U_task = self._estimate_task_subspace(train_loader, self._known_classes, self._total_classes)
            self._log(
                f"[InfoGAM] task={self._cur_task} "
                f"known={self._known_classes} total={self._total_classes} "
                f"U_t_dim={0 if self._U_task is None else self._U_task.shape[1]} "
                f"U_hist_dim={0 if self._U_hist is None else self._U_hist.shape[1]}"
            )

        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage=stage)
        lr = optimizer.param_groups[0]["lr"]
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("milestones", []),
            gamma=float(self.args.get("lrate_decay", 1.0)),
            T_max=self.args.get("epochs", None),
            eta_min=self.args.get("min_lr", 0.1 * lr),
        )
        if self._optimizer_type in {"arwp", "rwp"}:
            self._rwp_lr0 = float(lr)

        # Supported optimizers for InfoGAM objective; fallback otherwise.
        if self._optimizer_type in {"sgd", "sam", "gam"}:
            self._run_info_epochs(train_loader, test_loader, optimizer, scheduler, epochs=epochs)
        else:
            if self._cur_task == 0:
                super()._init_train(train_loader, test_loader, optimizer, scheduler)
            else:
                super()._update_representation(train_loader, test_loader, optimizer, scheduler)

        # Update historical protected basis after finishing the task
        self._U_hist = self._merge_orth_basis(self._U_hist, self._U_task, self._infogam_basis_cap)
        self._save_infogam_stats(self._cur_task)

        # Persist LoRA and FC (same behavior as SeqLoRA)
        save_dir = self.args.get("filepath", "./")
        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_dir, self._cur_task)

