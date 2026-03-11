"""GeoLoRA (orthogonalized LoRA) implementation aligned with relatedPaper/GeoLoRA.

Key points:
- Base ViT weights are frozen.
- Each attention qkv linear gets a low-rank update ΔW = B diag(softplus(s)+eps) A^T
  with A, B column-orthonormal (Stiefel). Orthogonalization is performed on-the-fly
  via QR retraction of unconstrained Theta_A, Theta_B.
- Applies updates to Q and V only (mirroring common LoRA practice), K untouched.
- Scaling factor alpha/r follows LoRA convention.

This learner reuses the SeqLoRA training loop but swaps the backbone builder.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


def _orthonormal(mat: torch.Tensor) -> torch.Tensor:
    # mat: [d, r]
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q


class GeoLoRA_QKV(nn.Module):
    def __init__(self, qkv: nn.Linear, r: int, alpha: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.qkv = qkv
        dim = qkv.in_features
        self.r = r
        self.alpha = alpha
        self.eps = eps

        # unconstrained params
        self.theta_a_q = nn.Parameter(torch.randn(dim, r) * 0.02)
        self.theta_b_q = nn.Parameter(torch.randn(dim, r) * 0.02)
        self.s_q = nn.Parameter(torch.zeros(r))

        self.theta_a_v = nn.Parameter(torch.randn(dim, r) * 0.02)
        self.theta_b_v = nn.Parameter(torch.randn(dim, r) * 0.02)
        self.s_v = nn.Parameter(torch.zeros(r))

        # freeze base
        for p in self.qkv.parameters():
            p.requires_grad = False

    def _delta(self, x: torch.Tensor, theta_a: torch.Tensor, theta_b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: [B, N, dim]
        A = _orthonormal(theta_a)  # [dim, r]
        B = _orthonormal(theta_b)  # [dim, r]
        sigma = F.softplus(s) + self.eps  # [r]
        scale = self.alpha / float(self.r)
        # (x @ A) -> [B,N,r]; * sigma; @ B^T -> [B,N,dim]
        u = torch.matmul(x, A) * sigma
        delta = torch.matmul(u, B.t()) * scale
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)  # [B,N,3*dim]
        B, N, threeD = qkv.shape
        dim = threeD // 3
        q, k, v = qkv.chunk(3, dim=2)
        q = q + self._delta(x, self.theta_a_q, self.theta_b_q, self.s_q)
        v = v + self._delta(x, self.theta_a_v, self.theta_b_v, self.s_v)
        return torch.cat([q, k, v], dim=2)


class GeoLoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: nn.Module, r: int, alpha: float = 1.0):
        super().__init__()
        self.vit = vit_model
        self.r = r
        self.alpha = alpha
        # freeze backbone
        for p in self.vit.parameters():
            p.requires_grad = False
        self._inject_geolora()
        self.out_dim = getattr(self.vit, "num_features", 768)

    def _inject_geolora(self):
        for blk in self.vit.blocks:
            qkv = blk.attn.qkv
            geolora_qkv = GeoLoRA_QKV(qkv, r=self.r, alpha=self.alpha)
            blk.attn.qkv = geolora_qkv

    def forward(self, x):
        return self.vit(x)

    def extract_features(self, x):
        # follow timm ViT: forward_features
        return self.vit.forward_features(x)


class Learner(LoraBaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._lora_initialized = True
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        self._geolora_rank = int(args.get("geolora_rank", args.get("lora_rank", 8)))
        self._geolora_alpha = float(args.get("geolora_alpha", 1.0))

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def build_lora_backbone(self, index=True, eval_mode=False):
        vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        model = GeoLoRA_ViT_timm(vit_model=vit.eval(), r=self._geolora_rank, alpha=self._geolora_alpha)
        model.out_dim = 768
        return model

    def incremental_train(self, data_manager):
        self._refresh_distributed_context()
        self.data_manager = data_manager

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._log("Learning on {}-{}".format(self._known_classes, self._total_classes))

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
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args.get("train_num_workers", 8)
        )

        self._train(self.train_loader, self.test_loader)
        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._unwrap_network()

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as _nme_exc:  # pylint: disable=broad-except
            self._log(f"[GeoLoRA][NME] Failed to compute class means: {_nme_exc}")

    def _train(self, train_loader, test_loader):
        network = self._unwrap_network()
        if not self._lora_initialized:
            network.backbone = self.build_lora_backbone()
            network.backbone.to(self._device)
            self._lora_initialized = True
        self._network = network
        self._prepare_network()

        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage="init" if self._cur_task == 0 else "update")
        lr = optimizer.param_groups[0]["lr"]
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("milestones", []),
            gamma=float(self.args.get("lrate_decay", 1.0)),
            T_max=self.args.get("epochs", None),
            eta_min=self.args.get("min_lr", 0.1 * lr),
        )
        epochs = int(self.args.get("init_epoch" if self._cur_task == 0 else "epochs", 1))

        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                optimizer.zero_grad()
                logits = self._network(inputs)["logits"]
                if self._cur_task == 0:
                    loss = F.cross_entropy(logits, targets)
                    eval_logits, eval_targets = logits, targets
                else:
                    fake_targets = targets - self._known_classes
                    loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets

                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(eval_logits, dim=1)
                correct += preds.eq(eval_targets).cpu().sum()
                total += len(eval_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)

