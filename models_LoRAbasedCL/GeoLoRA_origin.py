"""Original OrthoGeoLoRA-style adapter for the CL framework.

This implementation keeps only the GEO core (no slow-merge/shared dictionary):
    - Delta update: ΔW = B diag(softplus(s) + eps) A^T
    - A, B are orthonormal-column factors obtained from unconstrained hidden
      parameters via PyTorch orthogonal parametrization (Householder map).
    - Optimizer updates only Euclidean hidden parameters (Theta_A, Theta_B, s).
"""

from __future__ import annotations

import logging

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


def _build_householder_orth_linear(out_dim: int, rank: int) -> nn.Linear:
    """Create a linear layer whose weight has orthonormal columns.

    Weight shape is [out_dim, rank], so for out_dim >= rank we enforce W^T W = I.
    The trainable parameter seen by the optimizer is the unconstrained
    `parametrizations.weight.original`.
    """
    if rank > out_dim:
        raise ValueError(f"rank ({rank}) must be <= out_dim ({out_dim}) for Stiefel columns.")

    layer = nn.Linear(rank, out_dim, bias=False)
    parametrizations.orthogonal(
        layer,
        name="weight",
        orthogonal_map="householder",
        use_trivialization=False,
    )
    nn.init.normal_(layer.parametrizations.weight.original, mean=0.0, std=1.0)
    return layer


class GeoLoRA_QKV(nn.Module):
    """GeoLoRA adapter for a timm ViT qkv projection.

    Applies updates to Q and V branches only:
        q <- q + alpha/r * B_q diag(sigma_q) A_q^T x
        v <- v + alpha/r * B_v diag(sigma_v) A_v^T x
    """

    def __init__(self, qkv: nn.Linear, r: int, alpha: float = 1.0, eps: float = 1e-6):
        super().__init__()
        if not isinstance(qkv, nn.Linear):
            raise TypeError(f"Expected nn.Linear qkv, got {type(qkv)}")
        if r <= 0:
            raise ValueError(f"GeoLoRA rank must be > 0, got {r}")
        if qkv.out_features % 3 != 0:
            raise ValueError(f"qkv.out_features must be divisible by 3, got {qkv.out_features}")

        in_dim = int(qkv.in_features)
        out_dim = int(qkv.out_features // 3)
        if r > min(in_dim, out_dim):
            raise ValueError(
                f"GeoLoRA rank r={r} must satisfy r <= min(in_dim={in_dim}, out_dim={out_dim})"
            )

        self.qkv = qkv
        self.r = int(r)
        self.alpha = float(alpha)
        self.eps = float(eps)

        # Hidden unconstrained factors; orthogonalized on-the-fly via parametrization map.
        self.theta_a_q = _build_householder_orth_linear(in_dim, self.r)
        self.theta_b_q = _build_householder_orth_linear(out_dim, self.r)
        self.theta_a_v = _build_householder_orth_linear(in_dim, self.r)
        self.theta_b_v = _build_householder_orth_linear(out_dim, self.r)

        # Singular-value logits (Euclidean parameters)
        self.s_q = nn.Parameter(torch.zeros(self.r))
        self.s_v = nn.Parameter(torch.zeros(self.r))

        # Freeze pretrained base projection
        for p in self.qkv.parameters():
            p.requires_grad = False

    def _delta(self, x: torch.Tensor, theta_a: nn.Linear, theta_b: nn.Linear, s: torch.Tensor) -> torch.Tensor:
        # x: [B, N, in_dim]
        A = theta_a.weight  # [in_dim, r], orthonormal columns
        B = theta_b.weight  # [out_dim, r], orthonormal columns
        sigma = F.softplus(s) + self.eps  # [r], non-negative

        # x -> A^T -> diag(sigma) -> B
        u = torch.matmul(x, A) * sigma
        delta = torch.matmul(u, B.t())
        return delta * (self.alpha / float(self.r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)  # [B, N, 3*out_dim]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self._delta(x, self.theta_a_q, self.theta_b_q, self.s_q)
        v = v + self._delta(x, self.theta_a_v, self.theta_b_v, self.s_v)
        return torch.cat([q, k, v], dim=-1)


class GeoLoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: nn.Module, r: int, alpha: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.vit = vit_model
        self.r = int(r)
        self.alpha = float(alpha)
        self.eps = float(eps)

        for p in self.vit.parameters():
            p.requires_grad = False

        self._inject_geolora()
        self.out_dim = getattr(self.vit, "num_features", 768)

    def _inject_geolora(self):
        for blk in self.vit.blocks:
            qkv = blk.attn.qkv
            blk.attn.qkv = GeoLoRA_QKV(qkv=qkv, r=self.r, alpha=self.alpha, eps=self.eps)

    def forward(self, x):
        return self.vit(x)


class Learner(LoraBaseLearner):
    """GEO-only learner aligned with OrthoGeoLoRA parameterization/training."""

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._lora_initialized = False

        # Paper-compatible default optimizer for Euclidean hidden parameters.
        self._optimizer_type = str(args.get("optimizer_type", "adamw")).lower()

        self._backbone_type = str(args.get("backbone_type", "vit_base_patch16_224"))
        self._geolora_rank = int(args.get("geolora_rank", args.get("lora_rank", 8)))
        self._geolora_alpha = float(args.get("geolora_alpha", 1.0))
        self._geolora_eps = float(args.get("geolora_eps", 1e-6))

        if self._geolora_rank <= 0:
            raise ValueError(f"geolora_rank must be > 0, got {self._geolora_rank}")

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def build_lora_backbone(self, index=True, eval_mode=False):  # pylint: disable=unused-argument
        backbone_name = self._backbone_type.lower()
        if backbone_name in {"pretrained_vit_b16_224", "vit_base_patch16_224"}:
            timm_name = "vit_base_patch16_224"
        elif backbone_name in {"pretrained_vit_b16_224_in21k", "vit_base_patch16_224_in21k"}:
            timm_name = "vit_base_patch16_224_in21k"
        else:
            timm_name = self._backbone_type

        vit = timm.create_model(timm_name, pretrained=True, num_classes=0)
        model = GeoLoRA_ViT_timm(
            vit_model=vit.eval(),
            r=self._geolora_rank,
            alpha=self._geolora_alpha,
            eps=self._geolora_eps,
        )
        model.out_dim = getattr(model, "out_dim", 768)
        return model

    def _build_eval_backbone(self, task_idx: int):  # pylint: disable=unused-argument
        return self.build_lora_backbone(eval_mode=True)

    def incremental_train(self, data_manager):
        self._refresh_distributed_context()
        self.data_manager = data_manager

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

        self._train(self.train_loader, self.test_loader)
        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._unwrap_network()

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            self._log(f"[GeoLoRA_origin][NME] Failed to compute class means: {exc}")

    def _train(self, train_loader, test_loader):
        network = self._unwrap_network()
        if not self._lora_initialized:
            network.backbone = self.build_lora_backbone()
            network.backbone.to(self._device)
            self._lora_initialized = True
        self._network = network
        self._prepare_network()

        stage = "init" if self._cur_task == 0 else "update"
        lr, _, weight_decay = self._resolve_optimizer_hyper(stage)
        params = [p for p in self._network.parameters() if p.requires_grad]

        if self._optimizer_type == "adam":
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif self._optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif self._optimizer_type == "sgd":
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
        else:
            raise ValueError(
                f"GeoLoRA_origin supports optimizer_type in {{adam, adamw, sgd}}, got {self._optimizer_type}"
            )

        epochs = int(self.args.get("init_epoch" if stage == "init" else "epochs", 1))
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("init_milestones" if stage == "init" else "milestones", []),
            gamma=float(self.args.get("init_lrate_decay" if stage == "init" else "lrate_decay", 1.0)),
            T_max=epochs,
            eta_min=self.args.get("min_lr", 0.0),
        )

        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                optimizer.zero_grad()
                logits = self._network(inputs)["logits"]

                if stage == "init":
                    loss = F.cross_entropy(logits, targets)
                    eval_logits, eval_targets = logits, targets
                else:
                    fake_targets = targets - self._known_classes
                    eval_logits = logits[:, self._known_classes :]
                    eval_targets = fake_targets
                    loss = F.cross_entropy(eval_logits, eval_targets)

                loss.backward()
                optimizer.step()
                losses += float(loss.item())

                preds = torch.max(eval_logits, dim=1)[1]
                correct += preds.eq(eval_targets).cpu().sum()
                total += len(eval_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = (
                f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                f"Loss {losses / len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            )
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            if self._is_main_process:
                prog_bar.set_description(info)

        if self._is_main_process:
            self._log(info)
