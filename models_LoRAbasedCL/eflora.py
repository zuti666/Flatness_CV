"""EFLoRA: Effective-Subspace Aligned LoRA for class-incremental learning.

Core idea
---------
1) Estimate a task-sensitive feature subspace from a lightweight EFM proxy.
2) Initialize current LoRA A-matrices to align with that subspace.
3) Optionally project updates to the complement of previously accumulated
   effective subspaces to reduce interference.

This implementation intentionally reuses SeqLoRA's training flow so it can run
in the current pipeline without changing trainer logic.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_LoRAbasedCL.seqlora import Learner as SeqLoRALearner

logger = logging.getLogger(__name__)


class Learner(SeqLoRALearner):
    """SeqLoRA + effective-subspace alignment and complement projection."""

    def __init__(self, args):
        super().__init__(args)
        self._efalign_enable = bool(args.get("efalign_enable", True))
        self._efalign_topk = int(args.get("efalign_topk", max(1, int(args.get("lora_rank", 4)))))
        self._efalign_max_batches = int(args.get("efalign_max_batches", 4))
        self._efalign_eps = float(args.get("efalign_eps", 1e-12))
        self._efalign_init_task0 = bool(args.get("efalign_init_task0", True))
        self._efalign_init_updates = bool(args.get("efalign_init_updates", True))
        self._efalign_complement = bool(args.get("efalign_complement", True))
        self._efalign_basis_cap = int(args.get("efalign_basis_cap", 128))
        self._efalign_init_scale = float(args.get("efalign_init_scale", 1.0))
        self._efalign_save_json = bool(args.get("efalign_save_json", True))
        self._efalign_prev_basis: Optional[torch.Tensor] = None  # [D, K] on CPU

    # ------------------------------------------------------------------
    # EFM / effective-subspace helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap_batch(batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                _, inputs, targets = batch
                return inputs, targets
            if len(batch) == 2:
                return batch
        raise ValueError("Unexpected batch format for EFLoRA subspace estimation.")

    @staticmethod
    def _classifier_weight(module: nn.Module) -> torch.Tensor:
        head = getattr(module, "fc", None) or getattr(module, "classifier", None)
        if head is None:
            raise AttributeError("Classifier module not found (expected 'fc' or 'classifier').")
        if hasattr(head, "weight") and head.weight is not None:
            return head.weight
        if hasattr(head, "fc1") and hasattr(head, "fc2"):
            return torch.cat((head.fc1.weight, head.fc2.weight), dim=0)
        raise AttributeError("Unsupported classifier head type for EFLoRA.")

    def _estimate_effective_subspace(
        self,
        loader,
        *,
        topk: Optional[int] = None,
        max_batches: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Estimate a task-sensitive feature subspace using an EFM matrix."""
        module = self._unwrap_network()
        if module is None:
            return None, None

        was_training = module.training
        module.eval()
        try:
            head_weight = self._classifier_weight(module).detach()
            if head_weight.ndim != 2:
                return None, None
            device = next(module.parameters()).device
            head_weight = head_weight.to(device=device, dtype=torch.float32)

            d = head_weight.shape[1]
            efm = torch.zeros((d, d), device=device, dtype=torch.float64)
            total = 0

            max_batches = self._efalign_max_batches if max_batches is None else int(max_batches)
            eps = float(self._efalign_eps)

            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    if max_batches is not None and batch_idx >= max_batches:
                        break
                    inputs, _targets = self._unwrap_batch(batch)
                    inputs = inputs.to(device, non_blocking=True)
                    out = module(inputs)
                    if not isinstance(out, dict) or "logits" not in out:
                        continue
                    logits = out["logits"].to(device=device, dtype=torch.float32)
                    probs = torch.clamp(F.softmax(logits, dim=1), min=eps)
                    bs = logits.size(0)

                    weight_expanded = head_weight.unsqueeze(0).expand(bs, -1, -1)
                    weighted_mean = torch.matmul(probs, head_weight)
                    centered = weight_expanded - weighted_mean.unsqueeze(1)
                    weighted_centered = centered * torch.sqrt(probs).unsqueeze(2)
                    efm_batch = torch.matmul(weighted_centered.transpose(1, 2), weighted_centered)
                    efm += efm_batch.sum(dim=0).double()
                    total += bs

            if total == 0:
                return None, None

            efm = efm / float(total)
            efm = 0.5 * (efm + efm.transpose(0, 1))
            evals, evecs = torch.linalg.eigh(efm)  # ascending
            evals = torch.clamp(evals, min=0.0)

            topk = int(topk if topk is not None else self._efalign_topk)
            topk = max(1, min(topk, evecs.shape[1]))
            basis = evecs[:, -topk:].to(dtype=torch.float32)  # [D, K]
            eigvals = evals[-topk:].to(dtype=torch.float32)
            return basis, eigvals
        finally:
            module.train(was_training)

    def _collect_trainable_lora_A(self) -> List[torch.nn.Parameter]:
        """Collect trainable LoRA A matrices from the current network."""
        params = []
        module = self._unwrap_network()
        for name, p in module.named_parameters():
            if not p.requires_grad or p.ndim != 2:
                continue
            if "linear_a" in name:
                params.append(p)
        return params

    def _init_lora_with_basis(self, basis: torch.Tensor) -> None:
        """Initialize LoRA A rows from effective basis directions."""
        params = self._collect_trainable_lora_A()
        if not params:
            return
        for p in params:
            r, d = p.shape
            if basis.shape[0] != d:
                continue
            k = min(r, basis.shape[1])
            init = torch.randn((r, d), device=p.device, dtype=p.dtype) * 0.01
            init[:k, :] = basis[:, -k:].T.to(device=p.device, dtype=p.dtype)
            init = F.normalize(init, dim=1) * float(self._efalign_init_scale)
            with torch.no_grad():
                p.copy_(init)

    def _project_lora_to_complement(self, prev_basis: torch.Tensor) -> None:
        """Project LoRA A rows to the complement of previous effective basis."""
        params = self._collect_trainable_lora_A()
        if not params:
            return
        for p in params:
            if prev_basis.shape[0] != p.shape[1]:
                continue
            U = prev_basis.to(device=p.device, dtype=p.dtype)
            with torch.no_grad():
                proj = (p @ U) @ U.transpose(0, 1)
                p.sub_(proj)
                p.copy_(F.normalize(p, dim=1))

    def _alignment_ratio(self, basis: Optional[torch.Tensor]) -> float:
        """Approximate effective alignment: ||A U||_F^2 / ||A||_F^2."""
        if basis is None:
            return float("nan")
        params = self._collect_trainable_lora_A()
        if not params:
            return float("nan")
        ratios = []
        for p in params:
            if basis.shape[0] != p.shape[1]:
                continue
            U = basis.to(device=p.device, dtype=p.dtype)
            num = torch.linalg.matrix_norm(p @ U, ord="fro") ** 2
            den = torch.linalg.matrix_norm(p, ord="fro") ** 2 + self._efalign_eps
            ratios.append(float((num / den).item()))
        if not ratios:
            return float("nan")
        return float(sum(ratios) / len(ratios))

    def _update_prev_basis(self, basis: Optional[torch.Tensor]) -> None:
        if basis is None:
            return
        cur = basis.detach().cpu()
        if self._efalign_prev_basis is None:
            merged = cur
        else:
            merged = torch.cat([self._efalign_prev_basis, cur], dim=1)
        q, _ = torch.linalg.qr(merged, mode="reduced")
        cap = max(1, min(self._efalign_basis_cap, q.shape[1]))
        self._efalign_prev_basis = q[:, :cap].contiguous()

    def _save_efalign_stats(
        self,
        task_id: int,
        pre_align: float,
        post_align: float,
        eigvals: Optional[torch.Tensor],
    ) -> None:
        if not self._efalign_save_json:
            return
        save_dir = self.args.get("filepath", "./")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "efalign_metrics.json")
        payload = []
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = []
        payload.append(
            {
                "task": int(task_id),
                "pre_align": float(pre_align),
                "post_align": float(post_align),
                "top_eigenvalues": eigvals.detach().cpu().tolist() if eigvals is not None else None,
            }
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Training integration
    # ------------------------------------------------------------------
    def _train(self, train_loader, test_loader):
        """Reuse SeqLoRA optimizer/training flow with EF-aligned setup."""
        network = self._unwrap_network()
        pre_align = float("nan")
        post_align = float("nan")
        eigvals = None

        if self._cur_task == 0:
            if not self._lora_initialized:
                network.backbone = self.build_lora_backbone()
                network.backbone.to(self._device)
                self._lora_initialized = True
            self._network = network
            self._prepare_network()

            if self._efalign_enable:
                basis, eigvals = self._estimate_effective_subspace(train_loader)
                if basis is not None and self._efalign_init_task0:
                    self._init_lora_with_basis(basis)
                pre_align = self._alignment_ratio(basis)

            params = [p for p in self._network.parameters() if p.requires_grad]
            optimizer = self._build_optimizer(params, stage="init")
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
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._network = network
            self._prepare_network()

            if self._efalign_enable:
                basis, eigvals = self._estimate_effective_subspace(train_loader)
                if basis is not None and self._efalign_init_updates:
                    self._init_lora_with_basis(basis)
                if self._efalign_complement and self._efalign_prev_basis is not None:
                    self._project_lora_to_complement(self._efalign_prev_basis)
                pre_align = self._alignment_ratio(basis)

            params = [p for p in self._network.parameters() if p.requires_grad]
            optimizer = self._build_optimizer(params, stage="update")
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
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        if self._efalign_enable:
            post_basis, _ = self._estimate_effective_subspace(train_loader)
            post_align = self._alignment_ratio(post_basis)
            self._update_prev_basis(post_basis)
            self._save_efalign_stats(self._cur_task, pre_align, post_align, eigvals)
            self._log(
                f"[EFLoRA] task={self._cur_task} pre_align={pre_align:.4f} "
                f"post_align={post_align:.4f} basis_dim="
                f"{0 if self._efalign_prev_basis is None else self._efalign_prev_basis.shape[1]}"
            )

        # Keep SeqLoRA persistence behavior
        save_dir = self.args.get("filepath", "./")
        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_dir, self._cur_task)

