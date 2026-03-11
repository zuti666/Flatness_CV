"""InfoBudget-GAM with rank recycling (fixed trainable rank budget).

Goal:
- Keep trainable LoRA rank fixed (e.g., 8) for every task.
- After each task, freeze one selected rank into a non-trainable bank.
- Re-initialize that rank slot so the next task still has full trainable rank.

This yields per-task frozen rank components while keeping trainable parameter
count constant across tasks.
"""

from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Tuple

import torch

from models_LoRAbasedCL.infobudget_gam_lora import Learner as InfoBudgetLearner


class Learner(InfoBudgetLearner):
    """InfoBudget-GAM + rank recycle/freeze bank."""

    def __init__(self, args):
        super().__init__(args)
        self._rank_recycle_enable = bool(args.get("rank_recycle_enable", True))
        self._rank_recycle_per_task = max(0, int(args.get("rank_recycle_per_task", 1)))
        self._rank_recycle_strategy = str(args.get("rank_recycle_strategy", "importance")).lower()
        self._rank_recycle_reset = str(args.get("rank_recycle_reset", "kaiming_zero_b")).lower()
        self._rank_recycle_save_json = bool(args.get("rank_recycle_save_json", True))
        self._rank_recycle_start_task = max(0, int(args.get("rank_recycle_start_task", 0)))

        self._rank_recycle_history = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_backbone(self):
        net = self._unwrap_network()
        return getattr(net, "backbone", None)

    def _iter_qkv_train_modules(self):
        """Yield qkv train wrappers that own current A/B and saved bank dicts."""
        backbone = self._get_backbone()
        if backbone is None:
            return
        model = getattr(backbone, "lora_vit", None)
        if model is None:
            return
        blocks = getattr(model, "blocks", [])
        for blk in blocks:
            qkv = getattr(getattr(blk, "attn", blk), "qkv", None)
            if qkv is None:
                continue
            if all(hasattr(qkv, k) for k in ["linear_a_q", "linear_b_q", "linear_a_v", "linear_b_v", "saved_A", "saved_B"]):
                yield qkv

    @staticmethod
    def _next_saved_id(saved_dict: dict) -> int:
        max_id = -1
        for k in saved_dict.keys():
            if not isinstance(k, str) or "_" not in k:
                continue
            try:
                idx = int(k.rsplit("_", 1)[-1])
            except Exception:
                continue
            if idx > max_id:
                max_id = idx
        return max_id + 1

    def _rank_dim(self) -> int:
        backbone = self._get_backbone()
        if backbone is None:
            return 0
        w_as = getattr(backbone, "w_As", None)
        w_bs = getattr(backbone, "w_Bs", None)
        if not isinstance(w_as, list) or not isinstance(w_bs, list) or len(w_as) == 0 or len(w_bs) == 0:
            return 0
        r = int(w_as[0].weight.shape[0])
        for a_mod, b_mod in zip(w_as, w_bs):
            r = min(r, int(a_mod.weight.shape[0]), int(b_mod.weight.shape[1]))
        return max(0, r)

    def _importance_scores(self, rank_dim: int) -> torch.Tensor:
        backbone = self._get_backbone()
        scores = torch.zeros(rank_dim, dtype=torch.float32)
        if backbone is None:
            return scores
        w_as = getattr(backbone, "w_As", [])
        w_bs = getattr(backbone, "w_Bs", [])
        with torch.no_grad():
            for a_mod, b_mod in zip(w_as, w_bs):
                a_w = a_mod.weight.detach().float()  # [r, in]
                b_w = b_mod.weight.detach().float()  # [out, r]
                r = min(rank_dim, int(a_w.shape[0]), int(b_w.shape[1]))
                if r <= 0:
                    continue
                a_norm = torch.linalg.vector_norm(a_w[:r, :], dim=1)
                b_norm = torch.linalg.vector_norm(b_w[:, :r], dim=0)
                # Keep accumulator on CPU to avoid mixed-device additions.
                scores[:r] += (a_norm * b_norm).detach().cpu()
        return scores

    def _select_ranks(self, rank_dim: int, n_pick: int) -> List[int]:
        if rank_dim <= 0 or n_pick <= 0:
            return []
        all_idx = list(range(rank_dim))
        if self._rank_recycle_strategy == "sequential":
            # Rotate through rank slots by task id.
            start = int(self._cur_task) % rank_dim
            return [all_idx[(start + i) % rank_dim] for i in range(n_pick)]
        if self._rank_recycle_strategy == "random":
            perm = torch.randperm(rank_dim).tolist()
            return perm[:n_pick]

        # Default: importance (freeze strongest active rank).
        scores = self._importance_scores(rank_dim)
        ordered = sorted(all_idx, key=lambda k: float(scores[k]), reverse=True)
        return ordered[:n_pick]

    @staticmethod
    def _make_rank1_pair(a_mod: torch.nn.Linear, b_mod: torch.nn.Linear, rank_idx: int) -> Tuple[torch.nn.Linear, torch.nn.Linear]:
        """Create frozen rank-1 (A,B) modules from one rank slot."""
        in_dim = int(a_mod.weight.shape[1])
        out_dim = int(b_mod.weight.shape[0])

        a_new = torch.nn.Linear(in_dim, 1, bias=False)
        b_new = torch.nn.Linear(1, out_dim, bias=False)
        with torch.no_grad():
            a_new.weight.copy_(a_mod.weight.detach()[rank_idx : rank_idx + 1, :].cpu())
            b_new.weight.copy_(b_mod.weight.detach()[:, rank_idx : rank_idx + 1].cpu())
        a_new.weight.requires_grad_(False)
        b_new.weight.requires_grad_(False)
        return a_new, b_new

    def _append_rank_to_bank(self, rank_idx: int):
        """Append one selected rank into saved_A/saved_B bank as a new pseudo-task."""
        backbone = self._get_backbone()
        if backbone is None:
            return
        w_as = getattr(backbone, "w_As", None)
        w_bs = getattr(backbone, "w_Bs", None)
        saved_a = getattr(backbone, "saved_A", None)
        saved_b = getattr(backbone, "saved_B", None)
        if not isinstance(w_as, list) or not isinstance(w_bs, list) or not isinstance(saved_a, dict) or not isinstance(saved_b, dict):
            return

        bank_a = []
        bank_b = []
        for a_mod, b_mod in zip(w_as, w_bs):
            if rank_idx >= int(a_mod.weight.shape[0]) or rank_idx >= int(b_mod.weight.shape[1]):
                continue
            a1, b1 = self._make_rank1_pair(a_mod, b_mod, rank_idx)
            bank_a.append(a1)
            bank_b.append(b1)
        if len(bank_a) == 0:
            return

        sid = self._next_saved_id(saved_a)
        saved_a[f"saved_A_{sid}"] = bank_a
        saved_b[f"saved_B_{sid}"] = bank_b

        # Update qkv wrappers to consume all bank entries.
        qkv_count = 0
        for qkv in self._iter_qkv_train_modules() or []:
            qkv.saved_A = saved_a
            qkv.saved_B = saved_b
            qkv.task_id = int(sid + 1)
            qkv_count += 1

        self._log(f"[RankRecycle] appended rank={rank_idx} to bank sid={sid}, qkv_modules={qkv_count}")

    def _reset_rank_slot(self, rank_idx: int):
        """Re-initialize the selected rank slot so it can be reused by next task."""
        backbone = self._get_backbone()
        if backbone is None:
            return
        w_as = getattr(backbone, "w_As", [])
        w_bs = getattr(backbone, "w_Bs", [])
        with torch.no_grad():
            for a_mod, b_mod in zip(w_as, w_bs):
                a_w = a_mod.weight
                b_w = b_mod.weight
                if rank_idx >= int(a_w.shape[0]) or rank_idx >= int(b_w.shape[1]):
                    continue

                if self._rank_recycle_reset == "zero":
                    a_w[rank_idx, :].zero_()
                    b_w[:, rank_idx].zero_()
                else:
                    # LoRA-style reset: random A row, zero B column.
                    fan_in = max(1, int(a_w.shape[1]))
                    bound = math.sqrt(6.0 / fan_in)
                    a_w[rank_idx, :].uniform_(-bound, bound)
                    b_w[:, rank_idx].zero_()

    def _save_rank_recycle_state(self):
        if not self._rank_recycle_save_json:
            return
        save_dir = self.args.get("filepath", "./")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "rank_recycle_state.json")
        payload = {
            "strategy": self._rank_recycle_strategy,
            "per_task": int(self._rank_recycle_per_task),
            "start_task": int(self._rank_recycle_start_task),
            "reset": self._rank_recycle_reset,
            "history": self._rank_recycle_history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _recycle_after_task(self):
        if not self._rank_recycle_enable:
            return
        if int(self._cur_task) < int(self._rank_recycle_start_task):
            return

        rank_dim = self._rank_dim()
        if rank_dim <= 0:
            self._log("[RankRecycle] skip, invalid rank dim")
            return
        n_pick = min(rank_dim, int(self._rank_recycle_per_task))
        chosen = self._select_ranks(rank_dim, n_pick)
        if not chosen:
            return

        for ridx in chosen:
            self._append_rank_to_bank(int(ridx))
            self._reset_rank_slot(int(ridx))

        self._rank_recycle_history.append(
            {
                "task": int(self._cur_task),
                "rank_dim": int(rank_dim),
                "chosen": [int(k) for k in chosen],
            }
        )
        self._save_rank_recycle_state()

    # ------------------------------------------------------------------
    # Main hook
    # ------------------------------------------------------------------
    def _train(self, train_loader, test_loader):
        # Keep the original InfoBudget-GAM behavior untouched.
        super()._train(train_loader, test_loader)
        # Then convert selected rank(s) into frozen bank entries.
        self._recycle_after_task()
