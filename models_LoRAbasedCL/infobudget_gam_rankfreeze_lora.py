"""InfoBudget-GAM with progressive rank freezing.

This learner keeps the original InfoBudget-GAM logic and adds a hard
"freeze one rank per task" mechanism inside the same fixed-rank LoRA module:

1) Train with InfoBudget-GAM as usual.
2) After each task, select one rank (importance or sequential strategy).
3) Freeze that rank for all subsequent tasks by masking A/B gradients.

The capacity is still fixed by `lora_rank`; no new LoRA blocks are added.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from models_LoRAbasedCL.infobudget_gam_lora import Learner as InfoBudgetLearner
from utils.toolkit import tensor2numpy


class Learner(InfoBudgetLearner):
    """InfoBudget-GAM + progressive per-rank freezing."""

    def __init__(self, args):
        super().__init__(args)
        self._rank_freeze_enable = bool(args.get("rank_freeze_enable", True))
        self._rank_freeze_per_task = max(0, int(args.get("rank_freeze_per_task", 1)))
        self._rank_freeze_strategy = str(args.get("rank_freeze_strategy", "importance")).lower()
        self._rank_freeze_save_json = bool(args.get("rank_freeze_save_json", True))
        self._rank_freeze_max = args.get("rank_freeze_max", None)  # default handled at runtime

        self._frozen_rank_indices: Set[int] = set()
        self._rank_freeze_history: List[dict] = []
        self._rank_anchor_values = {}

    # ------------------------------------------------------------------
    # Rank freeze helpers
    # ------------------------------------------------------------------
    def _iter_lora_rank_pairs(self) -> List[Tuple[str, torch.nn.Parameter, str, torch.nn.Parameter, int]]:
        """Collect (A, B) LoRA pairs where rank lives on A rows / B cols."""
        module = self._unwrap_network()
        pairs: List[Tuple[str, torch.nn.Parameter, str, torch.nn.Parameter, int]] = []
        seen = set()

        attr_pairs = [
            ("linear_a_q", "linear_b_q"),
            ("linear_a_v", "linear_b_v"),
            ("linear_a_k", "linear_b_k"),
            ("linear_a", "linear_b"),
        ]

        for mod_name, mod in module.named_modules():
            for a_attr, b_attr in attr_pairs:
                a_mod = getattr(mod, a_attr, None)
                b_mod = getattr(mod, b_attr, None)
                if not isinstance(a_mod, torch.nn.Linear) or not isinstance(b_mod, torch.nn.Linear):
                    continue
                a_w = a_mod.weight
                b_w = b_mod.weight
                if a_w.ndim != 2 or b_w.ndim != 2:
                    continue
                rank_dim = min(int(a_w.shape[0]), int(b_w.shape[1]))
                if rank_dim <= 0:
                    continue
                a_name = f"{mod_name}.{a_attr}.weight" if mod_name else f"{a_attr}.weight"
                b_name = f"{mod_name}.{b_attr}.weight" if mod_name else f"{b_attr}.weight"
                key = (a_name, b_name)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((a_name, a_w, b_name, b_w, rank_dim))
        return pairs

    @staticmethod
    def _anchor_key(param_name: str, rank_idx: int, axis: str) -> str:
        return f"{param_name}::{axis}::{int(rank_idx)}"

    def _rank_scores(self, pairs: Sequence[Tuple[str, torch.nn.Parameter, str, torch.nn.Parameter, int]], rank_dim: int) -> torch.Tensor:
        """Importance score of each rank using ||A_k|| * ||B_k|| aggregated over layers."""
        scores = torch.zeros(rank_dim, dtype=torch.float32)
        with torch.no_grad():
            for _a_name, a_w, _b_name, b_w, rdim in pairs:
                r = min(rank_dim, rdim)
                a_norm = torch.linalg.vector_norm(a_w.detach().float(), dim=1)[:r]
                b_norm = torch.linalg.vector_norm(b_w.detach().float(), dim=0)[:r]
                scores[:r] += a_norm * b_norm
        return scores

    def _max_freeze_allowed(self, rank_dim: int) -> int:
        user_max = self._rank_freeze_max
        if user_max is None:
            # Keep at least one free rank by default.
            return max(0, rank_dim - 1)
        try:
            return max(0, min(rank_dim, int(user_max)))
        except Exception:
            return max(0, rank_dim - 1)

    def _snapshot_rank_anchor(self, pairs, rank_idx: int):
        """Store frozen reference values for the selected rank."""
        with torch.no_grad():
            for a_name, a_w, b_name, b_w, rdim in pairs:
                if rank_idx >= rdim:
                    continue
                ka = self._anchor_key(a_name, rank_idx, "row")
                kb = self._anchor_key(b_name, rank_idx, "col")
                self._rank_anchor_values[ka] = a_w.detach()[rank_idx, :].cpu().clone()
                self._rank_anchor_values[kb] = b_w.detach()[:, rank_idx].cpu().clone()

    def _freeze_ranks_after_task(self):
        """Freeze `rank_freeze_per_task` new ranks after finishing current task."""
        if not self._rank_freeze_enable or self._rank_freeze_per_task <= 0:
            return
        pairs = self._iter_lora_rank_pairs()
        if not pairs:
            self._log("[RankFreeze] no LoRA A/B pairs found; skip")
            return

        rank_dim = min(int(rdim) for *_rest, rdim in pairs)
        if rank_dim <= 0:
            return

        max_allowed = self._max_freeze_allowed(rank_dim)
        remain_slots = max_allowed - len(self._frozen_rank_indices)
        if remain_slots <= 0:
            self._log(
                f"[RankFreeze] task={self._cur_task} skip (frozen={len(self._frozen_rank_indices)}/{max_allowed})"
            )
            return

        candidates = [k for k in range(rank_dim) if k not in self._frozen_rank_indices]
        if not candidates:
            return
        n_new = min(len(candidates), remain_slots, self._rank_freeze_per_task)

        if self._rank_freeze_strategy == "sequential":
            chosen = candidates[:n_new]
        else:
            scores = self._rank_scores(pairs, rank_dim)
            chosen = sorted(candidates, key=lambda k: float(scores[k]), reverse=True)[:n_new]

        for ridx in chosen:
            self._frozen_rank_indices.add(int(ridx))
            self._snapshot_rank_anchor(pairs, int(ridx))

        self._rank_freeze_history.append(
            {
                "task": int(self._cur_task),
                "selected_ranks": [int(k) for k in chosen],
                "frozen_ranks": sorted(int(k) for k in self._frozen_rank_indices),
                "rank_dim": int(rank_dim),
            }
        )
        self._log(
            f"[RankFreeze] task={self._cur_task} selected={chosen} frozen_total={len(self._frozen_rank_indices)}/{rank_dim}"
        )

    def _apply_rank_freeze_mask(self):
        """Mask gradients on frozen ranks (A row and B column)."""
        if not self._frozen_rank_indices:
            return
        pairs = self._iter_lora_rank_pairs()
        if not pairs:
            return
        with torch.no_grad():
            for _a_name, a_w, _b_name, b_w, rdim in pairs:
                for ridx in self._frozen_rank_indices:
                    if ridx >= rdim:
                        continue
                    if a_w.grad is not None:
                        a_w.grad[ridx, :].zero_()
                    if b_w.grad is not None:
                        b_w.grad[:, ridx].zero_()

    def _restore_frozen_rank_values(self):
        """Hard-restore frozen rank values after optimizer steps."""
        if not self._frozen_rank_indices:
            return
        pairs = self._iter_lora_rank_pairs()
        if not pairs:
            return
        with torch.no_grad():
            for a_name, a_w, b_name, b_w, rdim in pairs:
                for ridx in self._frozen_rank_indices:
                    if ridx >= rdim:
                        continue
                    ka = self._anchor_key(a_name, int(ridx), "row")
                    kb = self._anchor_key(b_name, int(ridx), "col")
                    if ka in self._rank_anchor_values:
                        ref_a = self._rank_anchor_values[ka].to(device=a_w.device, dtype=a_w.dtype)
                        a_w.data[ridx, :].copy_(ref_a)
                    if kb in self._rank_anchor_values:
                        ref_b = self._rank_anchor_values[kb].to(device=b_w.device, dtype=b_w.dtype)
                        b_w.data[:, ridx].copy_(ref_b)

    def _save_rank_freeze_state(self):
        if not self._rank_freeze_save_json:
            return
        save_dir = self.args.get("filepath", "./")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "rank_freeze_state.json")
        payload = {
            "frozen_ranks": sorted(int(k) for k in self._frozen_rank_indices),
            "history": self._rank_freeze_history,
            "strategy": self._rank_freeze_strategy,
            "per_task": int(self._rank_freeze_per_task),
            "max_freeze": None if self._rank_freeze_max is None else int(self._rank_freeze_max),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _project_gam_direction(self, _param_groups=None):
        """Apply parent InfoBudget projection, then enforce rank freeze mask."""
        super()._project_gam_direction(_param_groups)
        self._apply_rank_freeze_mask()

    def _run_budget_epochs(self, train_loader, test_loader, optimizer, scheduler, epochs: int):
        """Same as parent loop, with rank freeze mask/restore around updates."""
        if self._optimizer_type == "gam" and hasattr(optimizer, "set_direction_projector"):
            # Always route GAM direction through InfoBudget+RankFreeze callback.
            optimizer.set_direction_projector(self._project_gam_direction)

        prog_bar = tqdm(range(int(epochs)), disable=not self._is_main_process)
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            losses_task = 0.0
            losses_drift = 0.0
            correct, total = 0, 0

            for _i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                mem_batch = self._next_old_mem_batch()

                if self._optimizer_type == "gam":
                    def closure():
                        optimizer.zero_grad()
                        loss, logits, diag = self._composite_loss(inputs, targets, mem_batch)
                        loss.backward()
                        return {"logits": logits.detach(), "diag": diag}, loss.detach()

                    outputs, loss_value = optimizer.step(closure=closure)
                    # Hard freeze after optimizer update.
                    self._restore_frozen_rank_values()
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    diag = outputs.get("diag", {"task": 0.0, "drift": 0.0}) if isinstance(outputs, dict) else {}
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    losses_task += float(diag.get("task", 0.0))
                    losses_drift += float(diag.get("drift", 0.0))

                elif self._optimizer_type == "sam":
                    optimizer.zero_grad()
                    loss, logits, diag = self._composite_loss(inputs, targets, mem_batch)
                    loss.backward()
                    self._apply_rank_freeze_mask()
                    optimizer.first_step(zero_grad=True)
                    second_loss, logits2, diag2 = self._composite_loss(inputs, targets, mem_batch)
                    second_loss.backward()
                    self._apply_rank_freeze_mask()
                    optimizer.second_step(zero_grad=True)
                    self._restore_frozen_rank_values()
                    logits = logits2.detach()
                    losses += float(second_loss.detach().item())
                    losses_task += float(diag2["task"])
                    losses_drift += float(diag2["drift"])

                else:
                    optimizer.zero_grad()
                    loss, logits, diag = self._composite_loss(inputs, targets, mem_batch)
                    loss.backward()
                    self._apply_rank_freeze_mask()
                    optimizer.step()
                    self._restore_frozen_rank_values()
                    logits = logits.detach()
                    losses += float(loss.detach().item())
                    losses_task += float(diag["task"])
                    losses_drift += float(diag["drift"])

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
                    f"drift {losses_drift / len(train_loader):.3f}), "
                    f"Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}"
                )
            elif self._is_main_process:
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / len(train_loader):.3f} "
                    f"(task {losses_task / len(train_loader):.3f}, "
                    f"drift {losses_drift / len(train_loader):.3f}), "
                    f"Train_accy {train_acc:.2f}"
                )
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)

    def _train(self, train_loader, test_loader):
        # Keep current InfoBudget behavior (including checkpointing and stats).
        super()._train(train_loader, test_loader)
        # Then freeze one (or more) rank(s) for future tasks.
        self._freeze_ranks_after_task()
        self._save_rank_freeze_state()
