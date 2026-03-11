"""InfoProbabilisticCL (Algorithm B) LoRA learner.

Core idea:
- Maintain head-free task subspaces U_j (whitened cross-cov) and task budgets Kb_j.
- During training, penalize budget deficit, spectral forget, and subspace misalignment.
- Add subspace InfoNCE in historical discriminative subspace.
- Optional KL-like surrogates for task/hyper complexity.

This file is intentionally independent from InfoBudget (no projection/drift).
"""

from __future__ import annotations

import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models_LoRAbasedCL.seqlora import Learner as SeqLoRALearner
from utils.toolkit import tensor2numpy


class Learner(SeqLoRALearner):
    """SeqLoRA + InfoProbabilisticCL losses."""

    def __init__(self, args):
        super().__init__(args)

        # Main switches
        self._ipb_enable = bool(args.get("ipb_enable", True))

        # Subspace estimation
        self._ipb_topk = int(args.get("ipb_topk", max(1, int(args.get("lora_rank", 4)))))
        self._ipb_max_batches = int(args.get("ipb_max_batches", 4))
        self._ipb_tau_h = float(args.get("ipb_tau_h", 1e-4))
        self._ipb_tau_y = float(args.get("ipb_tau_y", 1e-6))
        self._ipb_eps = float(args.get("ipb_eps", 1e-12))
        self._ipb_whiten_h = bool(args.get("ipb_whiten_h", True))
        self._ipb_whiten_y = bool(args.get("ipb_whiten_y", True))

        # Auxiliary sampling
        self._ipb_past_tasks_per_step = int(args.get("ipb_past_tasks_per_step", 3))
        self._ipb_mem_batches_per_task = int(args.get("ipb_mem_batches_per_task", 1))
        self._ipb_mem_batch_size = int(args.get("ipb_mem_batch_size", args.get("batch_size", 128)))
        self._ipb_aux_interval = int(args.get("ipb_aux_interval", 10))
        self._ipb_spec_batches = int(args.get("ipb_spec_batches", 1))
        self._ipb_spec_ridge = float(args.get("ipb_spec_ridge", 1e-3))

        # InfoNCE
        self._ipb_use_nce = bool(args.get("ipb_use_nce", True))
        self._ipb_nce_tau = float(args.get("ipb_nce_tau", 0.1))
        self._ipb_nce_norm = bool(args.get("ipb_nce_norm", True))

        # Loss weights
        self._lam_mix = float(args.get("lambda_mix", 0.0))
        self._lam_spec = float(args.get("lambda_spec", 0.0))
        self._lam_align = float(args.get("lambda_align", 0.0))
        self._lam_nce = float(args.get("lambda_nce", 0.0))
        self._lam_task_kl = float(args.get("lambda_taskkl", 0.0))
        self._lam_hyper_kl = float(args.get("lambda_hyperkl", 0.0))
        self._ipb_budget_eps = float(args.get("ipb_budget_eps", 1e-3))

        # KL surrogate mode
        self._ipb_task_kl_mode = str(args.get("ipb_task_kl_mode", "l2")).lower()
        self._ipb_hyper_kl_mode = str(args.get("ipb_hyper_kl_mode", "l2")).lower()

        # Logging
        self._ipb_save_json = bool(args.get("ipb_save_json", True))
        self._ipb_track_mechanism = bool(args.get("ipb_track_mechanism", True))

        # Old-data source
        self._ipb_mem_source = str(args.get("ipb_mem_source", "memory")).lower()
        self._ipb_mem_fallback_train = bool(args.get("ipb_mem_fallback_train", True))

        # Runtime state
        self._teacher_net = None
        self._U_task: Optional[torch.Tensor] = None
        self._U_hist_ref: Optional[torch.Tensor] = None
        self._U_ref_list: Dict[int, torch.Tensor] = {}
        self._kb_ref: Dict[int, float] = {}
        self._kb_budget: Dict[int, float] = {}
        self._eig_ref_sum: Dict[int, float] = {}
        self._spec_ref: Dict[int, float] = {}
        self._energy_ref_ratio: Dict[int, float] = {}
        self._kb_ratio_ref: Dict[int, float] = {}
        self._last_task_svals: Optional[torch.Tensor] = None
        self._mem_loader = None
        self._old_mem_iter = None

        # Stats
        self._last_task_stats: Dict[str, float] = {}
        self._last_task_curve: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap_batch(batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                _, inputs, targets = batch
                return inputs, targets
            if len(batch) == 2:
                return batch
        raise ValueError("Unexpected batch format for InfoProbabilisticCL.")

    @staticmethod
    def _extract_features(outputs):
        feat = outputs["features"] if isinstance(outputs, dict) and "features" in outputs else outputs
        if feat.dim() == 3:
            feat = feat[:, 0, ...]
        elif feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        return feat.float()

    def _class_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self._known_classes > 0:
            fake_targets = targets - self._known_classes
            return F.cross_entropy(logits[:, self._known_classes :], fake_targets)
        return F.cross_entropy(logits, targets)

    def _collect_lora_params(self) -> List[torch.nn.Parameter]:
        module = self._unwrap_network()
        picked: List[torch.nn.Parameter] = []
        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if ("linear_a" in n) or ("linear_b" in n) or ("wrapped_param" in n):
                picked.append(p)
        return picked

    def _flatten_params(self, params: List[torch.nn.Parameter]) -> torch.Tensor:
        if not params:
            return torch.zeros(1, device=self._device)
        chunks = [p.detach().reshape(-1).to(device=self._device) for p in params]
        return torch.cat(chunks, dim=0)

    def _snapshot_teacher(self):
        base = self._unwrap_network()
        self._teacher_net = copy.deepcopy(base).to(self._device).eval()
        for p in self._teacher_net.parameters():
            p.requires_grad_(False)

    def _task_class_range(self, task_id: int) -> Tuple[int, int]:
        dm = getattr(self, "data_manager", None)
        if dm is not None and hasattr(dm, "get_task_class_range"):
            return dm.get_task_class_range(task_id)
        init_cls = int(self.args.get("init_cls", 0))
        inc = int(self.args.get("increment", 0))
        if task_id == 0:
            return 0, init_cls
        start = init_cls + (task_id - 1) * inc
        end = start + inc
        return start, end

    # ------------------------------------------------------------------
    # Memory loaders
    # ------------------------------------------------------------------
    def _build_old_train_loader(self):
        if self._known_classes <= 0:
            return None
        dm = getattr(self, "data_manager", None)
        if dm is None:
            return None
        try:
            dataset_old = dm.get_dataset(
                np.arange(0, int(self._known_classes)),
                source="train",
                mode="train",
            )
        except Exception:
            return None
        from torch.utils.data import DataLoader
        nw = int(self.args.get("train_num_workers", 0))
        return DataLoader(dataset_old, batch_size=self._ipb_mem_batch_size, shuffle=True, num_workers=nw)

    def _build_replay_loader(self):
        mem = self._get_memory()
        if mem is None:
            return None
        mem_data, mem_targets = mem
        if mem_targets is None or len(mem_targets) == 0:
            return None
        dm = getattr(self, "data_manager", None)
        if dm is None:
            return None
        try:
            dataset_mem = dm.get_dataset(
                [],
                source="train",
                mode="train",
                appendent=(mem_data, mem_targets),
            )
        except Exception:
            try:
                from utils.data_manager import DummyDataset
                from torchvision import transforms
                trsf = transforms.Compose([*dm._train_trsf, *dm._common_trsf])
                dataset_mem = DummyDataset(mem_data, mem_targets, trsf, dm.use_path)
            except Exception:
                return None
        from torch.utils.data import DataLoader
        nw = int(self.args.get("train_num_workers", 0))
        return DataLoader(dataset_mem, batch_size=self._ipb_mem_batch_size, shuffle=True, num_workers=nw)

    def _build_task_mem_loader(self, task_id: int):
        mem = self._get_memory()
        if mem is None:
            return None
        mem_data, mem_targets = mem
        if mem_targets is None or len(mem_targets) == 0:
            return None
        start, end = self._task_class_range(task_id)
        mask = (mem_targets >= start) & (mem_targets < end)
        if not np.any(mask):
            return None
        mem_data_sel = mem_data[mask]
        mem_targets_sel = mem_targets[mask]
        dm = getattr(self, "data_manager", None)
        if dm is None:
            return None
        try:
            dataset_mem = dm.get_dataset(
                [],
                source="train",
                mode="train",
                appendent=(mem_data_sel, mem_targets_sel),
            )
        except Exception:
            try:
                from utils.data_manager import DummyDataset
                from torchvision import transforms
                trsf = transforms.Compose([*dm._train_trsf, *dm._common_trsf])
                dataset_mem = DummyDataset(mem_data_sel, mem_targets_sel, trsf, dm.use_path)
            except Exception:
                return None
        from torch.utils.data import DataLoader
        nw = int(self.args.get("train_num_workers", 0))
        return DataLoader(dataset_mem, batch_size=self._ipb_mem_batch_size, shuffle=True, num_workers=nw)

    def _build_task_train_loader(self, task_id: int):
        dm = getattr(self, "data_manager", None)
        if dm is None:
            return None
        start, end = self._task_class_range(task_id)
        try:
            dataset = dm.get_dataset(np.arange(start, end), source="train", mode="train")
        except Exception:
            return None
        from torch.utils.data import DataLoader
        nw = int(self.args.get("train_num_workers", 0))
        return DataLoader(dataset, batch_size=self._ipb_mem_batch_size, shuffle=True, num_workers=nw)

    def _next_old_mem_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self._mem_loader is None or self._known_classes <= 0:
            return None
        if self._old_mem_iter is None:
            self._old_mem_iter = iter(self._mem_loader)
        for _ in range(8):
            try:
                batch = next(self._old_mem_iter)
            except StopIteration:
                self._old_mem_iter = iter(self._mem_loader)
                batch = next(self._old_mem_iter)
            inputs, targets = self._unwrap_batch(batch)
            mask = targets < int(self._known_classes)
            if mask.any():
                return inputs[mask], targets[mask]
        return None

    # ------------------------------------------------------------------
    # Subspace estimation + spectral stats
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _estimate_task_subspace(
        self, loader, start_class: int, end_class: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if loader is None:
            return None, None
        module = self._unwrap_network()
        was_training = module.training
        module.eval()
        try:
            feat_dim = int(self.feature_dim)
            c_task = max(1, int(end_class - start_class))

            sum_h = torch.zeros(feat_dim, device=self._device, dtype=torch.float64)
            sum_h2 = torch.zeros(feat_dim, device=self._device, dtype=torch.float64)
            sum_y = torch.zeros(c_task, device=self._device, dtype=torch.float64)
            sum_hy = torch.zeros((feat_dim, c_task), device=self._device, dtype=torch.float64)
            n_total = 0

            for bidx, batch in enumerate(loader):
                if bidx >= int(self._ipb_max_batches):
                    break
                inputs, targets = self._unwrap_batch(batch)
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                y_local = targets - int(start_class)
                valid = (y_local >= 0) & (y_local < c_task)
                if not valid.any():
                    continue
                inputs = inputs[valid]
                y_local = y_local[valid]

                out = module(inputs)
                h = self._extract_features(out).to(dtype=torch.float64)
                y = F.one_hot(y_local.long(), num_classes=c_task).to(dtype=torch.float64)

                sum_h += h.sum(dim=0)
                sum_h2 += (h * h).sum(dim=0)
                sum_y += y.sum(dim=0)
                sum_hy += h.transpose(0, 1) @ y
                n_total += int(h.size(0))

            if n_total <= 1:
                return None, None

            n = float(n_total)
            mu_h = sum_h / n
            p_y = sum_y / n
            c_hy = (sum_hy / n) - mu_h.unsqueeze(1) * p_y.unsqueeze(0)
            var_h = torch.clamp((sum_h2 / n) - mu_h * mu_h, min=0.0)

            if self._ipb_whiten_h:
                inv_sqrt_h = torch.rsqrt(var_h + float(self._ipb_tau_h))
            else:
                inv_sqrt_h = torch.ones_like(var_h)
            if self._ipb_whiten_y:
                inv_sqrt_y = torch.rsqrt(p_y + float(self._ipb_tau_y))
            else:
                inv_sqrt_y = torch.ones_like(p_y)
            m = (inv_sqrt_h.unsqueeze(1) * c_hy) * inv_sqrt_y.unsqueeze(0)

            u, s, _vh = torch.linalg.svd(m.to(dtype=torch.float32), full_matrices=False)
            if u.numel() == 0:
                return None, None
            k = max(1, min(int(self._ipb_topk), int(u.shape[1])))
            u_t = u[:, :k].contiguous()
            return u_t.detach().cpu(), s.detach().cpu()
        finally:
            module.train(was_training)

    def _spec_loss(self, loader, start_class: int, end_class: int) -> Optional[float]:
        if loader is None:
            return None
        module = self._unwrap_network()
        was_training = module.training
        module.eval()
        try:
            feat_dim = int(self.feature_dim)
            c_task = max(1, int(end_class - start_class))
            H_list = []
            Y_list = []
            for bidx, batch in enumerate(loader):
                if bidx >= int(self._ipb_spec_batches):
                    break
                inputs, targets = self._unwrap_batch(batch)
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                y_local = targets - int(start_class)
                valid = (y_local >= 0) & (y_local < c_task)
                if not valid.any():
                    continue
                inputs = inputs[valid]
                y_local = y_local[valid]

                out = module(inputs)
                h = self._extract_features(out)
                y = F.one_hot(y_local.long(), num_classes=c_task).to(dtype=h.dtype)
                H_list.append(h)
                Y_list.append(y)

            if not H_list:
                return None
            H = torch.cat(H_list, dim=0)
            Y = torch.cat(Y_list, dim=0)
            n = float(H.size(0))
            HtH = H.t() @ H
            HtY = H.t() @ Y
            ridge = float(self._ipb_spec_ridge)
            HtH = HtH + ridge * torch.eye(HtH.size(0), device=HtH.device, dtype=HtH.dtype)
            W = torch.linalg.solve(HtH, HtY)
            Y_hat = H @ W
            loss = (Y - Y_hat).pow(2).sum() / max(1.0, n)
            return float(loss.detach().item())
        finally:
            module.train(was_training)

    @torch.no_grad()
    def _energy_ratio_ref(self, loader, u_ref: torch.Tensor) -> Optional[float]:
        if loader is None or u_ref is None or u_ref.numel() == 0:
            return None
        module = self._unwrap_network()
        was_training = module.training
        module.eval()
        try:
            u_ref = u_ref.to(device=self._device, dtype=torch.float32)
            energy_sum = 0.0
            total_sum = 0.0
            count = 0
            for bidx, batch in enumerate(loader):
                if bidx >= int(self._ipb_max_batches):
                    break
                inputs, _targets = self._unwrap_batch(batch)
                inputs = inputs.to(self._device, non_blocking=True)
                out = module(inputs)
                h = self._extract_features(out)
                proj = h @ u_ref
                energy = proj.pow(2).sum(dim=1).mean()
                total = h.pow(2).sum(dim=1).mean()
                energy_sum += float(energy.item())
                total_sum += float(total.item())
                count += 1
            if count <= 0 or total_sum <= 0:
                return None
            return float(energy_sum / max(1e-12, total_sum))
        finally:
            module.train(was_training)

    # ------------------------------------------------------------------
    # Auxiliary stats per past task
    # ------------------------------------------------------------------
    def _compute_task_stats(self, task_id: int) -> Optional[Dict[str, float]]:
        if self._ipb_mem_source == "train":
            loader = self._build_task_train_loader(task_id)
        elif self._ipb_mem_source == "memory":
            loader = self._build_task_mem_loader(task_id)
            if loader is None and self._ipb_mem_fallback_train:
                loader = self._build_task_train_loader(task_id)
        else:
            loader = None
        if loader is None:
            return None
        start, end = self._task_class_range(task_id)
        U_cur, svals = self._estimate_task_subspace(loader, start, end)
        if U_cur is None or svals is None:
            return None
        k = min(int(self._ipb_topk), int(U_cur.shape[1]))
        svals = svals[:k]
        lamb = (svals * svals).clamp(min=0.0, max=1.0 - 1e-6)
        kb = -0.5 * torch.log1p(-lamb).sum().item()

        align = float("nan")
        if task_id in self._U_ref_list:
            U_ref = self._U_ref_list[task_id]
            p = min(U_ref.shape[1], U_cur.shape[1])
            if p > 0:
                align = float((U_ref[:, :p].T @ U_cur[:, :p]).pow(2).sum().item() / max(1, p))

        energy = float("nan")
        if task_id in self._eig_ref_sum:
            ref_sum = float(self._eig_ref_sum[task_id])
            if ref_sum > 0:
                energy = float(lamb.sum().item() / (ref_sum + float(self._ipb_eps)))

        spec_forget = float("nan")
        if task_id in self._spec_ref:
            cur_spec = self._spec_loss(loader, start, end)
            if cur_spec is not None:
                spec_forget = float(cur_spec - float(self._spec_ref[task_id]))

        return {
            "kb": float(kb),
            "align": float(align),
            "energy": float(energy),
            "spec_forget": float(spec_forget),
        }

    def _sample_past_tasks(self) -> List[int]:
        keys = sorted([k for k in self._U_ref_list.keys() if k < self._cur_task])
        if not keys:
            return []
        if len(keys) <= self._ipb_past_tasks_per_step:
            return keys
        return list(np.random.choice(keys, size=int(self._ipb_past_tasks_per_step), replace=False))

    def _compute_aux_terms(self) -> Dict[str, float]:
        past_tasks = self._sample_past_tasks()
        return {
            "task_ids": past_tasks,
        }

    def _build_U_hist_ref(self) -> Optional[torch.Tensor]:
        if not self._U_ref_list:
            return None
        mats = [u for _, u in sorted(self._U_ref_list.items(), key=lambda x: x[0]) if u is not None]
        if not mats:
            return None
        merged = torch.cat(mats, dim=1)
        q, _ = torch.linalg.qr(merged, mode="reduced")
        return q.contiguous()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _subspace_nce(self, mem_inputs: torch.Tensor) -> torch.Tensor:
        if self._teacher_net is None or self._U_hist_ref is None or self._U_hist_ref.numel() == 0:
            return torch.zeros((), device=self._device)
        mem_inputs = mem_inputs.to(self._device, non_blocking=True)
        out_new = self._network(mem_inputs)
        h_new = self._extract_features(out_new)
        with torch.no_grad():
            out_old = self._teacher_net(mem_inputs)
            h_old = self._extract_features(out_old).to(device=h_new.device, dtype=h_new.dtype)
        u = self._U_hist_ref.to(device=h_new.device, dtype=h_new.dtype)
        z_new = h_new @ u
        z_old = h_old @ u
        if self._ipb_nce_norm:
            z_new = F.normalize(z_new, dim=1)
            z_old = F.normalize(z_old, dim=1)
        logits = (z_new @ z_old.t()) / float(self._ipb_nce_tau)
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def _kl_task(self) -> torch.Tensor:
        if self._lam_task_kl <= 0:
            return torch.zeros((), device=self._device)
        params = self._collect_lora_params()
        vec = self._flatten_params(params)
        if self._ipb_task_kl_mode == "l2":
            return (vec * vec).mean()
        return (vec * vec).mean()

    def _kl_hyper(self) -> torch.Tensor:
        if self._lam_hyper_kl <= 0:
            return torch.zeros((), device=self._device)
        if self._teacher_net is None:
            return torch.zeros((), device=self._device)
        cur_params = self._collect_lora_params()
        tea_params = [p for n, p in self._teacher_net.named_parameters() if ("linear_a" in n) or ("linear_b" in n) or ("wrapped_param" in n)]
        vec_cur = self._flatten_params(cur_params)
        vec_old = self._flatten_params(tea_params)
        diff = vec_cur - vec_old
        if self._ipb_hyper_kl_mode == "l2":
            return (diff * diff).mean()
        return (diff * diff).mean()

    def _composite_loss(self, inputs, targets, mem_batch, aux_stats: Dict[str, float]):
        out = self._network(inputs)
        logits = out["logits"] if isinstance(out, dict) else out
        loss_task = self._class_loss(logits, targets)

        # Differentiable proxy losses using frozen historical subspaces.
        loss_mix = torch.zeros((), device=logits.device, dtype=logits.dtype)
        loss_spec = torch.zeros((), device=logits.device, dtype=logits.dtype)
        loss_align = torch.zeros((), device=logits.device, dtype=logits.dtype)
        energy_ratio_mean = float("nan")

        task_ids = aux_stats.get("task_ids", []) if isinstance(aux_stats, dict) else []
        if task_ids:
            if mem_batch is not None:
                mem_inputs, _ = mem_batch
                mem_inputs = mem_inputs.to(self._device, non_blocking=True)
                out_mem = self._network(mem_inputs)
                h_cur = self._extract_features(out_mem)
            else:
                h_cur = self._extract_features(out)
            total_energy = h_cur.pow(2).sum(dim=1).mean() + float(self._ipb_eps)
            ratios = []
            for tid in task_ids:
                u_ref = self._U_ref_list.get(tid, None)
                if u_ref is None or u_ref.numel() == 0:
                    continue
                u_ref = u_ref.to(device=h_cur.device, dtype=h_cur.dtype)
                proj = h_cur @ u_ref
                energy = proj.pow(2).sum(dim=1).mean()
                ratio = energy / total_energy
                ratios.append(ratio)

                # Align: encourage projection energy to remain high
                loss_align = loss_align + (1.0 - ratio)

                # Spec/Mix: enforce reference energy budget if available
                ref_energy = self._energy_ref_ratio.get(tid, None)
                if ref_energy is not None:
                    loss_spec = loss_spec + F.relu(torch.as_tensor(ref_energy, device=ratio.device, dtype=ratio.dtype) - ratio)

                ref_kb_ratio = self._kb_ratio_ref.get(tid, None)
                if ref_kb_ratio is not None:
                    loss_mix = loss_mix + F.relu(torch.as_tensor(ref_kb_ratio, device=ratio.device, dtype=ratio.dtype) - ratio)

            if ratios:
                energy_ratio_mean = float(torch.stack(ratios).mean().detach().item())

        loss_nce = torch.zeros((), device=logits.device)
        if self._lam_nce > 0 and self._ipb_use_nce and mem_batch is not None:
            mem_inputs, _ = mem_batch
            loss_nce = self._subspace_nce(mem_inputs)

        loss_task_kl = self._kl_task()
        loss_hyper_kl = self._kl_hyper()

        total = (
            loss_task
            + self._lam_mix * loss_mix
            + self._lam_spec * loss_spec
            + self._lam_align * loss_align
            + self._lam_nce * loss_nce
            + self._lam_task_kl * loss_task_kl
            + self._lam_hyper_kl * loss_hyper_kl
        )

        diag = {
            "task": float(loss_task.detach().item()),
            "mix": float(loss_mix.detach().item()) if torch.is_tensor(loss_mix) else float(loss_mix),
            "spec": float(loss_spec.detach().item()) if torch.is_tensor(loss_spec) else float(loss_spec),
            "align": float(loss_align.detach().item()) if torch.is_tensor(loss_align) else float(loss_align),
            "nce": float(loss_nce.detach().item()) if torch.is_tensor(loss_nce) else 0.0,
            "task_kl": float(loss_task_kl.detach().item()) if torch.is_tensor(loss_task_kl) else 0.0,
            "hyper_kl": float(loss_hyper_kl.detach().item()) if torch.is_tensor(loss_hyper_kl) else 0.0,
            "kb": float("nan"),
            "align_mean": float("nan"),
            "energy": float("nan"),
            "spec_forget": float("nan"),
            "energy_ratio": float(energy_ratio_mean) if np.isfinite(energy_ratio_mean) else float("nan"),
            "total": float(total.detach().item()),
        }
        return total, logits, diag

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _run_prob_epochs(self, train_loader, test_loader, optimizer, scheduler, epochs: int):
        prog_bar = tqdm(range(int(epochs)), disable=not self._is_main_process)

        # task-level accumulators
        sum_mix = sum_spec = sum_align = 0.0
        sum_nce = sum_task_kl = sum_hyper_kl = 0.0
        sum_kb = sum_align_m = sum_energy = sum_spec_forget = 0.0
        cnt_aux = 0
        sum_energy_ratio = 0.0
        cnt_energy_ratio = 0

        curve_mix = []
        curve_spec = []
        curve_align = []
        curve_nce = []
        curve_task_kl = []
        curve_hyper_kl = []
        curve_test_acc = []
        curve_energy_ratio = []

        aux_cache: Dict[str, float] = {}
        step_idx = 0

        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for _i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                mem_batch = self._next_old_mem_batch()

                if self._ipb_enable and (self._ipb_aux_interval <= 1 or step_idx % self._ipb_aux_interval == 0):
                    aux_cache = self._compute_aux_terms()

                optimizer.zero_grad()
                loss, logits, diag = self._composite_loss(inputs, targets, mem_batch, aux_cache)
                loss.backward()
                optimizer.step()

                losses += float(loss.detach().item())

                sum_mix += float(diag.get("mix", 0.0))
                sum_spec += float(diag.get("spec", 0.0))
                sum_align += float(diag.get("align", 0.0))
                sum_nce += float(diag.get("nce", 0.0))
                sum_task_kl += float(diag.get("task_kl", 0.0))
                sum_hyper_kl += float(diag.get("hyper_kl", 0.0))
                v_energy_ratio = float(diag.get("energy_ratio", float("nan")))
                if np.isfinite(v_energy_ratio):
                    sum_energy_ratio += v_energy_ratio
                    cnt_energy_ratio += 1

                cnt_aux += 1

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                step_idx += 1

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)
            curve_mix.append(sum_mix / max(1, cnt_aux))
            curve_spec.append(sum_spec / max(1, cnt_aux))
            curve_align.append(sum_align / max(1, cnt_aux))
            curve_nce.append(sum_nce / max(1, cnt_aux))
            curve_task_kl.append(sum_task_kl / max(1, cnt_aux))
            curve_hyper_kl.append(sum_hyper_kl / max(1, cnt_aux))
            curve_energy_ratio.append(
                (sum_energy_ratio / max(1, cnt_energy_ratio)) if cnt_energy_ratio > 0 else float("nan")
            )

            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                curve_test_acc.append(float(test_acc))
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / len(train_loader):.3f} "
                    f"Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}"
                )
            elif self._is_main_process:
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / len(train_loader):.3f} "
                    f"Train_accy {train_acc:.2f}"
                )
            if self._is_main_process:
                prog_bar.set_description(info)

        if self._is_main_process:
            self._log(info)

        # task-level stats
        denom = max(1, cnt_aux)
        self._last_task_stats = {
            "avg_mix": sum_mix / denom,
            "avg_spec": sum_spec / denom,
            "avg_align": sum_align / denom,
            "avg_nce": sum_nce / denom,
            "avg_task_kl": sum_task_kl / denom,
            "avg_hyper_kl": sum_hyper_kl / denom,
            "avg_kb": float("nan"),
            "avg_align_mean": float("nan"),
            "avg_energy": float("nan"),
            "avg_spec_forget": float("nan"),
            "avg_energy_ratio": (sum_energy_ratio / max(1, cnt_energy_ratio)) if cnt_energy_ratio > 0 else float("nan"),
        }
        self._last_task_curve = {
            "mix": curve_mix,
            "spec": curve_spec,
            "align": curve_align,
            "nce": curve_nce,
            "task_kl": curve_task_kl,
            "hyper_kl": curve_hyper_kl,
            "test_acc": curve_test_acc,
            "energy_ratio": curve_energy_ratio,
        }

    def _save_ipb_stats(self, task_id: int):
        if not self._ipb_save_json:
            return
        save_dir = self.args.get("filepath", "./")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "infoprob_stats.json")
        payload = []
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = []

        # Offline (non-differentiable) stats for reporting
        past_tasks = [k for k in self._U_ref_list.keys() if k <= int(task_id)]
        kb_list = []
        align_list = []
        energy_list = []
        spec_list = []
        for tid in past_tasks:
            stats = self._compute_task_stats(tid)
            if stats is None:
                continue
            if np.isfinite(stats.get("kb", float("nan"))):
                kb_list.append(float(stats["kb"]))
            if np.isfinite(stats.get("align", float("nan"))):
                align_list.append(float(stats["align"]))
            if np.isfinite(stats.get("energy", float("nan"))):
                energy_list.append(float(stats["energy"]))
            if np.isfinite(stats.get("spec_forget", float("nan"))):
                spec_list.append(float(stats["spec_forget"]))

        def _mean(lst):
            return float(np.mean(lst)) if lst else float("nan")

        self._last_task_stats["avg_kb"] = _mean(kb_list)
        self._last_task_stats["avg_align_mean"] = _mean(align_list)
        self._last_task_stats["avg_energy"] = _mean(energy_list)
        self._last_task_stats["avg_spec_forget"] = _mean(spec_list)
        payload.append(
            {
                "task": int(task_id),
                "u_task_dim": 0 if self._U_task is None else int(self._U_task.shape[1]),
                "u_hist_dim": 0 if self._U_hist_ref is None else int(self._U_hist_ref.shape[1]),
                **self._last_task_stats,
                "epoch_curve": self._last_task_curve,
            }
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Main train entry
    # ------------------------------------------------------------------
    def _train(self, train_loader, test_loader):
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

        # Resolve old-data source
        self._mem_loader = None
        if self._ipb_mem_source == "memory":
            self._mem_loader = self._build_replay_loader()
            if self._mem_loader is None and self._ipb_mem_fallback_train:
                self._mem_loader = self._build_old_train_loader()
        elif self._ipb_mem_source == "train":
            self._mem_loader = self._build_old_train_loader()
        else:
            self._mem_loader = None

        if self._ipb_enable:
            self._snapshot_teacher()
            # Estimate current task subspace for logging only
            start, end = self._task_class_range(self._cur_task)
            self._U_task, self._last_task_svals = self._estimate_task_subspace(train_loader, start, end)
            # Build historical reference union for differentiable losses
            self._U_hist_ref = self._build_U_hist_ref()

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

        self._run_prob_epochs(train_loader, test_loader, optimizer, scheduler, epochs=epochs)

        # Task-end: store reference subspace and budget
        if self._ipb_enable and self._U_task is not None:
            task_id = int(self._cur_task)
            self._U_ref_list[task_id] = self._U_task
            if self._last_task_svals is not None:
                svals = self._last_task_svals[: int(self._ipb_topk)]
                lamb = (svals * svals).clamp(min=0.0, max=1.0 - 1e-6)
                kb = -0.5 * torch.log1p(-lamb).sum().item()
                self._kb_ref[task_id] = float(kb)
                self._kb_budget[task_id] = float(kb) - float(self._ipb_budget_eps)
                self._eig_ref_sum[task_id] = float(lamb.sum().item())
                kb_ratio = float(kb) / (float(kb) + 1.0)
                self._kb_ratio_ref[task_id] = kb_ratio
            # Spec reference (in-time)
            start, end = self._task_class_range(task_id)
            spec_ref = self._spec_loss(train_loader, start, end)
            if spec_ref is not None:
                self._spec_ref[task_id] = float(spec_ref)
            # Energy ratio reference (in-time)
            energy_ref = self._energy_ratio_ref(train_loader, self._U_task)
            if energy_ref is not None:
                self._energy_ref_ratio[task_id] = float(energy_ref)

            self._U_hist_ref = self._build_U_hist_ref()

        self._save_ipb_stats(self._cur_task)

        save_dir = self.args.get("filepath", "./")
        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_dir, self._cur_task)
