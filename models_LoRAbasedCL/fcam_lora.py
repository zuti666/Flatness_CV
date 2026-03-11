"""FCAM-LoRA: optimizer-agnostic curvature-weighted overlap minimization.

Key idea:
- No explicit gradient projection.
- Penalize feature drift projected onto historical discriminative subspaces.
- Works with SGD/SAM/GAM by optimizing the same objective.
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
    """SeqLoRA + FCAM regularization (optimizer-agnostic)."""

    def __init__(self, args):
        super().__init__(args)

        # Main switches
        self._fcam_enable = bool(args.get("fcam_enable", True))

        # Subspace estimation
        self._fcam_topk = int(args.get("fcam_topk", max(1, int(args.get("lora_rank", 4)))))
        self._fcam_max_batches = int(args.get("fcam_max_batches", 4))
        self._fcam_tau_h = float(args.get("fcam_tau_h", 1e-4))
        self._fcam_tau_y = float(args.get("fcam_tau_y", 1e-6))
        self._fcam_eps = float(args.get("fcam_eps", 1e-12))
        self._fcam_whiten_h = bool(args.get("fcam_whiten_h", True))
        self._fcam_whiten_y = bool(args.get("fcam_whiten_y", True))
        self._fcam_subspace_solver = str(args.get("fcam_subspace_solver", "gram_eig")).lower()

        # Sampling / bank
        self._fcam_past_tasks_per_step = int(args.get("fcam_past_tasks_per_step", 2))
        self._fcam_mem_batch_size = int(args.get("fcam_mem_batch_size", args.get("batch_size", 128)))
        self._fcam_aux_interval = int(args.get("fcam_aux_interval", 1))

        # Loss weights
        self._lam_cwo = float(args.get("lambda_cwo", 0.0))
        self._lam_nce = float(args.get("lambda_nce", 0.0))
        self._lam_reg = float(args.get("lambda_reg", 0.0))

        # NCE
        self._fcam_use_nce = bool(args.get("fcam_use_nce", False))
        self._fcam_nce_tau = float(args.get("fcam_nce_tau", 0.1))
        self._fcam_nce_norm = bool(args.get("fcam_nce_norm", True))

        # Weighting
        self._fcam_weight_mode = str(args.get("fcam_weight_mode", "uniform")).lower()
        self._fcam_align_grad = bool(args.get("fcam_align_grad", False))
        self._fcam_diag_specforget = bool(args.get("fcam_diag_specforget", True))
        self._fcam_mem_mode = str(args.get("fcam_mem_mode", "test")).lower()
        self._fcam_hist_max_dim = int(args.get("fcam_hist_max_dim", 128))
        self._fcam_trace_taskids = bool(args.get("fcam_trace_taskids", False))
        self._fcam_trace_max_steps = int(args.get("fcam_trace_max_steps", 20))
        self._fcam_phase0_check = bool(args.get("fcam_phase0_check", False))
        self._fcam_phase0_done = False
        self._fcam_cwo_ratio = bool(args.get("fcam_cwo_ratio", False))
        self._fcam_cwo_lam_norm = str(args.get("fcam_cwo_lam_norm", "none")).lower()
        self._fcam_cwo_lam_clip = float(args.get("fcam_cwo_lam_clip", 0.0))

        # Logging
        self._fcam_save_json = bool(args.get("fcam_save_json", True))
        self._fcam_track_mechanism = bool(args.get("fcam_track_mechanism", True))

        # Old-data source
        self._fcam_mem_source = str(args.get("fcam_mem_source", "memory")).lower()
        self._fcam_mem_fallback_train = bool(args.get("fcam_mem_fallback_train", True))

        # Runtime state
        self._teacher_net = None
        self._task_bank: Dict[int, Dict[str, object]] = {}
        self._U_task: Optional[torch.Tensor] = None
        self._U_hist: Optional[torch.Tensor] = None

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
        raise ValueError("Unexpected batch format for FCAM-LoRA.")

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

    def _flatten_grads_from_autograd(
        self, grads, params: List[torch.nn.Parameter], device: torch.device
    ) -> Optional[torch.Tensor]:
        chunks = []
        has_any = False
        for g, p in zip(grads, params):
            if g is None:
                chunks.append(torch.zeros(p.numel(), device=device, dtype=torch.float32))
            else:
                chunks.append(g.detach().reshape(-1).to(device=device, dtype=torch.float32))
                has_any = True
        if not has_any:
            return None
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
    # Subspace estimation
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
                if bidx >= int(self._fcam_max_batches):
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
            p_y = torch.clamp(p_y, min=float(self._fcam_eps))

            if self._fcam_whiten_h:
                inv_sqrt_h = torch.rsqrt(var_h + float(self._fcam_tau_h))
            else:
                inv_sqrt_h = torch.ones_like(var_h)
            if self._fcam_whiten_y:
                inv_sqrt_y = torch.rsqrt(p_y + float(self._fcam_tau_y))
            else:
                inv_sqrt_y = torch.ones_like(p_y)
            m = (inv_sqrt_h.unsqueeze(1) * c_hy) * inv_sqrt_y.unsqueeze(0)
            if not torch.isfinite(m).all():
                m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

            solver = self._fcam_subspace_solver
            if solver not in {"gram_eig", "svd"}:
                solver = "gram_eig"

            if solver == "svd":
                m_f = m.to(dtype=torch.float32)
                try:
                    u, s, _vh = torch.linalg.svd(m_f, full_matrices=False)
                except Exception as _svd_exc:
                    try:
                        self._log(f"[FCAM][SVD] fallback to CPU: {_svd_exc}")
                    except Exception:
                        pass
                    try:
                        m_cpu = m_f.detach().cpu().double()
                        u, s, _vh = torch.linalg.svd(m_cpu, full_matrices=False)
                        u = u.to(device=self._device, dtype=torch.float32)
                        s = s.to(device=self._device, dtype=torch.float32)
                    except Exception:
                        return None, None
                if u.numel() == 0:
                    return None, None
                k = max(1, min(int(self._fcam_topk), int(u.shape[1])))
                u_t = u[:, :k].contiguous()
                return u_t.detach().cpu(), s.detach().cpu()

            # gram_eig path (stable for p >> c)
            try:
                m64 = m.to(dtype=torch.float64)
                B = m64.transpose(0, 1) @ m64  # [c, c]
                B = 0.5 * (B + B.transpose(0, 1))
                evals, V = torch.linalg.eigh(B)
                idx = torch.argsort(evals, descending=True)
                evals = torch.clamp(evals[idx], min=0.0)
                V = V[:, idx]
                s = torch.sqrt(evals + float(self._fcam_eps))
                valid = (s > 1e-8).nonzero(as_tuple=False).numel()
                k = max(1, min(int(self._fcam_topk), int(valid)))
                Vk = V[:, :k]
                U = m64 @ Vk
                U = U / (s[:k].unsqueeze(0) + float(self._fcam_eps))
                U, _ = torch.linalg.qr(U, mode="reduced")
                return U.to(dtype=torch.float32).detach().cpu(), s.to(dtype=torch.float32).detach().cpu()
            except Exception as _eig_exc:
                try:
                    self._log(f"[FCAM][GramEig] fallback to CPU: {_eig_exc}")
                except Exception:
                    pass
                try:
                    m_cpu = m.detach().cpu().double()
                    B = m_cpu.transpose(0, 1) @ m_cpu
                    B = 0.5 * (B + B.transpose(0, 1))
                    evals, V = torch.linalg.eigh(B)
                    idx = torch.argsort(evals, descending=True)
                    evals = torch.clamp(evals[idx], min=0.0)
                    V = V[:, idx]
                    s = torch.sqrt(evals + float(self._fcam_eps))
                    valid = (s > 1e-8).nonzero(as_tuple=False).numel()
                    k = max(1, min(int(self._fcam_topk), int(valid)))
                    Vk = V[:, :k]
                    U = m_cpu @ Vk
                    U = U / (s[:k].unsqueeze(0) + float(self._fcam_eps))
                    U, _ = torch.linalg.qr(U, mode="reduced")
                    return U.to(dtype=torch.float32).detach().cpu(), s.to(dtype=torch.float32).detach().cpu()
                except Exception:
                    return None, None
        finally:
            module.train(was_training)

    # ------------------------------------------------------------------
    # Task memory + cached reference features
    # ------------------------------------------------------------------
    def _build_task_mem_dataset(self, mem_data, mem_targets):
        dm = getattr(self, "data_manager", None)
        if dm is None:
            return None
        mode = "test" if self._fcam_mem_mode not in {"train", "test", "flip"} else self._fcam_mem_mode
        try:
            dataset_mem = dm.get_dataset(
                [],
                source="train",
                mode=mode,
                appendent=(mem_data, mem_targets),
            )
        except Exception:
            try:
                from utils.data_manager import DummyDataset
                from torchvision import transforms
                if mode == "train":
                    trsf = transforms.Compose([*dm._train_trsf, *dm._common_trsf])
                else:
                    trsf = transforms.Compose([*dm._test_trsf, *dm._common_trsf])
                dataset_mem = DummyDataset(mem_data, mem_targets, trsf, dm.use_path)
            except Exception:
                return None
        return dataset_mem

    def _compute_ref_features(self, dataset_mem):
        if dataset_mem is None:
            return None, None
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset_mem, batch_size=self._fcam_mem_batch_size, shuffle=False, num_workers=0)
        module = self._unwrap_network()
        was_training = module.training
        module.eval()
        feats = []
        idx_list = []
        with torch.no_grad():
            for _idx, inputs, _targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                out = module(inputs)
                h = self._extract_features(out)
                feats.append(h.detach().cpu())
                idx_list.append(_idx.detach().cpu())
        module.train(was_training)
        if not feats:
            return None, None
        ref_feat = torch.cat(feats, dim=0)
        idx_all = torch.cat(idx_list, dim=0)
        # build index map (dataset idx -> row)
        idx_map = {}
        for pos, idx in enumerate(idx_all.tolist()):
            idx_map[int(idx)] = int(pos)
        return ref_feat, idx_map

    def _get_task_mem_subset(self, task_id: int):
        mem = self._get_memory()
        if mem is None:
            return None, None
        mem_data, mem_targets = mem
        start, end = self._task_class_range(task_id)
        mask = (mem_targets >= start) & (mem_targets < end)
        if not np.any(mask):
            return None, None
        mem_data_sel = mem_data[mask]
        mem_targets_sel = mem_targets[mask]
        return mem_data_sel, mem_targets_sel

    def _build_task_bank_entry(self, task_id: int):
        mem_data_sel, mem_targets_sel = (None, None)
        if self._fcam_mem_source == "memory":
            mem_data_sel, mem_targets_sel = self._get_task_mem_subset(task_id)
            if mem_data_sel is None and not self._fcam_mem_fallback_train:
                return
        if self._fcam_mem_source == "train" or (mem_data_sel is None and self._fcam_mem_fallback_train):
            dm = getattr(self, "data_manager", None)
            if dm is None:
                return
            start, end = self._task_class_range(task_id)
            try:
                dataset_mem = dm.get_dataset(np.arange(start, end), source="train", mode=self._fcam_mem_mode)
            except Exception:
                return
        else:
            dataset_mem = self._build_task_mem_dataset(mem_data_sel, mem_targets_sel)
        if dataset_mem is None:
            return
        ref_feat, idx_map = self._compute_ref_features(dataset_mem)
        if ref_feat is None or idx_map is None:
            return
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset_mem, batch_size=self._fcam_mem_batch_size, shuffle=True, num_workers=0)
        self._task_bank[task_id] = {
            "U": self._U_task,
            "Lambda": self._last_task_svals,
            "mem_data": mem_data_sel,
            "mem_targets": mem_targets_sel,
            "ref_feat": ref_feat,
            "ref_idx_map": idx_map,
            "mem_size": int(len(dataset_mem)),
            "loader": loader,
            "iter": None,
        }

    def _sample_task_batch(self, task_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        entry = self._task_bank.get(task_id, None)
        if entry is None:
            return None
        loader = entry.get("loader", None)
        if loader is None:
            return None
        it = entry.get("iter", None)
        if it is None:
            it = iter(loader)
            entry["iter"] = it
        for _ in range(8):
            try:
                idx, inputs, targets = next(it)
                entry["iter"] = it
                return idx, inputs, targets
            except StopIteration:
                it = iter(loader)
                entry["iter"] = it
        return None

    # ------------------------------------------------------------------
    # Loss terms
    # ------------------------------------------------------------------
    def _map_ref_indices(self, entry, idx_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        idx_map = entry.get("ref_idx_map", None)
        if idx_map is None:
            return None
        idx_list = idx_tensor.detach().cpu().tolist()
        mapped = []
        for idx in idx_list:
            if int(idx) in idx_map:
                mapped.append(int(idx_map[int(idx)]))
        if not mapped:
            return None
        mapped_t = torch.tensor(mapped, dtype=torch.long)
        if mapped_t.numel() != idx_tensor.numel():
            return None
        return mapped_t

    def _cwo_loss(self, aux_batches: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not aux_batches:
            return torch.zeros((), device=self._device), {
                "energy": float("nan"),
                "specforget": float("nan"),
            }
        losses = []
        energies = []
        specforgets = []
        weights = []
        for tid, batch in aux_batches.items():
            entry = self._task_bank.get(tid, None)
            if entry is None:
                continue
            idx, inputs, _targets = batch
            inputs = inputs.to(self._device, non_blocking=True)
            out = self._network(inputs)
            h_cur = self._extract_features(out)
            ref_feat_cpu = entry["ref_feat"]
            if ref_feat_cpu is None:
                # fallback: use teacher snapshot if available
                if self._teacher_net is None:
                    continue
                with torch.no_grad():
                    out_old = self._teacher_net(inputs)
                    ref_feat = self._extract_features(out_old)
            else:
                mapped = self._map_ref_indices(entry, idx)
                if mapped is None:
                    continue
                ref_feat = ref_feat_cpu[mapped].to(device=h_cur.device, dtype=h_cur.dtype)
            delta = h_cur - ref_feat
            if not torch.isfinite(delta).all():
                delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

            U = entry["U"]
            svals = entry["Lambda"]
            if U is None or svals is None:
                continue
            k = min(int(self._fcam_topk), int(U.shape[1]), int(svals.numel()))
            if k <= 0:
                continue
            U = U[:, :k].to(device=delta.device, dtype=delta.dtype)
            # Use singular values as sqrt(lambda) weights
            lam = svals[:k].to(device=delta.device, dtype=delta.dtype)
            if self._fcam_cwo_lam_norm == "max":
                lam = lam / (lam.max().clamp_min(float(self._fcam_eps)))
            elif self._fcam_cwo_lam_norm == "l2":
                lam = lam / (lam.norm().clamp_min(float(self._fcam_eps)))
            if self._fcam_cwo_lam_clip > 0:
                lam = lam.clamp(max=float(self._fcam_cwo_lam_clip))
            w = torch.clamp(lam, min=0.0)
            proj = delta @ (U * w.unsqueeze(0))
            if self._fcam_cwo_ratio:
                denom = delta.pow(2).sum(dim=1).clamp_min(float(self._fcam_eps))
                loss = (proj.pow(2).sum(dim=1) / denom).mean()
            else:
                loss = proj.pow(2).sum(dim=1).mean()

            # diagnostics
            energy = (h_cur @ U).pow(2).sum(dim=1).mean()
            spec = loss.detach()

            losses.append(loss)
            energies.append(energy.detach())
            specforgets.append(spec)
            if self._fcam_weight_mode == "lambda_sum":
                weights.append(float((lam * lam).sum().detach().item()))
            elif self._fcam_weight_mode == "size":
                weights.append(float(entry.get("mem_size", len(idx))))
            elif self._fcam_weight_mode == "recency":
                # newer tasks get higher weight
                weights.append(float(1.0 / max(1, (self._cur_task - tid))))
            else:
                weights.append(1.0)

        if not losses:
            return torch.zeros((), device=self._device), {
                "energy": float("nan"),
                "specforget": float("nan"),
            }

        wsum = sum(weights)
        norm = wsum if wsum > 0 else len(losses)
        loss_total = sum(w * l for w, l in zip(weights, losses)) / norm

        return loss_total, {
            "energy": float(torch.stack(energies).mean().item()) if energies else float("nan"),
            "specforget": float(torch.stack(specforgets).mean().item()) if specforgets else float("nan"),
        }

    def _subspace_nce(self, aux_batches: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if not self._fcam_use_nce or self._U_hist is None or self._U_hist.numel() == 0:
            return torch.zeros((), device=self._device)
        if not aux_batches:
            return torch.zeros((), device=self._device)
        # collect one batch from each sampled task
        inputs_list = []
        ref_list = []
        for tid, batch in aux_batches.items():
            entry = self._task_bank.get(tid, None)
            if entry is None:
                continue
            idx, inputs, _targets = batch
            inputs_list.append(inputs)
            ref_feat_cpu = entry.get("ref_feat", None)
            if ref_feat_cpu is None:
                continue
            mapped = self._map_ref_indices(entry, idx)
            if mapped is None:
                continue
            ref_feat = ref_feat_cpu[mapped]
            ref_list.append(ref_feat)
        if not inputs_list:
            return torch.zeros((), device=self._device)
        inputs = torch.cat(inputs_list, dim=0).to(self._device, non_blocking=True)
        ref_feat = torch.cat(ref_list, dim=0).to(self._device, non_blocking=True)

        out_new = self._network(inputs)
        h_new = self._extract_features(out_new)
        h_old = ref_feat

        u = self._U_hist.to(device=h_new.device, dtype=h_new.dtype)
        z_new = h_new @ u
        z_old = h_old @ u
        if self._fcam_nce_norm:
            z_new = F.normalize(z_new, dim=1)
            z_old = F.normalize(z_old, dim=1)
        logits = (z_new @ z_old.t()) / float(self._fcam_nce_tau)
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _run_fcam_epochs(self, train_loader, test_loader, optimizer, scheduler, epochs: int):
        prog_bar = tqdm(range(int(epochs)), disable=not self._is_main_process)

        curve_cwo = []
        curve_nce = []
        curve_energy = []
        curve_spec = []
        curve_align = []
        curve_test_acc = []
        curve_reg = []
        curve_task_ids = []
        curve_task_weights = []

        step_idx = 0

        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            # per-epoch accumulators
            sum_cwo = sum_nce = sum_reg = 0.0
            sum_energy = sum_spec = 0.0
            sum_align = 0.0
            cnt = 0

            for _i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                task_ids = []
                if self._fcam_enable and step_idx % int(max(1, self._fcam_aux_interval)) == 0:
                    task_ids = self._sample_past_tasks()

                # lock aux batches for this step (consistent for SAM/GAM double forward)
                aux_batches = {}
                if task_ids:
                    for tid in task_ids:
                        batch = self._sample_task_batch(tid)
                        if batch is not None:
                            aux_batches[tid] = batch

                # record task ids and weights (trace)
                if self._fcam_trace_taskids and len(curve_task_ids) < int(self._fcam_trace_max_steps):
                    curve_task_ids.append([int(t) for t in task_ids])
                    if task_ids:
                        w_list = []
                        for tid in task_ids:
                            if self._fcam_weight_mode == "lambda_sum":
                                entry = self._task_bank.get(tid, None)
                                if entry is not None and entry.get("Lambda") is not None:
                                    lam = entry["Lambda"][: int(self._fcam_topk)]
                                    w_list.append(float((lam * lam).sum().item()))
                                else:
                                    w_list.append(1.0)
                            elif self._fcam_weight_mode == "size":
                                entry = self._task_bank.get(tid, None)
                                w_list.append(float(len(entry.get("mem_targets", []))) if entry is not None else 1.0)
                            elif self._fcam_weight_mode == "recency":
                                w_list.append(float(1.0 / max(1, (self._cur_task - tid))))
                            else:
                                w_list.append(1.0)
                        curve_task_weights.append(w_list)

                def _compute_loss():
                    # CWO + NCE (recomputed every forward for optimizer consistency)
                    loss_cwo, diag_aux = self._cwo_loss(aux_batches) if aux_batches else (torch.zeros((), device=self._device), {})
                    loss_nce = self._subspace_nce(aux_batches) if aux_batches else torch.zeros((), device=self._device)
                    debug_losses = {}

                    # regularizer
                    reg_loss = torch.zeros((), device=self._device)
                    if self._lam_reg > 0:
                        params = [p for p in self._network.parameters() if p.requires_grad]
                        if params:
                            reg_loss = sum((p.pow(2).sum() for p in params)) / sum(p.numel() for p in params)

                    out = self._network(inputs)
                    logits = out["logits"] if isinstance(out, dict) else out
                    loss_task = self._class_loss(logits, targets)
                    total_loss = (
                        loss_task
                        + self._lam_cwo * loss_cwo
                        + self._lam_nce * loss_nce
                        + self._lam_reg * reg_loss
                    )
                    debug_losses["task"] = float(loss_task.detach().item())
                    debug_losses["cwo"] = float(loss_cwo.detach().item()) if torch.is_tensor(loss_cwo) else 0.0
                    debug_losses["nce"] = float(loss_nce.detach().item()) if torch.is_tensor(loss_nce) else 0.0
                    debug_losses["reg"] = float(reg_loss.detach().item()) if torch.is_tensor(reg_loss) else 0.0
                    debug_losses["total"] = float(total_loss.detach().item())
                    return total_loss, logits, loss_cwo, loss_nce, reg_loss, diag_aux, debug_losses

                # Optional gradient alignment diagnostic (cosine between grad(task) and grad(cwo))
                align_val = float("nan")
                if self._fcam_align_grad and aux_batches:
                    params = [p for p in self._network.parameters() if p.requires_grad]
                    if params:
                        optimizer.zero_grad()
                        out_tmp = self._network(inputs)
                        logits_tmp = out_tmp["logits"] if isinstance(out_tmp, dict) else out_tmp
                        loss_task_tmp = self._class_loss(logits_tmp, targets)
                        grads_task = torch.autograd.grad(loss_task_tmp, params, retain_graph=True, allow_unused=True)
                        g_task = self._flatten_grads_from_autograd(grads_task, params, self._device)
                        loss_cwo_tmp, _diag_tmp = self._cwo_loss(aux_batches)
                        grads_cwo = torch.autograd.grad(loss_cwo_tmp, params, retain_graph=True, allow_unused=True)
                        g_cwo = self._flatten_grads_from_autograd(grads_cwo, params, self._device)
                        if g_task is not None and g_cwo is not None and g_task.numel() > 0 and g_cwo.numel() > 0:
                            denom = (g_task.norm() * g_cwo.norm() + 1e-12)
                            align_val = float((g_task @ g_cwo / denom).item())

                if self._optimizer_type == "gam":
                    def closure():
                        optimizer.zero_grad()
                        loss, logits, loss_cwo, loss_nce, reg_loss, diag_aux, debug_losses = _compute_loss()
                        loss.backward()
                        return {"logits": logits.detach(), "diag_aux": diag_aux, "loss_cwo": loss_cwo.detach(), "loss_nce": loss_nce.detach(), "loss_reg": reg_loss.detach(), "debug": debug_losses}, loss.detach()

                    outputs, loss_value = optimizer.step(closure=closure)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    diag_aux = outputs.get("diag_aux", {})
                    loss_cwo = outputs.get("loss_cwo", torch.zeros((), device=self._device))
                    loss_nce = outputs.get("loss_nce", torch.zeros((), device=self._device))
                    reg_loss = outputs.get("loss_reg", torch.zeros((), device=self._device))
                    debug_losses = outputs.get("debug", None)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)

                elif self._optimizer_type == "sam":
                    optimizer.zero_grad()
                    loss, logits, loss_cwo, loss_nce, reg_loss, diag_aux, debug_losses = _compute_loss()
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    loss2, logits2, loss_cwo, loss_nce, reg_loss, diag_aux, debug_losses2 = _compute_loss()
                    loss2.backward()
                    optimizer.second_step(zero_grad=True)
                    logits = logits2.detach()
                    losses += float(loss2.detach().item())
                    if self._fcam_phase0_check and (not self._fcam_phase0_done) and debug_losses is not None and debug_losses2 is not None:
                        phase0_losses = {
                            "task": int(self._cur_task),
                            "sam_first": debug_losses,
                            "sam_second": debug_losses2,
                        }
                        save_dir = self.args.get("filepath", "./")
                        os.makedirs(save_dir, exist_ok=True)
                        with open(os.path.join(save_dir, "fcam_phase0_losses.json"), "w", encoding="utf-8") as f:
                            json.dump(phase0_losses, f, indent=2, ensure_ascii=False)
                else:
                    optimizer.zero_grad()
                    loss, logits, loss_cwo, loss_nce, reg_loss, diag_aux, debug_losses = _compute_loss()
                    loss.backward()
                    optimizer.step()
                    logits = logits.detach()
                    losses += float(loss.detach().item())

                sum_cwo += float(loss_cwo.detach().item()) if torch.is_tensor(loss_cwo) else 0.0
                sum_nce += float(loss_nce.detach().item()) if torch.is_tensor(loss_nce) else 0.0
                sum_reg += float(reg_loss.detach().item()) if torch.is_tensor(reg_loss) else 0.0
                if diag_aux:
                    if np.isfinite(diag_aux.get("energy", float("nan"))):
                        sum_energy += float(diag_aux.get("energy"))
                    if np.isfinite(diag_aux.get("specforget", float("nan"))):
                        sum_spec += float(diag_aux.get("specforget"))
                if np.isfinite(align_val):
                    sum_align += float(align_val)
                cnt += 1

                # Phase-0: gradient-path check (once per run, task>0)
                if self._fcam_phase0_check and (not self._fcam_phase0_done) and self._cur_task > 0 and task_ids:
                    params = [p for p in self._network.parameters() if p.requires_grad]
                    if params:
                        optimizer.zero_grad()
                        out_tmp = self._network(inputs)
                        logits_tmp = out_tmp["logits"] if isinstance(out_tmp, dict) else out_tmp
                        loss_task_tmp = self._class_loss(logits_tmp, targets)
                        grads_task = torch.autograd.grad(loss_task_tmp, params, retain_graph=True, allow_unused=True)
                        g_task = self._flatten_grads_from_autograd(grads_task, params, self._device)
                        loss_cwo_tmp, _diag_tmp = self._cwo_loss(aux_batches)
                        grads_cwo = torch.autograd.grad(loss_cwo_tmp, params, retain_graph=True, allow_unused=True)
                        g_cwo = self._flatten_grads_from_autograd(grads_cwo, params, self._device)
                        phase0 = {
                            "task": int(self._cur_task),
                            "gnorm_task": float(g_task.norm().item()) if g_task is not None else float("nan"),
                            "gnorm_cwo": float(g_cwo.norm().item()) if g_cwo is not None else float("nan"),
                            "task_ids": [int(t) for t in task_ids],
                            "aux_batch_sizes": {int(tid): int(batch[1].shape[0]) for tid, batch in aux_batches.items()},
                        }
                        # save immediately
                        save_dir = self.args.get("filepath", "./")
                        os.makedirs(save_dir, exist_ok=True)
                        with open(os.path.join(save_dir, "fcam_phase0.json"), "w", encoding="utf-8") as f:
                            json.dump(phase0, f, indent=2, ensure_ascii=False)
                        self._fcam_phase0_done = True

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                step_idx += 1

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)
            curve_cwo.append(sum_cwo / max(1, cnt))
            curve_nce.append(sum_nce / max(1, cnt))
            curve_energy.append(sum_energy / max(1, cnt))
            curve_spec.append(sum_spec / max(1, cnt))
            curve_align.append(sum_align / max(1, cnt))
            curve_reg.append(sum_reg / max(1, cnt))

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

        def _mean(lst):
            return float(np.mean(lst)) if lst else float("nan")
        self._last_task_stats = {
            "avg_cwo": _mean(curve_cwo),
            "avg_nce": _mean(curve_nce),
            "avg_reg": _mean(curve_reg),
            "avg_energy": _mean(curve_energy),
            "avg_specforget": _mean(curve_spec),
            "avg_align": _mean(curve_align),
        }
        self._last_task_curve = {
            "cwo": curve_cwo,
            "nce": curve_nce,
            "energy": curve_energy,
            "specforget": curve_spec,
            "align": curve_align,
            "reg": curve_reg,
            "test_acc": curve_test_acc,
            "task_ids": curve_task_ids if self._fcam_trace_taskids else [],
            "task_weights": curve_task_weights if self._fcam_trace_taskids else [],
        }

    def _save_fcam_stats(self, task_id: int):
        if not self._fcam_save_json:
            return
        save_dir = self.args.get("filepath", "./")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "fcam_stats.json")
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
                "u_task_dim": 0 if self._U_task is None else int(self._U_task.shape[1]),
                "u_hist_dim": 0 if self._U_hist is None else int(self._U_hist.shape[1]),
                "weight_mode": str(self._fcam_weight_mode),
                "align_grad": bool(self._fcam_align_grad),
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

        if self._fcam_enable:
            self._snapshot_teacher()

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

        self._run_fcam_epochs(train_loader, test_loader, optimizer, scheduler, epochs=epochs)

        # Task-end: estimate U_t and update history
        start, end = self._task_class_range(self._cur_task)
        self._U_task, svals = self._estimate_task_subspace(train_loader, start, end)
        self._last_task_svals = svals
        if self._U_task is not None:
            if self._U_hist is None:
                self._U_hist = self._U_task
            else:
                merged = torch.cat([self._U_hist, self._U_task], dim=1)
                q, _ = torch.linalg.qr(merged, mode="reduced")
                self._U_hist = q.contiguous()
            if self._U_hist is not None and self._fcam_hist_max_dim > 0:
                cap = min(int(self._fcam_hist_max_dim), int(self._U_hist.shape[1]))
                self._U_hist = self._U_hist[:, :cap].contiguous()

        self._save_fcam_stats(self._cur_task)

        save_dir = self.args.get("filepath", "./")
        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_dir, self._cur_task)

    # ------------------------------------------------------------------
    # Task-end hook: build bank entry
    # ------------------------------------------------------------------
    def after_task(self):
        # Update known classes first
        super().after_task()

        if not self._fcam_enable:
            return
        # Build per-task bank entry using rehearsal memory
        task_id = int(self._cur_task)
        if self._U_task is None:
            # Fallback: estimate U_t from train data
            dm = getattr(self, "data_manager", None)
            if dm is not None:
                start, end = self._task_class_range(task_id)
                try:
                    dataset = dm.get_dataset(np.arange(start, end), source="train", mode="train")
                    from torch.utils.data import DataLoader
                    loader = DataLoader(dataset, batch_size=self._fcam_mem_batch_size, shuffle=True, num_workers=0)
                    self._U_task, self._last_task_svals = self._estimate_task_subspace(loader, start, end)
                except Exception:
                    pass
        if self._U_task is None:
            return
        self._build_task_bank_entry(task_id)

    # ------------------------------------------------------------------
    # Helper: sample past tasks
    # ------------------------------------------------------------------
    def _sample_past_tasks(self) -> List[int]:
        keys = sorted([k for k in self._task_bank.keys() if k < self._cur_task])
        if not keys:
            return []
        if len(keys) <= self._fcam_past_tasks_per_step:
            return keys
        return list(np.random.choice(keys, size=int(self._fcam_past_tasks_per_step), replace=False))
