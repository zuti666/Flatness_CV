"""InfoBudget-GAM LoRA learner.

Two modes are supported via config:
1) Projection-only:
   - Project GAM update direction away from parameter directions that induce
     feature motion along historical task-information subspace U_<t.
2) Projection + drift penalty:
   - Add a feature drift penalty on replay/old samples in U_<t.

Task-information subspace is built in a head-free way using the whitened
cross-covariance operator from (features, labels).
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
    """SeqLoRA + InfoBudget projection/drift + subspace InfoNCE for GAM."""

    def __init__(self, args):
        super().__init__(args)

        # Main switches
        self._ibg_enable = bool(args.get("ibg_enable", True))
        self._ibg_use_projection = bool(args.get("ibg_use_projection", True))
        self._ibg_use_drift_penalty = bool(args.get("ibg_use_drift_penalty", False))

        # Subspace estimation
        self._ibg_topk = int(args.get("ibg_topk", max(1, int(args.get("lora_rank", 4)))))
        self._ibg_max_batches = int(args.get("ibg_max_batches", 8))
        self._ibg_tau_h = float(args.get("ibg_tau_h", 1e-4))
        self._ibg_tau_y = float(args.get("ibg_tau_y", 1e-6))
        self._ibg_eps = float(args.get("ibg_eps", 1e-12))
        self._ibg_basis_cap = int(args.get("ibg_basis_cap", 256))
        self._ibg_random_subspace = bool(args.get("ibg_random_subspace", False))
        self._ibg_shuffle_labels_subspace = bool(args.get("ibg_shuffle_labels_subspace", False))
        self._ibg_whiten_h = bool(args.get("ibg_whiten_h", True))
        self._ibg_whiten_y = bool(args.get("ibg_whiten_y", True))

        # Projection basis construction
        self._ibg_probe_batches = int(args.get("ibg_probe_batches", 2))
        self._ibg_probe_topk = int(args.get("ibg_probe_topk", 8))
        self._ibg_proj_basis_cap = int(args.get("ibg_proj_basis_cap", 64))
        self._ibg_random_protection_basis = bool(args.get("ibg_random_protection_basis", False))
        self._ibg_spec_k = int(args.get("ibg_spec_k", self._ibg_topk))
        self._ibg_prot_stat = str(args.get("ibg_prot_stat", "mean")).lower()

        # Drift penalty
        self._ibg_lambda_drift = float(args.get("ibg_lambda_drift", 0.5))
        # Subspace InfoNCE
        self._ibg_use_nce = bool(args.get("ibg_use_nce", False))
        self._ibg_lambda_nce = float(args.get("ibg_lambda_nce", 0.1))
        self._ibg_nce_tau = float(args.get("ibg_nce_tau", 0.07))
        self._ibg_nce_norm = bool(args.get("ibg_nce_norm", True))

        # Logging/persistence
        self._ibg_save_json = bool(args.get("ibg_save_json", True))
        self._ibg_track_mechanism = bool(args.get("ibg_track_mechanism", True))
        self._ibg_save_subspace = bool(args.get("ibg_save_subspace", True))
        self._ibg_subspace_dir = str(args.get("ibg_subspace_dir", "")).strip() or None
        self._ibg_ref_subspace_dir = args.get("ibg_ref_subspace_dir", None)

        # Old-data source: train | memory | stats
        self._ibg_mem_source = str(args.get("ibg_mem_source", "stats")).lower()
        self._ibg_mem_fallback_train = bool(args.get("ibg_mem_fallback_train", True))
        self._ibg_stats_bank_enable = bool(
            args.get("ibg_stats_bank_enable", self._ibg_mem_source == "stats")
        )
        self._ibg_stats_basis_cap = int(args.get("ibg_stats_basis_cap", self._ibg_proj_basis_cap))
        self._ibg_proj_basis_source = str(args.get("ibg_proj_basis_source", "auto")).lower()
        self._ibg_proj_basis_refresh = str(args.get("ibg_proj_basis_refresh", "task")).lower()
        self._ibg_track_old_acc = bool(args.get("ibg_track_old_acc", False))

        # Runtime state
        self._teacher_net = None
        self._U_task: Optional[torch.Tensor] = None      # [D, k], cpu
        self._U_hist: Optional[torch.Tensor] = None      # [D, k_hist], cpu
        self._U_hist_prev: Optional[torch.Tensor] = None # [D, k_hist_prev], cpu
        self._last_task_svals: Optional[torch.Tensor] = None  # singular values of M_t, cpu
        self._U_ref: Optional[torch.Tensor] = None       # [D, k_ref], cpu
        self._B_prot: Optional[torch.Tensor] = None      # [P_lora, k_proj], cpu
        self._lora_proj_params: List[torch.nn.Parameter] = []
        self._old_mem_iter = None
        self._ibg_mem_loader = None
        self._ibg_stats_bank: List[torch.Tensor] = []
        self._proj_ratio_sum = 0.0
        self._proj_ratio_cnt = 0
        self._last_task_avg_interf = float("nan")
        self._last_task_avg_interf_abs = float("nan")
        self._last_task_avg_effalign = float("nan")
        self._last_task_avg_interf_ref = float("nan")
        self._last_task_avg_drift = float("nan")
        self._last_task_drift_nonzero_ratio = float("nan")
        self._last_task_drift_enabled = False
        self._last_task_avg_nce = float("nan")
        self._last_task_overlap = float("nan")
        self._last_task_novel = float("nan")
        self._last_task_rot = float("nan")
        self._last_task_spec_energy = float("nan")
        self._last_task_spec_gap = float("nan")
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
        raise ValueError("Unexpected batch format for InfoBudget-GAM.")

    @staticmethod
    def _extract_features(outputs):
        feat = outputs["features"] if isinstance(outputs, dict) and "features" in outputs else outputs
        if feat.dim() == 3:
            feat = feat[:, 0, ...]
        elif feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        return feat.float()

    def _collect_lora_params(self) -> List[torch.nn.Parameter]:
        module = self._unwrap_network()
        picked: List[torch.nn.Parameter] = []
        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if ("linear_a" in n) or ("linear_b" in n) or ("wrapped_param" in n):
                picked.append(p)
        return picked

    def _lora_param_dim(self) -> int:
        return int(sum(int(p.numel()) for p in self._lora_proj_params))

    def _build_random_protection_basis(self, k_dim: int) -> Optional[torch.Tensor]:
        p_dim = self._lora_param_dim()
        if p_dim <= 0:
            return None
        k = max(1, min(int(k_dim), int(p_dim)))
        mat = torch.randn((p_dim, k), device=self._device, dtype=torch.float32)
        q, _ = torch.linalg.qr(mat, mode="reduced")
        return q[:, :k].contiguous().cpu()

    def _subspace_dir(self) -> str:
        base = self._ibg_subspace_dir or self.args.get("filepath", "./")
        os.makedirs(base, exist_ok=True)
        return base

    def _save_subspace(self, task_id: int):
        if not self._ibg_save_subspace:
            return
        base = self._subspace_dir()
        if self._U_task is not None:
            torch.save(self._U_task, os.path.join(base, f"U_task_t{task_id:02d}.pt"))
        if self._U_hist is not None:
            torch.save(self._U_hist, os.path.join(base, f"U_hist_t{task_id:02d}.pt"))

    def _load_ref_subspace(self, task_id: int) -> Optional[torch.Tensor]:
        ref_dir = self._ibg_ref_subspace_dir
        if not ref_dir:
            return None
        try:
            path = os.path.join(ref_dir, f"U_hist_t{task_id:02d}.pt")
        except Exception:
            return None
        if not os.path.exists(path):
            return None
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            return None

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

    def _flatten_current_grads(self, params: List[torch.nn.Parameter]) -> Optional[torch.Tensor]:
        if not params:
            return None
        device = params[0].device
        chunks = []
        has_any = False
        for p in params:
            if p.grad is None:
                chunks.append(torch.zeros(p.numel(), device=device, dtype=torch.float32))
            else:
                chunks.append(p.grad.detach().reshape(-1).to(device=device, dtype=torch.float32))
                has_any = True
        if not has_any:
            return None
        return torch.cat(chunks, dim=0)

    def _assign_flat_grads(self, params: List[torch.nn.Parameter], vec: torch.Tensor):
        ptr = 0
        for p in params:
            n = p.numel()
            if p.grad is not None:
                p.grad.copy_(vec[ptr : ptr + n].view_as(p).to(dtype=p.grad.dtype, device=p.grad.device))
            ptr += n

    @torch.no_grad()
    def _snapshot_teacher(self):
        base = self._unwrap_network()
        self._teacher_net = copy.deepcopy(base).to(self._device).eval()
        for p in self._teacher_net.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Head-free task subspace
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _estimate_task_subspace(self, loader, known_classes: int, total_classes: int) -> Optional[torch.Tensor]:
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

            for bidx, batch in enumerate(loader):
                if bidx >= int(self._ibg_max_batches):
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
                if self._ibg_shuffle_labels_subspace and y_local.numel() > 1:
                    perm = torch.randperm(y_local.numel(), device=y_local.device)
                    y_local = y_local[perm]

                out = module(inputs)
                h = self._extract_features(out).to(dtype=torch.float64)
                y = F.one_hot(y_local.long(), num_classes=c_task).to(dtype=torch.float64)

                sum_h += h.sum(dim=0)
                sum_h2 += (h * h).sum(dim=0)
                sum_y += y.sum(dim=0)
                sum_hy += h.transpose(0, 1) @ y
                n_total += int(h.size(0))

            if n_total <= 1:
                return None

            n = float(n_total)
            mu_h = sum_h / n
            p_y = sum_y / n
            c_hy = (sum_hy / n) - mu_h.unsqueeze(1) * p_y.unsqueeze(0)
            var_h = torch.clamp((sum_h2 / n) - mu_h * mu_h, min=0.0)

            if self._ibg_whiten_h:
                inv_sqrt_h = torch.rsqrt(var_h + float(self._ibg_tau_h))
            else:
                inv_sqrt_h = torch.ones_like(var_h)
            if self._ibg_whiten_y:
                inv_sqrt_y = torch.rsqrt(p_y + float(self._ibg_tau_y))
            else:
                inv_sqrt_y = torch.ones_like(p_y)
            m = (inv_sqrt_h.unsqueeze(1) * c_hy) * inv_sqrt_y.unsqueeze(0)  # [D, C_t]

            u, s, _vh = torch.linalg.svd(m.to(dtype=torch.float32), full_matrices=False)
            if u.numel() == 0:
                return None
            k = max(1, min(int(self._ibg_topk), int(u.shape[1])))
            u_t = u[:, :k].contiguous()
            try:
                self._last_task_svals = s.detach().cpu()
            except Exception:
                self._last_task_svals = None

            if self._ibg_random_subspace:
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
    # Projection basis in LoRA parameter space
    # ------------------------------------------------------------------
    def _iter_old_memory_batches(self, loader, max_batches: int, *, filter_old: bool = True):
        if loader is None or self._known_classes <= 0:
            return
        yielded = 0
        for batch in loader:
            inputs, targets = self._unwrap_batch(batch)
            if filter_old:
                mask = targets < int(self._known_classes)
            else:
                mask = torch.ones_like(targets, dtype=torch.bool)
            if mask.any():
                yield inputs[mask], targets[mask]
                yielded += 1
                if yielded >= int(max_batches):
                    break

    def _build_old_train_loader(self):
        """Build a loader from old-class TRAIN split to avoid test leakage."""
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
        bs = int(self.args.get("ibg_mem_batch_size", self.args.get("batch_size", 128)))
        nw = int(self.args.get("train_num_workers", 0))
        return DataLoader(dataset_old, batch_size=bs, shuffle=True, num_workers=nw)

    def _build_replay_loader(self):
        """Build a loader from replay memory (exemplars) if available."""
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
        bs = int(self.args.get("ibg_mem_batch_size", self.args.get("batch_size", 128)))
        nw = int(self.args.get("train_num_workers", 0))
        return DataLoader(dataset_mem, batch_size=bs, shuffle=True, num_workers=nw)

    def _build_basis_from_bank(self) -> Optional[torch.Tensor]:
        if not self._ibg_stats_bank:
            return None
        try:
            mat = torch.cat(self._ibg_stats_bank, dim=1)
        except Exception:
            return None
        q, _ = torch.linalg.qr(mat, mode="reduced")
        cap = max(1, min(int(self._ibg_stats_basis_cap), int(q.shape[1])))
        return q[:, :cap].contiguous().cpu()

    def _build_protection_basis(self, mem_loader, *, u_basis: Optional[torch.Tensor] = None, filter_old: bool = True) -> Optional[torch.Tensor]:
        if u_basis is None:
            u_basis = self._U_hist
        if u_basis is None or u_basis.numel() == 0:
            return None
        params = self._lora_proj_params
        if not params:
            return None

        k_u = min(int(self._ibg_probe_topk), int(u_basis.shape[1]))
        if k_u <= 0:
            return None
        u_hist = u_basis[:, :k_u].to(device=self._device, dtype=torch.float32)

        vecs = []
        module = self._unwrap_network()
        was_training = module.training
        module.train()
        try:
            for inputs_old, _targets_old in self._iter_old_memory_batches(
                mem_loader, self._ibg_probe_batches, filter_old=filter_old
            ) or []:
                inputs_old = inputs_old.to(self._device, non_blocking=True)
                out = module(inputs_old)
                h = self._extract_features(out)
                raw = h @ u_hist  # [B, k_u]
                stat = self._ibg_prot_stat
                if stat == "energy":
                    scores = (raw ** 2).mean(dim=0)
                elif stat == "abs":
                    scores = raw.abs().mean(dim=0)
                else:
                    scores = raw.mean(dim=0)
                for i in range(k_u):
                    module.zero_grad(set_to_none=True)
                    grads = torch.autograd.grad(
                        scores[i],
                        params,
                        retain_graph=(i < k_u - 1),
                        allow_unused=True,
                    )
                    v = self._flatten_grads_from_autograd(grads, params, self._device)
                    if v is None:
                        continue
                    nrm = torch.linalg.norm(v)
                    if float(nrm.item()) <= 1e-12:
                        continue
                    vecs.append((v / nrm).detach())

            if not vecs:
                return None
            mat = torch.stack(vecs, dim=1)  # [P, M]
            q, _ = torch.linalg.qr(mat, mode="reduced")
            cap = max(1, min(int(self._ibg_proj_basis_cap), int(q.shape[1])))
            return q[:, :cap].contiguous().cpu()
        finally:
            module.train(was_training)
            module.zero_grad(set_to_none=True)

    def _project_current_grads(self, *, count_ratio: bool = True) -> None:
        """Project current LoRA gradients to complement of protection basis."""
        if self._B_prot is None or self._B_prot.numel() == 0:
            return
        params = self._lora_proj_params
        if not params:
            return
        g = self._flatten_current_grads(params)
        if g is None:
            return

        b = self._B_prot.to(device=g.device, dtype=g.dtype)
        if b.shape[0] != g.numel():
            return
        coeff = b.transpose(0, 1) @ g
        g_proj = b @ coeff
        g_new = g - g_proj

        if count_ratio:
            denom = float(torch.dot(g, g).item()) + float(self._ibg_eps)
            ratio = float(torch.dot(g_proj, g_proj).item()) / denom
            self._proj_ratio_sum += ratio
            self._proj_ratio_cnt += 1

        self._assign_flat_grads(params, g_new)

    def _project_gam_direction(self, _param_groups=None):
        """Project GAM direction via gradient projection helper."""
        self._project_current_grads(count_ratio=True)

    # ------------------------------------------------------------------
    # Loss and training loop
    # ------------------------------------------------------------------
    def _class_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self._known_classes > 0:
            fake_targets = targets - self._known_classes
            return F.cross_entropy(logits[:, self._known_classes :], fake_targets)
        return F.cross_entropy(logits, targets)

    def _next_old_mem_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self._ibg_mem_source == "stats":
            return None
        if not (self._ibg_use_drift_penalty or self._ibg_track_mechanism):
            return None
        if self._known_classes <= 0 or self._ibg_mem_loader is None:
            return None
        if self._old_mem_iter is None:
            self._old_mem_iter = iter(self._ibg_mem_loader)
        for _ in range(8):
            try:
                batch = next(self._old_mem_iter)
            except StopIteration:
                self._old_mem_iter = iter(self._ibg_mem_loader)
                batch = next(self._old_mem_iter)
            inputs, targets = self._unwrap_batch(batch)
            mask = targets < int(self._known_classes)
            if mask.any():
                return inputs[mask], targets[mask]
        return None

    def _composite_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mem_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        out = self._network(inputs)
        logits = out["logits"] if isinstance(out, dict) else out
        h_new_cur = self._extract_features(out)
        loss_task = self._class_loss(logits, targets)
        loss_drift = torch.zeros((), device=logits.device, dtype=logits.dtype)
        loss_nce = torch.zeros((), device=logits.device, dtype=logits.dtype)
        interf = torch.tensor(float("nan"), device=logits.device, dtype=logits.dtype)
        interf_ref = torch.tensor(float("nan"), device=logits.device, dtype=logits.dtype)
        eff_align = torch.tensor(float("nan"), device=logits.device, dtype=logits.dtype)

        if self._ibg_enable and self._teacher_net is not None and self._U_task is not None and self._U_task.numel() > 0:
            with torch.no_grad():
                out_old_cur = self._teacher_net(inputs)
                h_old_cur = self._extract_features(out_old_cur).to(device=h_new_cur.device, dtype=h_new_cur.dtype)
            delta_cur = h_new_cur - h_old_cur
            denom_cur = delta_cur.pow(2).sum(dim=1).mean() + float(self._ibg_eps)
            u_task = self._U_task.to(device=delta_cur.device, dtype=delta_cur.dtype)
            proj_cur = delta_cur @ u_task
            eff_align = proj_cur.pow(2).sum(dim=1).mean() / denom_cur

        if (
            self._ibg_enable
            and mem_batch is not None
            and self._teacher_net is not None
            and self._U_hist is not None
            and self._U_hist.numel() > 0
        ):
            mem_inputs, _mem_targets = mem_batch
            mem_inputs = mem_inputs.to(self._device, non_blocking=True)
            out_new_mem = self._network(mem_inputs)
            h_new = self._extract_features(out_new_mem)
            with torch.no_grad():
                out_old_mem = self._teacher_net(mem_inputs)
                h_old = self._extract_features(out_old_mem).to(device=h_new.device, dtype=h_new.dtype)
            delta = h_new - h_old
            u_hist = self._U_hist.to(device=delta.device, dtype=delta.dtype)
            proj = delta @ u_hist
            denom = delta.pow(2).sum(dim=1).mean() + float(self._ibg_eps)
            interf_abs = proj.pow(2).sum(dim=1).mean()
            interf = interf_abs / denom
            if self._ibg_use_drift_penalty:
                loss_drift = interf
            if self._ibg_use_nce:
                z_new = h_new @ u_hist
                z_old = h_old @ u_hist
                if self._ibg_nce_norm:
                    z_new = F.normalize(z_new, dim=1)
                    z_old = F.normalize(z_old, dim=1)
                logits_nce = (z_new @ z_old.t()) / float(self._ibg_nce_tau)
                labels = torch.arange(logits_nce.size(0), device=logits_nce.device)
                loss_nce = F.cross_entropy(logits_nce, labels)
            if self._U_ref is not None and self._U_ref.numel() > 0:
                u_ref = self._U_ref.to(device=delta.device, dtype=delta.dtype)
                proj_ref = delta @ u_ref
                interf_ref = proj_ref.pow(2).sum(dim=1).mean() / denom

        loss = (
            loss_task
            + float(self._ibg_lambda_drift) * loss_drift
            + float(self._ibg_lambda_nce) * loss_nce
        )
        diag = {
            "task": float(loss_task.detach().item()),
            "drift": float(loss_drift.detach().item()),
            "nce": float(loss_nce.detach().item()),
            "interf": float(interf.detach().item()) if torch.isfinite(interf) else float("nan"),
            "interf_abs": float(interf_abs.detach().item()) if "interf_abs" in locals() and torch.isfinite(interf_abs) else float("nan"),
            "delta_abs": float(denom.detach().item()) if "denom" in locals() and torch.isfinite(denom) else float("nan"),
            "interf_ref": float(interf_ref.detach().item()) if torch.isfinite(interf_ref) else float("nan"),
            "eff_align": float(eff_align.detach().item()) if torch.isfinite(eff_align) else float("nan"),
            "total": float(loss.detach().item()),
        }
        return loss, logits, diag

    def _run_budget_epochs(self, train_loader, test_loader, optimizer, scheduler, epochs: int):
        prog_bar = tqdm(range(int(epochs)), disable=not self._is_main_process)
        task_interf_sum = 0.0
        task_interf_cnt = 0
        task_interf_abs_sum = 0.0
        task_interf_abs_cnt = 0
        task_interf_ref_sum = 0.0
        task_interf_ref_cnt = 0
        task_eff_sum = 0.0
        task_eff_cnt = 0
        task_drift_sum = 0.0
        task_drift_cnt = 0
        task_drift_nonzero_cnt = 0
        task_drift_step_cnt = 0
        task_nce_sum = 0.0
        task_nce_cnt = 0
        curve_proj_ratio = []
        curve_interf = []
        curve_interf_abs = []
        curve_drift = []
        curve_nce = []
        curve_eff = []
        curve_test_acc = []
        curve_old_acc = []
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            losses_task = 0.0
            losses_drift = 0.0
            losses_nce = 0.0
            losses_interf = 0.0
            losses_interf_abs = 0.0
            losses_eff = 0.0
            cnt_interf = 0
            cnt_interf_abs = 0
            cnt_eff = 0
            correct, total = 0, 0
            proj_sum_before = self._proj_ratio_sum
            proj_cnt_before = self._proj_ratio_cnt

            if (
                self._ibg_enable
                and self._ibg_use_projection
                and self._ibg_proj_basis_refresh == "epoch"
                and self._ibg_mem_source != "stats"
                and self._U_hist is not None
                and self._U_hist.numel() > 0
            ):
                try:
                    self._B_prot = self._build_protection_basis(self._ibg_mem_loader)
                except Exception:
                    pass

            for _i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                mem_batch = self._next_old_mem_batch()
                do_proj = (
                    self._ibg_enable
                    and self._ibg_use_projection
                    and self._B_prot is not None
                    and self._B_prot.numel() > 0
                )

                if self._optimizer_type == "gam":
                    def closure():
                        optimizer.zero_grad()
                        loss, logits, diag = self._composite_loss(inputs, targets, mem_batch)
                        loss.backward()
                        return {"logits": logits.detach(), "diag": diag}, loss.detach()

                    outputs, loss_value = optimizer.step(closure=closure)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    diag = outputs.get("diag", {"task": 0.0, "drift": 0.0}) if isinstance(outputs, dict) else {}
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    losses_task += float(diag.get("task", 0.0))
                    losses_drift += float(diag.get("drift", 0.0))
                    losses_nce += float(diag.get("nce", 0.0))
                    task_drift_step_cnt += 1
                    if float(diag.get("drift", 0.0)) > 0.0:
                        task_drift_nonzero_cnt += 1
                    v_interf = float(diag.get("interf", float("nan")))
                    if np.isfinite(v_interf):
                        losses_interf += v_interf
                        cnt_interf += 1
                    v_interf_abs = float(diag.get("interf_abs", float("nan")))
                    if np.isfinite(v_interf_abs):
                        losses_interf_abs += v_interf_abs
                        cnt_interf_abs += 1
                    v_interf_ref = float(diag.get("interf_ref", float("nan")))
                    if np.isfinite(v_interf_ref):
                        task_interf_ref_sum += v_interf_ref
                        task_interf_ref_cnt += 1
                    v_eff = float(diag.get("eff_align", float("nan")))
                    if np.isfinite(v_eff):
                        losses_eff += v_eff
                        cnt_eff += 1

                elif self._optimizer_type == "sam":
                    optimizer.zero_grad()
                    loss, logits, diag = self._composite_loss(inputs, targets, mem_batch)
                    loss.backward()
                    if do_proj:
                        # Project perturbation gradient but do not count ratio yet
                        self._project_current_grads(count_ratio=False)
                    optimizer.first_step(zero_grad=True)
                    second_loss, logits2, diag2 = self._composite_loss(inputs, targets, mem_batch)
                    second_loss.backward()
                    if do_proj:
                        # Project update gradient and count projection ratio
                        self._project_current_grads(count_ratio=True)
                    optimizer.second_step(zero_grad=True)
                    logits = logits2.detach()
                    losses += float(second_loss.detach().item())
                    losses_task += float(diag2["task"])
                    losses_drift += float(diag2["drift"])
                    losses_nce += float(diag2.get("nce", 0.0))
                    task_drift_step_cnt += 1
                    if float(diag2.get("drift", 0.0)) > 0.0:
                        task_drift_nonzero_cnt += 1
                    v_interf = float(diag2.get("interf", float("nan")))
                    if np.isfinite(v_interf):
                        losses_interf += v_interf
                        cnt_interf += 1
                    v_interf_abs = float(diag2.get("interf_abs", float("nan")))
                    if np.isfinite(v_interf_abs):
                        losses_interf_abs += v_interf_abs
                        cnt_interf_abs += 1
                    v_interf_ref = float(diag2.get("interf_ref", float("nan")))
                    if np.isfinite(v_interf_ref):
                        task_interf_ref_sum += v_interf_ref
                        task_interf_ref_cnt += 1
                    v_eff = float(diag2.get("eff_align", float("nan")))
                    if np.isfinite(v_eff):
                        losses_eff += v_eff
                        cnt_eff += 1

                else:
                    optimizer.zero_grad()
                    loss, logits, diag = self._composite_loss(inputs, targets, mem_batch)
                    loss.backward()
                    if do_proj:
                        self._project_current_grads(count_ratio=True)
                    optimizer.step()
                    logits = logits.detach()
                    losses += float(loss.detach().item())
                    losses_task += float(diag["task"])
                    losses_drift += float(diag["drift"])
                    losses_nce += float(diag.get("nce", 0.0))
                    task_drift_step_cnt += 1
                    if float(diag.get("drift", 0.0)) > 0.0:
                        task_drift_nonzero_cnt += 1
                    v_interf = float(diag.get("interf", float("nan")))
                    if np.isfinite(v_interf):
                        losses_interf += v_interf
                        cnt_interf += 1
                    v_interf_abs = float(diag.get("interf_abs", float("nan")))
                    if np.isfinite(v_interf_abs):
                        losses_interf_abs += v_interf_abs
                        cnt_interf_abs += 1
                    v_interf_ref = float(diag.get("interf_ref", float("nan")))
                    if np.isfinite(v_interf_ref):
                        task_interf_ref_sum += v_interf_ref
                        task_interf_ref_cnt += 1
                    v_eff = float(diag.get("eff_align", float("nan")))
                    if np.isfinite(v_eff):
                        losses_eff += v_eff
                        cnt_eff += 1

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)
            epoch_interf = (losses_interf / cnt_interf) if cnt_interf > 0 else float("nan")
            epoch_interf_abs = (losses_interf_abs / cnt_interf_abs) if cnt_interf_abs > 0 else float("nan")
            epoch_eff = (losses_eff / cnt_eff) if cnt_eff > 0 else float("nan")
            epoch_drift = (losses_drift / max(1, len(train_loader))) if len(train_loader) > 0 else float("nan")
            epoch_nce = (losses_nce / max(1, len(train_loader))) if len(train_loader) > 0 else float("nan")
            proj_sum_delta = self._proj_ratio_sum - proj_sum_before
            proj_cnt_delta = self._proj_ratio_cnt - proj_cnt_before
            if proj_cnt_delta > 0:
                epoch_proj_ratio = float(proj_sum_delta / proj_cnt_delta)
            else:
                epoch_proj_ratio = float("nan")
            if np.isfinite(epoch_interf):
                task_interf_sum += epoch_interf
                task_interf_cnt += 1
            if np.isfinite(epoch_interf_abs):
                task_interf_abs_sum += epoch_interf_abs
                task_interf_abs_cnt += 1
            if np.isfinite(epoch_eff):
                task_eff_sum += epoch_eff
                task_eff_cnt += 1
            if np.isfinite(epoch_drift):
                task_drift_sum += epoch_drift
                task_drift_cnt += 1
            if np.isfinite(epoch_nce):
                task_nce_sum += epoch_nce
                task_nce_cnt += 1
            curve_proj_ratio.append(epoch_proj_ratio)
            curve_interf.append(epoch_interf)
            curve_interf_abs.append(epoch_interf_abs)
            curve_drift.append(epoch_drift)
            curve_nce.append(epoch_nce)
            curve_eff.append(epoch_eff)
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                curve_test_acc.append(float(test_acc))
                if self._ibg_track_old_acc and self._ibg_mem_loader is not None:
                    try:
                        old_acc = self._compute_accuracy(self._network, self._ibg_mem_loader)
                        curve_old_acc.append(float(old_acc))
                    except Exception:
                        curve_old_acc.append(float("nan"))
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / len(train_loader):.3f} "
                    f"(task {losses_task / len(train_loader):.3f}, "
                    f"drift {losses_drift / len(train_loader):.3f}, "
                    f"interf {epoch_interf:.3f}, eff {epoch_eff:.3f}), "
                    f"Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}"
                )
            elif self._is_main_process:
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / len(train_loader):.3f} "
                    f"(task {losses_task / len(train_loader):.3f}, "
                    f"drift {losses_drift / len(train_loader):.3f}, "
                    f"interf {epoch_interf:.3f}, eff {epoch_eff:.3f}), "
                    f"Train_accy {train_acc:.2f}"
                )
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)
        self._last_task_avg_interf = (task_interf_sum / task_interf_cnt) if task_interf_cnt > 0 else float("nan")
        self._last_task_avg_interf_abs = (
            task_interf_abs_sum / task_interf_abs_cnt
        ) if task_interf_abs_cnt > 0 else float("nan")
        self._last_task_avg_effalign = (task_eff_sum / task_eff_cnt) if task_eff_cnt > 0 else float("nan")
        self._last_task_avg_interf_ref = (
            task_interf_ref_sum / task_interf_ref_cnt
        ) if task_interf_ref_cnt > 0 else float("nan")
        self._last_task_avg_drift = (task_drift_sum / task_drift_cnt) if task_drift_cnt > 0 else float("nan")
        self._last_task_drift_nonzero_ratio = (
            float(task_drift_nonzero_cnt) / float(task_drift_step_cnt)
            if task_drift_step_cnt > 0
            else float("nan")
        )
        self._last_task_drift_enabled = bool(self._ibg_use_drift_penalty and self._ibg_mem_source != "stats")
        self._last_task_avg_nce = (task_nce_sum / task_nce_cnt) if task_nce_cnt > 0 else float("nan")
        self._last_task_curve = {
            "proj_ratio": curve_proj_ratio,
            "interf": curve_interf,
            "interf_abs": curve_interf_abs,
            "drift": curve_drift,
            "nce": curve_nce,
            "eff_align": curve_eff,
            "test_acc": curve_test_acc,
            "old_acc": curve_old_acc,
        }

    def _save_ibg_stats(self, task_id: int):
        if not self._ibg_save_json:
            return
        save_dir = self.args.get("filepath", "./")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "infobudget_stats.json")
        payload = []
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = []
        avg_proj_ratio = (
            float(self._proj_ratio_sum / self._proj_ratio_cnt)
            if self._proj_ratio_cnt > 0
            else float("nan")
        )

        # ---- feature-space subspace similarity / novelty ----
        overlap = float("nan")
        novel = float("nan")
        if self._U_task is not None and self._U_task.numel() > 0:
            U_t = self._U_task
            r_t = int(U_t.shape[1])
            if self._U_hist_prev is not None and self._U_hist_prev.numel() > 0 and r_t > 0:
                U_prev = self._U_hist_prev
                p = int(min(U_prev.shape[1], U_t.shape[1]))
                if p > 0:
                    overlap = float((U_prev[:, :p].T @ U_t[:, :p]).pow(2).sum().item() / max(1, r_t))
                try:
                    proj = U_prev @ (U_prev.T @ U_t)
                    resid = U_t - proj
                    novel = float(resid.pow(2).sum().item() / max(1, r_t))
                except Exception:
                    novel = float("nan")

        # ---- bank rotation (principal angles between U_hist_{t-1} and U_hist_t) ----
        rot = float("nan")
        if (
            self._U_hist_prev is not None
            and self._U_hist_prev.numel() > 0
            and self._U_hist is not None
            and self._U_hist.numel() > 0
        ):
            U_prev = self._U_hist_prev
            U_cur = self._U_hist
            m = int(min(U_prev.shape[1], U_cur.shape[1]))
            if m > 0:
                try:
                    svals = torch.linalg.svdvals(U_prev[:, :m].T @ U_cur[:, :m])
                    svals = torch.clamp(svals, 0.0, 1.0)
                    rot = float((float(m) - (svals ** 2).sum().item()))
                except Exception:
                    rot = float("nan")

        # ---- spectral diagnostics from whitened cross-cov ----
        spec_energy = float("nan")
        spec_gap = float("nan")
        if self._last_task_svals is not None and self._last_task_svals.numel() > 0:
            svals = self._last_task_svals.to(dtype=torch.float64)
            k = int(min(int(self._ibg_spec_k), int(svals.numel())))
            if k > 0:
                denom = float((svals * svals).sum().item())
                if denom > 0:
                    spec_energy = float((svals[:k] * svals[:k]).sum().item() / denom)
                if k < int(svals.numel()):
                    spec_gap = float(svals[k - 1].item() / (svals[k].item() + float(self._ibg_eps)))

        self._last_task_overlap = overlap
        self._last_task_novel = novel
        self._last_task_rot = rot
        self._last_task_spec_energy = spec_energy
        self._last_task_spec_gap = spec_gap
        avg_interf = float(self._last_task_avg_interf)
        avg_interf_ref = float(self._last_task_avg_interf_ref)
        interf_valid = True
        if self._ibg_mem_source == "stats":
            avg_interf = None
            avg_interf_ref = None
            interf_valid = False

        payload.append(
            {
                "task": int(task_id),
                "u_task_dim": 0 if self._U_task is None else int(self._U_task.shape[1]),
                "u_hist_dim": 0 if self._U_hist is None else int(self._U_hist.shape[1]),
                "proj_basis_dim": 0 if self._B_prot is None else int(self._B_prot.shape[1]),
                "avg_proj_ratio": avg_proj_ratio,
                "avg_interf": avg_interf,
                "avg_interf_ref": avg_interf_ref,
                "avg_interf_valid": bool(interf_valid),
                "avg_interf_abs": float(self._last_task_avg_interf_abs)
                if np.isfinite(self._last_task_avg_interf_abs)
                else None,
                "avg_eff_align": float(self._last_task_avg_effalign),
                "avg_drift_loss": float(self._last_task_avg_drift) if np.isfinite(self._last_task_avg_drift) else None,
                "avg_nce_loss": float(self._last_task_avg_nce) if np.isfinite(self._last_task_avg_nce) else None,
                "drift_nonzero_ratio": float(self._last_task_drift_nonzero_ratio)
                if np.isfinite(self._last_task_drift_nonzero_ratio)
                else None,
                "drift_enabled": bool(self._last_task_drift_enabled),
                "epoch_curve": self._last_task_curve if isinstance(self._last_task_curve, dict) else {},
                "overlap_u": float(self._last_task_overlap),
                "novel_u": float(self._last_task_novel),
                "rot_bank": float(self._last_task_rot),
                "spec_energy_k": float(self._last_task_spec_energy),
                "spec_gap_k": float(self._last_task_spec_gap),
                "spec_k": int(self._ibg_spec_k),
                "prot_stat": self._ibg_prot_stat,
                "proj_basis_source": self._ibg_proj_basis_source,
                "proj_basis_refresh": self._ibg_proj_basis_refresh,
                "variant": {
                    "random_B": bool(self._ibg_random_protection_basis),
                    "shuffle_label_U": bool(self._ibg_shuffle_labels_subspace),
                    "whiten_h": bool(self._ibg_whiten_h),
                    "whiten_y": bool(self._ibg_whiten_y),
                },
            }
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Main train entry
    # ------------------------------------------------------------------
    def _train(self, train_loader, test_loader):
        network = self._unwrap_network()
        # snapshot previous historical subspace for overlap/rotation metrics
        self._U_hist_prev = self._U_hist

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

        self._proj_ratio_sum = 0.0
        self._proj_ratio_cnt = 0
        self._old_mem_iter = None
        self._lora_proj_params = self._collect_lora_params()
        # Resolve old-data source
        self._ibg_mem_loader = None
        if self._ibg_mem_source == "memory":
            self._ibg_mem_loader = self._build_replay_loader()
            if self._ibg_mem_loader is None and self._ibg_mem_fallback_train:
                self._ibg_mem_loader = self._build_old_train_loader()
        elif self._ibg_mem_source == "stats":
            self._ibg_mem_loader = None
        else:
            self._ibg_mem_loader = self._build_old_train_loader()
        if self._ibg_mem_source == "stats" and self._ibg_use_drift_penalty:
            self._ibg_use_drift_penalty = False
            self._log("[InfoBudget] stats mem_source -> drift penalty disabled (no old samples).")

        if self._ibg_enable:
            self._snapshot_teacher()
            self._U_task = self._estimate_task_subspace(train_loader, self._known_classes, self._total_classes)
            self._U_ref = None
            if self._cur_task > 0 and self._ibg_ref_subspace_dir:
                self._U_ref = self._load_ref_subspace(self._cur_task - 1)
            self._B_prot = None
            if self._ibg_use_projection and self._U_hist is not None and self._U_hist.numel() > 0:
                src = self._ibg_proj_basis_source
                if src not in {"auto", "bank", "online"}:
                    src = "auto"
                if src == "bank":
                    learned_B = self._build_basis_from_bank()
                elif src == "online":
                    if self._ibg_mem_source == "stats":
                        learned_B = self._build_basis_from_bank()
                    else:
                        learned_B = self._build_protection_basis(self._ibg_mem_loader)
                else:  # auto
                    if self._ibg_mem_source == "stats":
                        learned_B = self._build_basis_from_bank()
                    else:
                        learned_B = self._build_protection_basis(self._ibg_mem_loader)
                if self._ibg_random_protection_basis:
                    if learned_B is not None and learned_B.numel() > 0:
                        k_dim = int(learned_B.shape[1])
                    else:
                        k_dim = max(1, min(int(self._ibg_probe_topk), int(self._ibg_proj_basis_cap)))
                    self._B_prot = self._build_random_protection_basis(k_dim)
                else:
                    self._B_prot = learned_B
            self._log(
                f"[InfoBudget] task={self._cur_task} "
                f"U_t_dim={0 if self._U_task is None else self._U_task.shape[1]} "
                f"U_hist_dim={0 if self._U_hist is None else self._U_hist.shape[1]} "
                f"B_dim={0 if self._B_prot is None else self._B_prot.shape[1]}"
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

        if (
            self._ibg_enable
            and self._optimizer_type == "gam"
            and self._ibg_use_projection
            and self._B_prot is not None
            and hasattr(optimizer, "set_direction_projector")
        ):
            optimizer.set_direction_projector(self._project_gam_direction)

        if self._optimizer_type in {"sgd", "sam", "gam"} and self._ibg_enable:
            self._run_budget_epochs(train_loader, test_loader, optimizer, scheduler, epochs=epochs)
        else:
            if self._cur_task == 0:
                super()._init_train(train_loader, test_loader, optimizer, scheduler)
            else:
                super()._update_representation(train_loader, test_loader, optimizer, scheduler)

        self._U_hist = self._merge_orth_basis(self._U_hist, self._U_task, self._ibg_basis_cap)
        self._save_subspace(self._cur_task)
        if self._ibg_stats_bank_enable and self._U_task is not None:
            try:
                basis_cur = self._build_protection_basis(
                    train_loader, u_basis=self._U_task, filter_old=False
                )
                if basis_cur is not None and basis_cur.numel() > 0:
                    self._ibg_stats_bank.append(basis_cur.detach().cpu())
            except Exception:
                pass
        self._save_ibg_stats(self._cur_task)

        save_dir = self.args.get("filepath", "./")
        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_dir, self._cur_task)
