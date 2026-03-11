"""SCM-Finetune: Slow Consolidation Module + standard fine-tuning fast layer.

Fast layer  : standard cross-entropy fine-tuning (zero modification).
Slow layer  : SlowConsolidationModule (SCM) — runs offline at every task boundary.

SCM algorithm (per layer):
  1. ΔW_t = W_t − W_init_t              (pure task-specific update;
                                          W_init_t = W_0 + ΔW_shared_{t-1})
  2. SVD(ΔW_t) → canonical atoms (u_i, v_i, σ_i)
  3. Store atoms in per-task history
  4. Greedy subspace selection: maximise  utility(a,b) − λ · risk(a,b)
       utility(a, b) = Σ_{s≤t} c_s(a,b)²   where c_s = Σ_i σ_i (a·u_i)(b·v_i)
                       ↑ cross-task historical reconstruction (Eckart-Young)
       risk(a, b)    = (a^T F_in  a / tr F_in)
                     · (b^T F_out b / tr F_out)
                       ↑ KFAC Kronecker Fisher  F_W ≈ F_out ⊗ F_in
  5. Re-set selected atom sigma  ← √utility  (reflects total historical energy)
  6. ΔW_shared = Σ_k σ_k · u_k · v_k^T
  7. Next task init: W_init_{t+1} = W_0 + ΔW_shared

Key corrections vs. first draft
  - Utility uses cross-task history (Σ c_t²), not just current-task σ²
  - Fisher uses KFAC (F_out ⊗ F_in) not diagonal
  - sigma re-set to √utility after selection

References:
  Task Arithmetic      Ilharco et al., ICLR 2023
  Fisher Merging       Matena & Raffel, NeurIPS 2022
  TIES-Merging         Yadav et al., NeurIPS 2023
  KFAC                 Martens & Grosse, ICML 2015
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_CL.baseLearner import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────
Atom      = Tuple[torch.Tensor, torch.Tensor, float]  # (u ∈ R^d_out, v ∈ R^d_in, σ)
LayerAtoms = List[Atom]
KFACPair  = Tuple[torch.Tensor, torch.Tensor]          # (F_in [d_in,d_in], F_out [d_out,d_out])


# ═════════════════════════════════════════════════════════════════════════════
#  KFAC Fisher estimator
# ═════════════════════════════════════════════════════════════════════════════

def estimate_kfac_fisher(
    model:         nn.Module,
    loader:        DataLoader,
    target_layers: Dict[str, nn.Linear],   # {param_name → Linear module}
    device:        torch.device,
    n_batches:     int = 20,
    gamma:         float = 1.0,            # decay for running accumulation (1 = sum)
) -> Dict[str, KFACPair]:
    """Estimate KFAC factors  F_in = X^T X / n,  F_out = G^T G / n  per layer.

    For weight W of a linear layer  y = x W^T:
      F_W ≈ F_out ⊗ F_in
    Risk of atom (u, v):
      risk(u,v) = (u^T F_out u / tr F_out) · (v^T F_in v / tr F_in)

    Returns:
        {param_name: (F_in [d_in,d_in], F_out [d_out,d_out])}
    """
    model.eval()

    # accumulators (CPU float32)
    acc_in:  Dict[str, torch.Tensor] = {}
    acc_out: Dict[str, torch.Tensor] = {}
    counts:  Dict[str, int]          = {}

    for name, layer in target_layers.items():
        d_in  = layer.weight.shape[1]
        d_out = layer.weight.shape[0]
        acc_in[name]  = torch.zeros(d_in,  d_in,  dtype=torch.float32)
        acc_out[name] = torch.zeros(d_out, d_out, dtype=torch.float32)
        counts[name]  = 0

    # ── hooks ──────────────────────────────────────────────────────────────
    input_cache: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_fwd_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach()               # [B, (N,) d_in]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            input_cache[name] = x.cpu()
        return hook

    def make_bwd_hook(name):
        def hook(module, grad_in, grad_out):
            g = grad_out[0].detach()          # [B, (N,) d_out]
            if g.dim() == 3:
                g = g.reshape(-1, g.shape[-1])
            g = g.cpu().float()
            x = input_cache.get(name)
            if x is None:
                return
            x = x.float()
            n = x.shape[0]
            acc_in[name]  += (x.t() @ x) / n
            acc_out[name] += (g.t() @ g) / n
            counts[name]  += 1
        return hook

    for name, layer in target_layers.items():
        hooks.append(layer.register_forward_hook(make_fwd_hook(name)))
        hooks.append(layer.register_full_backward_hook(make_bwd_hook(name)))

    # ── forward-backward passes ────────────────────────────────────────────
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= n_batches:
            break
        # handle both (idx, inputs, targets) and (inputs, targets) formats
        if len(batch) == 3:
            _, inputs, targets = batch
        else:
            inputs, targets = batch[0], batch[1]
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        out    = model(inputs)
        logits = out["logits"] if isinstance(out, dict) else out
        F.cross_entropy(logits, targets).backward()

    for h in hooks:
        h.remove()

    # ── normalise ──────────────────────────────────────────────────────────
    result: Dict[str, KFACPair] = {}
    for name in target_layers:
        c = max(1, counts[name])
        result[name] = (acc_in[name] / c, acc_out[name] / c)
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  Slow Consolidation Module
# ═════════════════════════════════════════════════════════════════════════════

class SlowConsolidationModule:
    """Offline knowledge consolidation at task boundaries.

    State (all CPU):
        _atoms         : {layer → current shared dict, list of (u,v,σ)}
        _history_atoms : {layer → list[per-task LayerAtoms]}  (never pruned)
        _kfac_in       : {layer → F_in  [d_in,  d_in ]}  accumulated
        _kfac_out      : {layer → F_out [d_out, d_out]}  accumulated
        _n_tasks       : tasks consolidated so far
    """

    def __init__(
        self,
        budget_K:    int   = 8,
        lambda_risk: float = 1.0,
        top_r:       int   = 8,
        kfac_gamma:  float = 1.0,   # 1 = sum; <1 = exponential decay
        sim_thresh:  float = 0.0,   # cosine sim threshold for early-exit dedup (0 = off)
    ):
        self.budget_K    = int(budget_K)
        self.lambda_risk = float(lambda_risk)
        self.top_r       = int(top_r)
        self.kfac_gamma  = float(kfac_gamma)
        self.sim_thresh  = float(sim_thresh)

        self._atoms:         Dict[str, LayerAtoms]          = {}
        self._history_atoms: Dict[str, List[LayerAtoms]]    = {}  # [task_id][atom_idx]
        self._kfac_in:       Dict[str, torch.Tensor]        = {}
        self._kfac_out:      Dict[str, torch.Tensor]        = {}
        self._n_tasks:       int                            = 0
        self._gamma          = float(kfac_gamma)   # local alias used in consolidate

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def consolidate(
        self,
        delta_w_dict: Dict[str, torch.Tensor],   # {layer → ΔW_t = W_t - W_0}
        kfac_dict:    Dict[str, KFACPair],        # {layer → (F_in, F_out)}
    ) -> None:
        """Run one consolidation step at the end of task t."""

        # ── 1. SVD → canonical atoms; append to history ───────────────────
        for name, dw in delta_w_dict.items():
            new_atoms = self._svd_atoms(dw.float().cpu(), self.top_r)
            if name not in self._history_atoms:
                self._history_atoms[name] = []
            self._history_atoms[name].append(new_atoms)

        # ── 2. Update KFAC accumulators ───────────────────────────────────
        for name, (fin, fout) in kfac_dict.items():
            fin_f  = fin.float().cpu()
            fout_f = fout.float().cpu()
            if name in self._kfac_in:
                self._kfac_in[name]  = self._gamma * self._kfac_in[name]  + fin_f
                self._kfac_out[name] = self._gamma * self._kfac_out[name] + fout_f
            else:
                self._kfac_in[name]  = fin_f.clone()
                self._kfac_out[name] = fout_f.clone()

        # ── 3. Greedy subspace selection per layer ────────────────────────
        for name, history in self._history_atoms.items():
            fin  = self._kfac_in.get(name)
            fout = self._kfac_out.get(name)

            # candidate pool = current shared dict ∪ atoms from latest task
            prev       = self._atoms.get(name, [])
            latest     = history[-1] if history else []
            candidates = prev + latest

            selected = self._greedy_select(
                candidates, history, self.budget_K,
                fin, fout, self.lambda_risk, self.sim_thresh,
            )
            self._atoms[name] = selected

        self._n_tasks += 1
        avg_k = np.mean([len(v) for v in self._atoms.values()]) if self._atoms else 0
        logger.info("[SCM] task=%d  layers=%d  avg_atoms=%.1f",
                    self._n_tasks, len(self._atoms), avg_k)

    def get_shared_init(self) -> Dict[str, torch.Tensor]:
        """Return {layer → ΔW_shared} for next-task parameter initialisation.

        ΔW_shared = Σ_k σ_k · u_k · v_k^T
        """
        out: Dict[str, torch.Tensor] = {}
        for name, atoms in self._atoms.items():
            if not atoms:
                continue
            d_out = atoms[0][0].shape[0]
            d_in  = atoms[0][1].shape[0]
            dw    = torch.zeros(d_out, d_in)
            for u, v, sigma in atoms:
                dw += sigma * torch.ger(u, v)
            out[name] = dw
        return out

    def summary(self) -> Dict[str, dict]:
        info = {}
        for name, atoms in self._atoms.items():
            sigmas = [a[2] for a in atoms]
            risks  = []
            fin    = self._kfac_in.get(name)
            fout   = self._kfac_out.get(name)
            for u, v, _ in atoms:
                risks.append(float(self._kfac_risk(u, v, fin, fout)))
            info[name] = {
                "n_atoms":   len(atoms),
                "sigma_max": max(sigmas) if sigmas else 0.0,
                "sigma_min": min(sigmas) if sigmas else 0.0,
                "risk_mean": float(np.mean(risks)) if risks else 0.0,
            }
        return info

    # ─────────────────────────────────────────────────────────────────────
    # Internal: SVD canonical decomposition
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _svd_atoms(dw: torch.Tensor, top_r: int) -> LayerAtoms:
        """Decompose ΔW → canonical atoms (u_i, v_i, σ_i).

        Convention:
          u_i ∈ R^{d_out}  left  singular vector
          v_i ∈ R^{d_in}   right singular vector
          Sign canon: largest-magnitude component of u_i > 0
        """
        try:
            U, S, Vh = torch.linalg.svd(dw, full_matrices=False)
        except torch.linalg.LinAlgError:
            return []

        atoms: LayerAtoms = []
        for i in range(min(top_r, S.shape[0])):
            sigma = float(S[i].item())
            if sigma < 1e-10:
                break
            u = U[:, i].clone()
            v = Vh[i].clone()
            # canonical sign
            idx = int(u.abs().argmax().item())
            if u[idx] < 0:
                u, v = -u, -v
            atoms.append((u.cpu(), v.cpu(), sigma))
        return atoms

    # ─────────────────────────────────────────────────────────────────────
    # Internal: cross-task utility  Σ_t c_t(a,b)²
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _coeff(
        task_atoms: LayerAtoms,
        u_cand:     torch.Tensor,
        v_cand:     torch.Tensor,
    ) -> float:
        """Projection of ΔW_t onto direction u_cand ⊗ v_cand.

        c_t = <ΔW_t, u_cand v_cand^T>_F
            = Σ_i σ_i (u_cand · u_i)(v_cand · v_i)
        """
        c = 0.0
        for u_i, v_i, sigma_i in task_atoms:
            c += sigma_i * float(u_cand @ u_i) * float(v_cand @ v_i)
        return c

    @classmethod
    def _utility(
        cls,
        history: List[LayerAtoms],   # all tasks up to current
        u_cand:  torch.Tensor,
        v_cand:  torch.Tensor,
    ) -> float:
        """utility(u,v) = Σ_{t≤T} c_t(u,v)²"""
        total = 0.0
        for task_atoms in history:
            c = cls._coeff(task_atoms, u_cand, v_cand)
            total += c * c
        return total

    # ─────────────────────────────────────────────────────────────────────
    # Internal: KFAC risk  (u^T F_out u / tr) · (v^T F_in v / tr)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _kfac_risk(
        u:    torch.Tensor,
        v:    torch.Tensor,
        fin:  Optional[torch.Tensor],   # F_in  [d_in,  d_in ]
        fout: Optional[torch.Tensor],   # F_out [d_out, d_out]
    ) -> float:
        """Normalised KFAC risk for atom (u, v).

        risk = (u^T F_out u / tr(F_out)) · (v^T F_in v / tr(F_in))
        Falls back to 0 if Fisher not available.
        """
        if fin is None or fout is None:
            return 0.0

        u_f = u.float()
        v_f = v.float()

        tr_out = float(fout.diagonal().sum().clamp(min=1e-12))
        tr_in  = float(fin.diagonal().sum().clamp(min=1e-12))

        risk_out = float(u_f @ fout @ u_f) / tr_out
        risk_in  = float(v_f @ fin  @ v_f) / tr_in
        return risk_out * risk_in

    # ─────────────────────────────────────────────────────────────────────
    # Internal: greedy selection with G-S orthogonalisation
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def _greedy_select(
        cls,
        candidates:  LayerAtoms,
        history:     List[LayerAtoms],
        budget_K:    int,
        fin:         Optional[torch.Tensor],
        fout:        Optional[torch.Tensor],
        lambda_risk: float,
        sim_thresh:  float,
    ) -> LayerAtoms:
        """Pick ≤ budget_K atoms maximising  utility − λ · risk.

        Algorithm:
          Repeat K times:
            For each unused candidate (u, v, σ):
              project u onto complement of already-selected U-span (G-S)
              if residual < ε: skip (linearly dependent)
              score = utility(u,v) · ||u_⊥||² − λ · risk(u,v)
            Select highest-scoring atom
          After selection: σ ← √utility  (re-normalise to historical energy)
        """
        if not candidates or budget_K <= 0:
            return []

        selected:     LayerAtoms             = []
        residual_U:   List[torch.Tensor]     = []   # orthonormal basis of selected u-span
        used          = [False] * len(candidates)

        # pre-compute scores (utility and risk do not depend on selection order)
        # Score = utility / (1 + λ·risk)
        # Rationale: utility ~ σ² (unbounded), risk ∈ [0,1] (normalised KFAC).
        # Additive form  utility − λ·risk  makes λ dimensionally inconsistent
        # (λ=1 contributes at most 1 to a score that can be 100+).
        # Ratio form keeps λ as a meaningful relative penalty regardless of ΔW scale.
        scores: List[Tuple[float, float, float]] = []   # (utility, risk, score)
        for u, v, _ in candidates:
            util  = cls._utility(history, u, v)
            risk  = cls._kfac_risk(u, v, fin, fout)
            score = util / (1.0 + lambda_risk * risk)
            scores.append((util, risk, score))

        for _ in range(budget_K):
            best_score = -float("inf")
            best_idx   = -1

            for j, (u, v, _) in enumerate(candidates):
                if used[j]:
                    continue

                # sim-threshold early exit (cheap dedup before G-S)
                if sim_thresh > 0.0:
                    redundant = False
                    for su, sv, _ in selected:
                        sim = abs(float(u @ su)) * abs(float(v @ sv))
                        if sim >= sim_thresh:
                            redundant = True
                            break
                    if redundant:
                        used[j] = True
                        continue

                # Gram-Schmidt residual in u-space
                u_f = u.float()
                for e in residual_U:
                    u_f = u_f - (u_f @ e) * e
                res_norm_sq = float((u_f * u_f).sum())
                if res_norm_sq < 1e-8:
                    used[j] = True  # linearly dependent
                    continue

                util, risk, base_score = scores[j]
                eff_score = base_score * res_norm_sq   # scale by residual energy
                if eff_score > best_score:
                    best_score = eff_score
                    best_idx   = j

            if best_idx < 0:
                break

            u_sel, v_sel, _ = candidates[best_idx]
            used[best_idx]  = True

            util_sel = scores[best_idx][0]
            sigma_new = float(np.sqrt(max(util_sel, 0.0)))   # re-set σ ← √utility
            selected.append((u_sel.cpu(), v_sel.cpu(), sigma_new))

            # update G-S basis
            u_f = u_sel.float()
            for e in residual_U:
                u_f = u_f - (u_f @ e) * e
            norm = float(u_f.norm())
            if norm > 1e-8:
                residual_U.append((u_f / norm).cpu())

        return selected


# ═════════════════════════════════════════════════════════════════════════════
#  Learner: fast (finetune) + slow (SCM)
# ═════════════════════════════════════════════════════════════════════════════

class Learner(BaseLearner):
    """CL learner: standard fine-tuning (fast) + SlowConsolidationModule (slow).

    Interface contract:
      Fast layer produces θ_t.
      SCM receives ΔW_t = W_t − W_pretrained and KFAC(F_in, F_out).
      SCM returns ΔW_shared for θ_{t+1} initialisation.
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

        # SCM hyper-parameters
        self._scm_budget_K       = int(args.get("scm_budget_K",       8))
        self._scm_lambda_risk    = float(args.get("scm_lambda_risk",   1.0))
        self._scm_top_r          = int(args.get("scm_top_r",          8))
        self._scm_kfac_batches   = int(args.get("scm_kfac_batches",   20))
        self._scm_kfac_gamma     = float(args.get("scm_kfac_gamma",   1.0))
        self._scm_sim_thresh     = float(args.get("scm_sim_thresh",   0.0))
        self._scm_target_kws     = args.get(
            "scm_target_keywords",
            ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        )

        self._scm = SlowConsolidationModule(
            budget_K    = self._scm_budget_K,
            lambda_risk = self._scm_lambda_risk,
            top_r       = self._scm_top_r,
            kfac_gamma  = self._scm_kfac_gamma,
            sim_thresh  = self._scm_sim_thresh,
        )

        self._pretrained_weights: Dict[str, torch.Tensor] = {}   # W_0 (never updated)
        self._task_init_weights:  Dict[str, torch.Tensor] = {}   # W_0 + ΔW_shared_{t-1}

    # ─────────────────────────────────────────────────────────────────────
    # CL interface
    # ─────────────────────────────────────────────────────────────────────

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = (self._known_classes
                               + data_manager.get_task_size(self._cur_task))
        self._network.update_fc(self._total_classes)
        logger.info("Task %d: classes %d→%d",
                    self._cur_task, self._known_classes, self._total_classes)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train",
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"],
            shuffle=True,  num_workers=self.args.get("num_workers", 8),
        )
        self.test_loader = DataLoader(
            test_dataset,  batch_size=self.args["batch_size"],
            shuffle=False, num_workers=self.args.get("num_workers", 8),
        )
        train_loader = self.train_loader
        test_loader  = self.test_loader

        # ── (a) inject shared init from previous SCM ──────────────────────
        self._apply_shared_init()

        # ── (b) snapshot pretrained reference (first task only) ───────────
        if self._cur_task == 0:
            self._snapshot_pretrained()

        # ── (b2) snapshot task-init weights (= W_0 + ΔW_shared_{t-1}) ────
        #         Must happen AFTER _apply_shared_init() so we capture the
        #         actual starting point for this task, not W_0.
        self._snapshot_task_init()

        # Move network to device once here so all subsequent operations
        # (KFAC estimation and fast training) see the same device.
        self._network.to(self._device)

        # ── (b3) KFAC Fisher estimation at W_init_t ───────────────────────
        # Estimated BEFORE fast_train so that Fisher reflects the curvature at
        # the consolidated starting point W_init_t = W_0 + ΔW_shared_{t-1},
        # NOT at W_t (which is biased toward task t after fast training).
        # SCM risk measures stability of shared-subspace directions at W_init_t.
        target_layers = self._get_target_layers()
        kfac = estimate_kfac_fisher(
            self._network, train_loader, target_layers,
            self._device, self._scm_kfac_batches, self._scm_kfac_gamma,
        )

        # ── (c) fast layer: plain SGD fine-tuning ─────────────────────────
        self._fast_train(train_loader, test_loader)

        # ── (d) extract task vector  ΔW_t = W_t − W_init_t ───────────────
        #         W_init_t = W_0 + ΔW_shared_{t-1}  (strips shared knowledge)
        delta_w = self._extract_delta_w()

        # ── (f) SCM slow consolidation ────────────────────────────────────
        self._scm.consolidate(delta_w, kfac)
        self._log_scm_summary()

        # ── (g) write consolidated model back to network ──────────────────
        # After consolidation, θ* = W_0 + ΔW_shared is the single global model
        # used for task-agnostic evaluation (no task-id required at test time).
        # Without this step the network still holds W_T (biased toward task T).
        self._apply_consolidated_model()

        self._known_classes = self._total_classes

    def after_task(self):
        self._known_classes = self._total_classes

    # ─────────────────────────────────────────────────────────────────────
    # Fast layer: standard CE fine-tuning
    # ─────────────────────────────────────────────────────────────────────

    def _fast_train(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        self._network.to(self._device)

        lr     = float(self.args.get("init_lr" if self._cur_task == 0 else "lrate", 1e-3))
        wd     = float(self.args.get("weight_decay", 0.0))
        epochs = int(self.args.get(
            "init_epoch" if self._cur_task == 0 else "epochs", 5))

        momentum_key = "init_momentum" if self._cur_task == 0 else "momentum"
        momentum = float(self.args.get(momentum_key, 0.9))
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self._network.parameters()),
            lr=lr, weight_decay=wd, momentum=momentum,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01,
        )

        prog = tqdm(range(epochs), desc=f"Task {self._cur_task} fast-train")
        for epoch in prog:
            self._network.train()
            total_loss, correct, total = 0.0, 0, 0

            for batch in train_loader:
                _, inputs, targets = batch if len(batch) == 3 else (None, *batch)
                inputs  = inputs.to(self._device)
                targets = targets.to(self._device)

                optimizer.zero_grad()
                out    = self._network(inputs)
                logits = out["logits"] if isinstance(out, dict) else out
                loss   = F.cross_entropy(
                    logits[:, self._known_classes:],
                    targets - self._known_classes,
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, preds = logits[:, self._known_classes:].max(1)
                correct  += preds.eq(targets - self._known_classes).sum().item()
                total    += targets.size(0)

            scheduler.step()
            prog.set_postfix(
                loss=f"{total_loss / len(train_loader):.3f}",
                acc =f"{100. * correct / total:.1f}%",
            )

    # ─────────────────────────────────────────────────────────────────────
    # ΔW extraction
    # ─────────────────────────────────────────────────────────────────────

    def _snapshot_pretrained(self) -> None:
        """Save CPU copy of W_0 (pretrained) as permanent reference for shared init."""
        self._pretrained_weights = {}
        for name, param in self._network.named_parameters():
            if self._is_target(name) and param.dim() == 2:
                self._pretrained_weights[name] = param.detach().cpu().clone()
        logger.info("[SCM] Snapshotted %d pretrained weight matrices (W_0).",
                    len(self._pretrained_weights))

    def _snapshot_task_init(self) -> None:
        """Save CPU copy of weights at the START of task t.

        W_init_t = W_0 + ΔW_shared_{t-1}
        This is called AFTER _apply_shared_init() so it captures the exact
        starting point for this task's fast training.  For task 0 this equals W_0.
        """
        self._task_init_weights = {}
        for name, param in self._network.named_parameters():
            if self._is_target(name) and param.dim() == 2:
                self._task_init_weights[name] = param.detach().cpu().clone()

    def _extract_delta_w(self) -> Dict[str, torch.Tensor]:
        """ΔW_t^l = W_t^l − W_init_t^l  (pure task-specific update).

        Subtracts W_init_t (= W_0 + ΔW_shared_{t-1}) rather than W_0,
        so the extracted vector captures only what the fast layer learned
        for task t, stripping out shared knowledge already baked into the init.
        """
        ref = self._task_init_weights if self._task_init_weights else self._pretrained_weights
        delta: Dict[str, torch.Tensor] = {}
        for name, param in self._network.named_parameters():
            if name in ref and param.dim() == 2:
                delta[name] = (param.detach().cpu() - ref[name]).clone()
        return delta

    # ─────────────────────────────────────────────────────────────────────
    # Shared init injection
    # ─────────────────────────────────────────────────────────────────────

    def _apply_shared_init(self) -> None:
        """W_init = W_pretrained + ΔW_shared  before each task."""
        if self._cur_task == 0:
            return
        self._apply_consolidated_model()

    def _apply_consolidated_model(self) -> None:
        """Write the consolidated global model θ* = W_0 + ΔW_shared into the network.

        This is called:
          (1) at the START of each new task (as the initialisation point), and
          (2) at the END of each task after SCM consolidation (so that evaluation
              and next-task init both see the same task-agnostic single model).

        At test time no task-id is required — every input is forwarded through θ*.
        """
        shared = self._scm.get_shared_init()
        if not shared:
            return
        with torch.no_grad():
            for name, param in self._network.named_parameters():
                if name in shared and name in self._pretrained_weights:
                    consolidated_w = (self._pretrained_weights[name]
                                      + shared[name]).to(param.device)
                    param.copy_(consolidated_w)
        logger.info("[SCM] Network set to consolidated model W_0 + ΔW_shared (%d layers).",
                    len(shared))

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _is_target(self, name: str) -> bool:
        return any(kw in name for kw in self._scm_target_kws)

    def _get_target_layers(self) -> Dict[str, nn.Linear]:
        """Return {param_name → Linear module} for all target weight matrices."""
        result: Dict[str, nn.Linear] = {}
        for mod_name, module in self._network.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            full = mod_name + ".weight"
            if self._is_target(full) and full in self._task_init_weights:
                result[full] = module
        return result

    def _log_scm_summary(self) -> None:
        summary = self._scm.summary()
        for name, info in list(summary.items())[:3]:
            logger.info(
                "[SCM] %-50s  atoms=%d  σ=[%.3f,%.3f]  risk=%.4f",
                name, info["n_atoms"],
                info["sigma_min"], info["sigma_max"], info["risk_mean"],
            )
