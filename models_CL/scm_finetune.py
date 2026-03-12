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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models_CL.finetune import Learner as FinetuneLearner
from optimer_PerturabtionType.util import (
    disable_running_stats,
    enable_running_stats,
    generate_pertubation,
)
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
    known_classes: int = 0,
    loss_mode:     str = "task_ce",
    total_classes: Optional[int] = None,
    global_head: bool = False,
    local_weight: float = 1.0,
    seen_weight: float = 1.0,
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
    loss_mode = str(loss_mode).lower()

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
        loss = _build_kfac_objective(
            outputs=out,
            logits=logits,
            targets=targets,
            known_classes=known_classes,
            total_classes=total_classes,
            loss_mode=loss_mode,
            global_head=global_head,
            local_weight=local_weight,
            seen_weight=seen_weight,
        )
        loss.backward()

    for h in hooks:
        h.remove()

    # ── normalise ──────────────────────────────────────────────────────────
    result: Dict[str, KFACPair] = {}
    for name in target_layers:
        c = max(1, counts[name])
        result[name] = (acc_in[name] / c, acc_out[name] / c)
    return result


def _build_kfac_objective(
    *,
    outputs,
    logits: torch.Tensor,
    targets: torch.Tensor,
    known_classes: int,
    total_classes: Optional[int],
    loss_mode: str,
    global_head: bool = False,
    local_weight: float = 1.0,
    seen_weight: float = 1.0,
) -> torch.Tensor:
    """Build the scalar objective used to induce KFAC gradients.

    `task_ce` is the default because it matches the incremental fast loss:
        CE(logits[:, known:], targets-known)

    `feature_l2` and `logit_l2` are classifier-agnostic surrogates. They can be
    useful for ablations, but they are no longer Fisher of the training loss.
    """
    if global_head and loss_mode in {"task_ce", "new_ce", "incremental_ce", "aligned_ce"}:
        return _build_masked_global_head_loss(
            logits=logits,
            targets=targets,
            known_classes=known_classes,
            total_classes=total_classes,
            local_weight=local_weight,
            seen_weight=seen_weight,
        )

    if global_head and loss_mode in {"seen_ce", "masked_seen_ce", "mask_vr", "global_masked_ce"}:
        return _build_masked_global_head_loss(
            logits=logits,
            targets=targets,
            known_classes=known_classes,
            total_classes=total_classes,
            local_weight=0.0,
            seen_weight=1.0,
        )

    if loss_mode in {"task_ce", "new_ce", "incremental_ce", "aligned_ce"}:
        if known_classes > 0 and logits.shape[1] > known_classes:
            fake_targets = targets - known_classes
            return F.cross_entropy(logits[:, known_classes:], fake_targets)
        return F.cross_entropy(logits, targets)

    if loss_mode in {"full_ce", "global_ce"}:
        return F.cross_entropy(logits, targets)

    if loss_mode in {"logit_l2", "task_logit_l2", "new_logit_l2"}:
        logits_slice = logits[:, known_classes:] if (known_classes > 0 and logits.shape[1] > known_classes) else logits
        return 0.5 * logits_slice.pow(2).sum(dim=1).mean()

    if loss_mode in {"feature_l2", "feat_l2"}:
        if not isinstance(outputs, dict) or "features" not in outputs:
            raise ValueError("feature_l2 KFAC objective requires outputs['features'].")
        features = outputs["features"]
        return 0.5 * features.pow(2).sum(dim=1).mean()

    raise ValueError(f"Unknown scm_kfac_loss mode: {loss_mode}")


def _masked_logits_for_range(
    logits: torch.Tensor,
    start_class: int,
    end_class: int,
    fill_value: float = -1.0e9,
) -> torch.Tensor:
    start_class = max(0, int(start_class))
    end_class = min(int(end_class), int(logits.shape[1]))
    if end_class <= start_class:
        raise ValueError(
            f"Invalid masked-logit range [{start_class}, {end_class}) for shape {tuple(logits.shape)}."
        )
    masked = logits.new_full(logits.shape, float(fill_value))
    masked[:, start_class:end_class] = logits[:, start_class:end_class]
    return masked


def _build_masked_global_head_loss(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    known_classes: int,
    total_classes: Optional[int],
    local_weight: float,
    seen_weight: float,
) -> torch.Tensor:
    if total_classes is None or int(total_classes) <= 0:
        raise ValueError("Masked global-head loss requires total_classes > 0.")

    total_classes = min(int(total_classes), int(logits.shape[1]))
    local_start = max(0, int(known_classes))
    local_end = total_classes

    loss = logits.new_zeros(())

    if local_weight > 0.0:
        local_logits = _masked_logits_for_range(logits, local_start, local_end)
        loss = loss + float(local_weight) * F.cross_entropy(local_logits, targets)

    if seen_weight > 0.0 and (local_start > 0 or local_weight <= 0.0):
        seen_logits = _masked_logits_for_range(logits, 0, total_classes)
        loss = loss + float(seen_weight) * F.cross_entropy(seen_logits, targets)

    if local_weight <= 0.0 and not (seen_weight > 0.0 and local_start > 0):
        raise ValueError("Masked global-head loss needs at least one active term.")

    return loss


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
        delta_w_dict: Dict[str, torch.Tensor],   # {layer → ΔW_t = W_t - W_init_t}
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
    # Internal: greedy selection with matrix-space G-S orthogonalisation
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
              project rank-1 matrix uv^T onto complement of selected matrix span
              if residual < ε: skip (linearly dependent in matrix space)
              score = utility(u,v) · ||(uv^T)_⊥||² − λ · risk(u,v)
            Select highest-scoring atom
          After selection: σ ← √utility  (re-normalise to historical energy)
        """
        if not candidates or budget_K <= 0:
            return []

        selected:   LayerAtoms         = []
        basis_mats: List[torch.Tensor] = []   # orthonormal basis in Frobenius matrix space
        used        = [False] * len(candidates)

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

                # Gram-Schmidt residual in full matrix space, not only u-space.
                cand_mat = torch.outer(u.float(), v.float())
                for e in basis_mats:
                    cand_mat = cand_mat - float((cand_mat * e).sum()) * e
                res_norm_sq = float((cand_mat * cand_mat).sum())
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

            # update matrix-space G-S basis with the selected residual direction
            basis_mat = torch.outer(u_sel.float(), v_sel.float())
            for e in basis_mats:
                basis_mat = basis_mat - float((basis_mat * e).sum()) * e
            norm = float(basis_mat.norm())
            if norm > 1e-8:
                basis_mats.append((basis_mat / norm).cpu())

        return selected


# ═════════════════════════════════════════════════════════════════════════════
#  Learner: fast (finetune) + slow (SCM)
# ═════════════════════════════════════════════════════════════════════════════

class Learner(FinetuneLearner):
    """CL learner: standard fine-tuning (fast) + SlowConsolidationModule (slow).

    Interface contract:
      Fast layer produces θ_t.
      SCM receives ΔW_t = W_t − W_init_t and KFAC(F_in, F_out).
      SCM returns ΔW_shared for θ_{t+1} initialisation.
    """

    def __init__(self, args):
        super().__init__(args)

        # SCM hyper-parameters
        self._scm_budget_K       = int(args.get("scm_budget_K",       8))
        self._scm_lambda_risk    = float(args.get("scm_lambda_risk",   1.0))
        self._scm_top_r          = int(args.get("scm_top_r",          8))
        self._scm_kfac_batches   = int(args.get("scm_kfac_batches",   20))
        self._scm_kfac_gamma     = float(args.get("scm_kfac_gamma",   1.0))
        self._scm_kfac_loss      = str(args.get("scm_kfac_loss", "task_ce")).lower()
        self._scm_sim_thresh     = float(args.get("scm_sim_thresh",   0.0))
        # Default to every trainable 2D backbone matrix. A keyword list can be
        # passed to explicitly narrow the consolidated set.
        self._scm_target_kws     = args.get("scm_target_keywords", None)
        self._scm_freeze_non_targets = bool(args.get("scm_freeze_non_targets", True))
        self._scm_eval_fast_model = bool(args.get("scm_eval_fast_model", True))
        self._scm_eval_consolidated_model = bool(
            args.get("scm_eval_consolidated_model", True)
        )
        self._scm_global_head = bool(args.get("scm_global_head", True))
        self._scm_global_head_local_weight = float(
            args.get("scm_global_head_local_weight", 1.0)
        )
        self._scm_global_head_seen_weight = float(
            args.get("scm_global_head_seen_weight", 1.0)
        )
        self._scm_global_head_freeze_old_rows = bool(
            args.get("scm_global_head_freeze_old_rows", True)
        )
        self._scm_global_head_eval_seen_only = bool(
            args.get("scm_global_head_eval_seen_only", True)
        )
        self._scm_logit_mask_fill = float(args.get("scm_logit_mask_fill", -1.0e9))
        self._scm_head_recalibration = bool(args.get("scm_head_recalibration", False))
        self._scm_head_recalibration_method = str(
            args.get("scm_head_recalibration_method", "linear_probe")
        ).lower()
        self._scm_head_recalibration_epochs = int(
            args.get("scm_head_recalibration_epochs", 3)
        )
        self._scm_head_recalibration_lr = float(
            args.get("scm_head_recalibration_lr", 1.0e-2)
        )
        self._scm_head_recalibration_weight_decay = float(
            args.get("scm_head_recalibration_weight_decay", 0.0)
        )
        self._scm_head_recalibration_batch_size = int(
            args.get(
                "scm_head_recalibration_batch_size",
                args.get("eval_batch_size", args.get("batch_size", 128)),
            )
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
        self._scm_target_names:   Set[str]                = set()
        self._scm_snapshot_metrics: Dict[str, Dict[str, Optional[dict]]] = {}
        self._scm_global_head_num_classes: Optional[int] = None

        if self._scm_global_head and self._scm_head_recalibration:
            logger.warning(
                "[SCM] scm_global_head=True with scm_head_recalibration=True. "
                "This mixes training-time masking with task-end head refit."
            )

    # ─────────────────────────────────────────────────────────────────────
    # CL interface
    # ─────────────────────────────────────────────────────────────────────

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = (self._known_classes
                               + data_manager.get_task_size(self._cur_task))
        if self._scm_global_head:
            self._scm_global_head_num_classes = int(data_manager.nb_classes)
            fc = self._get_fc_module()
            fc_out = getattr(fc, "out_features", None) if fc is not None else None
            if fc_out != self._scm_global_head_num_classes:
                self._network.update_fc(self._scm_global_head_num_classes)
        else:
            self._network.update_fc(self._total_classes)
        self._configure_fast_slow_parameters()
        self._scm_snapshot_metrics = {}
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
            shuffle=True,  num_workers=self.args.get("train_num_workers", 8),
        )
        self.test_loader = DataLoader(
            test_dataset,  batch_size=self.args["batch_size"],
            shuffle=False, num_workers=self.args.get("train_num_workers", 8),
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
            self._known_classes, self._scm_kfac_loss,
            self._total_classes,
            self._scm_global_head,
            self._scm_global_head_local_weight,
            self._scm_global_head_seen_weight,
        )

        # ── (c) fast layer: plain SGD fine-tuning ─────────────────────────
        self._fast_train(train_loader, test_loader)
        if self._scm_eval_fast_model:
            self._log_task_snapshot(data_manager, tag="fast")

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

        consolidated_bank = self._extract_seen_feature_bank(data_manager)
        if self._scm_head_recalibration:
            self._recalibrate_classifier_from_feature_bank(*consolidated_bank)

        if self._scm_eval_consolidated_model:
            self._log_task_snapshot(
                data_manager,
                tag="consolidated",
                feature_bank=consolidated_bank,
            )
        else:
            self._set_class_means_from_feature_bank(*consolidated_bank)

    def after_task(self):
        super().after_task()

    def _use_masked_global_head(self) -> bool:
        return bool(self._scm_global_head and self._scm_global_head_num_classes)

    def _mask_logits_range(
        self,
        logits: torch.Tensor,
        start_class: int,
        end_class: int,
    ) -> torch.Tensor:
        return _masked_logits_for_range(
            logits,
            start_class,
            end_class,
            fill_value=self._scm_logit_mask_fill,
        )

    def _mask_seen_logits(
        self,
        logits: torch.Tensor,
        seen_classes: Optional[int] = None,
    ) -> torch.Tensor:
        if not self._use_masked_global_head() or not self._scm_global_head_eval_seen_only:
            return logits
        seen_classes = self._total_classes if seen_classes is None else int(seen_classes)
        return self._mask_logits_range(logits, 0, seen_classes)

    def _compute_global_head_objective(
        self,
        outputs,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = outputs["logits"]
        loss = _build_masked_global_head_loss(
            logits=logits,
            targets=targets,
            known_classes=self._known_classes,
            total_classes=self._total_classes,
            local_weight=self._scm_global_head_local_weight,
            seen_weight=self._scm_global_head_seen_weight,
        )
        return loss, self._mask_seen_logits(logits, self._total_classes), targets

    def _apply_classifier_row_grad_mask(self) -> None:
        if not self._use_masked_global_head() or not self._scm_global_head_freeze_old_rows:
            return

        fc = self._get_fc_module()
        if fc is None or not hasattr(fc, "weight") or fc.weight.grad is None:
            return

        old_end = max(0, min(int(self._known_classes), int(fc.weight.shape[0])))
        unseen_start = max(0, min(int(self._total_classes), int(fc.weight.shape[0])))

        if old_end > 0:
            fc.weight.grad[:old_end].zero_()
            if getattr(fc, "bias", None) is not None and fc.bias.grad is not None:
                fc.bias.grad[:old_end].zero_()

        if unseen_start < int(fc.weight.shape[0]):
            fc.weight.grad[unseen_start:].zero_()
            if getattr(fc, "bias", None) is not None and fc.bias.grad is not None:
                fc.bias.grad[unseen_start:].zero_()

    def _run_masked_global_head_train(
        self,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        epochs: int,
    ) -> None:
        prog_bar = tqdm(range(int(epochs)))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                if self._optimizer_type == "cflat":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        loss, _, _ = self._compute_global_head_objective(outputs, targets)
                        loss.backward()
                        self._apply_classifier_row_grad_mask()
                        return outputs, [loss]

                    _, loss_list = optimizer.step(closure=closure)
                    loss_value = torch.stack([loss_term.detach() for loss_term in loss_list]).sum()
                    losses += loss_value.item()
                    with torch.no_grad():
                        logits = self._mask_seen_logits(self._network(inputs)["logits"], self._total_classes)
                    eval_targets = targets
                elif self._optimizer_type == "gam":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        loss, _, _ = self._compute_global_head_objective(outputs, targets)
                        loss_value = loss.detach()
                        loss.backward()
                        self._apply_classifier_row_grad_mask()
                        return outputs, loss_value

                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = self._mask_seen_logits(outputs["logits"].detach(), self._total_classes)
                    eval_targets = targets
                elif self._optimizer_type == "arwp":
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = (
                                float(scheduler.get_last_lr()[0])
                                if hasattr(scheduler, "get_last_lr")
                                else float(optimizer.param_groups[0]["lr"])
                            )
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"])
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr) or cur_lr)
                        optimizer.std = float(self._rwp_std) * ((cur_lr / base_lr) if base_lr > 0 else 1.0)

                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        loss, _, _ = self._compute_global_head_objective(outputs, targets)
                        loss_value = loss.detach()
                        loss.backward()
                        self._apply_classifier_row_grad_mask()
                        return outputs, loss_value

                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = self._mask_seen_logits(outputs["logits"].detach(), self._total_classes)
                    eval_targets = targets
                elif self._optimizer_type == "rwp":
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = (
                                float(scheduler.get_last_lr()[0])
                                if hasattr(scheduler, "get_last_lr")
                                else float(optimizer.param_groups[0]["lr"])
                            )
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"])
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr) or cur_lr)
                        rwp_std = float(self._rwp_std) * ((cur_lr / base_lr) if base_lr > 0 else 1.0)
                    else:
                        rwp_std = float(self._rwp_std)

                    enable_running_stats(self._network)
                    optimizer.zero_grad()
                    outputs = self._network(inputs)
                    loss_clean, logits_clean, eval_targets = self._compute_global_head_objective(outputs, targets)
                    loss_clean.backward()
                    self._apply_classifier_row_grad_mask()
                    g0 = {}
                    for name, param in self._network.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            g0[name] = param.grad.detach().clone()

                    disable_running_stats(self._network)
                    noise_dict = {}
                    with torch.no_grad():
                        std_for_noise = float(self.args.get("noise_std", rwp_std))
                        for name, param in self._network.named_parameters():
                            if self._rwp_range == "lora" and (not param.requires_grad or param.numel() == 0):
                                continue
                            fisher_param = getattr(self, "_rwp_fisher", {}).get(name, None)
                            noise = generate_pertubation(
                                param,
                                pertubation_mode=self.rwp_noise_type,
                                std=std_for_noise,
                                fisher_param=fisher_param,
                                fisher_scaler=float(self._rwp_eta),
                            )
                            param.data.add_(noise)
                            noise_dict[name] = noise

                    optimizer.zero_grad()
                    outputs_noisy = self._network(inputs)
                    loss_noisy, _, _ = self._compute_global_head_objective(outputs_noisy, targets)
                    loss_noisy.backward()
                    self._apply_classifier_row_grad_mask()

                    with torch.no_grad():
                        if hasattr(self, "_rwp_fisher"):
                            for name, param in self._network.named_parameters():
                                if param.requires_grad and param.grad is not None:
                                    grad_sq = param.grad.detach() ** 2
                                    if name not in self._rwp_fisher:
                                        self._rwp_fisher[name] = grad_sq
                                    else:
                                        self._rwp_fisher[name] = (
                                            float(self._rwp_beta) * self._rwp_fisher[name] + grad_sq
                                        )

                    with torch.no_grad():
                        for name, param in self._network.named_parameters():
                            if name in noise_dict:
                                param.data.sub_(noise_dict[name])

                    lam = float(self._rwp_lambda)
                    for name, param in self._network.named_parameters():
                        if self._rwp_range == "lora" and not param.requires_grad:
                            continue
                        if param.grad is not None:
                            g1 = param.grad.detach()
                            g0_n = g0.get(name, torch.zeros_like(g1))
                            param.grad.data.copy_(lam * g1 + (1.0 - lam) * g0_n)

                    optimizer.step()
                    logits = logits_clean.detach()
                    losses += (
                        lam * float(loss_noisy.detach().item())
                        + (1.0 - lam) * float(loss_clean.detach().item())
                    )
                else:
                    optimizer.zero_grad()
                    outputs = self._network(inputs)
                    loss, masked_eval_logits, eval_targets = self._compute_global_head_objective(outputs, targets)

                    if self._optimizer_type == "sam":
                        loss.backward()
                        self._apply_classifier_row_grad_mask()
                        optimizer.first_step(zero_grad=True)

                        outputs = self._network(inputs)
                        second_loss, masked_eval_logits, eval_targets = self._compute_global_head_objective(outputs, targets)
                        second_loss.backward()
                        self._apply_classifier_row_grad_mask()
                        optimizer.second_step(zero_grad=True)
                        logits = masked_eval_logits.detach()
                        losses += second_loss.item()
                    else:
                        loss.backward()
                        self._apply_classifier_row_grad_mask()
                        optimizer.step()
                        logits = masked_eval_logits.detach()
                        losses += loss.item()

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 4:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / len(train_loader):.3f}, Train_accy {train_acc:.2f}, "
                    f"Test_accy {test_acc:.2f}"
                )
            else:
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / len(train_loader):.3f}, Train_accy {train_acc:.2f}"
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if not self._use_masked_global_head():
            return super()._init_train(train_loader, test_loader, optimizer, scheduler)
        self._run_masked_global_head_train(
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            self.args["init_epoch"],
        )

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        if not self._use_masked_global_head():
            return super()._update_representation(train_loader, test_loader, optimizer, scheduler)
        self._run_masked_global_head_train(
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            self.args["epochs"],
        )

    def _compute_accuracy(self, model, loader):
        if not self._use_masked_global_head():
            return super()._compute_accuracy(model, loader)

        model.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
                outputs = self._mask_seen_logits(outputs, self._total_classes)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        if not self._use_masked_global_head():
            return super()._eval_cnn(loader)

        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
                outputs = self._mask_seen_logits(outputs, self._total_classes)
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    # ─────────────────────────────────────────────────────────────────────
    # Fast layer: reuse finetune.py training stack
    # ─────────────────────────────────────────────────────────────────────

    def _fast_train(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(train_loader, test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

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

    @staticmethod
    def _is_classifier_param(name: str) -> bool:
        return name.startswith("fc.")

    def _configure_fast_slow_parameters(self) -> None:
        """Align fast-train and slow-consolidation coverage.

        SCM only supports stable 2D backbone matrices. We therefore train exactly
        that set in the backbone and keep the task classifier trainable as the
        explicit head exception.
        """
        target_names: Set[str] = set()
        classifier_names: Set[str] = set()

        for name, param in self._network.named_parameters():
            if self._is_classifier_param(name):
                classifier_names.add(name)
                continue
            if not param.requires_grad and name not in self._scm_target_names:
                continue
            if param.dim() != 2:
                continue
            if self._scm_target_kws and not any(kw in name for kw in self._scm_target_kws):
                continue
            target_names.add(name)

        self._scm_target_names = target_names

        if self._scm_freeze_non_targets:
            for name, param in self._network.named_parameters():
                param.requires_grad = (name in self._scm_target_names) or (name in classifier_names)

        logger.info(
            "[SCM] target_matrices=%d classifier_params=%d freeze_non_targets=%s",
            len(self._scm_target_names),
            len(classifier_names),
            self._scm_freeze_non_targets,
        )

    def _is_target(self, name: str) -> bool:
        return name in self._scm_target_names

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

    def _extract_seen_feature_bank(
        self,
        data_manager,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a clean feature bank for all seen train samples in one pass."""
        seen_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="train",
            mode="test",
        )
        loader = DataLoader(
            seen_dataset,
            batch_size=int(self.args.get("eval_batch_size", self.args.get("batch_size", 128))),
            shuffle=False,
            num_workers=int(self.args.get("eval_num_workers", self.args.get("train_num_workers", 8))),
        )
        vectors, targets = self._extract_vectors(loader)
        return vectors.astype(np.float32, copy=False), targets.astype(np.int64, copy=False)

    def _set_class_means_from_feature_bank(
        self,
        vectors: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Build NME class means from a single feature-bank extraction."""
        nb_classes = int(self._total_classes)
        feat_dim = int(self.feature_dim)
        class_means = np.zeros((nb_classes, feat_dim), dtype=np.float64)
        if vectors.size == 0:
            self._class_means = class_means
            return

        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
        for cls_idx in range(nb_classes):
            cls_vectors = vectors[targets == cls_idx]
            if cls_vectors.size == 0:
                continue
            mu = np.mean(cls_vectors, axis=0)
            class_means[cls_idx] = mu / (np.linalg.norm(mu) + 1e-12)

        self._class_means = class_means

    def _log_task_snapshot(
        self,
        data_manager,
        *,
        tag: str,
        feature_bank: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Evaluate and log the current model state under a descriptive snapshot tag."""
        if feature_bank is None:
            feature_bank = self._extract_seen_feature_bank(data_manager)
        self._set_class_means_from_feature_bank(*feature_bank)
        cnn_accy, nme_accy = self.eval_task()
        self._scm_snapshot_metrics[tag] = {"cnn": cnn_accy, "nme": nme_accy}
        logger.info("[SCM][%s] CNN: %s", tag, cnn_accy["grouped"])
        if nme_accy is not None:
            logger.info("[SCM][%s] NME: %s", tag, nme_accy["grouped"])

    def _get_fc_module(self) -> Optional[nn.Module]:
        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        return getattr(net, "fc", None)

    def _recalibrate_classifier_from_feature_bank(
        self,
        vectors: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        fc = self._get_fc_module()
        if fc is None or not hasattr(fc, "weight"):
            logger.warning("[SCM][head] Skip recalibration: unsupported classifier head.")
            return

        method = self._scm_head_recalibration_method
        if method in {"prototype", "proto", "class_mean"}:
            self._apply_prototype_head(vectors, targets)
            logger.info("[SCM][head] Recalibrated classifier with prototype head.")
            return
        if method in {"linear_probe", "probe", "logreg"}:
            self._apply_prototype_head(vectors, targets)
            self._fit_linear_probe_head(vectors, targets)
            return
        raise ValueError(
            f"Unknown scm_head_recalibration_method: {self._scm_head_recalibration_method}"
        )

    def _apply_prototype_head(self, vectors: np.ndarray, targets: np.ndarray) -> None:
        """Initialise fc as a nearest-centroid linear classifier in feature space."""
        fc = self._get_fc_module()
        if fc is None or not hasattr(fc, "weight"):
            return

        device = fc.weight.device
        feature_bank = torch.from_numpy(vectors).float()
        label_bank = torch.from_numpy(targets).long()
        prototypes = torch.zeros(self._total_classes, feature_bank.shape[1], dtype=torch.float32)

        for cls_idx in range(self._total_classes):
            mask = label_bank == cls_idx
            if not torch.any(mask):
                continue
            prototypes[cls_idx] = feature_bank[mask].mean(dim=0)

        with torch.no_grad():
            fc.weight.data.copy_((2.0 * prototypes).to(device))
            if getattr(fc, "bias", None) is not None:
                fc.bias.data.copy_((-prototypes.pow(2).sum(dim=1)).to(device))

    def _fit_linear_probe_head(self, vectors: np.ndarray, targets: np.ndarray) -> None:
        """Refit the classifier head on frozen consolidated features."""
        fc = self._get_fc_module()
        if fc is None or not hasattr(fc, "weight"):
            return

        device = fc.weight.device
        features = torch.from_numpy(vectors).float()
        labels = torch.from_numpy(targets).long()
        dataset = TensorDataset(features, labels)
        batch_size = max(1, min(self._scm_head_recalibration_batch_size, len(dataset)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        params = [fc.weight]
        if getattr(fc, "bias", None) is not None:
            params.append(fc.bias)
        optimizer = torch.optim.Adam(
            params,
            lr=self._scm_head_recalibration_lr,
            weight_decay=self._scm_head_recalibration_weight_decay,
        )

        fc.train()
        for epoch in range(self._scm_head_recalibration_epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for feat_batch, target_batch in loader:
                feat_batch = feat_batch.to(device)
                target_batch = target_batch.to(device)
                logits = F.linear(feat_batch, fc.weight, getattr(fc, "bias", None))
                loss = F.cross_entropy(logits, target_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * target_batch.size(0)
                total_correct += int((logits.argmax(dim=1) == target_batch).sum().item())
                total_samples += int(target_batch.size(0))

            avg_loss = total_loss / max(1, total_samples)
            avg_acc = 100.0 * total_correct / max(1, total_samples)
            logger.info(
                "[SCM][head] linear_probe epoch %d/%d => Loss %.4f, Acc %.2f",
                epoch + 1,
                self._scm_head_recalibration_epochs,
                avg_loss,
                avg_acc,
            )
        fc.eval()
