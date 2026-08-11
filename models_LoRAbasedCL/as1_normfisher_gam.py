"""AS^1 + Trace-Normalized Fisher + Adaptive Dual Ascent (Paper A method).

Improvements over EWCLoRA-GAM:

1. Trace-normalized Fisher: F̃ = F / (trace(F) + eps).
   Reason: raw Fisher in ΔW = BA space has values ~1e-6 (due to low-rank projection),
   making the EWC penalty <0.1% of task loss regardless of λ_ewc. Trace normalization
   makes the penalty scale predictable and λ_ewc directly controls penalty/task_loss ratio.

2. Adaptive dual variable (true Lagrangian): after each task, update λ_ewc via dual ascent:
      λ_new = max(0, λ_old + α * (D^Fisher_normalized - δ))
   where δ is the forgetting budget and D^Fisher_normalized is the normalized drift.
   This makes the method a penalty-method approximation to the constrained optimization:
      min L_task + λ_flat S^(1)_Δ   s.t.  D^Fisher_normalized ≤ δ

Both improvements are lossless: if λ_ewc remains constant and Fisher is not normalized,
this class reduces exactly to ewclora_youyue_fitarchitecture_gam.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch

from models_LoRAbasedCL.ewclora_youyue_fitArchitechture_gam import Learner as _BaseGAMLearner

logger = logging.getLogger(__name__)


class Learner(_BaseGAMLearner):
    """AS^1 + trace-normalized Fisher + optional adaptive dual variable."""

    def __init__(self, args):
        super().__init__(args)

        self._normalize_fisher = bool(args.get("ewc_normalize_fisher", True))
        self._fisher_norm_eps = float(args.get("ewc_fisher_norm_eps", 1e-12))

        # Dual ascent (adaptive λ)
        self._dual_ascent = bool(args.get("ewc_dual_ascent", False))
        self._dual_alpha = float(args.get("ewc_dual_alpha", 1.0))
        self._dual_delta = float(args.get("ewc_dual_delta", 1e-4))
        self._dual_lambda_max = float(args.get("ewc_dual_lambda_max", 1e6))
        self._dual_lambda_min = float(args.get("ewc_dual_lambda_min", 0.0))

        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher] normalize_fisher=%s, dual_ascent=%s, "
                "dual_alpha=%s, dual_delta=%s, ewc_lambda_init=%s",
                self._normalize_fisher,
                self._dual_ascent,
                self._dual_alpha,
                self._dual_delta,
                self._lambda_ewc,
            )

    # ------------------------------------------------------------------
    # Override Fisher computation to add trace normalization
    # ------------------------------------------------------------------
    def _compute_delta_fisher(self, loader, model, class_offset: int) -> Dict[str, torch.Tensor]:
        raw_fisher = super()._compute_delta_fisher(loader, model, class_offset)

        if not self._normalize_fisher or not raw_fisher:
            return raw_fisher

        # Trace-normalize: F̃ = F / (Σ_k trace(F_k) + eps)
        total_trace = sum(f.sum().item() for f in raw_fisher.values())
        denom = float(total_trace) + self._fisher_norm_eps
        return {key: f / denom for key, f in raw_fisher.items()}

    # ------------------------------------------------------------------
    # Override after_task to add dual ascent update
    # ------------------------------------------------------------------
    def after_task(self):
        # Compute drift metrics BEFORE calling parent (which clears delta_reference)
        if self._mechanism_eval or self._dual_ascent:
            drift_before = self._compute_delta_drift_metrics()
        else:
            drift_before = {}

        super().after_task()  # Fisher accumulation + super lifecycle

        if self._dual_ascent and self._cur_task > 0:
            # Use normalized_fisher_drift as the Lagrangian constraint value D^Fisher
            d_fisher = float(drift_before.get("normalized_fisher_drift", 0.0))
            violation = d_fisher - self._dual_delta
            old_lambda = float(self._lambda_ewc)
            new_lambda = float(
                min(
                    self._dual_lambda_max,
                    max(self._dual_lambda_min, old_lambda + self._dual_alpha * violation),
                )
            )
            self._lambda_ewc = new_lambda
            self._ewc_lambda = new_lambda  # keep parent attribute in sync
            if self._is_main_process:
                logger.info(
                    "[AS1-NormFisher] Dual ascent task %d: D^Fisher=%.3e, δ=%.3e, "
                    "violation=%.3e, λ_ewc: %.2f → %.2f",
                    int(self._cur_task),
                    d_fisher,
                    self._dual_delta,
                    violation,
                    old_lambda,
                    new_lambda,
                )
