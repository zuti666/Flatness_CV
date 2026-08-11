"""
cl_evaluator.py
---------------
Main orchestrator.  Called once per task boundary by OGD_Fisher3_mech.

Orchestration order at task t:
  1. Build old / new / all loaders
  2. Compute L_old, L_new, L_all  (+ ΔL if ref snapshot available)
  3. Compute S_old, S_new via emp_fisher power iteration
  4. Compute tr(H_old · Σ) — analytic + Monte Carlo
  5. Compute cos(ε, u_i^old) — noise–eigenvector alignment
  6. Save per-task JSON
  7. Return results dict

All heavy math is delegated to:
  - evaluation_CL_mechanism.loss_tracker       → L_{old/new/all}
  - evaluation_CL_mechanism.noise_curvature    → S, tr(HΣ), cos alignment
  - evaluation_CL_mechanism.loader_factory     → data loaders
  - evaluation_CL_mechanism.io_utils           → JSON persistence
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Any, Sequence

import torch
import torch.nn as nn

from evaluation_CL_mechanism.loader_factory import (
    make_old_loader,
    make_new_loader,
    make_all_loader,
)
from evaluation_CL_mechanism.loss_tracker import (
    compute_tripled_loss,
    compute_delta_loss,
)
from evaluation_CL_mechanism.noise_curvature import (
    tr_h_sigma_analytic,
    tr_h_sigma_mc,
    tr_h_sigma_mc_for_geometry,
    noise_eigvec_alignment,
    noise_eigvec_alignment_for_geometry,
    compute_lambda_max,
    expected_sharpness,
    estimate_fisher_subspace,
)
from evaluation_CL_mechanism.feature_tracker import (
    compute_triplet_feature_flatness,
    compute_old_task_feature_drift,
    save_reference_task_features,
)
from evaluation_CL_mechanism.loss_landscape_interface import run_loss_landscape_suite
from evaluation_CL_mechanism.drift_tracker import compute_drift_interference
from evaluation_CL_mechanism.io_utils import save_task_json
from sharpness_evaluation_core.curvature import build_mvp_fns

logger = logging.getLogger(__name__)


class CLMechanismEvaluator:
    """
    Online per-task-boundary evaluator.

    Usage (inside OGD_Fisher3_mech.incremental_train):

        evaluator = CLMechanismEvaluator(args, save_root)

        # snapshot before training (optional, for ΔL)
        ref_params = evaluator.snapshot_params(network)

        # ... train task t ...

        evaluator.evaluate(
            network        = learner._network,
            data_manager   = data_manager,
            cur_task       = learner._cur_task,
            fisher_basis   = learner._fisher_basis,
            fisher_eigvals = learner._fisher_eigvals,
            rwp_std        = learner._rwp_std,
            noise_geometry = learner._noise_geometry,
            device         = learner._device,
            ref_params     = ref_params,    # optional
        )
    """

    def __init__(self, args: dict, save_root: str):
        self.args        = args
        self.save_root   = save_root

        # budget knobs (configurable via YAML)
        self.max_batches  = int(args.get("cl_eval_max_batches",   20))
        self.n_noise      = int(args.get("cl_eval_n_noise_samples", 100))
        self.power_iters  = int(args.get("cl_eval_power_iters",   15))
        self.noise_topk   = int(args.get("cl_eval_noise_topk",    3))
        self.backend      = str(args.get("cl_eval_backend",       "emp_fisher"))
        self.fisher_eps   = float(args.get("fisher_eps",          1e-5))
        self.es_samples   = int(args.get("cl_eval_es_samples",    8))
        self.eval_feature   = bool(args.get("cl_eval_feature_eval", True))
        self.eval_es        = bool(args.get("cl_eval_expected_sharpness", True))
        self.eval_per_old   = bool(args.get("cl_eval_per_old_task", True))
        self.eval_drift     = bool(args.get("cl_eval_drift", True))
        self.drift_batches  = int(args.get("cl_eval_drift_max_batches", 3))
        # Separate smaller budget for S (λ_max) to avoid OOM
        self.s_max_batches  = int(args.get("cl_eval_s_max_batches", 5))
        self.s_power_iters  = int(args.get("cl_eval_s_power_iters", 5))
        self.eval_fisher_rank = int(args.get("cl_eval_fisher_rank", max(int(args.get("fisher_rank", 0)), 20)))
        self.eval_fisher_samples = int(args.get("cl_eval_fisher_samples", max(int(args.get("fisher_samples", 0)), 32)))
        self.eval_fisher_max_batches = int(
            args.get("cl_eval_fisher_max_batches", max(int(args.get("fisher_max_batches", 0)), self.max_batches))
        )

    def _config_snapshot(self) -> Dict[str, Any]:
        return {
            "model_name": self.args.get("model_name"),
            "prefix": self.args.get("prefix"),
            "seed": self.args.get("seed"),
            "cl_eval_backend": self.backend,
            "cl_eval_max_batches": self.max_batches,
            "cl_eval_n_noise_samples": self.n_noise,
            "cl_eval_power_iters": self.power_iters,
            "cl_eval_noise_topk": self.noise_topk,
            "cl_eval_es_samples": self.es_samples,
            "cl_eval_feature_eval": self.eval_feature,
            "cl_eval_expected_sharpness": self.eval_es,
            "cl_eval_per_old_task": self.eval_per_old,
            "cl_eval_drift": self.eval_drift,
            "cl_eval_drift_max_batches": self.drift_batches,
            "cl_eval_s_max_batches": self.s_max_batches,
            "cl_eval_s_power_iters": self.s_power_iters,
            "cl_eval_fisher_rank": self.eval_fisher_rank,
            "cl_eval_fisher_samples": self.eval_fisher_samples,
            "cl_eval_fisher_max_batches": self.eval_fisher_max_batches,
            "cl_eval_lossland": bool(self.args.get("cl_eval_lossland", False)),
            "cl_eval_lossland_targets": self.args.get("cl_eval_lossland_targets", "old,new,all"),
            "fisher_eps": self.fisher_eps,
        }

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        network: nn.Module,
        data_manager,
        cur_task: int,
        fisher_basis: List[torch.Tensor],
        fisher_eigvals: torch.Tensor,
        rwp_std: float,
        noise_geometry: str,
        device: torch.device,
        ref_params: Optional[torch.Tensor] = None,
        ogd_directions: Optional[Sequence[torch.Tensor]] = None,
        known_classes: int = 0,
        total_classes: int = 0,
    ) -> Dict[str, Any]:
        """
        Run all mechanism metrics and write per-task JSON.

        Returns the results dict (also written to disk).
        """
        t_start = time.time()
        results: Dict[str, Any] = {
            "task":           cur_task,
            "noise_geometry": noise_geometry,
            "rwp_std":        rwp_std,
            "config_snapshot": self._config_snapshot(),
        }

        # ── 1. Loaders ────────────────────────────────────────────────────────
        old_loader = make_old_loader(data_manager, cur_task, self.args)
        new_loader = make_new_loader(data_manager, cur_task, self.args)
        all_loader = make_all_loader(data_manager, cur_task, self.args)

        # ── 2. Loss ───────────────────────────────────────────────────────────
        loss_out = compute_tripled_loss(
            network, old_loader, new_loader, all_loader, device, self.max_batches
        )
        results.update(loss_out)

        # ΔL (only if caller provided a θ_{t-1} snapshot)
        if ref_params is not None:
            delta_out = compute_delta_loss(
                network, old_loader, new_loader, device,
                self.max_batches, ref_params, exclude_prefix=None
            )
            results.update(delta_out)
            # derive ΔL values from current + prev
            if "L_old" in results and "delta_L_old_prev" in results:
                results["delta_L_old"] = results["L_old"] - results["delta_L_old_prev"]
            if "L_new" in results and "delta_L_new_prev" in results:
                results["delta_L_new"] = results["delta_L_new_prev"] - results["L_new"]

        # ── 2.5 Drift interference  v^T q  ────────────────────────────────────
        # Layer 1 of Table 3: verify OGD zeroes the drift term.
        # Only meaningful when there are old tasks (cur_task >= 1).
        if self.eval_drift and cur_task >= 1:
            try:
                drift_out = compute_drift_interference(
                    network        = network,
                    old_loader     = old_loader,
                    new_loader     = new_loader,
                    device         = device,
                    ogd_directions = list(ogd_directions) if ogd_directions else [],
                    max_batches    = self.drift_batches,
                    known_classes  = known_classes,
                    total_classes  = total_classes,
                )
                results.update(drift_out)
            except Exception as exc:
                logger.warning("[cl_evaluator] drift interference failed: %s", exc)

        # ── 3. S_old / S_new  (λ_max) ─────────────────────────────────────────
        # Use a smaller dedicated budget (s_max_batches, s_power_iters) to
        # reduce OOM risk — separate from the general max_batches knob.
        import torch as _torch
        _torch.cuda.empty_cache()
        network.eval()
        if old_loader is not None and cur_task >= 1:
            try:
                s_old = compute_lambda_max(
                    network, old_loader, device,
                    self.s_max_batches, self.s_power_iters, self.backend
                )
                results["S_old"] = s_old
            except Exception as exc:
                logger.warning("[cl_evaluator] S_old failed: %s", exc)

        try:
            s_new = compute_lambda_max(
                network, new_loader, device,
                self.s_max_batches, self.s_power_iters, self.backend
            )
            results["S_new"] = s_new
        except Exception as exc:
            logger.warning("[cl_evaluator] S_new failed: %s", exc)
        try:
            s_all = compute_lambda_max(
                network, all_loader, device,
                self.s_max_batches, self.s_power_iters, self.backend
            )
            results["S_all"] = s_all
        except Exception as exc:
            logger.warning("[cl_evaluator] S_all failed: %s", exc)

        # ── 4. tr(H_old · Σ_noise) ─────────────────────────────────────────────
        has_fisher = bool(fisher_basis) and fisher_eigvals.numel() > 0
        total_dim = int(sum(p.numel() for p in network.parameters() if p.requires_grad))
        analysis_fisher_basis = list(fisher_basis) if has_fisher else []
        analysis_fisher_eigvals = fisher_eigvals.detach().clone() if has_fisher else torch.tensor([], dtype=torch.float32)
        analysis_fisher_source = "history_cache" if has_fisher else "unavailable"

        if cur_task >= 1 and old_loader is not None and (
            not analysis_fisher_basis or analysis_fisher_eigvals.numel() == 0
        ):
            try:
                est = estimate_fisher_subspace(
                    network,
                    old_loader,
                    device,
                    rank=self.eval_fisher_rank,
                    max_batches=self.eval_fisher_max_batches,
                    sample_cap=self.eval_fisher_samples,
                    fisher_eps=self.fisher_eps,
                )
                analysis_fisher_basis = est["basis"]
                analysis_fisher_eigvals = est["eigvals"]
                analysis_fisher_source = f"estimated_old_loader:{est['status']}"
                results["analysis_fisher_est_batches"] = int(est["batches"])
            except Exception as exc:
                analysis_fisher_source = f"estimate_failed:{exc}"

        has_analysis_fisher = bool(analysis_fisher_basis) and analysis_fisher_eigvals.numel() > 0
        results["analysis_fisher_source"] = analysis_fisher_source
        results["analysis_fisher_rank_used"] = int(len(analysis_fisher_basis))

        # 4a. Analytic (free)
        if has_analysis_fisher:
            analytic_out = tr_h_sigma_analytic(
                analysis_fisher_eigvals,
                analysis_fisher_basis,
                rwp_std,
                fisher_eps=self.fisher_eps,
            )
            results.update(analytic_out)

        # 4b. Monte Carlo (uses old_loader MVP)
        if has_analysis_fisher and old_loader is not None and cur_task >= 1:
            try:
                _, _, mvp_f_old = build_mvp_fns(
                    network, old_loader, device,
                    loss_eval_max_batches=self.max_batches,
                )
                mc_out = tr_h_sigma_mc(
                    mvp_fn         = mvp_f_old,
                    fisher_basis   = analysis_fisher_basis,
                    fisher_eigvals = analysis_fisher_eigvals,
                    sigma          = rwp_std,
                    n_samples      = self.n_noise,
                    device         = device,
                    fisher_eps     = self.fisher_eps,
                )
                results.update(mc_out)
            except Exception as exc:
                logger.warning("[cl_evaluator] MC tr(HΣ) failed: %s", exc)

        method_project_ogd = bool(ogd_directions)
        method_geometry = str(noise_geometry or "none").lower()
        if old_loader is not None and cur_task >= 1:
            try:
                _, _, mvp_f_old = build_mvp_fns(
                    network, old_loader, device,
                    loss_eval_max_batches=self.max_batches,
                )
                method_mc = tr_h_sigma_mc_for_geometry(
                    mvp_f_old,
                    geometry=method_geometry,
                    sigma=rwp_std,
                    n_samples=self.n_noise,
                    device=device,
                    fisher_basis=analysis_fisher_basis if has_analysis_fisher else None,
                    fisher_eigvals=analysis_fisher_eigvals if has_analysis_fisher else None,
                    ogd_directions=ogd_directions,
                    fisher_eps=self.fisher_eps,
                    project_ogd=method_project_ogd,
                    dim=total_dim,
                )
                results.update(method_mc)
                if "mc_tr_h_sigma_gaussian" in results and f"mc_tr_h_sigma_{method_geometry}" in method_mc:
                    denom = float(method_mc.get(f"mc_tr_h_sigma_{method_geometry}", float("nan")))
                    if denom != 0.0 and not torch.isnan(torch.tensor(denom)):
                        results["mc_tr_ratio_gauss_method"] = float(results["mc_tr_h_sigma_gaussian"]) / abs(denom)
            except Exception as exc:
                logger.warning("[cl_evaluator] method-specific MC tr(HΣ) failed: %s", exc)

        # ── 5. Noise–eigenvector alignment ─────────────────────────────────────
        if has_analysis_fisher:
            try:
                align_out = noise_eigvec_alignment(
                    fisher_basis   = analysis_fisher_basis,
                    fisher_eigvals = analysis_fisher_eigvals,
                    sigma          = rwp_std,
                    n_samples      = self.n_noise,
                    topk           = self.noise_topk,
                    fisher_eps     = self.fisher_eps,
                )
                results.update(align_out)
            except Exception as exc:
                logger.warning("[cl_evaluator] noise alignment failed: %s", exc)
            try:
                method_align = noise_eigvec_alignment_for_geometry(
                    fisher_basis=analysis_fisher_basis,
                    fisher_eigvals=analysis_fisher_eigvals,
                    geometry=method_geometry,
                    sigma=rwp_std,
                    ogd_directions=ogd_directions,
                    n_samples=self.n_noise,
                    topk=self.noise_topk,
                    fisher_eps=self.fisher_eps,
                    project_ogd=method_project_ogd,
                )
                results.update(method_align)
            except Exception as exc:
                logger.warning("[cl_evaluator] method noise alignment failed: %s", exc)

        # ── 6. Expected sharpness ──────────────────────────────────────────────
        if self.eval_es:
            loader_map = {
                "old": old_loader,
                "new": new_loader,
                "all": all_loader,
            }
            base_loss_map = {
                "old": results.get("L_old"),
                "new": results.get("L_new"),
                "all": results.get("L_all"),
            }
            for split, loader in loader_map.items():
                if loader is None:
                    results[f"ES_{split}_status"] = "skipped_no_loader"
                    continue
                try:
                    iso_out = expected_sharpness(
                        network,
                        loader,
                        device,
                        sigma=rwp_std,
                        geometry="gaussian",
                        max_batches=self.max_batches,
                        n_samples=self.es_samples,
                        base_loss=base_loss_map.get(split),
                    )
                    results[f"ES_{split}_iso"] = iso_out.get("es_gaussian", float("nan"))
                    results[f"ES_{split}_iso_std"] = iso_out.get("es_gaussian_std", 0.0)
                    results[f"ES_{split}_iso_status"] = iso_out.get("es_gaussian_status", "unknown")

                    method_out = expected_sharpness(
                        network,
                        loader,
                        device,
                        sigma=rwp_std,
                        geometry=method_geometry,
                        max_batches=self.max_batches,
                        n_samples=self.es_samples,
                        fisher_basis=analysis_fisher_basis if has_analysis_fisher else None,
                        fisher_eigvals=analysis_fisher_eigvals if has_analysis_fisher else None,
                        ogd_directions=ogd_directions,
                        fisher_eps=self.fisher_eps,
                        project_ogd=method_project_ogd,
                        base_loss=base_loss_map.get(split),
                    )
                    results[f"ES_{split}_method"] = method_out.get(f"es_{method_geometry}", float("nan"))
                    results[f"ES_{split}_method_std"] = method_out.get(f"es_{method_geometry}_std", 0.0)
                    results[f"ES_{split}_method_status"] = method_out.get(f"es_{method_geometry}_status", "unknown")
                except Exception as exc:
                    logger.warning("[cl_evaluator] expected sharpness failed on %s: %s", split, exc)
                    results[f"ES_{split}_status"] = f"failed:{exc}"

        # ── 7. Feature flatness + per-old-task feature drift ─────────────────
        if self.eval_feature:
            try:
                results.update(
                    compute_triplet_feature_flatness(
                        network,
                        old_loader,
                        new_loader,
                        all_loader,
                        self.args,
                    )
                )
            except Exception as exc:
                logger.warning("[cl_evaluator] feature flatness failed: %s", exc)

            try:
                # Save current task as future reference before comparing older tasks.
                ref_cache_path = save_reference_task_features(
                    network,
                    data_manager,
                    cur_task,
                    self.save_root,
                    device,
                    self.args,
                )
                if ref_cache_path:
                    results["feature_ref_cache"] = ref_cache_path
            except Exception as exc:
                logger.warning("[cl_evaluator] feature cache save failed: %s", exc)

            if self.eval_per_old:
                try:
                    results.update(
                        compute_old_task_feature_drift(
                            network,
                            data_manager,
                            cur_task,
                            self.save_root,
                            device,
                            self.args,
                        )
                    )
                except Exception as exc:
                    logger.warning("[cl_evaluator] old-task feature drift failed: %s", exc)

        # ── 8. Optional loss landscape artifacts ──────────────────────────────
        try:
            results.update(
                run_loss_landscape_suite(
                    network,
                    {"old": old_loader, "new": new_loader, "all": all_loader},
                    device=device,
                    save_root=self.save_root,
                    task_idx=cur_task,
                    args=self.args,
                )
            )
        except Exception as exc:
            logger.warning("[cl_evaluator] loss landscape suite failed: %s", exc)

        # ── 9. Save JSON ───────────────────────────────────────────────────────
        results["eval_time_s"] = round(time.time() - t_start, 1)
        save_path = os.path.join(self.save_root, f"cl_mechanism_t{cur_task:02d}.json")
        save_task_json(save_path, results)

        logger.info(
            "[cl_evaluator] task %2d  "
            "L_old=%.4f  L_new=%.4f  dL_old=%.4f  "
            "drift_cos(vq)=%.3f  drift_cos(vg)=%.3f  "
            "S_old=%.3e  S_new=%.3e  "
            "tr_ratio(analytic)=%.2f  ES_old_iso=%.4f  (%.1fs)",
            cur_task,
            results.get("L_old",                           float("nan")),
            results.get("L_new",                           float("nan")),
            results.get("delta_L_old",                     float("nan")),
            results.get("drift_cos_vq",                    float("nan")),
            results.get("drift_cos_vg",                    float("nan")),
            results.get("S_old",                           float("nan")),
            results.get("S_new",                           float("nan")),
            results.get("analytic_tr_ratio_gauss_fisher",  float("nan")),
            results.get("ES_old_iso",                      float("nan")),
            results["eval_time_s"],
        )
        return results

    @staticmethod
    def snapshot_params(network: nn.Module, exclude_prefix: Optional[str] = "fc.") -> torch.Tensor:
        """Lightweight θ_{t-1} snapshot. Call before incremental_train.
        Excludes fc.* by default; pass ``exclude_prefix=None`` to snapshot all
        trainable parameters after the head has already been expanded."""
        from evaluation_CL_mechanism.loss_tracker import snapshot_params
        return snapshot_params(network, exclude_prefix=exclude_prefix)
