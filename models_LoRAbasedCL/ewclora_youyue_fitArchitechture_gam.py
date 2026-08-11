from __future__ import annotations

import logging
from typing import Dict, List

import torch

from models_LoRAbasedCL.ewclora_youyue_fitArchitechture import Learner as _BaseFitArchLearner

logger = logging.getLogger(__name__)


class Learner(_BaseFitArchLearner):
    """Explicit GAM + past-Fisher version on top of the framework-native EWCLoRA shell.

    Flatness and forgetting are handled by two separate components:

    - GAM provides the current-task flatness-aware direction.
    - The past-task Fisher penalty is differentiated explicitly in delta-W space
      and added to the final update direction.

    The final batch update direction is:

        g_clean + lambda_flat * (g_gam - g_clean) + g_fisher

    where:

    - g_clean  : clean current-task gradient
    - g_gam    : GAM flatness-aware gradient
    - g_fisher : gradient of the explicit past-Fisher quadratic penalty
    """

    def __init__(self, args):
        super().__init__(args)
        if self._optimizer_type != "gam":
            raise ValueError(
                "ewclora_youyue_fitArchitechture_gam only supports optimizer_type=gam, "
                f"got {self._optimizer_type}."
            )

        self._lambda_flat = float(args.get("lambda_flat", 1.0))
        self._lambda_ewc = float(args.get("ewc_lambda", args.get("lambda", 20.0)))
        self._fisher_eta = float(args.get("ewc_eta", args.get("eta", 0.0)))
        self._num_finished_tasks = 0
        self._mechanism_eval = bool(args.get("mechanism_eval", False))
        self._mechanism_grad_stats = bool(args.get("mechanism_grad_stats", self._mechanism_eval))
        self._mechanism_top_fraction = float(args.get("mechanism_top_fraction", 0.1))
        topk = args.get("mechanism_topk", None)
        self._mechanism_topk = None if topk is None else int(topk)
        self._mechanism_eps = float(args.get("mechanism_eps", 1e-12))
        self._kl_diag_sigma2 = float(args.get("kl_diag_sigma2", 1.0))
        self._mechanism_grad_sums: Dict[str, float] = {}
        self._mechanism_grad_count = 0
        self._mechanism_loss_sums: Dict[str, float] = {}
        self._mechanism_loss_count = 0
        self._last_mechanism_metrics: Dict[str, float] = {}

        self._as1_rho = float(args.get("as1_rho", getattr(self, "_gam_grad_rho", 0.2)))
        self._as1_norm_rho = float(
            args.get("as1_norm_rho", getattr(self, "_gam_grad_norm_rho", 0.2))
        )

        # Keep the parent implementation consistent with the explicit names used here.
        self._ewc_lambda = self._lambda_ewc
        self._ewc_eta = self._fisher_eta
        self._gam_grad_rho = self._as1_rho
        self._gam_grad_norm_rho = self._as1_norm_rho
        if hasattr(self, "_gam_args"):
            self._gam_args.grad_rho = self._as1_rho
            self._gam_args.grad_norm_rho = self._as1_norm_rho

    def after_task(self):
        drift_metrics = self._compute_delta_drift_metrics() if self._mechanism_eval else {}
        super().after_task()
        self._num_finished_tasks = int(self._cur_task + 1)
        if self._mechanism_eval:
            self._last_mechanism_metrics = {
                "task": int(self._cur_task),
                "lambda_flat": float(self._lambda_flat),
                "lambda_ewc": float(self._lambda_ewc),
                "fisher_eta": float(self._fisher_eta),
                **self._summarize_mechanism_loss_stats(),
                **self._summarize_mechanism_grad_stats(),
                **drift_metrics,
            }
            if self._is_main_process:
                self._log(f"[EWCLoRA-GAM][Mechanism] {self._last_mechanism_metrics}")

    def get_mechanism_metrics(self) -> Dict[str, float]:
        return dict(self._last_mechanism_metrics)

    def _reset_mechanism_stats(self) -> None:
        self._mechanism_grad_sums = {}
        self._mechanism_grad_count = 0
        self._mechanism_loss_sums = {}
        self._mechanism_loss_count = 0
        self._last_mechanism_metrics = {}

    def _snapshot_task_reference(self) -> None:
        super()._snapshot_task_reference()
        if self._mechanism_eval:
            self._reset_mechanism_stats()

    @staticmethod
    def _flatten_optimizer_params(optimizer) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        seen = set()
        for group in optimizer.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                pid = id(param)
                if pid in seen:
                    continue
                seen.add(pid)
                params.append(param)
        return params

    def _compute_clean_task_grad_cache(
        self,
        params: List[torch.nn.Parameter],
        inputs: torch.Tensor,
        targets: torch.Tensor,
        class_offset: int,
    ):
        with torch.enable_grad():
            outputs = self._network(inputs)
            task_loss, _, _ = self._task_loss(outputs["logits"], targets, class_offset)
        grads = torch.autograd.grad(task_loss, params, allow_unused=True)
        grad_cache = {
            id(param): grad.detach().clone()
            for param, grad in zip(params, grads)
            if grad is not None
        }
        return float(task_loss.detach().item()), grad_cache

    def _compute_fisher_grad_cache(self, params: List[torch.nn.Parameter]):
        if not self._fisher_past_delta:
            return 0.0, {}

        base_model = self._unwrap_network()
        with torch.enable_grad():
            penalty = self._past_fisher_penalty(base_model)
        grads = torch.autograd.grad(penalty, params, allow_unused=True)
        grad_cache = {
            id(param): grad.detach().clone()
            for param, grad in zip(params, grads)
            if grad is not None
        }
        return float(penalty.detach().item()), grad_cache

    @staticmethod
    def _cosine_from_sums(dot: float, norm_a_sq: float, norm_b_sq: float, eps: float) -> float:
        denom = (max(norm_a_sq, 0.0) ** 0.5) * (max(norm_b_sq, 0.0) ** 0.5)
        if denom <= eps:
            return 0.0
        return float(dot / (denom + eps))

    def _accumulate_mechanism_grad_stats(self, stats: Dict[str, float]) -> None:
        if not self._mechanism_grad_stats:
            return
        for key, value in stats.items():
            self._mechanism_grad_sums[key] = self._mechanism_grad_sums.get(key, 0.0) + float(value)
        self._mechanism_grad_count += 1

    def _record_mechanism_loss_stats(self, clean_loss: float, fisher_loss: float) -> None:
        if not self._mechanism_eval:
            return
        self._mechanism_loss_sums["clean_loss"] = (
            self._mechanism_loss_sums.get("clean_loss", 0.0) + float(clean_loss)
        )
        self._mechanism_loss_sums["fisher_loss"] = (
            self._mechanism_loss_sums.get("fisher_loss", 0.0) + float(fisher_loss)
        )
        self._mechanism_loss_count += 1

    def _summarize_mechanism_loss_stats(self) -> Dict[str, float]:
        if self._mechanism_loss_count <= 0:
            return {}
        denom = float(self._mechanism_loss_count)
        return {
            "train_clean_loss_mean": self._mechanism_loss_sums.get("clean_loss", 0.0) / denom,
            "train_fisher_loss_mean": self._mechanism_loss_sums.get("fisher_loss", 0.0) / denom,
        }

    def _summarize_mechanism_grad_stats(self) -> Dict[str, float]:
        if self._mechanism_grad_count <= 0:
            return {}
        denom = float(self._mechanism_grad_count)
        return {
            f"grad_{key}_mean": value / denom
            for key, value in sorted(self._mechanism_grad_sums.items())
        }

    def _compute_delta_drift_metrics(self) -> Dict[str, float]:
        if not self._fisher_past_delta or not self._delta_reference:
            return {
                "delta_drift_available": 0,
                "delta_tensors": 0,
                "delta_numel": 0,
                "delta_norm": 0.0,
                "fisher_weighted_drift": 0.0,
                "fisher_eta_weighted_drift": 0.0,
                "normalized_fisher_drift": 0.0,
                "normalized_fisher_eta_drift": 0.0,
                "high_fisher_delta_energy_ratio": 0.0,
                "high_fisher_weighted_energy_ratio": 0.0,
                "kl_iso_delta": 0.0,
                "kl_step_iso_proxy": 0.0,
                "kl_gaussian_l2_sigma2": 0.0,
                "kl_gaussian_l2_per_dim": 0.0,
                "kl_fisher_delta": 0.0,
                "kl_fisher_delta_weighted": 0.0,
                "kl_fisher_to_iso_ratio": 0.0,
                "kl_fisher_per_tensor_mean": 0.0,
                "kl_fisher_per_layer_mean": 0.0,
                "kl_fisher_per_task": 0.0,
                "kl_step_fisher_proxy": 0.0,
                "kl_fisher_proxy": 0.0,
                "kl_fisher_eta_proxy": 0.0,
                "kl_fisher_proxy_per_dim": 0.0,
                "kl_fisher_eta_proxy_per_dim": 0.0,
                "kl_fisher_eta_proxy_scaled_by_lambda": 0.0,
                "ewc_penalty_value": 0.0,
            }

        base_model = self._unwrap_network()
        eps = float(self._mechanism_eps)
        kl_sigma2 = max(float(self._kl_diag_sigma2), eps)
        delta_norm = 0.0
        fisher_weighted = 0.0
        fisher_eta_weighted = 0.0
        high_delta_energy = 0.0
        high_weighted_energy = 0.0
        fisher_sum = 0.0
        fisher_max = 0.0
        fisher_numel = 0
        delta_numel = 0
        tensor_count = 0

        with torch.no_grad():
            for key, delta_now in self._iter_delta_terms(base_model):
                fisher = self._fisher_past_delta.get(key)
                delta_ref = self._delta_reference.get(key)
                if fisher is None or delta_ref is None:
                    continue

                delta_ref_t = delta_ref.to(device=delta_now.device, dtype=delta_now.dtype)
                fisher_t = fisher.to(device=delta_now.device, dtype=delta_now.dtype)
                update_sq = (delta_now - delta_ref_t).pow(2)
                fisher_update = fisher_t * update_sq

                delta_norm += float(update_sq.sum().item())
                delta_numel += int(update_sq.numel())
                fisher_weighted += float(fisher_update.sum().item())
                if self._fisher_eta != 0.0:
                    fisher_eta_weighted += float(((fisher_t + self._fisher_eta) * update_sq).sum().item())
                else:
                    fisher_eta_weighted += float(fisher_update.sum().item())

                fisher_sum += float(fisher_t.sum().item())
                fisher_max = max(fisher_max, float(fisher_t.max().item()) if fisher_t.numel() else 0.0)
                fisher_numel += int(fisher_t.numel())
                tensor_count += 1

                flat_fisher = fisher_t.reshape(-1)
                flat_update_sq = update_sq.reshape(-1)
                if flat_fisher.numel() > 0:
                    if self._mechanism_topk is not None:
                        k = min(max(int(self._mechanism_topk), 1), int(flat_fisher.numel()))
                    else:
                        frac = min(max(float(self._mechanism_top_fraction), 0.0), 1.0)
                        k = min(max(int(round(float(flat_fisher.numel()) * frac)), 1), int(flat_fisher.numel()))
                    top_idx = torch.topk(flat_fisher, k=k, largest=True, sorted=False).indices
                    high_delta_energy += float(flat_update_sq[top_idx].sum().item())
                    high_weighted_energy += float((flat_fisher[top_idx] * flat_update_sq[top_idx]).sum().item())

        kl_gaussian_l2 = 0.5 * delta_norm / kl_sigma2
        kl_fisher = 0.5 * fisher_weighted
        kl_fisher_eta = 0.5 * fisher_eta_weighted
        ewc_penalty = self._lambda_ewc * kl_fisher_eta
        denom_dim = float(max(delta_numel, 1))
        denom_tensor = float(max(tensor_count, 1))

        return {
            "delta_drift_available": 1,
            "delta_tensors": int(tensor_count),
            "delta_numel": int(delta_numel),
            "delta_norm": float(delta_norm),
            "fisher_weighted_drift": float(fisher_weighted),
            "fisher_eta_weighted_drift": float(fisher_eta_weighted),
            "normalized_fisher_drift": float(fisher_weighted / (delta_norm + eps)),
            "normalized_fisher_eta_drift": float(fisher_eta_weighted / (delta_norm + eps)),
            "high_fisher_delta_energy_ratio": float(high_delta_energy / (delta_norm + eps)),
            "high_fisher_weighted_energy_ratio": float(high_weighted_energy / (fisher_weighted + eps)),
            "fisher_mean": float(fisher_sum / max(fisher_numel, 1)),
            "fisher_max": float(fisher_max),
            "kl_iso_delta": float(kl_gaussian_l2),
            "kl_step_iso_proxy": float(kl_gaussian_l2),
            "kl_gaussian_l2_sigma2": float(kl_gaussian_l2),
            "kl_gaussian_l2_per_dim": float(kl_gaussian_l2 / denom_dim),
            "kl_fisher_delta": float(kl_fisher_eta),
            "kl_fisher_delta_weighted": float(ewc_penalty),
            "kl_fisher_to_iso_ratio": float(kl_fisher_eta / (kl_gaussian_l2 + eps)),
            "kl_fisher_per_tensor_mean": float(kl_fisher_eta / denom_tensor),
            "kl_fisher_per_layer_mean": float(kl_fisher_eta / denom_tensor),
            "kl_fisher_per_task": float(kl_fisher_eta),
            "kl_step_fisher_proxy": float(kl_fisher_eta),
            "kl_fisher_proxy": float(kl_fisher),
            "kl_fisher_eta_proxy": float(kl_fisher_eta),
            "kl_fisher_proxy_per_dim": float(kl_fisher / denom_dim),
            "kl_fisher_eta_proxy_per_dim": float(kl_fisher_eta / denom_dim),
            "kl_fisher_eta_proxy_scaled_by_lambda": float(ewc_penalty),
            "ewc_penalty_value": float(ewc_penalty),
        }

    def _build_direction_projector(
        self,
        clean_grad_cache: Dict[int, torch.Tensor],
        fisher_grad_cache: Dict[int, torch.Tensor],
    ):
        lambda_flat = float(self._lambda_flat)
        eps = float(self._mechanism_eps)

        def projector(param_groups):
            clean_norm_sq = 0.0
            gam_norm_sq = 0.0
            flat_norm_sq = 0.0
            fisher_norm_sq = 0.0
            combined_norm_sq = 0.0
            clean_gam_dot = 0.0
            clean_flat_dot = 0.0
            clean_fisher_dot = 0.0
            flat_fisher_dot = 0.0
            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue

                    clean_grad = clean_grad_cache.get(id(param))
                    fisher_grad = fisher_grad_cache.get(id(param))
                    gam_grad = None if param.grad is None else param.grad.detach().clone()

                    if clean_grad is None:
                        combined_grad = gam_grad
                    elif gam_grad is None:
                        combined_grad = clean_grad.clone()
                    else:
                        combined_grad = clean_grad + lambda_flat * (gam_grad - clean_grad)

                    if fisher_grad is not None:
                        combined_grad = (
                            fisher_grad.clone()
                            if combined_grad is None
                            else combined_grad + fisher_grad
                        )

                    if combined_grad is None:
                        continue

                    if self._mechanism_grad_stats:
                        if clean_grad is not None:
                            clean_norm_sq += float(clean_grad.pow(2).sum().item())
                        if gam_grad is not None:
                            gam_norm_sq += float(gam_grad.pow(2).sum().item())
                        if fisher_grad is not None:
                            fisher_norm_sq += float(fisher_grad.pow(2).sum().item())
                        combined_norm_sq += float(combined_grad.pow(2).sum().item())
                        if clean_grad is not None and gam_grad is not None:
                            flat_grad = gam_grad - clean_grad
                            flat_norm_sq += float(flat_grad.pow(2).sum().item())
                            clean_gam_dot += float((clean_grad * gam_grad).sum().item())
                            clean_flat_dot += float((clean_grad * flat_grad).sum().item())
                            if fisher_grad is not None:
                                flat_fisher_dot += float((flat_grad * fisher_grad).sum().item())
                        if clean_grad is not None and fisher_grad is not None:
                            clean_fisher_dot += float((clean_grad * fisher_grad).sum().item())

                    if param.grad is None:
                        param.grad = combined_grad.clone()
                    else:
                        param.grad.data.copy_(combined_grad)

            if self._mechanism_grad_stats:
                clean_norm = clean_norm_sq ** 0.5
                gam_norm = gam_norm_sq ** 0.5
                flat_norm = flat_norm_sq ** 0.5
                fisher_norm = fisher_norm_sq ** 0.5
                combined_norm = combined_norm_sq ** 0.5
                self._accumulate_mechanism_grad_stats(
                    {
                        "clean_norm": clean_norm,
                        "gam_norm": gam_norm,
                        "flat_component_norm": flat_norm,
                        "fisher_norm": fisher_norm,
                        "combined_norm": combined_norm,
                        "flat_to_clean_norm_ratio": flat_norm / (clean_norm + eps),
                        "fisher_to_clean_norm_ratio": fisher_norm / (clean_norm + eps),
                        "cos_clean_gam": self._cosine_from_sums(
                            clean_gam_dot, clean_norm_sq, gam_norm_sq, eps
                        ),
                        "cos_clean_flat": self._cosine_from_sums(
                            clean_flat_dot, clean_norm_sq, flat_norm_sq, eps
                        ),
                        "cos_clean_fisher": self._cosine_from_sums(
                            clean_fisher_dot, clean_norm_sq, fisher_norm_sq, eps
                        ),
                        "cos_flat_fisher": self._cosine_from_sums(
                            flat_fisher_dot, flat_norm_sq, fisher_norm_sq, eps
                        ),
                    }
                )

        return projector

    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset: int):
        del scheduler
        params = self._flatten_optimizer_params(optimizer)
        clean_loss_value, clean_grad_cache = self._compute_clean_task_grad_cache(
            params=params,
            inputs=inputs,
            targets=targets,
            class_offset=class_offset,
        )
        fisher_loss_value, fisher_grad_cache = self._compute_fisher_grad_cache(params)
        self._record_mechanism_loss_stats(clean_loss_value, fisher_loss_value)

        optimizer.set_direction_projector(
            self._build_direction_projector(clean_grad_cache, fisher_grad_cache)
        )

        def closure():
            optimizer.zero_grad()
            outputs = self._network(inputs)
            task_loss, _, _ = self._task_loss(outputs["logits"], targets, class_offset)
            loss_value = task_loss.detach()
            task_loss.backward()
            return outputs, loss_value

        try:
            outputs, _ = optimizer.step(closure=closure)
        finally:
            optimizer.set_direction_projector(None)

        total_loss_value = clean_loss_value + fisher_loss_value
        return outputs["logits"].detach(), float(total_loss_value)


class EWCLoRA(Learner):
    """Backward-compatible alias."""
