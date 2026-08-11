"""IncLoRA-specific FS-LoRA learner.

This variant keeps IncLoRA's task-specific branch lifecycle:

- task t creates and trains a fresh LoRA branch;
- previous branches are loaded from disk and kept frozen by the IncLoRA backbone;
- the current branch is saved through IncLoRA's existing save_lora_parameters path.

The FS-LoRA update is attached only to the current branch:

    g_clean + lambda_flat * (g_gam - g_clean) + g_fisher

where the Fisher term penalizes the current branch's effective update ``Delta W = BA``
with the accumulated old-task Fisher in the same update space. This matches the
low-rank-cl EWCLoRA design, where the regularizer is applied to ``B_new @ A_new``.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict

import torch

from models_LoRAbasedCL.inclora import Learner as _IncLoRALearner
from models_LoRAbasedCL.ewclora_youyue_fitArchitechture import (
    Learner as _DeltaEWCMixin,
)
from models_LoRAbasedCL.ewclora_youyue_fitArchitechture_gam import (
    Learner as _FSGAMMixin,
)

logger = logging.getLogger(__name__)


class Learner(_IncLoRALearner):
    """FS-LoRA on top of IncLoRA's per-task LoRA branches."""

    # Reuse the DeltaW/Fisher mechanics from the SeqLoRA-based implementation.
    _is_delta_module = staticmethod(_DeltaEWCMixin._is_delta_module)
    _iter_delta_modules = _DeltaEWCMixin._iter_delta_modules
    _iter_delta_terms = _DeltaEWCMixin._iter_delta_terms
    _set_delta_hook_state = _DeltaEWCMixin._set_delta_hook_state
    _clear_delta_hook_grads = _DeltaEWCMixin._clear_delta_hook_grads
    _iter_hook_grads = _DeltaEWCMixin._iter_hook_grads
    _task_loss = _DeltaEWCMixin._task_loss
    _past_fisher_penalty = _DeltaEWCMixin._past_fisher_penalty
    _regularized_task_loss = _DeltaEWCMixin._regularized_task_loss
    _accumulate_fisher = _DeltaEWCMixin._accumulate_fisher
    _run_regularized_train = _DeltaEWCMixin._run_regularized_train

    # Reuse the decoupled GAM direction and mechanism diagnostics.
    _flatten_optimizer_params = staticmethod(_FSGAMMixin._flatten_optimizer_params)
    _compute_clean_task_grad_cache = _FSGAMMixin._compute_clean_task_grad_cache
    _compute_fisher_grad_cache = _FSGAMMixin._compute_fisher_grad_cache
    _cosine_from_sums = staticmethod(_FSGAMMixin._cosine_from_sums)
    _accumulate_mechanism_grad_stats = _FSGAMMixin._accumulate_mechanism_grad_stats
    _record_mechanism_loss_stats = _FSGAMMixin._record_mechanism_loss_stats
    _summarize_mechanism_loss_stats = _FSGAMMixin._summarize_mechanism_loss_stats
    _summarize_mechanism_grad_stats = _FSGAMMixin._summarize_mechanism_grad_stats
    _compute_delta_drift_metrics = _FSGAMMixin._compute_delta_drift_metrics
    _build_direction_projector = _FSGAMMixin._build_direction_projector
    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset: int):
        if self._optimizer_type != "gam":
            # Plain SGD path: combined loss backward (no perturbation).
            optimizer.zero_grad()
            outputs = self._network(inputs)
            task_loss, _, _ = self._task_loss(outputs["logits"], targets, class_offset)
            clean_loss_value = float(task_loss.detach().item())
            if self._fisher_past_delta:
                base_model = self._unwrap_network()
                fisher_penalty = self._past_fisher_penalty(base_model)
                fisher_loss_value = float(fisher_penalty.detach().item())
                total_loss = task_loss + fisher_penalty
            else:
                fisher_loss_value = 0.0
                total_loss = task_loss
            self._record_mechanism_loss_stats(clean_loss_value, fisher_loss_value)
            total_loss.backward()
            optimizer.step()
            return outputs["logits"].detach(), clean_loss_value + fisher_loss_value
        return _FSGAMMixin._step_batch(self, optimizer, scheduler, inputs, targets, class_offset)

    def __init__(self, args):
        super().__init__(args)
        if self._optimizer_type not in {"gam", "sgd"}:
            raise ValueError(
                "as1_normfisher_gam_inclora supports optimizer_type in {gam, sgd}, "
                f"got {self._optimizer_type}."
            )

        self._lambda_flat = float(args.get("lambda_flat", 1.0))
        self._lambda_ewc = float(args.get("ewc_lambda", args.get("lambda", 20.0)))
        self._fisher_eta = float(args.get("ewc_eta", args.get("eta", 0.0)))
        self._ewc_lambda = self._lambda_ewc
        self._ewc_eta = self._fisher_eta
        self._ewc_gamma = float(args.get("ewc_gamma", args.get("gamma", 1.0)))

        fisher_max = args.get("ewc_max_batches", 100)
        if fisher_max is None or str(fisher_max).lower() in {"none", "all", "full"}:
            self._ewc_max_batches = None
        else:
            fisher_max = int(fisher_max)
            self._ewc_max_batches = None if fisher_max <= 0 else fisher_max

        self._fisher_past_delta: Dict[str, torch.Tensor] = {}
        self._delta_reference: Dict[str, torch.Tensor] = {}

        self._normalize_fisher = bool(args.get("ewc_normalize_fisher", True))
        self._fisher_norm_eps = float(args.get("ewc_fisher_norm_eps", 1e-12))

        self._dual_ascent = bool(args.get("ewc_dual_ascent", False))
        self._dual_alpha = float(args.get("ewc_dual_alpha", 1.0))
        self._dual_delta = float(args.get("ewc_dual_delta", 1e-4))
        self._dual_lambda_max = float(args.get("ewc_dual_lambda_max", 1e6))
        self._dual_lambda_min = float(args.get("ewc_dual_lambda_min", 0.0))

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
        self._last_mechanism_metrics: Dict[str, Any] = {}

        self._as1_rho = float(args.get("as1_rho", getattr(self, "_gam_grad_rho", 0.2)))
        self._as1_norm_rho = float(
            args.get("as1_norm_rho", getattr(self, "_gam_grad_norm_rho", 0.2))
        )
        self._gam_grad_rho = self._as1_rho
        self._gam_grad_norm_rho = self._as1_norm_rho
        if hasattr(self, "_gam_args"):
            self._gam_args.grad_rho = self._as1_rho
            self._gam_args.grad_norm_rho = self._as1_norm_rho

        if self._is_main_process:
            logger.info(
                "[IncLoRA-FS] normalize_fisher=%s, gamma=%s, lambda_ewc=%s, "
                "lambda_flat=%s, dual_ascent=%s",
                self._normalize_fisher,
                self._ewc_gamma,
                self._lambda_ewc,
                self._lambda_flat,
                self._dual_ascent,
            )

    def _snapshot_task_reference(self) -> None:
        _DeltaEWCMixin._snapshot_task_reference(self)
        if self._mechanism_eval:
            self._reset_mechanism_stats()

    def _compute_delta_fisher(self, loader, model, class_offset: int) -> Dict[str, torch.Tensor]:
        raw_fisher = _DeltaEWCMixin._compute_delta_fisher(self, loader, model, class_offset)
        if not self._normalize_fisher or not raw_fisher:
            return raw_fisher

        total_trace = sum(f.sum().item() for f in raw_fisher.values())
        denom = float(total_trace) + self._fisher_norm_eps
        return {key: f / denom for key, f in raw_fisher.items()}

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        self._run_regularized_train(
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            epochs=int(self.args["init_epoch"]),
            class_offset=0,
        )

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        self._run_regularized_train(
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            epochs=int(self.args["epochs"]),
            class_offset=int(self._known_classes),
        )

    def after_task(self):
        # Compute drift before Fisher accumulation clears the current task reference.
        if self._mechanism_eval or self._dual_ascent:
            drift_before = self._compute_delta_drift_metrics()
        else:
            drift_before = {}

        base_model = self._unwrap_network()
        new_fisher = self._compute_delta_fisher(
            loader=self.train_loader,
            model=base_model,
            class_offset=int(self._known_classes),
        )
        self._fisher_past_delta = self._accumulate_fisher(self._fisher_past_delta, new_fisher)
        self._delta_reference = {}

        if self._is_main_process:
            self._log(
                "[IncLoRA-FS] Updated past Fisher for "
                f"{len(new_fisher)} delta tensors after task {self._cur_task}."
            )
        self._save_fisher_state(self._cur_task)

        # Preserve IncLoRA's task lifecycle bookkeeping.
        _IncLoRALearner.after_task(self)
        self._num_finished_tasks = int(self._cur_task + 1)

        if self._dual_ascent and self._cur_task > 0:
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
            self._ewc_lambda = new_lambda
            if self._is_main_process:
                self._log(
                    "[IncLoRA-FS] Dual ascent task "
                    f"{self._cur_task}: D^Fisher={d_fisher:.3e}, "
                    f"delta={self._dual_delta:.3e}, violation={violation:.3e}, "
                    f"lambda_ewc: {old_lambda:.2f} -> {new_lambda:.2f}"
                )

        if self._mechanism_eval:
            self._last_mechanism_metrics = {
                "task": int(self._cur_task),
                "lambda_flat": float(self._lambda_flat),
                "lambda_ewc": float(self._lambda_ewc),
                "fisher_eta": float(self._fisher_eta),
                "branch_mode": "inclora",
                **self._summarize_mechanism_loss_stats(),
                **self._summarize_mechanism_grad_stats(),
                **drift_before,
            }
            if self._is_main_process:
                self._log(f"[IncLoRA-FS][Mechanism] {self._last_mechanism_metrics}")

    def get_mechanism_metrics(self) -> Dict[str, Any]:
        return dict(self._last_mechanism_metrics)

    def _reset_mechanism_stats(self) -> None:
        self._mechanism_grad_sums = {}
        self._mechanism_grad_count = 0
        self._mechanism_loss_sums = {}
        self._mechanism_loss_count = 0
        self._last_mechanism_metrics = {}

    def load_task_checkpoint(self, checkpoint_dir: str, task_idx: int, data_manager) -> None:
        """Restore IncLoRA task files for resume/forked runs.

        IncLoRA reconstructs historical branches from files in ``args["filepath"]``.
        When resuming from another run, copy all task files up to ``task_idx`` into
        the active run's checkpoint directory, then restore the classifier head.
        """
        checkpoint_dir = os.path.abspath(os.path.expanduser(str(checkpoint_dir)))
        task_idx = int(task_idx)
        if task_idx < 0:
            raise ValueError(f"resume task_idx must be non-negative, got {task_idx}")

        dst_dir = os.path.abspath(os.path.expanduser(str(self.args.get("filepath", checkpoint_dir))))
        os.makedirs(dst_dir, exist_ok=True)
        if not dst_dir.endswith(os.sep):
            self.args["filepath"] = dst_dir + os.sep

        for t in range(task_idx + 1):
            for stem in ("lora_w_a", "lora_w_b", "lora_meta", "fc_state", "CLs_weight", "CLs_bias"):
                ext = ".json" if stem == "lora_meta" else ".pt"
                src = os.path.join(checkpoint_dir, f"{stem}_{t}{ext}")
                if stem in {"CLs_weight", "CLs_bias"}:
                    src = os.path.join(checkpoint_dir, f"{stem}{t}.pt")
                if not os.path.exists(src):
                    continue
                dst = os.path.join(dst_dir, os.path.basename(src))
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)
            fisher_src = os.path.join(checkpoint_dir, f"fs_lora_fisher_state_{t}.pt")
            if os.path.exists(fisher_src):
                fisher_dst = os.path.join(dst_dir, os.path.basename(fisher_src))
                if os.path.abspath(fisher_src) != os.path.abspath(fisher_dst):
                    shutil.copy2(fisher_src, fisher_dst)

        _, task_end = data_manager.get_task_class_range(task_idx)
        self._cur_task = task_idx
        self._known_classes = int(task_end)
        self._total_classes = int(task_end)
        self._refresh_distributed_context()

        network = self._unwrap_network()
        network.update_fc(self._total_classes)
        if hasattr(network, "load_fc"):
            network.load_fc(self.args["filepath"], task_idx)
        self._load_fisher_state(task_idx)
        self._network = network
        self._prepare_network()
        self._log(
            f"[IncLoRA-FS][Resume] Restored task {task_idx} files from {checkpoint_dir}; "
            f"next task will be {task_idx + 1}"
        )

    def _fisher_state_path(self, task_idx: int) -> str:
        save_dir = os.path.abspath(os.path.expanduser(str(self.args.get("filepath", "./"))))
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f"fs_lora_fisher_state_{int(task_idx)}.pt")

    def _save_fisher_state(self, task_idx: int) -> None:
        state = {
            "task": int(task_idx),
            "fisher_past_delta": {
                key: value.detach().cpu() for key, value in self._fisher_past_delta.items()
            },
            "lambda_ewc": float(self._lambda_ewc),
            "ewc_gamma": float(self._ewc_gamma),
            "fisher_eta": float(self._fisher_eta),
            "normalize_fisher": bool(self._normalize_fisher),
        }
        torch.save(state, self._fisher_state_path(task_idx))

    def _load_fisher_state(self, task_idx: int) -> None:
        path = self._fisher_state_path(task_idx)
        if not os.path.exists(path):
            if self._is_main_process:
                self._log(f"[IncLoRA-FS][Resume] No Fisher state found at {path}")
            return

        state = torch.load(path, map_location="cpu")
        self._fisher_past_delta = dict(state.get("fisher_past_delta", {}))
        if "lambda_ewc" in state:
            self._lambda_ewc = float(state["lambda_ewc"])
            self._ewc_lambda = self._lambda_ewc
        if self._is_main_process:
            self._log(
                "[IncLoRA-FS][Resume] Loaded Fisher state for task "
                f"{task_idx} with {len(self._fisher_past_delta)} tensors."
            )


class EWCLoRA(Learner):
    """Backward-compatible alias."""
