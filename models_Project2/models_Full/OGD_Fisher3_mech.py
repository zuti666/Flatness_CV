"""
OGD_Fisher3_mech.py
-------------------
OGD_Fisher3 + online CL mechanism evaluation hook.

This subclass mirrors ``OGD_Fisher3.incremental_train`` but inserts two hooks:
  1. Snapshot θ_{t-1} *after* head expansion so ΔL uses the real task-t model.
  2. Snapshot the pre-training OGD / Fisher histories actually used during task t.
  3. Call CLMechanismEvaluator.evaluate after training using those pre-training histories.

NO existing code is modified.  All new evaluation logic lives in
``evaluation_CL_mechanism/``.

Model name (used in YAML):
  model_name: ogd_fisher3_mech

Registered via monkey-patch in evaluation_CL_mechanism/run_mech_experiment.py.

New YAML keys:
  cl_mechanism_eval      : true          # enable/disable hook
  cl_eval_max_batches    : 20            # batches per loader per metric
  cl_eval_n_noise_samples: 100           # noise samples for MC / alignment
  cl_eval_power_iters    : 15            # power-iter steps for λ_max
  cl_eval_noise_topk     : 3             # top-k eigvecs for cos alignment
  cl_eval_backend        : emp_fisher    # curvature backend
  cl_eval_batch_size     : 64            # batch size for eval loaders
  cl_eval_num_workers    : 0
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models_Project2.models_Full.OGD_Fisher3 import Learner as _BaseOGDFisher3

logger = logging.getLogger(__name__)


def _seed_to_str(seed_value) -> str:
    if isinstance(seed_value, (list, tuple)) and seed_value:
        seed_value = seed_value[0]
    return str(seed_value if seed_value is not None else "0")


class Learner(_BaseOGDFisher3):
    """
    OGD_Fisher3 with online mechanistic evaluation at each task boundary.

    Inherits the training/update implementation from OGD_Fisher3 and only
    reorders the task-level orchestration to preserve pre-training histories for
    mechanism evaluation.
    """

    def incremental_train(self, data_manager) -> None:
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        self._align_directions_to_model(self._network)
        self._align_fisher_to_model(self._network)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        logging.info(
            "[OGD_FISHER3] noise_geometry=%s, rwp_std=%.4e, rwp_start_task=%d, fisher_rank=%d",
            self._noise_geometry,
            self._rwp_std,
            self._rwp_start_task,
            self._fisher_rank,
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args.get("train_num_workers", 8),
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args.get("train_num_workers", 8),
        )

        # ── 0. Snapshot θ_{t-1} and the exact history used during task t ─────
        ref_params: Optional[torch.Tensor] = None
        eval_ogd_dirs: List[torch.Tensor] = []
        eval_fisher_basis: List[torch.Tensor] = []
        eval_fisher_eigvals = torch.tensor([], dtype=torch.float32)
        if self.args.get("cl_mechanism_eval", False):
            try:
                from evaluation_CL_mechanism.cl_evaluator import CLMechanismEvaluator
                ref_params = CLMechanismEvaluator.snapshot_params(
                    self._unwrap_network(), exclude_prefix=None
                )
                eval_ogd_dirs, eval_fisher_basis, eval_fisher_eigvals = self._snapshot_pretrain_history()
            except Exception as exc:
                logger.warning("[OGD_Fisher3_mech] snapshot failed: %s", exc)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        model = self._unwrap_network()
        self._update_fisher_lowrank(self.train_loader, model)
        self._update_ogd_directions(self.train_loader)

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[OGD_FISHER3] Failed to compute class means: %s", exc)

        # ── 2. Online mechanism evaluation  ──────────────────────────────────
        if self.args.get("cl_mechanism_eval", False):
            self._run_mechanism_eval(
                data_manager,
                ref_params,
                eval_ogd_dirs,
                eval_fisher_basis,
                eval_fisher_eigvals,
            )

    # ── private helper ────────────────────────────────────────────────────────

    def _snapshot_pretrain_history(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        ogd_dirs = [d.detach().clone().cpu() for d in getattr(self, "_directions", [])]
        fisher_basis = [u.detach().clone().cpu() for u in getattr(self, "_fisher_basis", [])]
        fisher_eigvals = getattr(self, "_fisher_eigvals", torch.tensor([], dtype=torch.float32))
        fisher_eigvals = fisher_eigvals.detach().clone().cpu()
        return ogd_dirs, fisher_basis, fisher_eigvals

    def _run_mechanism_eval(
        self,
        data_manager,
        ref_params,
        eval_ogd_dirs: List[torch.Tensor],
        eval_fisher_basis: List[torch.Tensor],
        eval_fisher_eigvals: torch.Tensor,
    ) -> None:
        """
        Build evaluator, collect class_ranges, call evaluate().
        Written as a separate method so any exception here does NOT
        crash the training run — just logs a warning.
        """
        try:
            from evaluation_CL_mechanism.cl_evaluator import CLMechanismEvaluator

            # Reuse trainer-managed filepath when available. Otherwise fall back
            # to a run-specific directory to avoid cross-run overwrites.
            base_path = self.args.get("filepath", None)
            if base_path:
                save_root = os.path.join(base_path, "cl_mechanism")
            else:
                save_root = os.path.join(
                    "outputs_logs",
                    "cl_mechanism",
                    str(self.args.get("model_name", "model")),
                    str(self.args.get("prefix", "run")),
                    f"seed_{_seed_to_str(self.args.get('seed', 0))}",
                )

            evaluator = CLMechanismEvaluator(self.args, save_root)

            evaluator.evaluate(
                network        = self._unwrap_network(),
                data_manager   = data_manager,
                cur_task       = self._cur_task,
                fisher_basis   = eval_fisher_basis,
                fisher_eigvals = eval_fisher_eigvals,
                rwp_std        = self._rwp_std,
                noise_geometry = self._noise_geometry,
                device         = self._device,
                ref_params     = ref_params,
                ogd_directions = eval_ogd_dirs,
                known_classes  = self._known_classes,
                total_classes  = self._total_classes,
            )

        except Exception as exc:
            logger.warning(
                "[OGD_Fisher3_mech] mechanism eval failed at task %d: %s",
                self._cur_task, exc,
            )
