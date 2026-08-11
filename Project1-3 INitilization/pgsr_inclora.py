"""PGSR-LoRA implemented as a minimal extension of the repository's IncLoRA.

This learner keeps IncLoRA's additive historical branches. Before training a
new branch it evaluates task-level historical right-singular subspaces using a
small, class-balanced preview of the incoming task. The selected basis is used
as ``A`` while ``B`` remains exactly zero, so the current predictor is kept
unchanged. During a short warm-up, gradients of ``A`` are suppressed.

The first executable version intentionally supports the ViT q/v LoRA path and
ordinary SGD/Adam optimizers. That makes the preview gradient and the local
descent diagnostic match the method's derivation without silently changing the
meaning of a convolutional LoRA subspace or a multi-step sharpness optimizer.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from models_LoRAbasedCL.inclora import Learner as IncLoRALearner
from pgsr.bank import SubspaceBank
from pgsr.initialization import initialize_lora_factors, maximum_function_deviation
from pgsr.selector import (
    candidate_energy,
    make_fresh_candidate,
    make_orthogonal_complement_candidate,
    select_candidate,
)
from utils.toolkit import tensor2numpy


class Learner(IncLoRALearner):
    """IncLoRA with preview-guided, function-preserving subspace reuse."""

    def __init__(self, args):
        super().__init__(args)
        self._pgsr_preview_samples = int(args.get("pgsr_preview_samples", 32))
        self._pgsr_preview_batch_size = int(
            args.get("pgsr_preview_batch_size", self._pgsr_preview_samples)
        )
        self._pgsr_fresh_prior = float(args.get("pgsr_fresh_prior", 0.5))
        self._pgsr_gamma = float(args.get("pgsr_gamma", 12.0))
        self._pgsr_warmup_steps = int(args.get("pgsr_warmup_steps", 10))
        self._pgsr_bank_max_tasks = int(args.get("pgsr_bank_max_tasks", 0))
        self._pgsr_function_tolerance = float(args.get("pgsr_function_tolerance", 1e-6))
        self._pgsr_max_train_steps = int(args.get("pgsr_max_train_steps", 0))
        self._pgsr_selection_mode = str(args.get("pgsr_selection_mode", "map")).lower()
        self._pgsr_forced_history_task = int(args.get("pgsr_forced_history_task", -1))
        self._pgsr_heldout_samples = int(args.get("pgsr_heldout_samples", 250))
        self._pgsr_heldout_batch_size = int(args.get("pgsr_heldout_batch_size", 125))
        self._pgsr_function_check_samples = int(
            args.get("pgsr_function_check_samples", 300)
        )
        self._pgsr_probe_lr = float(args.get("pgsr_probe_lr", args.get("lrate", 0.01)))
        self._pgsr_random_null_candidates = int(
            args.get("pgsr_random_null_candidates", 32)
        )
        diagnostic_steps = args.get("pgsr_diagnostic_steps", [0, 1, 2, 5, 10, 20, 40, 60])
        if isinstance(diagnostic_steps, str):
            diagnostic_steps = [
                int(value.strip()) for value in diagnostic_steps.split(",") if value.strip()
            ]
        self._pgsr_diagnostic_steps = sorted(
            {max(0, int(value)) for value in diagnostic_steps}
        )
        self._pgsr_preview_loader = None
        self._pgsr_heldout_loader = None
        self._pgsr_function_loader = None
        self._pgsr_preview_batches = []
        self._pgsr_heldout_batches = []
        self._pgsr_factor_snapshots = []
        self._pgsr_split_manifest: dict = {}
        self._pgsr_diagnostics: dict = {}

        supported = {"sgd", "adam", "adamw"}
        if self._optimizer_type not in supported:
            raise ValueError(
                "The first PGSR implementation supports optimizer_type in "
                f"{sorted(supported)}, got {self._optimizer_type!r}."
            )
        selection_modes = {
            "map",
            "raw_history",
            "fresh",
            "latest",
            "random_history",
            "history_task",
            "perpendicular",
        }
        if self._pgsr_selection_mode not in selection_modes:
            raise ValueError(
                f"pgsr_selection_mode must be in {sorted(selection_modes)}, "
                f"got {self._pgsr_selection_mode!r}"
            )

    # ------------------------------------------------------------------
    # Data and task lifecycle
    # ------------------------------------------------------------------
    def incremental_train(self, data_manager):
        self._refresh_distributed_context()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._log(f"Learning on {self._known_classes}-{self._total_classes}")

        class_ids = np.arange(self._known_classes, self._total_classes)
        full_train_dataset = data_manager.get_dataset(
            class_ids, source="train", mode="train"
        )
        diagnostic_train_dataset = data_manager.get_dataset(
            class_ids, source="train", mode="test"
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )

        raw_order = list(self.args.get("class_order", range(self._total_classes)))
        raw_classes = [
            int(value) for value in raw_order[self._known_classes : self._total_classes]
        ]
        split_seed = int(self.args["seed"]) + 104729 * (self._cur_task + 1)
        if self._cur_task > 0:
            labels = np.asarray(getattr(diagnostic_train_dataset, "labels", []))
            preview_indices, heldout_indices = self._balanced_partition_indices(
                labels,
                self._pgsr_preview_samples,
                self._pgsr_heldout_samples,
                seed=split_seed,
            )
            excluded = set(preview_indices) | set(heldout_indices)
            train_indices = [
                index for index in range(len(full_train_dataset)) if index not in excluded
            ]
            train_dataset = Subset(full_train_dataset, train_indices)
            self._pgsr_preview_loader = self._subset_loader(
                diagnostic_train_dataset,
                preview_indices,
                batch_size=self._pgsr_preview_batch_size,
            )
            self._pgsr_heldout_loader = self._subset_loader(
                diagnostic_train_dataset,
                heldout_indices,
                batch_size=self._pgsr_heldout_batch_size,
            )
        else:
            preview_indices, heldout_indices = [], []
            train_indices = list(range(len(full_train_dataset)))
            train_dataset = full_train_dataset
            self._pgsr_preview_loader = None
            self._pgsr_heldout_loader = None

        function_labels = np.asarray(getattr(test_dataset, "labels", []))
        function_indices, _ = self._balanced_partition_indices(
            function_labels,
            min(self._pgsr_function_check_samples, len(test_dataset)),
            0,
            seed=split_seed + 65537,
        )
        self._pgsr_function_loader = self._subset_loader(
            test_dataset,
            function_indices,
            batch_size=self.args["batch_size"],
        )
        self._pgsr_split_manifest = {
            "task_id": self._cur_task,
            "remapped_classes": [int(value) for value in class_ids.tolist()],
            "raw_cifar100_classes": raw_classes,
            "source": "cifar100_train",
            "train_indices": train_indices,
            "preview_indices": preview_indices,
            "heldout_indices": heldout_indices,
            "pairwise_index_intersections": {
                "train_preview": len(set(train_indices) & set(preview_indices)),
                "train_heldout": len(set(train_indices) & set(heldout_indices)),
                "preview_heldout": len(set(preview_indices) & set(heldout_indices)),
            },
        }

        train_generator = torch.Generator(device="cpu")
        train_generator.manual_seed(split_seed + 31337)
        self.train_loader = data_manager.build_dataloader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args.get("train_num_workers", 8),
            generator=train_generator,
        )
        self.test_loader = data_manager.build_dataloader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args.get("eval_num_workers", 8),
        )

        self._train(self.train_loader, self.test_loader)
        self._network = self._unwrap_network()

        if not bool(self.args.get("pgsr_skip_nme", False)):
            try:
                self.compute_all_seen_class_means(data_manager)
            except Exception as exc:  # pylint: disable=broad-except
                self._log(f"[PGSR][NME] Failed to compute class means: {exc}")

    def load_task_checkpoint(self, checkpoint_dir: str, task_idx: int, data_manager) -> None:
        """Restore an additive IncLoRA history into the active fork directory.

        The parent implementation restores a single trainable SeqLoRA adapter.
        PGSR instead needs every historical A/B pair on disk because both the
        IncLoRA backbone constructor and the subspace bank rebuild from them.
        """

        source_dir = Path(checkpoint_dir).expanduser().resolve()
        task_idx = int(task_idx)
        if task_idx < 0:
            raise ValueError(f"resume task_idx must be non-negative, got {task_idx}")
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {source_dir}")

        target_dir = Path(str(self.args.get("filepath", source_dir))).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        # The repository's ViT LoRA loader concatenates filenames onto this
        # string, so retain an explicit separator.
        self.args["filepath"] = str(target_dir) + os.sep

        def copy_file(source: Path, *, required: bool = False) -> None:
            if not source.exists():
                if required:
                    raise FileNotFoundError(f"Required checkpoint file is missing: {source}")
                return
            destination = target_dir / source.name
            if source != destination:
                shutil.copy2(source, destination)

        for restored_task in range(task_idx + 1):
            copy_file(source_dir / f"lora_w_a_{restored_task}.pt", required=True)
            copy_file(source_dir / f"lora_w_b_{restored_task}.pt", required=True)
            for filename in (
                f"lora_meta_{restored_task}.json",
                f"fc_state_{restored_task}.pt",
                f"CLs_weight{restored_task}.pt",
                f"CLs_bias{restored_task}.pt",
                f"pgsr_subspace_task_{restored_task}.pt",
            ):
                copy_file(source_dir / filename)

            diagnostics_dir = source_dir / "pgsr"
            if diagnostics_dir.is_dir():
                target_diagnostics = target_dir / "pgsr"
                target_diagnostics.mkdir(parents=True, exist_ok=True)
                for diagnostic in diagnostics_dir.glob(f"task_{restored_task:02d}_*.json"):
                    destination = target_diagnostics / diagnostic.name
                    if diagnostic.resolve() != destination.resolve():
                        shutil.copy2(diagnostic, destination)

        classifier_candidates = (
            target_dir / f"fc_state_{task_idx}.pt",
            target_dir / f"CLs_weight{task_idx}.pt",
        )
        if not any(path.exists() for path in classifier_candidates):
            raise FileNotFoundError(
                f"Task {task_idx} classifier checkpoint is missing in {source_dir}"
            )

        _, task_end = data_manager.get_task_class_range(task_idx)
        self._cur_task = task_idx
        self._known_classes = int(task_end)
        self._total_classes = int(task_end)
        self._refresh_distributed_context()

        network = self._unwrap_network()
        network.update_fc(self._total_classes)
        if hasattr(network, "load_fc"):
            network.load_fc(self.args["filepath"], task_idx)
        self._network = network
        self._prepare_network()
        self._log(
            f"[PGSR][Resume] restored additive tasks 0..{task_idx} from {source_dir} "
            f"into {target_dir}; next task is {task_idx + 1}"
        )

    @staticmethod
    def _balanced_partition_indices(
        labels: np.ndarray,
        first_count: int,
        second_count: int,
        *,
        seed: int,
    ) -> tuple[list[int], list[int]]:
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError("PGSR requires a one-dimensional label array")
        total_requested = max(0, int(first_count)) + max(0, int(second_count))
        if total_requested > len(labels):
            raise ValueError(
                f"Requested {total_requested} diagnostic samples from {len(labels)} examples"
            )

        rng = np.random.default_rng(int(seed))
        class_pools: dict[int, list[int]] = {}
        for class_id in sorted(np.unique(labels).tolist()):
            indices = np.flatnonzero(labels == class_id)
            class_pools[int(class_id)] = rng.permutation(indices).astype(int).tolist()

        def take(count: int) -> list[int]:
            selected: list[int] = []
            while len(selected) < int(count):
                made_progress = False
                for class_id in sorted(class_pools):
                    if class_pools[class_id] and len(selected) < int(count):
                        selected.append(class_pools[class_id].pop())
                        made_progress = True
                if not made_progress:
                    raise RuntimeError("Could not construct a class-balanced diagnostic split")
            return selected

        return take(max(0, int(first_count))), take(max(0, int(second_count)))

    @staticmethod
    def _subset_loader(dataset, indices: list[int], *, batch_size: int) -> DataLoader:
        if not indices:
            return None
        return DataLoader(
            Subset(dataset, indices),
            batch_size=min(max(1, int(batch_size)), len(indices)),
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    # ------------------------------------------------------------------
    # PGSR preparation
    # ------------------------------------------------------------------
    def _materialize_preview(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._materialize_loader(self._pgsr_preview_loader, "preview")

    def _materialize_heldout(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._materialize_loader(self._pgsr_heldout_loader, "held-out")

    def _materialize_function_check(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._materialize_loader(self._pgsr_function_loader, "function-check")

    def _materialize_loader(self, loader, name: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if loader is None:
            raise RuntimeError(f"{name} loader was not constructed")
        batches = []
        for _, inputs, targets in loader:
            batches.append((inputs.to(self._device), targets.to(self._device)))
        if not batches:
            raise RuntimeError(f"{name} loader is empty")
        return batches

    def _calibrate_new_classifier(
        self, network, preview_batches: Iterable[tuple[torch.Tensor, torch.Tensor]]
    ) -> dict:
        """Feature-imprint the new class rows before taking backbone gradients."""

        was_training = network.training
        network.eval()
        feature_sums: dict[int, torch.Tensor] = {}
        counts: dict[int, int] = {}
        with torch.no_grad():
            for inputs, targets in preview_batches:
                features = network.backbone(inputs)
                if isinstance(features, dict):
                    features = features["features"]
                for class_id in targets.unique(sorted=True).tolist():
                    mask = targets == int(class_id)
                    feature_sums[int(class_id)] = feature_sums.get(
                        int(class_id), torch.zeros_like(features[0])
                    ) + features[mask].sum(dim=0)
                    counts[int(class_id)] = counts.get(int(class_id), 0) + int(mask.sum().item())

            fc = network.fc
            if self._known_classes > 0:
                reference_norm = fc.weight[: self._known_classes].norm(dim=1).mean()
            else:
                reference_norm = fc.weight.norm(dim=1).mean()
            reference_norm = reference_norm.clamp_min(1e-6)
            for class_id, feature_sum in feature_sums.items():
                prototype = feature_sum / max(counts[class_id], 1)
                fc.weight[class_id].copy_(F.normalize(prototype, dim=0) * reference_norm)
                if fc.bias is not None:
                    fc.bias[class_id].zero_()

        network.train(was_training)
        return {
            "mode": "feature_imprinting",
            "classes": sorted(feature_sums),
            "samples_per_class": {str(key): value for key, value in sorted(counts.items())},
            "reference_weight_norm": float(reference_norm.item()),
        }

    @staticmethod
    def _vit_qv_wrappers(backbone) -> list[object]:
        vit = getattr(backbone, "lora_vit", None)
        blocks = getattr(vit, "blocks", None)
        if blocks is None:
            raise NotImplementedError(
                "PGSR preview gradients currently support the ViT q/v IncLoRA backbone only"
            )
        wrappers = []
        for block in blocks:
            wrapped = block.attn.qkv
            required = ("qkv", "linear_a_q", "linear_b_q", "linear_a_v", "linear_b_v")
            if all(hasattr(wrapped, name) for name in required):
                wrappers.append(wrapped)
        if not wrappers:
            raise RuntimeError("No train-time ViT q/v LoRA wrappers were found")
        return wrappers

    def _preview_gradients(self, network, preview_batches) -> list[torch.Tensor]:
        wrappers = self._vit_qv_wrappers(network.backbone)
        original_requires_grad = []
        for wrapped in wrappers:
            weight = wrapped.qkv.weight
            original_requires_grad.append(bool(weight.requires_grad))
            weight.requires_grad_(True)

        was_training = network.training
        network.eval()  # deterministic dropout/drop-path behavior during selection
        network.zero_grad(set_to_none=True)
        total_examples = sum(int(targets.numel()) for _, targets in preview_batches)
        for inputs, targets in preview_batches:
            local_targets = targets - self._known_classes
            logits = network(inputs)["logits"][:, self._known_classes : self._total_classes]
            loss = F.cross_entropy(logits, local_targets, reduction="sum") / max(total_examples, 1)
            loss.backward()

        gradients: list[torch.Tensor] = []
        for wrapped in wrappers:
            full_gradient = wrapped.qkv.weight.grad
            if full_gradient is None:
                raise RuntimeError("Full qkv preview gradient was not populated")
            dim = int(wrapped.dim)
            gradients.extend(
                [
                    full_gradient[:dim].detach().float().cpu().clone(),
                    full_gradient[-dim:].detach().float().cpu().clone(),
                ]
            )

        network.zero_grad(set_to_none=True)
        for wrapped, requires_grad in zip(wrappers, original_requires_grad):
            wrapped.qkv.weight.requires_grad_(requires_grad)
        network.train(was_training)
        return gradients

    def _evaluate_task_batches(self, network, batches) -> dict:
        was_training = network.training
        network.eval()
        loss_sum = 0.0
        task_correct = 0
        class_il_correct = 0
        sample_count = 0
        with torch.no_grad():
            for inputs, targets in batches:
                logits = network(inputs)["logits"]
                local_targets = targets - self._known_classes
                local_logits = logits[:, self._known_classes : self._total_classes]
                loss_sum += float(
                    F.cross_entropy(local_logits, local_targets, reduction="sum").item()
                )
                task_correct += int(local_logits.argmax(dim=1).eq(local_targets).sum().item())
                class_il_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
                sample_count += int(targets.numel())
        network.train(was_training)
        return {
            "samples": sample_count,
            "task_local_loss": loss_sum / max(sample_count, 1),
            "task_aware_accuracy": 100.0 * task_correct / max(sample_count, 1),
            "class_il_accuracy": 100.0 * class_il_correct / max(sample_count, 1),
        }

    @staticmethod
    def _collect_logits(network, batches) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = network.training
        network.eval()
        logits_rows = []
        target_rows = []
        with torch.no_grad():
            for inputs, targets in batches:
                logits_rows.append(network(inputs)["logits"].detach().float().cpu())
                target_rows.append(targets.detach().cpu())
        network.train(was_training)
        return torch.cat(logits_rows, dim=0), torch.cat(target_rows, dim=0)

    def _one_step_probe(self, network, preview_batches, heldout_batches) -> dict:
        """Apply and revert one deterministic B-only SGD step on preview data."""

        before_preview = self._evaluate_task_batches(network, preview_batches)
        before_heldout = self._evaluate_task_batches(network, heldout_batches)
        b_weights = [module.weight for module in network.backbone.w_Bs]
        original_b = [weight.detach().clone() for weight in b_weights]

        was_training = network.training
        network.eval()
        network.zero_grad(set_to_none=True)
        sample_count = sum(int(targets.numel()) for _, targets in preview_batches)
        probe_loss = torch.zeros((), device=self._device)
        for inputs, targets in preview_batches:
            local_targets = targets - self._known_classes
            local_logits = network(inputs)["logits"][
                :, self._known_classes : self._total_classes
            ]
            probe_loss = probe_loss + F.cross_entropy(
                local_logits, local_targets, reduction="sum"
            ) / max(sample_count, 1)
        gradients = torch.autograd.grad(probe_loss, b_weights, create_graph=False)

        with torch.no_grad():
            for weight, gradient in zip(b_weights, gradients):
                weight.add_(gradient, alpha=-self._pgsr_probe_lr)

        after_preview = self._evaluate_task_batches(network, preview_batches)
        after_heldout = self._evaluate_task_batches(network, heldout_batches)
        effective_update_norm = math.sqrt(
            sum(
                float(
                    (
                        module_b.weight.detach().float()
                        @ module_a.weight.detach().float()
                    )
                    .square()
                    .sum()
                    .item()
                )
                for module_a, module_b in zip(
                    network.backbone.w_As, network.backbone.w_Bs
                )
            )
        )
        b_gradient_norm = math.sqrt(
            sum(float(gradient.detach().float().square().sum().item()) for gradient in gradients)
        )

        with torch.no_grad():
            for weight, original in zip(b_weights, original_b):
                weight.copy_(original)
        restore_max_abs = max(
            float((weight.detach() - original).abs().max().item())
            for weight, original in zip(b_weights, original_b)
        )
        network.zero_grad(set_to_none=True)
        network.train(was_training)

        return {
            "optimizer": "b_only_sgd",
            "learning_rate": self._pgsr_probe_lr,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "preview_before": before_preview,
            "preview_after": after_preview,
            "preview_loss_decrease": (
                before_preview["task_local_loss"] - after_preview["task_local_loss"]
            ),
            "heldout_before": before_heldout,
            "heldout_after": after_heldout,
            "heldout_loss_decrease": (
                before_heldout["task_local_loss"] - after_heldout["task_local_loss"]
            ),
            "b_gradient_norm": b_gradient_norm,
            "effective_ba_update_norm": effective_update_norm,
            "restore_max_abs": restore_max_abs,
        }

    @staticmethod
    def _factor_snapshot(backbone, step: int) -> dict:
        return {
            "step": int(step),
            "a": [module.weight.detach().cpu().half().clone() for module in backbone.w_As],
            "b": [module.weight.detach().cpu().half().clone() for module in backbone.w_Bs],
        }

    def _save_factor_trajectory(self) -> None:
        if not self._is_main_process:
            return
        output_dir = Path(str(self.args.get("filepath", "./"))) / "pgsr"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"task_{self._cur_task:02d}_factor_trajectory.pt"
        torch.save(
            {
                "task_id": self._cur_task,
                "selection": self._pgsr_diagnostics.get("applied_selection", {}),
                "snapshots": self._pgsr_factor_snapshots,
            },
            path,
        )
        logging.info("[PGSR] wrote %s", path)

    def _prepare_pgsr(self, network) -> None:
        preview_batches = self._materialize_preview()
        heldout_batches = self._materialize_heldout()
        function_batches = self._materialize_function_check()
        self._pgsr_preview_batches = preview_batches
        self._pgsr_heldout_batches = heldout_batches
        head_diagnostics = self._calibrate_new_classifier(network, preview_batches)
        gradients = self._preview_gradients(network, preview_batches)

        checkpoint_dir = Path(str(self.args.get("filepath", "./")))
        max_tasks = self._pgsr_bank_max_tasks or None
        bank = SubspaceBank.from_checkpoint_dir(
            checkpoint_dir,
            upto_task=self._cur_task,
            max_tasks=max_tasks,
        )
        if not bank.tasks:
            raise RuntimeError(
                f"Task {self._cur_task} has no historical LoRA checkpoint in {checkpoint_dir}"
            )
        expected_sites = len(network.backbone.w_As)
        if any(len(task.bases) != expected_sites for task in bank.tasks):
            raise ValueError("Historical bank and current backbone have different LoRA site counts")

        fresh_seed = int(self.args["seed"]) + 1009 * (self._cur_task + 1)
        fresh = make_fresh_candidate(gradients, int(self.args.get("lora_rank", 10)), seed=fresh_seed)
        history = bank.candidates
        random_null = []
        for null_index in range(max(0, self._pgsr_random_null_candidates)):
            null_seed = fresh_seed + 7919 * (null_index + 1)
            null_candidate = make_fresh_candidate(
                gradients,
                int(self.args.get("lora_rank", 10)),
                seed=null_seed,
            )
            null_raw, null_normalized = candidate_energy(
                gradients, null_candidate, 1e-12
            )
            random_null.append(
                {
                    "index": null_index,
                    "seed": null_seed,
                    "energy_raw": null_raw,
                    "energy_normalized": null_normalized,
                }
            )
        perpendicular_seed = fresh_seed + 424243
        perpendicular = make_orthogonal_complement_candidate(
            history,
            gradients,
            int(self.args.get("lora_rank", 10)),
            seed=perpendicular_seed,
        )
        map_selection = select_candidate(
            gradients,
            history,
            fresh,
            historical_ids=[f"history_task_{task_id}" for task_id in bank.task_ids],
            fresh_prior=self._pgsr_fresh_prior,
            gamma=self._pgsr_gamma,
        )
        all_candidates = [fresh, *history]
        candidate_ids = ["fresh", *[f"history_task_{task_id}" for task_id in bank.task_ids]]
        if self._pgsr_selection_mode == "fresh":
            selected_index = 0
            selected_id = candidate_ids[selected_index]
            selected_bases = all_candidates[selected_index]
        elif self._pgsr_selection_mode == "latest":
            selected_index = len(all_candidates) - 1
            selected_id = candidate_ids[selected_index]
            selected_bases = all_candidates[selected_index]
        elif self._pgsr_selection_mode == "random_history":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(fresh_seed + 17)
            selected_index = 1 + int(torch.randint(len(history), (1,), generator=generator).item())
            selected_id = candidate_ids[selected_index]
            selected_bases = all_candidates[selected_index]
        elif self._pgsr_selection_mode == "history_task":
            if self._pgsr_forced_history_task not in bank.task_ids:
                raise ValueError(
                    f"Requested history task {self._pgsr_forced_history_task}, "
                    f"available tasks are {bank.task_ids}"
                )
            selected_index = 1 + bank.task_ids.index(self._pgsr_forced_history_task)
            selected_id = candidate_ids[selected_index]
            selected_bases = all_candidates[selected_index]
        elif self._pgsr_selection_mode == "raw_history":
            history_rows = list(map_selection.rows[1:])
            history_offset = int(
                np.argmax([row.energy_normalized for row in history_rows])
            )
            selected_index = 1 + history_offset
            selected_id = candidate_ids[selected_index]
            selected_bases = all_candidates[selected_index]
        elif self._pgsr_selection_mode == "perpendicular":
            selected_index = -1
            selected_id = "perpendicular_to_history"
            selected_bases = perpendicular
        else:
            selected_index = map_selection.selected_index
            selected_id = candidate_ids[selected_index]
            selected_bases = all_candidates[selected_index]

        applied_energy_raw, applied_energy_normalized = candidate_energy(
            gradients, selected_bases, 1e-12
        )
        max_history_overlap = max(
            float(
                torch.linalg.matrix_norm(
                    selected.detach().float().t() @ historical.detach().float()
                ).item()
            )
            for selected, site_histories in zip(
                selected_bases, zip(*history)
            )
            for historical in site_histories
        )
        max_basis_orthogonality_error = max(
            float(
                torch.linalg.matrix_norm(
                    basis.detach().float().t() @ basis.detach().float()
                    - torch.eye(basis.shape[1])
                ).item()
            )
            for basis in selected_bases
        )

        logits_before, function_targets = self._collect_logits(network, function_batches)
        initialize_lora_factors(
            network.backbone.w_As,
            network.backbone.w_Bs,
            selected_bases,
        )
        logits_after, function_targets_after = self._collect_logits(network, function_batches)
        if not torch.equal(function_targets, function_targets_after):
            raise RuntimeError("Function-check sample order changed during initialization")
        function_deviation = maximum_function_deviation(logits_before, logits_after)
        if function_deviation > self._pgsr_function_tolerance:
            raise RuntimeError(
                "PGSR initialization changed the predictor: "
                f"D_func={function_deviation:.3e} > {self._pgsr_function_tolerance:.3e}"
            )

        self._pgsr_initial_a = [module.weight.detach().clone() for module in network.backbone.w_As]
        self._pgsr_initial_b = [module.weight.detach().clone() for module in network.backbone.w_Bs]
        one_step_probe = self._one_step_probe(network, preview_batches, heldout_batches)
        self._pgsr_factor_snapshots = [self._factor_snapshot(network.backbone, 0)]
        initial_a_norm = math.sqrt(
            sum(
                float(module.weight.detach().float().square().sum().item())
                for module in network.backbone.w_As
            )
        )
        self._pgsr_diagnostics = {
            "task_id": self._cur_task,
            "preview_samples": sum(int(targets.numel()) for _, targets in preview_batches),
            "preview_batches": len(preview_batches),
            "heldout_samples": sum(int(targets.numel()) for _, targets in heldout_batches),
            "data_split": self._pgsr_split_manifest,
            "head_calibration": head_diagnostics,
            "fresh_seed": fresh_seed,
            "perpendicular_seed": perpendicular_seed,
            "fresh_prior": self._pgsr_fresh_prior,
            "gamma": self._pgsr_gamma,
            "warmup_steps": self._pgsr_warmup_steps,
            "function_deviation": function_deviation,
            "function_check_samples": int(function_targets.numel()),
            "gradient_norms": [float(value.norm().item()) for value in gradients],
            "selection": map_selection.to_dict(),
            "random_null": random_null,
            "applied_selection": {
                "mode": self._pgsr_selection_mode,
                "selected_index": selected_index,
                "selected_id": selected_id,
                "selected_is_fresh": selected_index == 0,
                "energy_raw": applied_energy_raw,
                "energy_normalized": applied_energy_normalized,
                "max_overlap_with_history": max_history_overlap,
                "max_basis_orthogonality_error": max_basis_orthogonality_error,
                "initial_a_frobenius_norm": initial_a_norm,
            },
            "one_step_probe": one_step_probe,
        }
        if self._is_main_process:
            output_dir = Path(str(self.args.get("filepath", "./"))) / "pgsr"
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "task_id": self._cur_task,
                    "selection": self._pgsr_diagnostics["applied_selection"],
                    "targets": function_targets,
                    "logits": logits_after,
                    "logits_before_candidate_initialization": logits_before,
                },
                output_dir / f"task_{self._cur_task:02d}_initial_outputs.pt",
            )
        self._write_task_json("selection", self._pgsr_diagnostics)
        self._log(
            f"[PGSR] task={self._cur_task} mode={self._pgsr_selection_mode} "
            f"selected={selected_id} map={map_selection.selected_id} "
            f"D_func={function_deviation:.3e} entropy={map_selection.posterior_entropy:.4f}"
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train(self, train_loader, test_loader):
        network = self._unwrap_network()
        network.backbone = self._build_incremental_lora()
        network.backbone.to(self._device)
        self._network = network
        self._prepare_network()
        network = self._unwrap_network()

        if self._cur_task > 0:
            self._prepare_pgsr(network)

        params = [parameter for parameter in self._network.parameters() if parameter.requires_grad]
        stage = "init" if self._cur_task == 0 else "update"
        optimizer = self._build_optimizer(params, stage=stage)
        lr = optimizer.param_groups[0]["lr"]
        epoch_count = int(self.args["init_epoch"] if self._cur_task == 0 else self.args["epochs"])
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("milestones", []),
            gamma=float(self.args.get("lrate_decay", 1.0)),
            T_max=epoch_count,
            eta_min=self.args.get("min_lr", 0.1 * lr),
        )

        if self._cur_task == 0:
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._update_representation_pgsr(train_loader, test_loader, optimizer, scheduler)

        save_dir = self.args.get("filepath", "./")
        owner = self._unwrap_network()
        owner.backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(owner, "save_fc"):
            owner.save_fc(save_dir, self._cur_task)
        self._save_current_subspace(owner.backbone)

    def _update_representation_pgsr(self, train_loader, test_loader, optimizer, scheduler):
        progress = tqdm(range(int(self.args["epochs"])), disable=not self._is_main_process)
        global_step = 0
        trajectory = []
        evaluation_curve = [
            {
                "step": 0,
                "preview": self._pgsr_diagnostics["one_step_probe"]["preview_before"],
                "heldout": self._pgsr_diagnostics["one_step_probe"]["heldout_before"],
            }
        ]
        warmup_boundary = None
        stop = False
        a_modules = list(self._unwrap_network().backbone.w_As)

        for epoch in progress:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            epoch_steps = 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                fake_targets = targets - self._known_classes
                optimizer.zero_grad(set_to_none=True)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(
                    logits[:, self._known_classes : self._total_classes], fake_targets
                )
                loss.backward()

                warmup_active = global_step < self._pgsr_warmup_steps
                if warmup_active:
                    for module in a_modules:
                        module.weight.grad = None
                optimizer.step()
                step_number = global_step + 1

                if self._pgsr_warmup_steps > 0 and step_number == self._pgsr_warmup_steps:
                    owner = self._unwrap_network()
                    warmup_boundary = {
                        "step": step_number,
                        "a_parameter_displacement": math.sqrt(
                            sum(
                                float(
                                    (module.weight.detach() - initial)
                                    .float()
                                    .square()
                                    .sum()
                                    .item()
                                )
                                for module, initial in zip(
                                    owner.backbone.w_As, self._pgsr_initial_a
                                )
                            )
                        ),
                        "b_parameter_displacement": math.sqrt(
                            sum(
                                float(
                                    (module.weight.detach() - initial)
                                    .float()
                                    .square()
                                    .sum()
                                    .item()
                                )
                                for module, initial in zip(
                                    owner.backbone.w_Bs, self._pgsr_initial_b
                                )
                            )
                        ),
                    }

                losses += float(loss.item())
                epoch_steps += 1
                _, predictions = torch.max(logits, dim=1)
                correct += predictions.eq(targets.expand_as(predictions)).cpu().sum()
                total += len(targets)
                if step_number in self._pgsr_diagnostic_steps:
                    owner = self._unwrap_network()
                    preview_metrics = self._evaluate_task_batches(
                        owner, self._pgsr_preview_batches
                    )
                    heldout_metrics = self._evaluate_task_batches(
                        owner, self._pgsr_heldout_batches
                    )
                    trajectory.append(
                        {
                            "step": step_number,
                            "epoch": epoch,
                            "train_loss_before_step": float(loss.item()),
                            "warmup_active": warmup_active,
                        }
                    )
                    evaluation_curve.append(
                        {
                            "step": step_number,
                            "preview": preview_metrics,
                            "heldout": heldout_metrics,
                        }
                    )
                    self._pgsr_factor_snapshots.append(
                        self._factor_snapshot(owner.backbone, step_number)
                    )
                global_step += 1
                if self._pgsr_max_train_steps > 0 and global_step >= self._pgsr_max_train_steps:
                    stop = True
                    break

            scheduler.step()
            train_accuracy = np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)
            mean_loss = losses / max(epoch_steps, 1)
            info = (
                f"Task {self._cur_task}, Epoch {epoch + 1}/{self.args['epochs']} => "
                f"Loss {mean_loss:.3f}, Train_accy {train_accuracy:.2f}, Steps {global_step}"
            )
            if self._is_main_process:
                progress.set_description(info)
            if stop:
                break

        owner = self._unwrap_network()
        if not evaluation_curve or evaluation_curve[-1]["step"] != global_step:
            evaluation_curve.append(
                {
                    "step": global_step,
                    "preview": self._evaluate_task_batches(
                        owner, self._pgsr_preview_batches
                    ),
                    "heldout": self._evaluate_task_batches(
                        owner, self._pgsr_heldout_batches
                    ),
                }
            )
        if not self._pgsr_factor_snapshots or (
            self._pgsr_factor_snapshots[-1]["step"] != global_step
        ):
            self._pgsr_factor_snapshots.append(
                self._factor_snapshot(owner.backbone, global_step)
            )
        a_displacement = math.sqrt(
            sum(
                float((module.weight.detach() - initial).float().square().sum().item())
                for module, initial in zip(owner.backbone.w_As, self._pgsr_initial_a)
            )
        )
        b_displacement = math.sqrt(
            sum(
                float((module.weight.detach() - initial).float().square().sum().item())
                for module, initial in zip(owner.backbone.w_Bs, self._pgsr_initial_b)
            )
        )
        training_diagnostics = {
            "task_id": self._cur_task,
            "optimizer_steps": global_step,
            "warmup_steps": self._pgsr_warmup_steps,
            "a_parameter_displacement": a_displacement,
            "b_parameter_displacement": b_displacement,
            "warmup_boundary": warmup_boundary,
            "trajectory": trajectory,
            "evaluation_curve": evaluation_curve,
        }
        self._pgsr_diagnostics["training"] = training_diagnostics
        self._write_task_json("diagnostics", self._pgsr_diagnostics)
        self._save_factor_trajectory()
        self._log(
            f"[PGSR] task={self._cur_task} steps={global_step} "
            f"A_shift={a_displacement:.4e} B_shift={b_displacement:.4e}"
        )

    def _save_current_subspace(self, backbone) -> None:
        bank = SubspaceBank()
        bank.add_from_factors(
            self._cur_task,
            zip(backbone.w_As, backbone.w_Bs),
            rank=int(self.args.get("lora_rank", 10)),
        )
        checkpoint_dir = Path(str(self.args.get("filepath", "./")))
        bank.save(checkpoint_dir / f"pgsr_subspace_task_{self._cur_task}.pt")
        self._write_task_json("subspace", bank.summary())

    def _write_task_json(self, kind: str, payload: dict) -> None:
        if not self._is_main_process:
            return
        output_dir = Path(str(self.args.get("filepath", "./"))) / "pgsr"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"task_{self._cur_task:02d}_{kind}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        logging.info("[PGSR] wrote %s", path)
