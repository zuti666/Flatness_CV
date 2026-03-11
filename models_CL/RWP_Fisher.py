"""RWP with Fisher-aware Gaussian perturbation (no OGD constraints).

Implementation summary:
- RWP perturbation is sampled before forward/backward.
- Fisher-aware diffusion is applied by masking top-k Fisher-sensitive
  coordinates.

Fisher shaping here uses a diagonal-Fisher approximation:
- Keep a running diagonal Fisher per parameter (EWCon-style accumulation).
- Build a global top-k sensitive coordinate set.
- Project perturbation to the complement by zeroing those sensitive coordinates.
"""

from __future__ import annotations

import heapq
import logging
from typing import Dict, List, Tuple

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


def _gram_schmidt(vectors: List[torch.Tensor], eps: float = 1e-10) -> List[torch.Tensor]:
    ortho: List[torch.Tensor] = []
    for v in vectors:
        w = v.clone()
        for u in ortho:
            w -= torch.dot(u, v) * u
        norm = w.norm()
        if norm > eps:
            ortho.append(w / norm)
    return ortho


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # OGD
        self._ogd_max_dirs = int(args.get("ogd_max_dirs", 60))
        self._ogd_store_batches = int(args.get("ogd_store_batches", 50))
        self._ogd_eps = float(args.get("ogd_eps", 1e-8))

        # RWP Gaussian
        self._rwp_std = float(args.get("rwp_std", args.get("noise_std", 0.01)))
        self._rwp_start_task = int(args.get("rwp_start_task", args.get("noise_start_task", 1)))

        # Fisher (EWCon-style running diagonal)
        self._fisher_gamma = float(args.get("fisher_gamma", args.get("ewc_gamma", 1.0)))
        self._fisher_max_batches = int(args.get("fisher_max_batches", args.get("ewc_max_batches", 100)))
        self._fisher_eps = float(args.get("fisher_eps", args.get("ewc_eps", 1e-5)))
        self._fisher_topk = int(args.get("fisher_topk", args.get("fisher_rank", 256)))

        # Global OGD basis (CPU)
        self._directions: List[torch.Tensor] = []

        # Running Fisher diagonal (CPU)
        self._fisher: Dict[str, torch.Tensor] | None = None

        # Fisher-sensitive coordinate set for perturbation projection.
        # For each parameter name, store flattened indices (CPU long) to REMOVE.
        self._fisher_drop_idx: Dict[str, torch.Tensor] = {}
        self._fisher_drop_idx_device_cache: Dict[Tuple[str, str], torch.Tensor] = {}

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._align_fisher_to_model(self._network)
        self._rebuild_fisher_projection()
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

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

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        model = self._unwrap_network()
        self._fisher = self._compute_fisher(self.train_loader, model)
        self._rebuild_fisher_projection()

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[RWP_FISHER] Failed to compute class means: %s", exc)

    def _train(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)

        stage = "init" if self._cur_task == 0 else "update"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage=stage)

        if stage == "init":
            epochs = int(self.args.get("init_epoch", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("init_milestones", []),
                gamma=float(self.args.get("init_lrate_decay", 1.0)),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )
        else:
            epochs = int(self.args.get("epochs", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )

        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            model.train()
            total_loss = 0.0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                optimizer.zero_grad()
                perturbations = self._sample_and_apply_fisher_rwp(model, optimizer)
                try:
                    outputs = model(inputs)
                    logits = outputs["logits"]

                    if stage == "init":
                        loss = F.cross_entropy(logits, targets)
                        eval_logits, eval_targets = logits, targets
                    else:
                        fake_targets = targets - self._known_classes
                        loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                        eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets

                    # RWP gradient: evaluated at perturbed parameters.
                    loss.backward()

                finally:
                    # Apply step from clean theta using gradient from theta + delta.
                    self._restore_rwp_perturbation(perturbations)

                optimizer.step()

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss.detach().item())

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => "
                f"Loss {total_loss/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            )
            if epoch % 5 == 4:
                test_acc = self._compute_accuracy(model, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            prog_bar.set_description(info)
        logging.info(info)

    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    @torch.no_grad()
    def _sample_and_apply_fisher_rwp(self, model: nn.Module, optimizer) -> List[Tuple[torch.nn.Parameter, torch.Tensor]]:
        if self._cur_task < self._rwp_start_task:
            return []
        if self._rwp_std <= 0.0:
            return []

        lr_by_pid = {}
        for group in optimizer.param_groups:
            lr = float(group.get("lr", 0.0))
            if lr <= 0.0:
                continue
            for p in group.get("params", []):
                if p is not None:
                    lr_by_pid[id(p)] = lr
        if not lr_by_pid:
            return []

        sigma = float(self._rwp_std)
        perturbations: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        for name, p in self._iter_trainable(model):
            if id(p) not in lr_by_pid:
                continue
            delta = self._sample_weight_normalized_noise(p, sigma)

            drop_idx = self._get_drop_idx_on_device(name, p.device)
            if drop_idx is not None and drop_idx.numel() > 0:
                flat = delta.reshape(-1)
                flat.index_fill_(0, drop_idx, 0.0)
                delta = flat.reshape_as(p)

            p.add_(delta)
            perturbations.append((p, delta))
        return perturbations

    @torch.no_grad()
    def _sample_weight_normalized_noise(self, param: torch.nn.Parameter, std: float) -> torch.Tensor:
        if param.dim() > 1:
            shape = tuple(param.shape)
            row_norms = param.detach().reshape(shape[0], -1).norm(dim=1, keepdim=True)
            row_norms = row_norms.reshape(shape[0], *([1] * (param.dim() - 1)))
            return torch.randn_like(param) * (float(std) * row_norms)

        scale = float(std) * (param.detach().reshape(-1).norm().item() + 1e-16)
        return torch.randn_like(param) * scale

    @torch.no_grad()
    def _restore_rwp_perturbation(self, perturbations: List[Tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
        for p, delta in perturbations:
            p.sub_(delta)

    def _project_gradients(self, model: nn.Module):
        if not self._directions:
            return
        params = [p for _, p in self._iter_trainable(model)]
        flat_grad = self._flatten_current_grads(params)
        if flat_grad.numel() == 0:
            return
        flat_grad = self._project(flat_grad)
        self._assign_flat_grad(params, flat_grad)

    def _flatten_current_grads(self, params: List[torch.nn.Parameter]) -> torch.Tensor:
        flats = []
        for p in params:
            if p.grad is not None:
                flats.append(p.grad.reshape(-1))
        return torch.cat(flats) if flats else torch.tensor([], device=self._device)

    def _assign_flat_grad(self, params: List[torch.nn.Parameter], flat: torch.Tensor) -> None:
        offset = 0
        for p in params:
            if p.grad is None:
                continue
            numel = p.numel()
            p.grad.copy_(flat[offset : offset + numel].reshape_as(p))
            offset += numel

    def _project(self, g: torch.Tensor) -> torch.Tensor:
        if not self._directions or g.numel() == 0:
            return g
        g_proj = g.clone()
        for s in self._directions:
            if g_proj.numel() == s.numel():
                s_use = s
                if s_use.device != g_proj.device or s_use.dtype != g_proj.dtype:
                    s_use = s_use.to(device=g_proj.device, dtype=g_proj.dtype)
                dot = torch.dot(g_proj, s_use)
                g_proj = g_proj - dot * s_use
            else:
                n = min(g_proj.numel(), s.numel())
                if n == 0:
                    continue
                s_part = s[:n]
                if s_part.device != g_proj.device or s_part.dtype != g_proj.dtype:
                    s_part = s_part.to(device=g_proj.device, dtype=g_proj.dtype)
                dot = torch.dot(g_proj[:n], s_part)
                g_proj[:n] = g_proj[:n] - dot * s_part
        return g_proj

    def _align_directions_to_model(self, model: nn.Module) -> None:
        target_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._align_directions_to_dim(target_dim)

    def _align_directions_to_dim(self, target_dim: int) -> None:
        if not self._directions:
            return
        changed = False
        aligned: List[torch.Tensor] = []
        for v in self._directions:
            flat = v.detach().clone().reshape(-1).cpu()
            if flat.numel() == target_dim:
                aligned.append(flat)
                continue
            changed = True
            out = torch.zeros(target_dim, dtype=flat.dtype, device="cpu")
            n = min(target_dim, flat.numel())
            out[:n] = flat[:n]
            aligned.append(out)
        if not changed:
            return
        self._directions = _gram_schmidt(aligned, eps=self._ogd_eps)
        if len(self._directions) > self._ogd_max_dirs:
            self._directions = self._directions[: self._ogd_max_dirs]
        logger.info("[RWP_FISHER] Aligned %d directions to dim=%d", len(self._directions), target_dim)

    def _match_tensor(self, stored: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if stored.shape == param.shape:
            return stored
        new = torch.zeros_like(param.detach().cpu())
        slices = tuple(slice(0, min(a, b)) for a, b in zip(stored.shape, param.shape))
        new[slices] = stored[slices]
        return new

    def _align_fisher_to_model(self, model: nn.Module) -> None:
        if self._fisher is None:
            return
        for name, p in model.named_parameters():
            if name in self._fisher:
                self._fisher[name] = self._match_tensor(self._fisher[name], p)

    def _compute_fisher(self, loader, model: nn.Module) -> Dict[str, torch.Tensor]:
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        total = 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= self._fisher_max_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            model.zero_grad()
            with torch.enable_grad():
                logits = model(inputs)["logits"]
                logp = F.log_softmax(logits, dim=1)
                nll = -F.nll_loss(logp, targets, reduction="none")
                exp_cp = torch.mean(torch.exp(nll.detach()))
                loss = nll.mean()
                loss.backward()

            for name, p in self._iter_trainable(model):
                if p.grad is None:
                    continue
                g2 = (p.grad.detach().cpu() ** 2) * exp_cp.cpu()
                fisher[name] = g2 if name not in fisher else fisher[name] + g2
            total += 1

        if total > 0:
            for name in fisher:
                fisher[name] = (fisher[name] / float(total)).clamp_min(self._fisher_eps)

        if self._fisher is not None:
            for name, new_f in fisher.items():
                old = self._fisher.get(name)
                self._fisher[name] = new_f if old is None else self._fisher_gamma * old + new_f
            for name, old in self._fisher.items():
                if name not in fisher:
                    self._fisher[name] = self._fisher_gamma * old
        else:
            self._fisher = fisher

        logger.info("[RWP_FISHER] Fisher estimated over %d batches", total)
        return self._fisher

    def _rebuild_fisher_projection(self) -> None:
        self._fisher_drop_idx = {}
        self._fisher_drop_idx_device_cache.clear()

        if self._fisher is None or self._fisher_topk <= 0:
            return

        total_dim = 0
        for f in self._fisher.values():
            total_dim += int(f.numel())
        k = min(int(self._fisher_topk), total_dim)
        if k <= 0:
            return

        heap: List[Tuple[float, str, int]] = []
        for name, f in self._fisher.items():
            flat = f.reshape(-1)
            if flat.numel() == 0:
                continue
            local_k = min(k, int(flat.numel()))
            vals, idxs = torch.topk(flat, k=local_k, largest=True, sorted=False)
            vals_l = vals.tolist()
            idxs_l = idxs.tolist()
            for value, idx in zip(vals_l, idxs_l):
                item = (float(value), name, int(idx))
                if len(heap) < k:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

        if not heap:
            return

        grouped: Dict[str, set] = {}
        for _, name, idx in heap:
            if name not in grouped:
                grouped[name] = set()
            grouped[name].add(idx)

        for name, idx_set in grouped.items():
            self._fisher_drop_idx[name] = torch.tensor(sorted(idx_set), dtype=torch.long)

        covered = sum(int(v.numel()) for v in self._fisher_drop_idx.values())
        logger.info(
            "[RWP_FISHER] Fisher projection built: topk=%d, covered_dims=%d, params=%d",
            k,
            covered,
            len(self._fisher_drop_idx),
        )

    def _get_drop_idx_on_device(self, name: str, device: torch.device) -> torch.Tensor | None:
        idx = self._fisher_drop_idx.get(name)
        if idx is None or idx.numel() == 0:
            return None
        key = (name, str(device))
        cached = self._fisher_drop_idx_device_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        out = idx.to(device=device, dtype=torch.long)
        self._fisher_drop_idx_device_cache[key] = out
        return out

    @torch.no_grad()
    def _update_ogd_directions(self, loader) -> None:
        model = self._unwrap_network()
        model.eval()
        params = [p for _, p in self._iter_trainable(model)]
        collected_global: List[torch.Tensor] = []

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= self._ogd_store_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            model.zero_grad()
            with torch.enable_grad():
                logits = model(inputs)["logits"]
                idx = torch.arange(len(targets), device=targets.device)
                # Match OGD: collect correct-class logit gradient directions.
                anchor = logits[idx, targets].sum()
                anchor.backward()

            flat = self._flatten_current_grads(params).detach().cpu()
            if flat.numel() > 0:
                collected_global.append(flat)

        if collected_global:
            new_dirs = _gram_schmidt(collected_global, eps=self._ogd_eps)
            merged = _gram_schmidt(self._directions + new_dirs, eps=self._ogd_eps)
            if len(merged) > self._ogd_max_dirs:
                merged = merged[: self._ogd_max_dirs]
            self._directions = merged

        logger.info("[RWP_FISHER] Stored global dirs %d", len(self._directions))

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
