"""FLAD with explicit OGD/Fisher perturbation filtering.

Design goals (aligned with user requirement):
- Keep FLAD core optimization logic: g_final = g0 + gamma * g1.
- Maintain first-order history like GAM_OGD_FisherMethod:
  global OGD orthonormal direction library.
- Maintain second-order history like GAM_OGD_FisherMethod:
  online diagonal Fisher accumulation and flattened Fisher vectors.
- Explicitly filter perturbations:
  * delta0_raw filtered by first-order OGD references.
  * delta1_raw filtered by second-order Fisher references.
- Filtering rule follows orthogonal projection style used by GAM filters.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_CL.baseLearner import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


def gram_schmidt(vectors: List[torch.Tensor], eps: float = 1e-10) -> List[torch.Tensor]:
    ortho: List[torch.Tensor] = []
    for v in vectors:
        w = v.clone()
        for u in ortho:
            w -= torch.dot(u, v) * u
        norm = w.norm()
        if norm > eps:
            ortho.append(w / norm)
    return ortho


def _flatten(grads: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads]) if grads else torch.tensor([])


def _double_backward_safe_context():
    """Force math SDPA kernels to avoid unsupported double-backward paths."""
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        return contextlib.nullcontext()
    if hasattr(cuda_backend, "sdp_kernel"):
        return cuda_backend.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    return contextlib.nullcontext()


def _split(vec: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    idx = 0
    for p in params:
        n = p.numel()
        out.append(vec[idx : idx + n].view_as(p))
        idx += n
    return out


@torch.no_grad()
def _add_vector(params: List[torch.nn.Parameter], vec: torch.Tensor) -> None:
    idx = 0
    for p in params:
        n = p.numel()
        p.add_(vec[idx : idx + n].view_as(p))
        idx += n


@torch.no_grad()
def _set_flat_grad(params: List[torch.nn.Parameter], grad_vec: torch.Tensor) -> None:
    idx = 0
    for p in params:
        n = p.numel()
        g = grad_vec[idx : idx + n].view_as(p)
        if p.grad is None:
            p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)
        p.grad.copy_(g)
        idx += n


def _project_orthogonal_flat(
    direction: torch.Tensor,
    references: List[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    projected = direction.clone()
    for ref in references:
        ref_norm = torch.linalg.vector_norm(ref, ord=2)
        if ref_norm <= eps:
            continue
        unit_ref = ref / (ref_norm + eps)
        projected = projected - torch.dot(projected, unit_ref) * unit_ref
    return projected


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = str(args.get("optimizer_type", "sgd")).lower()

        # FLAD core hyperparameters.
        self._flad_rho = float(args.get("flad_rho", 0.5))
        self._flad_gamma = float(args.get("flad_gamma", 0.1))
        self._flad_lambda0 = float(args.get("flad_lambda0", 0.9))
        self._flad_lambda1 = float(args.get("flad_lambda1", 0.9))
        self._flad_sigma = float(args.get("flad_sigma", 0.3))
        self._flad_c = float(args.get("flad_c", 1e-6))

        # Historical first-order information (OGD directions).
        self._ogd_max_dirs = int(
            args.get("flad_filterogdfisher_max_directions", args.get("ogd_max_dirs", args.get("max_directions", 2000)))
        )
        self._ogd_store_batches = int(
            args.get(
                "flad_filterogdfisher_store_batches",
                args.get("ogd_store_batches", args.get("grads_per_task", args.get("flad_filterogdfisher_grads_per_task", 50))),
            )
        )
        self._ogd_eps = float(args.get("ogd_eps", 1e-10))
        self._directions: List[torch.Tensor] = []

        # Historical second-order information (online diagonal Fisher).
        self._fisher_gamma = float(args.get("flad_filterogdfisher_gamma", args.get("ewc_gamma", 1.0)))
        self._fisher_max_batches = int(args.get("flad_filterogdfisher_max_batches", args.get("ewc_max_batches", 100)))
        self._fisher_eps = float(args.get("flad_filterogdfisher_eps", args.get("ewc_eps", 1e-5)))
        self._fisher: Dict[str, torch.Tensor] | None = None
        self.max_second_order_history = int(
            args.get("flad_filterogdfisher_max_second_order_history", args.get("max_second_order_history", 100))
        )
        self.second_order_history: List[torch.Tensor] = []
        self.second_order_accum: Optional[torch.Tensor] = None

        # Explicit perturbation filtering strengths.
        self._first_order_filter_strength = float(
            args.get("first_order_filter_strength", args.get("flad_first_order_filter_strength", 1.0))
        )
        self._second_order_filter_strength = float(
            args.get("second_order_filter_strength", args.get("flad_second_order_filter_strength", 1.0))
        )
        self._delta0_filter_strength = float(
            args.get(
                "flad_filterogdfisher_delta0_filter_strength",
                args.get("perturb1_filter_strength", self._first_order_filter_strength),
            )
        )
        self._delta1_filter_strength = float(
            args.get(
                "flad_filterogdfisher_delta1_filter_strength",
                args.get("perturb2_filter_strength", self._second_order_filter_strength),
            )
        )
        self._filter_eps = float(args.get("projection_eps", args.get("flad_filter_eps", 1e-12)))

        self.m: Optional[torch.Tensor] = None
        self.n: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def after_task(self):
        model = self._unwrap_network()

        if hasattr(self, "train_loader") and self.train_loader is not None:
            logger.info("Estimating FLAD-OGD-Fisher second-order information for task %d...", self._cur_task)
            self._fisher = self._compute_fisher(self.train_loader, model)
            fisher_vec = self._fisher_to_vector(model)
            if fisher_vec is not None:
                self.second_order_history.append(fisher_vec)
                if len(self.second_order_history) > self.max_second_order_history:
                    self.second_order_history = self.second_order_history[-self.max_second_order_history :]
                self.second_order_accum = fisher_vec.clone()

            logger.info(
                "[FLAD_filterogdfisher] task=%d first_order_dirs=%d second_order_hist=%d fisher_dim=%d",
                self._cur_task,
                len(self._directions),
                len(self.second_order_history),
                0 if fisher_vec is None else int(fisher_vec.numel()),
            )

        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._align_state_to_model(self._unwrap_network())
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

        # Follow FLAD setting: reset EMA directions each task.
        self.m, self.n = None, None
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Align with OGD/GAM pattern: store first-order directions after task training.
        self._store_directions(self.train_loader, self._unwrap_network())

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[FLAD_filterogdfisher] Failed to compute class means: %s", exc)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def _stage_loss(self, logits: torch.Tensor, targets: torch.Tensor, stage: str):
        if stage == "init":
            loss = F.cross_entropy(logits, targets)
            eval_logits, eval_targets = logits, targets
        else:
            fake_targets = targets - self._known_classes
            eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets
            loss = F.cross_entropy(eval_logits, eval_targets)
        return loss, eval_logits, eval_targets

    def _filter_delta(
        self,
        delta_raw: torch.Tensor,
        references: Optional[List[torch.Tensor]],
        strength: float,
    ) -> torch.Tensor:
        if strength <= 0 or references is None or len(references) == 0:
            return delta_raw

        refs: List[torch.Tensor] = []
        total_numel = delta_raw.numel()
        for ref in references:
            flat_ref = ref.detach().reshape(-1)
            if flat_ref.numel() != total_numel:
                matched = torch.zeros(total_numel, dtype=flat_ref.dtype, device="cpu")
                n = min(total_numel, flat_ref.numel())
                matched[:n] = flat_ref[:n].cpu()
                flat_ref = matched
            refs.append(flat_ref.to(device=delta_raw.device, dtype=delta_raw.dtype))

        if len(refs) == 0:
            return delta_raw

        delta_orth = _project_orthogonal_flat(delta_raw, refs, eps=self._filter_eps)
        projection = delta_raw - delta_orth
        return delta_raw - strength * projection

    def _train(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)

        stage = "init" if self._cur_task == 0 else "update"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage=stage)

        epochs = int(self.args.get("init_epoch" if self._cur_task == 0 else "epochs", 1))
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("init_milestones" if self._cur_task == 0 else "milestones", []),
            gamma=float(self.args.get("init_lrate_decay" if self._cur_task == 0 else "lrate_decay", 1.0)),
            T_max=epochs,
            eta_min=self.args.get("min_lr", 0.0),
        )

        first_order_refs = self._first_order_information()
        second_order_refs = self._second_order_information(self._unwrap_network())

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

                with _double_backward_safe_context():
                    outputs = model(inputs)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    loss, eval_logits, eval_targets = self._stage_loss(logits, targets, stage=stage)

                    g_hat_list = autograd.grad(loss, params, retain_graph=True, create_graph=True)
                    g_hat = _flatten(g_hat_list)

                    if self.m is None:
                        self.m = torch.zeros_like(g_hat, device=self._device)
                    self.m = self._flad_lambda0 * self.m + (1 - self._flad_lambda0) * g_hat.detach()

                    # delta0_raw: explicitly filtered by first-order OGD references.
                    diff0 = g_hat - self._flad_sigma * self.m
                    delta0_raw = self._flad_rho * diff0 / (diff0.norm() + self._flad_c)
                    delta0 = self._filter_delta(
                        delta_raw=delta0_raw,
                        references=first_order_refs,
                        strength=self._delta0_filter_strength,
                    )

                    with torch.no_grad():
                        _add_vector(params, delta0)
                    logits_perturb0 = model(inputs)["logits"]
                    loss_perturb0, _, _ = self._stage_loss(logits_perturb0, targets, stage=stage)
                    g0_list = autograd.grad(loss_perturb0, params, retain_graph=False, create_graph=False)
                    g0 = _flatten(g0_list).detach()
                    with torch.no_grad():
                        _add_vector(params, -delta0)

                    grad_norm = g_hat.norm()
                    grad_norm_grad_list = autograd.grad(grad_norm, params, retain_graph=True, create_graph=True)
                    grad_norm_grad = _flatten(grad_norm_grad_list)

                    if self.n is None:
                        self.n = torch.zeros_like(grad_norm_grad, device=self._device)
                    self.n = self._flad_lambda1 * self.n + (1 - self._flad_lambda1) * grad_norm_grad.detach()

                    # delta1_raw: explicitly filtered by second-order Fisher references.
                    diff1 = grad_norm_grad - self._flad_sigma * self.n
                    delta1_raw = self._flad_rho * diff1 / (diff1.norm() + self._flad_c)
                    delta1 = self._filter_delta(
                        delta_raw=delta1_raw,
                        references=second_order_refs,
                        strength=self._delta1_filter_strength,
                    )

                    with torch.no_grad():
                        _add_vector(params, delta1)
                    logits_hvp = model(inputs)["logits"]
                    loss_hvp, _, _ = self._stage_loss(logits_hvp, targets, stage=stage)
                    g_vec_list = autograd.grad(loss_hvp, params, create_graph=True)
                    g_vec = _flatten(g_vec_list)
                    v = g_vec / (g_vec.norm() + 1e-8)
                    hvp_list = autograd.grad(g_vec_list, params, grad_outputs=_split(v, params), retain_graph=False)
                    g1 = _flatten(hvp_list).detach()
                    with torch.no_grad():
                        _add_vector(params, -delta1)

                g_final = g0 + self._flad_gamma * g1
                optimizer.zero_grad(set_to_none=True)
                _set_flat_grad(params, g_final)
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
            if (epoch % 5 == 4):
                test_acc = self._compute_accuracy(model, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            prog_bar.set_description(info)
        logging.info(info)

    # ------------------------------------------------------------------ #
    # historical information helpers (GAM_OGD_Fisher style)
    # ------------------------------------------------------------------ #
    def _first_order_information(self) -> Optional[List[torch.Tensor]]:
        if not self._directions:
            return None
        return [v.detach().clone() for v in self._directions]

    def _second_order_information(self, model: nn.Module) -> Optional[List[torch.Tensor]]:
        fisher_vec = self._fisher_to_vector(model)
        if fisher_vec is not None and fisher_vec.numel() > 0:
            return [fisher_vec]

        infos: List[torch.Tensor] = []
        if self.second_order_history:
            infos.extend(v.detach().clone() for v in self.second_order_history)
        if self.second_order_accum is not None:
            infos.append(self.second_order_accum.detach().clone())
        return infos or None

    def _align_state_to_model(self, model: nn.Module):
        target_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)

        def _match(v: torch.Tensor) -> torch.Tensor:
            if v.numel() == target_dim:
                return v.detach().clone().cpu()
            out = torch.zeros(target_dim, dtype=v.dtype, device="cpu")
            n = min(target_dim, v.numel())
            out[:n] = v[:n].detach().cpu()
            return out

        if self._directions:
            aligned = [_match(v) for v in self._directions]
            self._directions = gram_schmidt(aligned, eps=self._ogd_eps)
            if len(self._directions) > self._ogd_max_dirs:
                self._directions = self._directions[: self._ogd_max_dirs]

        if self._fisher is not None:
            for name, p in model.named_parameters():
                if name in self._fisher:
                    self._fisher[name] = self._match_tensor(self._fisher[name], p)

        if self.second_order_history:
            self.second_order_history = [_match(v) for v in self.second_order_history]
        if self.second_order_accum is not None:
            self.second_order_accum = _match(self.second_order_accum)

    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    def _match_tensor(self, stored: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if stored.shape == param.shape:
            return stored
        new = torch.zeros_like(param.detach().cpu())
        slices = tuple(slice(0, min(a, b)) for a, b in zip(stored.shape, param.shape))
        new[slices] = stored[slices]
        return new

    def _fisher_to_vector(self, model: nn.Module) -> Optional[torch.Tensor]:
        if self._fisher is None:
            return None
        parts: List[torch.Tensor] = []
        for name, p in self._iter_trainable(model):
            f = self._fisher.get(name)
            if f is None:
                parts.append(torch.zeros(p.numel(), dtype=p.detach().cpu().dtype, device="cpu"))
                continue
            if f.shape != p.shape:
                f = self._match_tensor(f, p)
                self._fisher[name] = f
            parts.append(f.detach().reshape(-1).cpu())
        if not parts:
            return None
        return torch.cat(parts)

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
            updated: Dict[str, torch.Tensor] = {}
            for name, new_f in fisher.items():
                old = self._fisher.get(name)
                if old is None:
                    updated[name] = new_f
                else:
                    old_aligned = self._match_tensor(old, new_f)
                    updated[name] = self._fisher_gamma * old_aligned + new_f
            for name, old in self._fisher.items():
                if name not in updated:
                    updated[name] = self._fisher_gamma * old
            self._fisher = updated
        else:
            self._fisher = fisher

        logger.info("[FLAD_filterogdfisher] Fisher estimated over %d batches", total)
        return self._fisher

    @torch.no_grad()
    def _store_directions(self, loader, model: nn.Module):
        model.eval()
        collected: List[torch.Tensor] = []
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
                anchor = logits[idx, targets].sum()
                anchor.backward()

            flat = self._flatten_current_grads([p for p in model.parameters() if p.requires_grad]).detach().cpu()
            if flat.numel() > 0:
                collected.append(flat)

        if not collected:
            return

        new_dirs = gram_schmidt(collected, eps=self._ogd_eps)
        merged = gram_schmidt(self._directions + new_dirs, eps=self._ogd_eps)
        if len(merged) > self._ogd_max_dirs:
            merged = merged[: self._ogd_max_dirs]
        self._directions = merged
        logger.info("[FLAD_filterogdfisher] Stored %d directions (cap %d)", len(self._directions), self._ogd_max_dirs)

    def _flatten_current_grads(self, params: List[torch.nn.Parameter]) -> torch.Tensor:
        flats = []
        for p in params:
            if p.grad is not None:
                flats.append(p.grad.view(-1))
        return torch.cat(flats) if flats else torch.tensor([], device=self._device)

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
