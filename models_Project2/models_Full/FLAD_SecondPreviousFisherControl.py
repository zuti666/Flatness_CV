"""FLAD + second-previous Fisher-style perturbation filtering.

Framework adaptation of relatedPaper/flad_secondPreviousFisherControl.py:
- Keep FLAD core update rule: g_final = g0 + gamma * g1.
- Build replay-aware low-rank curvature subspace from memory gradients.
- Filter perturbations (delta0/delta1) along high-curvature replay directions.

Replay is integrated via BaseLearner exemplar memory.
"""

from __future__ import annotations

import contextlib
import logging
from typing import List, Optional, Tuple

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


def _curv_filter(
    delta: torch.Tensor,
    U: Optional[torch.Tensor],
    eig: Optional[torch.Tensor],
    alpha: float,
    mode: str = "soft",
    eps: float = 1e-8,
) -> torch.Tensor:
    if U is None or alpha <= 0 or delta.numel() == 0:
        return delta
    z = U.t() @ delta
    delta_parallel = U @ z
    delta_perp = delta - delta_parallel

    if mode == "hard":
        return delta_perp
    if mode == "spectral-soft" and eig is not None:
        scale = eig.clamp_min(eps)
        scale = scale / scale.max()
        weights = (alpha * scale).clamp(max=1.0)
        z_safe = (1 - weights) * z
        return delta_perp + U @ z_safe
    return delta_perp + (1 - alpha) * delta_parallel


def _top_eig(
    history: List[torch.Tensor],
    r: int,
    device: torch.device,
    eps: float = 1e-8,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if r <= 0 or len(history) == 0:
        return None, None
    G = torch.stack(history, dim=1)  # [p, k]
    gram = (G.t() @ G) / float(G.shape[0])
    evals, evecs = torch.linalg.eigh(gram)
    top_vals, idx = torch.topk(evals, k=min(r, evals.numel()))
    vec_small = evecs[:, idx]
    U = G @ vec_small
    U = U / torch.sqrt(top_vals.clamp_min(eps))
    return U.to(device), top_vals.to(device)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = str(args.get("optimizer_type", "sgd")).lower()

        self._flad_rho = float(args.get("flad_rho", 0.5))
        self._flad_gamma = float(args.get("flad_gamma", 0.1))
        self._flad_lambda0 = float(args.get("flad_lambda0", 0.9))
        self._flad_lambda1 = float(args.get("flad_lambda1", 0.9))
        self._flad_sigma = float(args.get("flad_sigma", 0.3))
        self._flad_c = float(args.get("flad_c", 1e-6))

        self._spfc_rank = int(args.get("spfc_rank", 4))
        self._spfc_beta = float(args.get("spfc_beta", 0.9))
        self._spfc_interval = int(args.get("spfc_interval", 20))
        self._spfc_history = int(args.get("spfc_history", 64))
        self._spfc_alpha0 = float(args.get("spfc_alpha0", 0.4))
        self._spfc_alpha1 = float(args.get("spfc_alpha1", 0.6))
        self._spfc_mode = str(args.get("spfc_mode", "spectral-soft")).lower()
        self._spfc_replay_batch = int(args.get("spfc_replay_batch", args.get("batch_size", 64)))

        self.m: Optional[torch.Tensor] = None
        self.n: Optional[torch.Tensor] = None
        self.curv_history: List[torch.Tensor] = []
        self.U_H: Optional[torch.Tensor] = None
        self.eig_H: Optional[torch.Tensor] = None
        self.curv_ema: Optional[torch.Tensor] = None
        self.step = 0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reset_task_state(self):
        self.m, self.n = None, None
        self.curv_history = []
        self.U_H = None
        self.eig_H = None
        self.curv_ema = None
        self.step = 0

    def _build_memory_loader(self, data_manager):
        mem = self._get_memory()
        if mem is None or len(mem[0]) == 0:
            return None
        mem_dataset = data_manager.get_dataset([], source="train", mode="train", appendent=mem)
        if len(mem_dataset) == 0:
            return None
        batch_size = max(1, min(self._spfc_replay_batch, len(mem_dataset)))
        return DataLoader(
            mem_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.args.get("train_num_workers", 8),
        )

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
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

        replay_loader = self._build_memory_loader(data_manager)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._reset_task_state()
        self._train(self.train_loader, self.test_loader, replay_loader=replay_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[FLAD-SPFC] Failed to compute class means: %s", exc)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def _maybe_update_curvature(self, g_old_vec: torch.Tensor):
        vec_cpu = g_old_vec.detach().cpu()
        if self.curv_ema is None:
            self.curv_ema = vec_cpu
        else:
            self.curv_ema = self._spfc_beta * self.curv_ema + (1 - self._spfc_beta) * vec_cpu

        self.curv_history.append(self.curv_ema.clone())
        if len(self.curv_history) > self._spfc_history:
            self.curv_history = self.curv_history[-self._spfc_history :]

        if self._spfc_interval <= 0:
            self.U_H, self.eig_H = _top_eig(self.curv_history, self._spfc_rank, self._device)
            return
        if self.step % self._spfc_interval != 0:
            return
        self.U_H, self.eig_H = _top_eig(self.curv_history, self._spfc_rank, self._device)

    def _train(self, train_loader, test_loader, replay_loader=None):
        model = self._unwrap_network()
        model.to(self._device)

        params = [p for p in model.parameters() if p.requires_grad]
        stage = "init" if self._cur_task == 0 else "update"
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

        replay_iter = iter(replay_loader) if replay_loader is not None else None

        def _next_replay_batch():
            nonlocal replay_iter
            if replay_loader is None:
                return None
            try:
                batch = next(replay_iter)
            except StopIteration:
                replay_iter = iter(replay_loader)
                batch = next(replay_iter)

            if len(batch) == 3:
                _, x, y = batch
            else:
                x, y = batch
            return x.to(self._device), y.to(self._device)

        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            model.train()
            total_loss = 0.0
            correct, total = 0, 0

            for batch in train_loader:
                self.step += 1
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                mix_inputs, mix_targets = inputs, targets

                replay_pair = _next_replay_batch()
                if replay_pair is not None:
                    buf_x, buf_y = replay_pair
                    mix_inputs = torch.cat([inputs, buf_x], dim=0)
                    mix_targets = torch.cat([targets, buf_y], dim=0)

                # Replay-only gradient for curvature estimate.
                replay_curv_pair = _next_replay_batch()
                if replay_curv_pair is not None:
                    rx, ry = replay_curv_pair
                    loss_rep = F.cross_entropy(model(rx)["logits"], ry)
                    g_old = autograd.grad(loss_rep, params, retain_graph=True, create_graph=False)
                    self._maybe_update_curvature(_flatten(g_old))

                with _double_backward_safe_context():
                    logits = model(mix_inputs)["logits"]
                    loss = F.cross_entropy(logits, mix_targets)
                    g_hat_list = autograd.grad(loss, params, retain_graph=True, create_graph=True)
                    g_hat = _flatten(g_hat_list)

                    if self.m is None:
                        self.m = torch.zeros_like(g_hat, device=self._device)
                    self.m = self._flad_lambda0 * self.m + (1 - self._flad_lambda0) * g_hat.detach()

                    diff0 = g_hat - self._flad_sigma * self.m
                    delta0_raw = self._flad_rho * diff0 / (diff0.norm() + self._flad_c)
                    delta0 = _curv_filter(delta0_raw, self.U_H, self.eig_H, self._spfc_alpha0, self._spfc_mode)
                    with torch.no_grad():
                        _add_vector(params, delta0)
                    loss_perturb0 = F.cross_entropy(model(mix_inputs)["logits"], mix_targets)
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

                    diff1 = grad_norm_grad - self._flad_sigma * self.n
                    delta1_raw = self._flad_rho * diff1 / (diff1.norm() + self._flad_c)
                    delta1 = _curv_filter(delta1_raw, self.U_H, self.eig_H, self._spfc_alpha1, self._spfc_mode)
                    with torch.no_grad():
                        _add_vector(params, delta1)
                    loss_hvp = F.cross_entropy(model(mix_inputs)["logits"], mix_targets)
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
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(mix_targets).cpu().sum()
                    total += len(mix_targets)
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

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
