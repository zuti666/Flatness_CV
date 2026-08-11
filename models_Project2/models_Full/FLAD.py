"""FLAD continual learner in the BaseLearner framework.

Paper-aligned core:
- Zeroth-order branch: perturb with delta0, evaluate g0 at perturbed weights.
- First-order branch:
  * `hvp` mode: perturb with delta1, estimate curvature-related g1 via HVP.
  * `mhp` mode: avoid second-order graph, approximate curvature terms with
    finite-difference MHP-style probes (first-order only).
- Final update direction: g_final = g0 + gamma * g1.

This variant does not use replay during optimization (same spirit as relatedPaper/flad.py).
"""

from __future__ import annotations

import contextlib
import logging
from typing import List, Optional

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
    attn_backend = getattr(torch.nn, "attention", None)
    if attn_backend is not None and hasattr(attn_backend, "sdpa_kernel") and hasattr(attn_backend, "SDPBackend"):
        return attn_backend.sdpa_kernel(backends=[attn_backend.SDPBackend.MATH])
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

        # Curvature branch mode:
        # - hvp: original FLAD with second-order graph (high memory).
        # - mhp/fd: first-order finite-difference proxy, avoids second-order graph.
        self._flad_second_order_mode = str(
            args.get("flad_second_order_mode", args.get("flad_curvature_mode", "mhp"))
        ).lower()
        if self._flad_second_order_mode == "fd":
            self._flad_second_order_mode = "mhp"
        if self._flad_second_order_mode not in {"hvp", "mhp"}:
            logging.warning(
                "[FLAD] Unknown flad_second_order_mode=%s, fallback to mhp.",
                self._flad_second_order_mode,
            )
            self._flad_second_order_mode = "mhp"

        # MHP(FD) step sizes:
        # 1) legacy single fd step (if provided), otherwise
        # 2) borrow GAM-like radii: probe->gam_grad_norm_rho, sharp->gam_grad_rho.
        legacy_rho_fd = args.get("flad_rho_fd", None)
        if legacy_rho_fd is not None:
            fd_rho = float(legacy_rho_fd)
            self._flad_mhp_probe_rho = fd_rho
            self._flad_mhp_sharp_rho = fd_rho
        else:
            self._flad_mhp_probe_rho = float(
                args.get(
                    "flad_mhp_probe_rho",
                    args.get("gam_grad_norm_rho", args.get("grad_norm_rho", self._flad_rho)),
                )
            )
            self._flad_mhp_sharp_rho = float(
                args.get(
                    "flad_mhp_sharp_rho",
                    args.get("gam_grad_rho", args.get("grad_rho", self._flad_rho)),
                )
            )

        self.m: Optional[torch.Tensor] = None
        self.n: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # Match the original method: reset EMA states at the beginning of each task.
        self.m, self.n = None, None
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[FLAD] Failed to compute class means: %s", exc)

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

                if self._flad_second_order_mode == "hvp":
                    with _double_backward_safe_context():
                        outputs = model(inputs)
                        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                        loss, eval_logits, eval_targets = self._stage_loss(logits, targets, stage=stage)

                        g_hat_list = autograd.grad(loss, params, retain_graph=True, create_graph=True)
                        g_hat = _flatten(g_hat_list)
                        g_hat_detached = g_hat.detach()

                        if self.m is None:
                            self.m = torch.zeros_like(g_hat_detached, device=self._device)
                        self.m = self._flad_lambda0 * self.m + (1 - self._flad_lambda0) * g_hat_detached

                        # delta0 -> g0
                        diff0 = g_hat_detached - self._flad_sigma * self.m
                        delta0 = self._flad_rho * diff0 / (diff0.norm() + self._flad_c)
                        with torch.no_grad():
                            _add_vector(params, delta0)
                        outputs_perturb0 = model(inputs)
                        logits_perturb0 = (
                            outputs_perturb0["logits"] if isinstance(outputs_perturb0, dict) else outputs_perturb0
                        )
                        loss_perturb0, _, _ = self._stage_loss(logits_perturb0, targets, stage=stage)
                        g0_list = autograd.grad(loss_perturb0, params, retain_graph=False, create_graph=False)
                        g0 = _flatten(g0_list).detach()
                        with torch.no_grad():
                            _add_vector(params, -delta0)

                        grad_norm = g_hat.norm()
                        grad_norm_grad_list = autograd.grad(grad_norm, params, retain_graph=True, create_graph=True)
                        grad_norm_grad = _flatten(grad_norm_grad_list)

                        # delta1 -> g1 (HVP branch)
                        if self.n is None:
                            self.n = torch.zeros_like(grad_norm_grad, device=self._device)
                        self.n = self._flad_lambda1 * self.n + (1 - self._flad_lambda1) * grad_norm_grad.detach()

                        diff1 = grad_norm_grad - self._flad_sigma * self.n
                        delta1 = self._flad_rho * diff1 / (diff1.norm() + self._flad_c)
                        with torch.no_grad():
                            _add_vector(params, delta1)
                        outputs_hvp = model(inputs)
                        logits_hvp = outputs_hvp["logits"] if isinstance(outputs_hvp, dict) else outputs_hvp
                        loss_hvp, _, _ = self._stage_loss(logits_hvp, targets, stage=stage)
                        g_vec_list = autograd.grad(loss_hvp, params, create_graph=True)
                        g_vec = _flatten(g_vec_list)
                        v = g_vec / (g_vec.norm() + 1e-8)
                        hvp_list = autograd.grad(g_vec_list, params, grad_outputs=_split(v, params), retain_graph=False)
                        g1 = _flatten(hvp_list).detach()
                        with torch.no_grad():
                            _add_vector(params, -delta1)
                else:
                    # MHP(FD) branch: no second-order graph, all first-order grads.
                    outputs = model(inputs)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    loss, eval_logits, eval_targets = self._stage_loss(logits, targets, stage=stage)
                    g_hat_list = autograd.grad(loss, params, retain_graph=False, create_graph=False)
                    g_hat = _flatten(g_hat_list).detach()

                    if self.m is None:
                        self.m = torch.zeros_like(g_hat, device=self._device)
                    self.m = self._flad_lambda0 * self.m + (1 - self._flad_lambda0) * g_hat

                    # delta0 -> g0
                    diff0 = g_hat - self._flad_sigma * self.m
                    delta0 = self._flad_rho * diff0 / (diff0.norm() + self._flad_c)
                    with torch.no_grad():
                        _add_vector(params, delta0)
                    outputs_perturb0 = model(inputs)
                    logits_perturb0 = (
                        outputs_perturb0["logits"] if isinstance(outputs_perturb0, dict) else outputs_perturb0
                    )
                    loss_perturb0, _, _ = self._stage_loss(logits_perturb0, targets, stage=stage)
                    g0_list = autograd.grad(loss_perturb0, params, retain_graph=False, create_graph=False)
                    g0 = _flatten(g0_list).detach()
                    with torch.no_grad():
                        _add_vector(params, -delta0)

                    rho_probe = max(float(self._flad_mhp_probe_rho), 1e-12)
                    rho_sharp = max(float(self._flad_mhp_sharp_rho), 1e-12)

                    # Finite-difference proxy for grad-norm direction at w.
                    step_plus = rho_probe * g_hat / (g_hat.norm() + self._flad_c)
                    with torch.no_grad():
                        _add_vector(params, step_plus)
                    logits_plus = model(inputs)["logits"]
                    loss_plus, _, _ = self._stage_loss(logits_plus, targets, stage=stage)
                    g_plus_list = autograd.grad(loss_plus, params, retain_graph=False, create_graph=False)
                    g_plus = _flatten(g_plus_list).detach()
                    with torch.no_grad():
                        _add_vector(params, -step_plus)
                    d_hat = (g_plus - g_hat) / rho_probe

                    # delta1 -> g1 (MHP/FDM branch)
                    if self.n is None:
                        self.n = torch.zeros_like(d_hat, device=self._device)
                    self.n = self._flad_lambda1 * self.n + (1 - self._flad_lambda1) * d_hat
                    diff1 = d_hat - self._flad_sigma * self.n
                    delta1 = self._flad_rho * diff1 / (diff1.norm() + self._flad_c)

                    with torch.no_grad():
                        _add_vector(params, delta1)
                    logits_w2 = model(inputs)["logits"]
                    loss_w2, _, _ = self._stage_loss(logits_w2, targets, stage=stage)
                    g2_list = autograd.grad(loss_w2, params, retain_graph=False, create_graph=False)
                    g2 = _flatten(g2_list).detach()

                    step3 = rho_sharp * g2 / (g2.norm() + self._flad_c)
                    with torch.no_grad():
                        _add_vector(params, step3)
                    logits_w3 = model(inputs)["logits"]
                    loss_w3, _, _ = self._stage_loss(logits_w3, targets, stage=stage)
                    g3_list = autograd.grad(loss_w3, params, retain_graph=False, create_graph=False)
                    g3 = _flatten(g3_list).detach()
                    g1 = (g3 - g2) / rho_sharp

                    with torch.no_grad():
                        _add_vector(params, -step3)
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

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
