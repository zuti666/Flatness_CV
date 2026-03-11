"""GPM_EWC: Gradient Projection Memory (first-order projection) + EWC (second-order diag Fisher).

- GPM preserves past subspaces of layer activations; gradients are projected to the orthogonal complement.
- EWC adds a Fisher-weighted quadratic penalty to resist drift in important parameters.
"""

from __future__ import annotations

import logging
from typing import Dict, List

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


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # GPM hyperparams
        self._gpm_threshold = float(args.get("gpm_threshold", 0.97))
        self._gpm_threshold_delta = float(args.get("gpm_threshold_delta", 0.0))
        self._gpm_max_batches = int(args.get("gpm_max_batches", 2))
        self._gpm_max_samples = int(args.get("gpm_max_samples", 256))

        # EWC hyperparams
        self._ewc_lambda = float(args.get("ewc_lambda", 20.0))
        self._ewc_gamma = float(args.get("ewc_gamma", 1.0))
        self._ewc_max_batches = int(args.get("ewc_max_batches", 100))
        self._ewc_eps = float(args.get("ewc_eps", 1e-5))

        # GPM state
        self._gpm_modules: List[nn.Module] = []
        self._gpm_bases: List[torch.Tensor] = []  # CPU

        # EWC state
        self._fisher: Dict[str, torch.Tensor] | None = None
        self._ewc_checkpoint: Dict[str, torch.Tensor] | None = None

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

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Post-task updates
        self._update_gpm_bases(self.train_loader)
        self._ewc_checkpoint = {n: p.detach().cpu().clone() for n, p in self._iter_trainable(self._unwrap_network())}
        self._fisher = self._compute_fisher(self.train_loader, self._unwrap_network())

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[GPM_EWC] Failed to compute class means: %s", exc)

    def _train(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)

        if not self._gpm_modules:
            self._init_gpm_modules(model)

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

                optimizer.zero_grad()
                outputs = model(inputs)
                logits = outputs["logits"]

                if self._cur_task == 0:
                    loss = F.cross_entropy(logits, targets)
                    eval_logits, eval_targets = logits, targets
                else:
                    fake_targets = targets - self._known_classes
                    loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets

                loss = loss + self._ewc_penalty(model)
                loss.backward()

                if self._cur_task > 0 and self._gpm_bases:
                    self._project_gradients_gpm()

                optimizer.step()

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss.detach().item())

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => Loss {total_loss/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            if (epoch % 5 == 4):
                test_acc = self._compute_accuracy(model, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            prog_bar.set_description(info)
        logging.info(info)

    # ---------- GPM helpers ----------
    def _init_gpm_modules(self, model: nn.Module):
        head = getattr(model, "fc", None)
        for module in model.modules():
            if module is head:
                continue
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._gpm_modules.append(module)
        logger.info("[GPM_EWC] Protected layers: %d", len(self._gpm_modules))

    @torch.no_grad()
    def _collect_activations(self, loader) -> List[torch.Tensor]:
        model = self._unwrap_network()
        acts: List[List[torch.Tensor]] = [[] for _ in self._gpm_modules]
        hooks = []

        def hook_fn(idx):
            def _fn(module, inp, out):
                x = inp[0].detach()
                if isinstance(module, nn.Conv2d):
                    k = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    dilation = module.dilation
                    x_unf = F.unfold(x, kernel_size=k, dilation=dilation, padding=padding, stride=stride)
                    acts[idx].append(x_unf.cpu())
                elif isinstance(module, nn.Linear):
                    acts[idx].append(x.view(x.size(0), -1).cpu())
            return _fn

        for i, m in enumerate(self._gpm_modules):
            hooks.append(m.register_forward_hook(hook_fn(i)))

        model.eval()
        total_seen = 0
        for b_idx, batch in enumerate(loader):
            if b_idx >= self._gpm_max_batches or total_seen >= self._gpm_max_samples:
                break
            if len(batch) == 3:
                _, inputs, _ = batch
            else:
                inputs, _ = batch
            bs = inputs.size(0)
            total_seen += bs
            model(inputs.to(self._device))

        for h in hooks:
            h.remove()

        mat_list: List[torch.Tensor] = []
        for lst in acts:
            if not lst:
                mat_list.append(torch.zeros(1, 1))
                continue
            x = torch.cat(lst, dim=0)
            if x.dim() == 3:
                B, d, L = x.shape
                mat = x.permute(1, 0, 2).reshape(d, B * L)
            else:
                B, d = x.shape
                mat = x.t()
            mat_list.append(mat)
        return mat_list

    def _update_gpm_bases(self, loader):
        mat_list = self._collect_activations(loader)
        threshold = self._gpm_threshold + self._cur_task * self._gpm_threshold_delta

        if not self._gpm_bases:
            for mat in mat_list:
                if mat.numel() <= 1:
                    self._gpm_bases.append(torch.zeros(1, 0))
                    continue
                U, S, _ = torch.linalg.svd(mat, full_matrices=False)
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / (sval_total + 1e-12)
                r = int(torch.sum(torch.cumsum(sval_ratio, dim=0) < threshold).item())
                self._gpm_bases.append(U[:, :r].cpu())
        else:
            for i, mat in enumerate(mat_list):
                if mat.numel() <= 1 or self._gpm_bases[i].numel() == 0:
                    continue
                B = self._gpm_bases[i]
                proj = B @ (B.t() @ mat)
                act_hat = mat - proj
                U1, S1, _ = torch.linalg.svd(act_hat, full_matrices=False)
                sval_total = (torch.linalg.svdvals(mat) ** 2).sum()
                sval_hat = (S1 ** 2).sum()
                sval_ratio = (S1 ** 2) / (sval_total + 1e-12)
                accumulated = (sval_total - sval_hat) / (sval_total + 1e-12)
                r = 0
                for j in range(sval_ratio.shape[0]):
                    if accumulated < threshold:
                        accumulated += sval_ratio[j].item()
                        r += 1
                    else:
                        break
                if r == 0:
                    continue
                Ui = torch.cat([B, U1[:, :r]], dim=1)
                if Ui.shape[1] > Ui.shape[0]:
                    Ui = Ui[:, : Ui.shape[0]]
                self._gpm_bases[i] = Ui.cpu()

        for idx, B in enumerate(self._gpm_bases):
            logger.info("[GPM_EWC] Layer %d basis rank: %d / %d", idx + 1, B.shape[1], B.shape[0])

    def _project_gradients_gpm(self):
        kk = 0
        for m in self._gpm_modules:
            for param_name, param in m.named_parameters(recurse=False):
                if param.grad is None:
                    continue
                if param_name == "bias":
                    param.grad.zero_()
                    continue
                basis = self._gpm_bases[kk] if kk < len(self._gpm_bases) else None
                kk += 1
                if basis is None or basis.numel() == 0:
                    continue
                B = basis.to(device=param.grad.device, dtype=param.grad.dtype)
                g = param.grad.view(param.grad.shape[0], -1)
                g_proj = g - (g @ B) @ B.t()
                param.grad.copy_(g_proj.view_as(param.grad))

    # ---------- EWC helpers ----------
    def _ewc_penalty(self, model: nn.Module):
        if self._fisher is None or self._ewc_checkpoint is None:
            return torch.tensor(0.0, device=self._device)
        pen = None
        for name, p in self._iter_trainable(model):
            ref = self._ewc_checkpoint.get(name)
            fisher = self._fisher.get(name)
            if ref is None or fisher is None:
                continue
            diff = p - ref.to(device=p.device, dtype=p.dtype)
            fisher_d = fisher.to(device=p.device, dtype=p.dtype)
            term = (fisher_d * diff.pow(2)).sum()
            pen = term if pen is None else pen + term
        if pen is None:
            return torch.tensor(0.0, device=self._device)
        return 0.5 * self._ewc_lambda * pen

    def _compute_fisher(self, loader, model: nn.Module) -> Dict[str, torch.Tensor]:
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        total = 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= self._ewc_max_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            model.zero_grad()
            with torch.enable_grad():
                logits = model(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                loss.backward()

            for name, p in self._iter_trainable(model):
                if p.grad is None:
                    continue
                g2 = (p.grad.detach().cpu() ** 2)
                fisher[name] = g2 if name not in fisher else fisher[name] + g2
            total += 1

        if total > 0:
            for name in fisher:
                fisher[name] = (fisher[name] / float(total)).clamp_min(self._ewc_eps)

        if self._fisher is not None:
            for name, new_f in fisher.items():
                old = self._fisher.get(name)
                self._fisher[name] = new_f if old is None else self._ewc_gamma * old + new_f
            for name, old in self._fisher.items():
                if name not in fisher:
                    self._fisher[name] = self._ewc_gamma * old
        else:
            self._fisher = fisher

        logger.info("[GPM_EWC] Fisher estimated over %d batches", total)
        return self._fisher

    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network

