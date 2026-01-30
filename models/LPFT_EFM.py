import copy
import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.inc_net import IncrementalNet
from models.baseLearner import BaseLearner
from utils.toolkit import tensor2numpy


class Learner(BaseLearner):
    """LP-FT with EFM regularization (optional in both stages).

    Structure mirrors models/finetune.py and models/LinearProbe_Finetune.py:
      - public API: incremental_train, finetune_all_data
      - internal phases: _init_train, _update_representation, _full_finetune
      - optimizer backends: sgd | sam | cflat | gam

    Additions:
      - Stage-A (LP head-only) optional EFM-Trace smoothing on the readout
      - Stage-B (FT full-network) optional EFM trust-region penalty on feature shift δf
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

        # Optimizer type & hyper-params (kept identical to finetune.py)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()
        if self._optimizer_type == "sam":
            self._sam_rho = float(args.get("sam_rho", 0.05))
            self._sam_adaptive = bool(args.get("sam_adaptive", False))
        elif self._optimizer_type == "cflat":
            self._cflat_rho = float(args.get("cflat_rho", 0.2))
            self._cflat_lambda = float(args.get("cflat_lambda", 0.2))
            self._cflat_adaptive = bool(args.get("cflat_adaptive", False))
            self._cflat_perturb_eps = float(args.get("cflat_perturb_eps", 1e-12))
            self._cflat_grad_reduce = args.get("cflat_grad_reduce", "mean")
        elif self._optimizer_type == "gam":
            self._gam_adaptive = bool(args.get("gam_adaptive", False))
            self._gam_grad_reduce = args.get("gam_grad_reduce", "mean")
            self._gam_perturb_eps = float(args.get("gam_perturb_eps", 1e-12))
            self._gam_grad_rho = float(args.get("gam_grad_rho", 0.2))
            self._gam_grad_norm_rho = float(args.get("gam_grad_norm_rho", 0.2))
            self._gam_beta1 = float(args.get("gam_grad_beta_1", 1.0))
            self._gam_beta2 = float(args.get("gam_grad_beta_2", 1.0))
            self._gam_beta3 = float(args.get("gam_grad_beta_3", 1.0))
            self._gam_gamma = float(args.get("gam_grad_gamma", 0.1))
        elif self._optimizer_type == "rwp":
            self._rwp_std = float(args.get("rwp_std", 0.01))
            self._rwp_eta = float(args.get("rwp_eta", 1.0))
            self._rwp_beta = float(args.get("rwp_beta", 0.9))

        # LP-FT head-stage epochs (re-using LP learner defaults)
        self._lp_init_head_epochs = int(args.get("lp_init_head_epochs", args.get("lp_head_epochs", 5)))
        self._lp_update_head_epochs = int(args.get("lp_update_head_epochs", args.get("lp_head_epochs", 5)))
        self._lp_full_head_epochs = int(args.get("lp_full_head_epochs", args.get("lp_head_epochs", 5)))

        # Optional LR overrides for LP stage
        # Optional LR overrides for LP stage; if None, reuse stage LR
        self._lp_init_lr = args.get("lp_init_lr", None)
        self._lp_init_wd = args.get("lp_init_wd", None)
        self._lp_init_moment = args.get("lp_init_moment", None)


        self._lp_update_lr = args.get("lp_update_lr", None)
        self._lp_update_wd = args.get("lp_update_wd", None)
        

        self._lp_full_lr = args.get("lp_full_lr", None)
        self._lp_full_wd = args.get("lp_full_wd", None)

        # EFM switches & hyper-params (incremental extension)
        self._efm_lp_enable = bool(args.get("efm_lp_enable", False))
        self._efm_ft_enable = bool(args.get("efm_ft_enable", False))
        self._efm_lambda_lp = float(args.get("efm_lambda_lp", 0.0))
        self._efm_lambda_ft = float(args.get("efm_lambda_ft", 0.0))
        self._efm_eta = float(args.get("efm_eta", 0.0))  # damping for FT trust-region
        self._efm_tau = float(args.get("efm_tau", 1.0))  # temperature for softmax probs
        # subset policy: 'auto' (new when incremental, else all) | 'new' | 'all'
        self._efm_subset = str(args.get("efm_subset", "auto")).lower()

        # teacher network captured after LP stage for FT trust-region penalty
        self._lp_teacher = None

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- Public training entrypoints ----------
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
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
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

    def finetune_all_data(self, data_manager):
        self._cur_task = 0
        self._known_classes = 0
        self._total_classes = data_manager.nb_classes
        self._network.update_fc(self._total_classes)

        full_idx = np.arange(0, self._total_classes)
        train_dataset = data_manager.get_dataset(full_idx, source="train", mode="train")
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args.get("train_num_workers", 8),
        )
        test_dataset = data_manager.get_dataset(full_idx, source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args.get("train_num_workers", 8),
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.to(self._device)
        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage="full")
        epochs = int(self.args.get("full_epochs", 20))
        # lr = optimizer.param_groups[0]["lr"] 
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("milestones", []),
            gamma=float(self.args.get("lrate_decay", 1.0)),
            T_max=epochs,
            eta_min=self.args.get("min_lr", 0.0),
        )
        
        self._full_finetune(self.train_loader, self.test_loader, optimizer, scheduler, epochs)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._known_classes = self._total_classes

    # ---------- Phase orchestration ----------
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            params = [p for p in self._network.parameters() if p.requires_grad]
            optimizer = self._build_optimizer(params, stage="init")
            # lr = optimizer.param_groups[0]["lr"] 
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=self.args.get("init_epoch", 20),
                eta_min=self.args.get("min_lr", 0.0),
                # eta_min=self.args.get("min_lr", 0.01*lr),
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            params = [p for p in self._network.parameters() if p.requires_grad]
            optimizer = self._build_optimizer(params, stage="update")
            # lr = optimizer.param_groups[0]["lr"] 
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=self.args.get("epochs", 20),
                eta_min=self.args.get("min_lr", 0.0)
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _freeze_backbone(self):
        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        for p in net.backbone.parameters():
            p.requires_grad = False

    def _unfreeze_all(self):
        for p in self._network.parameters():
            p.requires_grad = True

    def _head_parameters(self):
        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        return [p for p in net.fc.parameters() if p.requires_grad]

    def _get_fc(self, from_teacher: bool = False):
        if from_teacher:
            return self._lp_teacher.fc
        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        return net.fc

    def _build_head_optimizer(self, stage: str, override_lr: float | None, override_wd: float | None = None):
        head_params = self._head_parameters()
        opt = self._build_optimizer(head_params, stage=stage)
        # Override LR to match probe settings if provided
        if override_lr is not None:
            for g in opt.param_groups:
                g["lr"] = float(override_lr)
        # Override weight decay for head-only training to align with probe_fit_wd
        if override_wd is not None:
            for g in opt.param_groups:
                g["weight_decay"] = float(override_wd)
        return opt

    def _capture_lp_teacher(self):
        """Snapshot a frozen teacher after LP stage for FT trust-region penalty."""
        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        teacher = net.copy()
        teacher.to(self._device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        self._lp_teacher = teacher

    # ---------- Training stages ----------
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        # Stage A: LP head-only
        if self._lp_init_head_epochs > 0:
            self._freeze_backbone()
            # head_opt = optimizer
            head_opt = self._build_head_optimizer(
                stage="init",
                override_lr=self._lp_init_lr,
                override_wd=self._lp_init_wd,
            )
            # Force constant LR for head-only phase to mirror probe behavior
            head_sched = self.build_scheduler(
                head_opt,
                policy="constant",  # fallback to constant scheduler in BaseLearner
            )
            # head_sched = scheduler
            self._run_epoch_loop(
                epochs=self._lp_init_head_epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=head_opt,
                scheduler=head_sched,
                loss_on_new_classes=False,
                phase="lp_head",
            )
            # Capture teacher after LP stage
            if self._efm_ft_enable:
                self._capture_lp_teacher()
            self._unfreeze_all()

        # Stage B: FT full-network
        epochs = int(self.args.get("init_epoch", 0))
        if epochs <= 0:
            return
        self._run_epoch_loop(
            epochs=epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_on_new_classes=False,
            phase="ft_full",
        )

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        # Stage A: LP head-only on new classes
        if self._lp_update_head_epochs > 0:
            self._freeze_backbone()
            head_opt = self._build_head_optimizer(
                stage="update",
                override_lr=self._lp_update_lr,
                override_wd=self._lp_update_wd,
            )
            # Force constant LR for head-only phase
            head_sched = self.build_scheduler(
                head_opt,
                policy="constant"
            )
            # head_opt = optimizer
            # head_sched = scheduler
            self._run_epoch_loop(
                epochs=self._lp_update_head_epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=head_opt,
                scheduler=head_sched,
                loss_on_new_classes=True,
                phase="lp_head",
            )
            if self._efm_ft_enable:
                self._capture_lp_teacher()
            self._unfreeze_all()

        # Stage B: FT full-network on new classes
        epochs = int(self.args.get("epochs", 0))
        if epochs <= 0:
            return
        self._run_epoch_loop(
            epochs=epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_on_new_classes=True,
            phase="ft_full",
        )

    def _full_finetune(self, train_loader, test_loader, optimizer, scheduler, epochs):
        # Stage A: LP head-only on all classes
        if self._lp_full_head_epochs > 0:
            self._freeze_backbone()
            head_opt = self._build_head_optimizer(
                stage="full",
                override_lr=self._lp_full_lr,
                override_wd=self._lp_full_wd
            )
            # Force constant LR for head-only phase
            head_sched = self.build_scheduler(
                head_opt,
                policy="constant",
            )
            # head_opt = optimizer
            # head_sched = scheduler
            self._run_epoch_loop(
                epochs=self._lp_full_head_epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=head_opt,
                scheduler=head_sched,
                loss_on_new_classes=False,
                phase="lp_head",
            )
            if self._efm_ft_enable:
                self._capture_lp_teacher()
            self._unfreeze_all()

        # Stage B: FT full-network
        if epochs <= 0:
            return
        self._run_epoch_loop(
            epochs=epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_on_new_classes=False,
            phase="ft_full",
        )

    # ---------- EFM helpers ----------
    def _select_subset(self, loss_on_new_classes: bool):
        policy = self._efm_subset
        if policy == "auto":
            return (self._known_classes, self._total_classes) if loss_on_new_classes else (0, self._total_classes)
        if policy == "new":
            return (self._known_classes, self._total_classes)
        return (0, self._total_classes)

    def _efm_trace_penalty(self, logits: torch.Tensor, fc: nn.Module, start: int, end: int) -> torch.Tensor:
        """Compute EFM trace penalty on readout: tr(W^T (Diag(p)-pp^T) W).
        logits: [B, C]
        fc.weight: [C, D]
        subset classes in [start:end)
        returns scalar tensor
        """
        if end <= start:
            return logits.new_zeros(())
        logits_sub = logits[:, start:end] / max(self._efm_tau, 1e-8)
        p = F.softmax(logits_sub, dim=1)  # [B, C_sub]
        W = fc.weight[start:end]  # [C_sub, D]
        # term1 = sum_c p_c ||w_c||^2
        w_norm2 = (W ** 2).sum(dim=1)  # [C_sub]
        term1 = (p * w_norm2.unsqueeze(0)).sum(dim=1)  # [B]
        # term2 = || W^T p ||^2
        w_bar = torch.matmul(p, W)  # [B, D]
        term2 = (w_bar ** 2).sum(dim=1)  # [B]
        efm_trace = (term1 - term2).mean()
        return efm_trace

    def _efm_trust_region(self, inputs: torch.Tensor, outputs: dict, loss_on_new_classes: bool) -> torch.Tensor:
        """EFM trust-region style quadratic penalty on feature shift δf.
        Uses LP teacher network snapshot and its head to compute p_prev and weighting.
        returns scalar tensor
        """
        if self._lp_teacher is None:
            return outputs["logits"].new_zeros(())

        # current features/logits
        f_cur = outputs.get("features", None)
        if f_cur is None:
            return outputs["logits"].new_zeros(())

        with torch.no_grad():
            t_out = self._lp_teacher(inputs)
            f_prev = t_out.get("features", None)
            if f_prev is None:
                return outputs["logits"].new_zeros(())

        delta_f = f_cur - f_prev  # [B, D]
        start, end = self._select_subset(loss_on_new_classes)
        # teacher head & probabilities
        W_prev = self._lp_teacher.fc.weight[start:end]  # [C_sub, D]
        with torch.no_grad():
            logits_prev = t_out["logits"][:, start:end] / max(self._efm_tau, 1e-8)
            p_prev = F.softmax(logits_prev, dim=1)  # [B, C_sub]

        q = torch.matmul(delta_f, W_prev.T)  # [B, C_sub]
        term1 = (p_prev * (q ** 2)).sum(dim=1)  # [B]
        term2 = ((p_prev * q).sum(dim=1)) ** 2  # [B]
        quad = term1 - term2  # [B]

        if self._efm_eta > 0.0:
            quad = quad + self._efm_eta * (delta_f ** 2).sum(dim=1)

        return quad.mean()

    # ---------- Shared training loop ----------
    def _run_epoch_loop(self, *, epochs, train_loader, test_loader, optimizer, scheduler, loss_on_new_classes: bool, phase: str):
        prog_bar = tqdm(range(epochs))

        def _compute_total_loss(inputs, targets, outputs):
            logits = outputs["logits"]
            if loss_on_new_classes:
                fake_targets = targets - self._known_classes
                ce = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
            else:
                ce = F.cross_entropy(logits, targets)

            loss = ce
            # LP-stage EFM trace smoothing (readout only)
            if self._efm_lp_enable and phase == "lp_head":
                start, end = self._select_subset(loss_on_new_classes)
                fc = self._get_fc(from_teacher=False)
                efm_trace = self._efm_trace_penalty(logits, fc, start, end)
                loss = loss + float(self._efm_lambda_lp) * efm_trace

            # FT-stage EFM trust-region (feature shift)
            if self._efm_ft_enable and phase == "ft_full":
                tr_pen = self._efm_trust_region(inputs, outputs, loss_on_new_classes)
                loss = loss + float(self._efm_lambda_ft) * tr_pen

            return loss

        for epoch in prog_bar:
            self._network.train()
            # Keep backbone in eval during head-only phase to mirror probe extraction
            if phase == "lp_head":
                net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
                if hasattr(net, "backbone"):
                    net.backbone.eval()
                if hasattr(net, "fc"):
                    try:
                        net.fc.train(True)
                    except Exception:
                        pass
            losses = 0.0
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                if self._optimizer_type == "cflat":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        loss = _compute_total_loss(inputs, targets, outputs)
                        loss.backward()
                        return outputs["logits"], [loss]

                    _, loss_list = optimizer.step(closure=closure)
                    loss_value = torch.stack([loss_term.detach() for loss_term in loss_list]).sum()
                    losses += loss_value.item()
                    with torch.no_grad():
                        logits = self._network(inputs)["logits"]

                elif self._optimizer_type == "gam":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        loss = _compute_total_loss(inputs, targets, outputs)
                        loss_value = loss.detach()
                        loss.backward()
                        return outputs, loss_value

                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()
                elif self._optimizer_type == "rwp":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        loss = _compute_total_loss(inputs, targets, outputs)
                        loss_value = loss.detach()
                        loss.backward()
                        return outputs, loss_value

                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()

                else:
                    optimizer.zero_grad()
                    outputs = self._network(inputs)
                    loss = _compute_total_loss(inputs, targets, outputs)

                    if self._optimizer_type == "sam":
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        outputs2 = self._network(inputs)
                        loss2 = _compute_total_loss(inputs, targets, outputs2)
                        loss2.backward()
                        optimizer.second_step(zero_grad=True)
                        losses += loss2.item()
                        logits = outputs2["logits"].detach()
                    else:
                        loss.backward()
                        optimizer.step()
                        losses += loss.item()
                        logits = outputs["logits"].detach()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            # periodic eval as in finetune.py
            if (epoch % 5 == 4):
                test_acc = self._compute_accuracy(self._network, test_loader)
                desc = (
                    f"Task {self._cur_task}, {phase} Epoch {epoch+1}/{epochs} => "
                    f"Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}"
                )
            else:
                desc = (
                    f"Task {self._cur_task}, {phase} Epoch {epoch+1}/{epochs} => "
                    f"Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
                )
            prog_bar.set_description(desc)
        logging.info(desc)
