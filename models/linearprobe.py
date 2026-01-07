import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from optimer.util import enable_running_stats, disable_running_stats, generate_pertubation

from models.baseLearner import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy
from types import SimpleNamespace

class Learner(BaseLearner):
    """Pure Linear Probe: freeze backbone; train only classifier head.

    - incremental_train: for each task, expand the classifier and train only
      the new class rows while keeping the backbone frozen.
    - finetune_all_data: train a single head on the full dataset with the
      frozen backbone.

    Optimizer/scheduler handling follows models/finetune.py. We also support
    SAM/C-Flat/GAM like other learners via args["optimizer_type"].
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

        # Optimizer type and optional hyper-params (keep parity with other learners)
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
            # Core GAM flags
            self._gam_adaptive = bool(args.get("gam_adaptive", False))
            self._gam_grad_reduce = args.get("gam_grad_reduce", "mean")
            self._gam_perturb_eps = float(args.get("gam_perturb_eps", 1e-12))
            # Two perturbation radii (ρ for loss-grad step; ρ' for norm-ascent step)
            self._gam_grad_rho = float(args.get("gam_grad_rho", 0.2))
            self._gam_grad_norm_rho = float(args.get("gam_grad_norm_rho", 0.2))
            # Gradient decomposition weights used in optimer.gam.GAM.gradient_decompose
            self._gam_beta1 = float(args.get("gam_grad_beta_1", 1.0))
            self._gam_beta2 = float(args.get("gam_grad_beta_2", 1.0))
            self._gam_beta3 = float(args.get("gam_grad_beta_3", 1.0))
            self._gam_gamma = float(args.get("gam_grad_gamma", 0.1))

            # Pack into a namespace for downstream optimizers (BaseLearner can read self._gam_args)
            self._gam_args = SimpleNamespace(
                # decomposition weights
                grad_beta_1=self._gam_beta1,
                grad_beta_2=self._gam_beta2,
                grad_beta_3=self._gam_beta3,
                grad_gamma=self._gam_gamma,
                # radii (optional convenience)
                grad_rho=self._gam_grad_rho,
                grad_norm_rho=self._gam_grad_norm_rho,
                # misc flags (optional convenience)
                adaptive=self._gam_adaptive,
                perturb_eps=self._gam_perturb_eps,
                grad_reduce=str(self._gam_grad_reduce),
            )
        elif self._optimizer_type == "arwp":
            self._rwp_std = float(args.get("rwp_std", 0.01))
            self._rwp_eta = float(args.get("rwp_eta", 1.0))
            self._rwp_beta = float(args.get("rwp_beta", 0.9))
            self._rwp_std_follow_lr = bool(args.get("rwp_std_follow_lr", False))
            self._rwp_range = str(args.get("rwp_range", "lora"))
        elif self._optimizer_type == "rwp":
            self._rwp_std = float(args.get("rwp_std", 0.01))
            self._rwp_eta = float(args.get("rwp_eta", 1.0))
            self._rwp_beta = float(args.get("rwp_beta", 0.9))
            self._rwp_lambda = float(args.get("rwp_lambda", 0.5))
            self._rwp_std_follow_lr = bool(args.get("rwp_std_follow_lr", False))
            self._rwp_range = str(args.get("rwp_range", "lora"))
            self._rwp_range = str(args.get("rwp_range", "lora"))
            
            self.rwp_noise_type = str(self.args.get("rwp_noise_type", "Gauss_standard"))
            if "fisher" in  self.rwp_noise_type:
                self._rwp_fisher = {}


        # Optional LR/WD overrides for head-only training (stage-specific)
        self._lp_init_lr = args.get("lp_init_lr", args.get("lp_lr", None))
        self._lp_init_wd = args.get("lp_init_wd", args.get("lp_wd", None))
        self._lp_update_lr = args.get("lp_update_lr", args.get("lp_lr", None))
        self._lp_update_wd = args.get("lp_update_wd", args.get("lp_wd", None))
        self._lp_full_lr = args.get("lp_full_lr", args.get("lp_lr", None))
        self._lp_full_wd = args.get("lp_full_wd", args.get("lp_wd", None))

    # ---------------- Lifecycle ----------------
    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        # cache data_manager for later NME prototype computation
        # self._data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{} (LinearProbe, backbone frozen)".format(self._known_classes, self._total_classes))

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

        # Compute class-means over all seen classes for NME evaluation (Class-IL)
        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as _nme_exc:  # pylint: disable=broad-except
            logging.exception("[Finetune][NME] Failed to compute class means: %s", _nme_exc)


    def finetune_all_data(self, data_manager):
        """Train a single linear head on the full dataset with frozen backbone."""
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
        # Freeze backbone, optimize head only
        self._freeze_backbone()
        head_opt = self._build_head_optimizer(
            stage="full", override_lr=self._lp_full_lr, override_wd=self._lp_full_wd
        )
        if self._optimizer_type == "arwp":
            try:
                self._rwp_lr0 = float(head_opt.param_groups[0]["lr"])  # base LR for std scaling
            except Exception:
                self._rwp_lr0 = None
        epochs = int(self.args.get("full_epochs", 20))
        head_sched = self.build_scheduler(head_opt, policy=self.args.get("scheduler", "constant"), T_max=epochs, eta_min=self.args.get("min_lr", 0.0))

        self._run_epoch_loop(
            epochs=epochs,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            optimizer=head_opt,
            scheduler=head_sched,
            loss_on_new_classes=False,
        )

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._known_classes = self._total_classes

    # ---------------- Internals ----------------
    def _freeze_backbone(self):
        if isinstance(self._network, nn.DataParallel):
            backbone = self._network.module.backbone
        else:
            backbone = self._network.backbone
        for p in backbone.parameters():
            p.requires_grad = False

    def _unfreeze_head_only(self):
        # Ensure classifier params require grad
        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        for p in net.fc.parameters():
            p.requires_grad = True

    def _head_parameters(self):
        if isinstance(self._network, nn.DataParallel):
            return [p for p in self._network.module.fc.parameters() if p.requires_grad]
        return [p for p in self._network.fc.parameters() if p.requires_grad]

    def _build_head_optimizer(self, stage: str, override_lr=None, override_wd=None):
        params = self._head_parameters()
        opt = self._build_optimizer(params, stage=stage)
        if override_lr is not None:
            for g in opt.param_groups:
                g["lr"] = float(override_lr)
        if override_wd is not None:
            for g in opt.param_groups:
                g["weight_decay"] = float(override_wd)
        return opt

    # Use base optimizer for true RWP (mixing in-loop), otherwise defer
    # def _build_optimizer(self, params, stage):
    #     if getattr(self, "_optimizer_type", "").lower() == "rwp":
    #         # Head-only training; respect provided LR/WD overrides via caller
    #         lr = float(self.args.get("lp_lr", self.args.get("lr", 0.01)))
    #         weight_decay = float(self.args.get("lp_wd", self.args.get("weight_decay", 0.0)))
    #         momentum = float(self.args.get("momentum", 0.0))
    #         base_name = str(self.args.get("optimizer", "sgd")).lower()
    #         if base_name == "adam":
    #             return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    #         if base_name == "adamw":
    #             return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    #         return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    #     return super()._build_optimizer(params, stage)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        # Always freeze backbone, train head only
        self._freeze_backbone()
        self._unfreeze_head_only()

        if self._cur_task == 0:
            head_opt = self._build_head_optimizer(stage="init", override_lr=self._lp_init_lr, override_wd=self._lp_init_wd)
            if self._optimizer_type in {"arwp", "rwp"}:
                try:
                    self._rwp_lr0 = float(head_opt.param_groups[0]["lr"])  # base LR for std scaling
                except Exception:
                    self._rwp_lr0 = None
            epochs = int(self.args.get("init_epoch", 0))
            sched = self.build_scheduler(head_opt, policy=self.args.get("scheduler", "constant"), T_max=epochs, eta_min=self.args.get("min_lr", 0.0))
            if epochs > 0:
                self._run_epoch_loop(
                    epochs=epochs,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    optimizer=head_opt,
                    scheduler=sched,
                    loss_on_new_classes=False,
                )
        else:
            head_opt = self._build_head_optimizer(stage="update", override_lr=self._lp_update_lr, override_wd=self._lp_update_wd)
            if self._optimizer_type in {"arwp", "rwp"}:
                try:
                    self._rwp_lr0 = float(head_opt.param_groups[0]["lr"])  # base LR for std scaling
                except Exception:
                    self._rwp_lr0 = None
            epochs = int(self.args.get("epochs", 0))
            sched = self.build_scheduler(head_opt, policy=self.args.get("scheduler", "constant"), T_max=epochs, eta_min=self.args.get("min_lr", 0.0))
            if epochs > 0:
                self._run_epoch_loop(
                    epochs=epochs,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    optimizer=head_opt,
                    scheduler=sched,
                    loss_on_new_classes=True,
                )

        # After training the (frozen-backbone) head for this task, compute
        # per-class feature means for NME evaluation like other learners.
        # try:
        #     dm = getattr(self, "_data_manager", None)
        #     if dm is not None:
        #         self.compute_all_seen_class_means(dm)
        # except Exception as _nme_exc:  # pylint: disable=broad-except
        #     logging.exception("[LinearProbe][NME] Failed to compute class means: %s", _nme_exc)

    # -----------------------------
    # Prototype/NME helper (same logic as finetune.py)
    # -----------------------------
    def compute_all_seen_class_means(self, data_manager) -> None:
        if getattr(self, "_total_classes", 0) <= 0:
            return
        nb_classes = int(self._total_classes)
        feat_dim = int(self.feature_dim)
        class_means = np.zeros((nb_classes, feat_dim), dtype=np.float64)

        bs = int(self.args.get("eval_batch_size", self.args.get("batch_size", 128)))
        nw = int(self.args.get("eval_num_workers", 8))

        for c in range(nb_classes):
            try:
                dataset_c = data_manager.get_dataset(np.arange(c, c + 1), source="train", mode="test")
                loader_c = DataLoader(dataset_c, batch_size=bs, shuffle=False, num_workers=nw)
                vectors, _ = self._extract_vectors(loader_c)
                if vectors.size == 0:
                    continue
                # L2-normalize features then mean; normalize prototype as well
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + 1e-12)).T
                mu = np.mean(vectors, axis=0)
                norm = np.linalg.norm(mu) + 1e-12
                class_means[c, :] = (mu / norm).astype(np.float64)
            except Exception as _exc:  # pylint: disable=broad-except
                logging.exception("[LinearProbe][NME] Failed to compute mean for class %d: %s", c, _exc)

        self._class_means = class_means

    def _run_epoch_loop(self, *, epochs, train_loader, test_loader, optimizer, scheduler, loss_on_new_classes: bool):
        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            self._network.train()
            # Keep backbone in eval if it's frozen, to mimic probe behavior
            
            net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
            bb = getattr(net, "backbone", None)
            if bb is not None:
                all_frozen = True
                for p in bb.parameters():
                    if getattr(p, "requires_grad", True):
                        all_frozen = False
                        break
                if all_frozen:
                    bb.eval()
                    if hasattr(net, "fc"):
                        net.fc.train(True)
                        
            

            losses = 0.0
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                if loss_on_new_classes:
                    fake_targets = targets - self._known_classes

                if self._optimizer_type == "cflat":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
                        if loss_on_new_classes:
                            loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                        else:
                            loss = F.cross_entropy(logits, targets)
                        loss.backward()
                        return outputs, [loss]

                    _, loss_list = optimizer.step(closure=closure)
                    loss_value = torch.stack([loss_term.detach() for loss_term in loss_list]).sum()
                    losses += loss_value.item()
                    with torch.no_grad():
                        logits = self._network(inputs)["logits"]

                elif self._optimizer_type == "gam":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
                        if loss_on_new_classes:
                            loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                        else:
                            loss = F.cross_entropy(logits, targets)
                        loss_value = loss.detach()
                        loss.backward()
                        return outputs, loss_value

                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()
                elif self._optimizer_type == "arwp":
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else float(optimizer.param_groups[0]["lr"]) 
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"]) 
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr) or cur_lr)
                        scale = (cur_lr / base_lr) if base_lr > 0 else 1.0
                        optimizer.std = float(self._rwp_std) * scale
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
                        if loss_on_new_classes:
                            loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                        else:
                            loss = F.cross_entropy(logits, targets)
                        loss_value = loss.detach()
                        loss.backward()
                        return outputs, loss_value

                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()
                elif self._optimizer_type == "rwp":
                    # RWP for LP: apply to head params (requires_grad)
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else float(optimizer.param_groups[0]["lr"])  # type: ignore
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"])  # type: ignore
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr) or cur_lr)
                        rwp_std = float(self._rwp_std) * ((cur_lr / base_lr) if base_lr > 0 else 1.0)
                    else:
                        rwp_std = float(self._rwp_std)

                    # clean
                    enable_running_stats(self._network)
                    optimizer.zero_grad()
                    outputs = self._network(inputs)
                    logits_clean = outputs["logits"]
                    loss_clean = F.cross_entropy(logits_clean[:, self._known_classes :], fake_targets) if loss_on_new_classes else F.cross_entropy(logits_clean, targets)
                    loss_clean.backward()
                    g0 = {}
                    for name, p in self._network.named_parameters():
                        if p.requires_grad and (p.grad is not None):
                            g0[name] = p.grad.detach().clone()

                    # noisy
                    disable_running_stats(self._network)
                    noise_dict = {}
                    with torch.no_grad():
                        mode = str(self.args.get("rwp_noise_type", "mARWP_fisher"))
                        std_for_noise = float(self.args.get("noise_std", rwp_std))
                        for name, p in self._network.named_parameters():
                            if self._rwp_range == "lora":
                                if not p.requires_grad or p.numel() == 0:
                                    continue
                            fisher_param = getattr(self, "_rwp_fisher", {}).get(name, None)
                            e = generate_pertubation(p, pertubation_mode=self.rwp_noise_type, std=std_for_noise, fisher_param=fisher_param, fisher_scaler=float(self._rwp_eta))
                            p.data.add_(e)
                            noise_dict[name] = e

                    optimizer.zero_grad()
                    outputs_noisy = self._network(inputs)
                    logits_noisy = outputs_noisy["logits"]
                    loss_noisy = F.cross_entropy(logits_noisy[:, self._known_classes :], fake_targets) if loss_on_new_classes else F.cross_entropy(logits_noisy, targets)
                    loss_noisy.backward()

                    with torch.no_grad():
                        if hasattr(self, "_rwp_fisher"):
                            # self._rwp_fisher = {}
                            for name, p in self._network.named_parameters():
                                if p.requires_grad and (p.grad is not None):
                                    g2 = p.grad.detach() ** 2
                                    self._rwp_fisher[name] = g2 if name not in self._rwp_fisher else float(self._rwp_beta) * self._rwp_fisher[name] + g2

                    with torch.no_grad():
                        for name, p in self._network.named_parameters():
                            if name in noise_dict:
                                p.data.sub_(noise_dict[name])

                    lam = float(self._rwp_lambda)
                    for name, p in self._network.named_parameters():
                        if self._rwp_range == "lora":
                            if not p.requires_grad:
                                continue
                        if p.grad is not None:
                            g1 = p.grad.detach()
                            g0_n = g0.get(name, torch.zeros_like(g1))
                            p.grad.data.copy_(lam * g1 + (1.0 - lam) * g0_n)
                    optimizer.step()
                    logits = logits_clean.detach()
                    losses += (lam * float(loss_noisy.detach().item()) + (1.0 - lam) * float(loss_clean.detach().item()))
                elif self._optimizer_type == "arwp":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
                        if loss_on_new_classes:
                            loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                        else:
                            loss = F.cross_entropy(logits, targets)
                        loss_value = loss.detach()
                        loss.backward()
                        return outputs, loss_value

                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()

                else:
                    optimizer.zero_grad()
                    logits = self._network(inputs)["logits"]
                    if loss_on_new_classes:
                        loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    else:
                        loss = F.cross_entropy(logits, targets)

                    if self._optimizer_type == "sam":
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        logits = self._network(inputs)["logits"]
                        if loss_on_new_classes:
                            second_loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                        else:
                            second_loss = F.cross_entropy(logits, targets)
                        second_loss.backward()
                        optimizer.second_step(zero_grad=True)
                        losses += second_loss.item()
                    else:
                        loss.backward()
                        # Important: old-class rows receive zero grad thanks to slicing the logits
                        optimizer.step()
                        losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch % 3 == 0):
                test_acc = self._compute_accuracy(self._network, test_loader)
                phase = "update" if loss_on_new_classes else ("init" if self._cur_task == 0 else "full")
                desc = "Task {}, LP {} Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, phase, epoch + 1, epochs, losses / len(train_loader), train_acc, test_acc
                )
            else:
                phase = "update" if loss_on_new_classes else ("init" if self._cur_task == 0 else "full")
                desc = "Task {}, LP {} Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, phase, epoch + 1, epochs, losses / len(train_loader), train_acc
                )
            prog_bar.set_description(desc)
        logging.info(desc)
