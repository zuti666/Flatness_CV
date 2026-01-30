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
    """LP-FT training: linear-probe head, then fine-tune the full network.

    Architecture and public methods mirror models/finetune.py so it can be
    drop-in used by the existing trainer. Internally, _init_train/_update_representation
    and _full_finetune perform a 2-stage routine:
      1) Freeze backbone, train only the classification head for a few epochs
      2) Unfreeze all parameters, fine-tune full network with the usual epochs
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

        # optimizer type and hyperparams consistent with finetune.py
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

        # LP stage knobs (incremental extension; defaults are modest to be safe)
        self._lp_init_head_epochs = int(args.get("lp_init_head_epochs", args.get("lp_head_epochs", 5)))
        self._lp_update_head_epochs = int(args.get("lp_update_head_epochs", args.get("lp_head_epochs", 5)))
        self._lp_full_head_epochs = int(args.get("lp_full_head_epochs", args.get("lp_head_epochs", 5)))

        # Optional LR overrides for LP stage; if None, reuse stage LR
        self._lp_init_lr = args.get("lp_init_lr", None)
        self._lp_init_wd = args.get("lp_init_wd", None)
        self._lp_init_moment = args.get("lp_init_moment", None)

        self._lp_update_lr = args.get("lp_update_lr", None)
        self._lp_update_wd= args.get("lp_update_wd", None)


        self._lp_full_lr = args.get("lp_full_lr", None)
        self._lp_full_wd = args.get("lp_full_wd", None)

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
        """Finetune the model on the full dataset (all classes).

        For LP-FT: head-only training on all classes, then full-network fine-tuning.
        """
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
            T_max= epochs,
            eta_min=self.args.get("min_lr", 0.0)
        )
        self._full_finetune(self.train_loader, self.test_loader, optimizer, scheduler, epochs)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._known_classes = self._total_classes

    # ---------- Internal training orchestration ----------
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
                eta_min=self.args.get("min_lr", 0.0) #1*lr
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    # ---------- LP-FT: stage A (head-only) helpers ----------
    def _freeze_backbone(self):
        if isinstance(self._network, nn.DataParallel):
            backbone = self._network.module.backbone
        else:
            backbone = self._network.backbone
        for p in backbone.parameters():
            p.requires_grad = False

    def _unfreeze_all(self):
        for p in self._network.parameters():
            p.requires_grad = True

    def _head_parameters(self):
        if isinstance(self._network, nn.DataParallel):
            return [p for p in self._network.module.fc.parameters() if p.requires_grad]
        return [p for p in self._network.fc.parameters() if p.requires_grad]

    def _build_head_optimizer(self, stage: str, override_lr: float | None,override_moment: float | None = None, override_wd: float | None = None):
        head_params = self._head_parameters()
        opt = self._build_optimizer(head_params, stage=stage)
        # Override LR if provided
        if override_lr is not None:
            for g in opt.param_groups:
                g["lr"] = float(override_lr)
        # 
        if override_moment is not None:
            for g in opt.param_groups:
                g["moment"] = float(override_moment)

        # Override WD for head-only phase to align with probe
        if override_wd is not None:
            for g in opt.param_groups:
                g["weight_decay"] = float(override_wd)
        
        return opt

    # ---------- LP-FT: Stage B integrated into these entrypoints ----------
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        # Stage A: head-only training
        if self._lp_init_head_epochs > 0:
            self._freeze_backbone()
            
            head_opt = self._build_head_optimizer(
                stage="init",
                override_lr=self._lp_init_lr,
                override_wd=self._lp_init_wd,
            )
            # Force constant LR for head-only phase to mirror probe
            head_sched = self.build_scheduler(
                head_opt,
                policy="constant"
            )
            # head_opt = optimizer
            head_sched = scheduler
            self._run_epoch_loop(
                epochs=self._lp_init_head_epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=head_opt,
                scheduler=head_sched,
                loss_on_new_classes=False,
            )
            self._unfreeze_all()

        # Stage B: full-network training (same as finetune _init_train)
        epochs = int(self.args["init_epoch"]) if "init_epoch" in self.args else 0
        if epochs <= 0:
            return
        self._run_epoch_loop(
            epochs=epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_on_new_classes=False,
        )

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        # Stage A: head-only training for new classes
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
            )
            self._unfreeze_all()

        # Stage B: full-network training for new classes
        epochs = int(self.args["epochs"]) if "epochs" in self.args else 0
        if epochs <= 0:
            return
        self._run_epoch_loop(
            epochs=epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_on_new_classes=True,
        )

    def _full_finetune(self, train_loader, test_loader, optimizer, scheduler, epochs):
        # Stage A: head-only training on all classes
        if self._lp_full_head_epochs > 0:
            self._freeze_backbone()
            head_opt = self._build_head_optimizer(
                stage="full",
                override_lr=self._lp_full_lr,
                override_wd=self._lp_full_wd,
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
            )
            self._unfreeze_all()

        # Stage B: full-network fine-tuning on all classes (same as finetune _full_finetune)
        if epochs <= 0:
            return
        self._run_epoch_loop(
            epochs=epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_on_new_classes=False,
        )

    # ---------- Shared training loop with optimizer_type branches ----------
    def _run_epoch_loop(self, *, epochs, train_loader, test_loader, optimizer, scheduler, loss_on_new_classes: bool):
        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            self._network.train()
            # If backbone is frozen (head-only phase), keep backbone in eval to match probe
            try:
                net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
                bb = getattr(net, "backbone", None)
                if bb is not None:
                    # Check if all backbone params are frozen
                    all_frozen = True
                    for p in bb.parameters():
                        if getattr(p, "requires_grad", True):
                            all_frozen = False
                            break
                    if all_frozen:
                        bb.eval()
                        if hasattr(net, "fc"):
                            try:
                                net.fc.train(True)
                            except Exception:
                                pass
            except Exception:
                pass
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
                elif self._optimizer_type == "rwp":
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
                        optimizer.step()
                        losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            # show test every 5 epochs to keep parity with finetune.py UX
            if (epoch % 3 == 0):
                test_acc = self._compute_accuracy(self._network, test_loader)
                if loss_on_new_classes:
                    desc = "Task {}, LP-FT-{} Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                        self._cur_task, "update", epoch + 1, epochs, losses / len(train_loader), train_acc, test_acc
                    )
                else:
                    phase = "init" if self._cur_task == 0 else "full"
                    desc = "Task {}, LP-FT-{} Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                        self._cur_task, phase, epoch + 1, epochs, losses / len(train_loader), train_acc, test_acc
                    )
            else:
                if loss_on_new_classes:
                    desc = "Task {}, LP-FT-update Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                        self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc
                    )
                else:
                    phase = "init" if self._cur_task == 0 else "full"
                    desc = "Task {}, LP-FT-{} Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                        self._cur_task, phase, epoch + 1, epochs, losses / len(train_loader), train_acc
                    )

            prog_bar.set_description(desc)
        logging.info(desc)
