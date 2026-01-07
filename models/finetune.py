import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from optimer.util import enable_running_stats, disable_running_stats, generate_pertubation
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.baseLearner import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from typing import Optional
from types import SimpleNamespace
num_workers = 8

from models_EFM.efm_trust import EFMTrustRegionHelper

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

        if args.get("efm_ft_enable", False):
            self._efm = EFMTrustRegionHelper(self.args)
            print("Init EFM")
        else:
            self._efm = None


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

        # --- GAM hyperparams (with safe defaults) ---
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
        
        # --- RWP hyperparams ---
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
            
            self.rwp_noise_type = str(self.args.get("rwp_noise_type", "Gauss_standard"))
            if "fisher" in  self.rwp_noise_type:
                self._rwp_fisher = {}

    def after_task(self):
        self._known_classes = self._total_classes

        # Release CUDA cache after each task to mitigate fragmentation/OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
    

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,      
            num_workers=self.args.get("train_num_workers", 8)
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.args["batch_size"], 
            shuffle=False, 
            num_workers=self.args.get("train_num_workers", 8)
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
        """Finetune the model on the full dataset (all classes at once)."""
        self._cur_task = 0
        self._known_classes = 0
        self._total_classes = data_manager.nb_classes
        self._network.update_fc(self._total_classes)

        full_class_indices = np.arange(0, self._total_classes)
        train_dataset = data_manager.get_dataset(
            full_class_indices, source="train", mode="train"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,      
            num_workers=self.args.get("train_num_workers", 8)
        )
        test_dataset = data_manager.get_dataset(
            full_class_indices, source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args.get("train_num_workers", 8)
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.to(self._device)
        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage="full")
        if self._optimizer_type in {"arwp", "rwp"}:
            try:
                self._rwp_lr0 = float(optimizer.param_groups[0]["lr"])  # base LR for std scaling
            except Exception:
                self._rwp_lr0 = None
        # lr = optimizer.param_groups[0]["lr"] 
        epochs = int(self.args.get("full_epochs"))
         
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("milestones", []),
            gamma=float(self.args.get("lrate_decay", 1.0)),
            T_max=epochs,
            eta_min=self.args.get("min_lr", 0.0),
            # eta_min=self.args.get("min_lr", 0.001*lr),
        )


        # 进行评估注释掉这一行
        self._full_finetune(self.train_loader, self.test_loader, optimizer, scheduler, epochs)


        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # After full finetuning, treat all classes as "known" for evaluation utilities
        self._known_classes = self._total_classes

        # Compute class means for all classes for NME evaluation
        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as _nme_exc:  # pylint: disable=broad-except
            logging.exception("[FinetuneAll][NME] Failed to compute class means: %s", _nme_exc)

    # -----------------------------
    # Prototype/NME helper (duplicated to avoid broad base changes)
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
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + 1e-12)).T
                mu = np.mean(vectors, axis=0)
                norm = np.linalg.norm(mu) + 1e-12
                class_means[c, :] = (mu / norm).astype(np.float64)
            except Exception as _exc:  # pylint: disable=broad-except
                logging.exception("[Finetune][NME] Failed to compute mean for class %d: %s", c, _exc)

        self._class_means = class_means

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:

            # 首任务阶段 teacher
            if self._efm:
                self._efm.capture_teacher(self._network)

            params = [p for p in self._network.parameters() if p.requires_grad]
            optimizer = self._build_optimizer(params, stage="init")
            if self._optimizer_type in {"arwp", "rwp"}:
                try:
                    self._rwp_lr0 = float(optimizer.param_groups[0]["lr"])  # base LR for std scaling
                except Exception:
                    self._rwp_lr0 = None
            # lr = optimizer.param_groups[0]["lr"] 
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("init_milestones", []),
                gamma=float(self.args.get("init_lrate_decay", 1.0)),
                T_max=self.args.get("init_epoch", None),
                eta_min=self.args.get("min_lr", 0.0),
                # eta_min=self.args.get("min_lr", 0.01*lr),
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            # 增量阶段 teacher（已扩展 fc 之后）
            if self._efm:
                self._efm.capture_teacher(self._network)
            params = [p for p in self._network.parameters() if p.requires_grad]
            optimizer = self._build_optimizer(params, stage="update")
            if self._optimizer_type in {"arwp", "rwp"}:
                try:
                    self._rwp_lr0 = float(optimizer.param_groups[0]["lr"])  # base LR for std scaling
                except Exception:
                    self._rwp_lr0 = None
            # lr = optimizer.param_groups[0]["lr"] 
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=self.args.get("epochs", None),
                eta_min=self.args.get("min_lr", 0.0),
                # eta_min=self.args.get("min_lr", 0.01*lr),
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0,0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)


                if self._optimizer_type == "cflat":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
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
                        loss = F.cross_entropy(logits, targets)
                        loss_value = loss.detach()
                        loss.backward()  # ← 必须：让 p.grad 生效
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
                        loss = F.cross_entropy(logits, targets)
                        loss_value = loss.detach()
                        loss.backward()
                        return outputs, loss_value
                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()
                elif self._optimizer_type == "rwp":
                    # RWP: two-pass grad mix (include EFM penalty if enabled)
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else float(optimizer.param_groups[0]["lr"])  # type: ignore
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"])  # type: ignore
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr) or cur_lr)
                        rwp_std = float(self._rwp_std) * ((cur_lr / base_lr) if base_lr > 0 else 1.0)
                    else:
                        rwp_std = float(self._rwp_std)

                    enable_running_stats(self._network)
                    optimizer.zero_grad()
                    outputs = self._network(inputs)
                    logits_clean = outputs["logits"]
                    loss_clean = F.cross_entropy(logits_clean, targets)
                    # if self._efm:
                    #     efm_pen = self._efm.penalty(inputs=inputs, outputs=outputs, known=self._known_classes, total=self._total_classes, loss_on_new=False)
                    #     if efm_pen is not None:
                    #         loss_clean = loss_clean + self._efm.lam * efm_pen
                    loss_clean.backward()
                    g0 = {}
                    for name, p in self._network.named_parameters():
                        if p.requires_grad and (p.grad is not None):
                            g0[name] = p.grad.detach().clone()

                    disable_running_stats(self._network)
                    noise_dict = {}
                    with torch.no_grad():
                        # mode = str(self.args.get("rwp_noise_type", "mARWP_fisher"))
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
                    loss_noisy = F.cross_entropy(logits_noisy, targets)
                    # if self._efm:
                    #     efm_pen = self._efm.penalty(inputs=inputs, outputs=outputs_noisy, known=self._known_classes, total=self._total_classes, loss_on_new=False)
                    #     if efm_pen is not None:
                    #         loss_noisy = loss_noisy + self._efm.lam * efm_pen
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
                else:
                    optimizer.zero_grad()
                    # logits = self._network(inputs)["logits"]
                    outputs = self._network(inputs)      # 需要 outputs 含 "features"
                    logits  = outputs["logits"]
                    loss = F.cross_entropy(logits, targets)
                    # if self._efm:
                    #     efm_pen = self._efm.penalty(
                    #     inputs=inputs, outputs=outputs,
                    #     known=self._known_classes, total=self._total_classes,
                    #     loss_on_new=False
                    #     )
                        
                    #     if efm_pen is not None:
                    #         loss = loss + self._efm.lam * efm_pen


                    if self._optimizer_type == "sam":
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        logits = self._network(inputs)["logits"]
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

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch % 5 == 4):
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                fake_targets = targets - self._known_classes

                if self._optimizer_type == "cflat":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
                        loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
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
                        loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                        loss_value = loss.detach()
                        loss.backward()  # ← 必须：让 p.grad 生效
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
                        loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                        loss_value = loss.detach()
                        loss.backward()
                        return outputs, loss_value
                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()
                elif self._optimizer_type == "rwp":
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else float(optimizer.param_groups[0]["lr"])  # type: ignore
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"])  # type: ignore
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr) or cur_lr)
                        rwp_std = float(self._rwp_std) * ((cur_lr / base_lr) if base_lr > 0 else 1.0)
                    else:
                        rwp_std = float(self._rwp_std)

                    enable_running_stats(self._network)
                    optimizer.zero_grad()
                    outputs = self._network(inputs)
                    logits_clean = outputs["logits"]
                    loss_clean = F.cross_entropy(logits_clean[:, self._known_classes :], fake_targets)
                    # if self._efm:
                    #     efm_pen = self._efm.penalty(inputs=inputs, outputs=outputs, known=self._known_classes, total=self._total_classes, loss_on_new=True)
                    #     if efm_pen is not None:
                    #         loss_clean = loss_clean + self._efm.lam * efm_pen
                    loss_clean.backward()
                    g0 = {}
                    for name, p in self._network.named_parameters():
                        if p.requires_grad and (p.grad is not None):
                            g0[name] = p.grad.detach().clone()

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
                    loss_noisy = F.cross_entropy(logits_noisy[:, self._known_classes :], fake_targets)
                    # if self._efm:
                    #     efm_pen = self._efm.penalty(inputs=inputs, outputs=outputs_noisy, known=self._known_classes, total=self._total_classes, loss_on_new=True)
                    #     if efm_pen is not None:
                    #         loss_noisy = loss_noisy + self._efm.lam * efm_pen
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
                        loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                        loss_value = loss.detach()
                        loss.backward()
                        return outputs, loss_value
                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()
                else:
                    optimizer.zero_grad()
                    # logits = self._network(inputs)["logits"]
                    outputs = self._network(inputs)
                    logits  = outputs["logits"]
                    loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    # ---- EFM（只新类切片）----
                    if self._efm:
                        efm_pen = self._efm.penalty(
                            inputs=inputs, outputs=outputs,
                            known=self._known_classes, total=self._total_classes,
                            loss_on_new=True
                        )
                        if efm_pen is not None:
                            loss = loss + self._efm.lam * efm_pen

                    if self._optimizer_type == "sam":
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        logits = self._network(inputs)["logits"]
                        second_loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
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

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch % 5 == 4) :
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["epochs"], losses / len(train_loader), train_acc, test_acc
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["epochs"], losses / len(train_loader), train_acc
                )
            
            prog_bar.set_description(info)
        
        logging.info(info)

    def _full_finetune(self, train_loader, test_loader, optimizer, scheduler, epochs):
        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                if self._optimizer_type == "cflat":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
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
                        loss = F.cross_entropy(logits, targets)
                        loss_value = loss.detach()
                        loss.backward()  # ← 必须：让 p.grad 生效
                        return outputs, loss_value
                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()
                elif self._optimizer_type == "arwp":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
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
                    loss = F.cross_entropy(logits, targets)

                    if self._optimizer_type == "sam":
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        logits = self._network(inputs)["logits"]
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
            if ((epoch) % 3 == 0) or epoch == epochs - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "FullFinetune Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "FullFinetune Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)
    
    # Use base optimizer for true RWP (mixing in-loop), otherwise defer
    def _build_optimizer(self, params, stage):
        if getattr(self, "_optimizer_type", "").lower() == "rwp":
            # reuse BaseLearner hyperparam resolver indirectly via args
            if stage == "init":
                lr = float(self.args.get("init_lr", self.args.get("lr", 0.01)))
                weight_decay = float(self.args.get("weight_decay", 0.0))
                momentum = float(self.args.get("momentum", 0.0))
            elif stage == "full":
                lr = float(self.args.get("full_lr", self.args.get("lr", 0.01)))
                weight_decay = float(self.args.get("full_weight_decay", self.args.get("weight_decay", 0.0)))
                momentum = float(self.args.get("full_momentum", self.args.get("momentum", 0.0)))
            else:
                lr = float(self.args.get("lr", 0.01))
                weight_decay = float(self.args.get("weight_decay", 0.0))
                momentum = float(self.args.get("momentum", 0.0))
            base_name = str(self.args.get("optimizer", "sgd")).lower()
            if base_name == "adam":
                return optim.Adam(params, lr=lr, weight_decay=weight_decay)
            if base_name == "adamw":
                return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        return super()._build_optimizer(params, stage)
    
