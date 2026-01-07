import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.inc_net import IncrementalNet
from loraCL.baseLoRA import LoraBaseLearner
from utils.toolkit import tensor2numpy
from optimer.util import enable_running_stats, disable_running_stats, generate_pertubation

import timm
from backbone.lora import LoRA_ViT_timm
from types import SimpleNamespace

num_workers = 8


class Learner(LoraBaseLearner):
    """
    oLoRA: Incremental LoRA with an orthogonality regularizer between previous
    LoRA_A and newly added LoRA_A factors plus L2 on new LoRA params.
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()
        # hyperparamter for SAM optimizer
        if self._optimizer_type == "sam":
            self._sam_rho = float(args.get("sam_rho", 0.05))
            self._sam_adaptive = bool(args.get("sam_adaptive", False))

        # hyperparamter for CF# hyperparamter for optimizerlat optimizer
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
        self._refresh_distributed_context()

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._log("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args.get("train_num_workers", 8)
        )

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args.get("train_num_workers", 8)
        )

        self._train(self.train_loader, self.test_loader)
        # self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._unwrap_network()

        # Compute class-means over all seen classes for NME evaluation (Class-IL)
        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as _nme_exc:  # pylint: disable=broad-except
            self._log(f"[LoRA-OLoRA][NME] Failed to compute class means: {_nme_exc}")

    def _build_incremental_lora(self, eval_mode: bool = False):
        vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        rank = self.args.get("lora_rank", 10)
        model = LoRA_ViT_timm(
            vit_model=vit.eval(),
            r=rank,
            num_classes=0,
            index=False,
            increment=self.args["increment"],
            filepath=self.args.get("filepath", "./"),
            cur_task_index=self._cur_task,
            learn_alpha=False,
            eval=eval_mode,
        )
        model.out_dim = 768
        return model

    def _train(self, train_loader, test_loader):
        network = self._unwrap_network()
        self._network = network

        if self._cur_task == 0:
            network.backbone = self._build_incremental_lora()
            network.backbone.to(self._device)
            self._network = network
            self._prepare_network()
            params = [p for p in self._network.parameters() if p.requires_grad]
            optimizer = self._build_optimizer(params, stage="init")
            lr = optimizer.param_groups[0]["lr"] 
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=self.args.get("epochs", None),
                eta_min=self.args.get("min_lr", 0.1*lr),
            )
            if self._optimizer_type in {"arwp", "rwp"}:
                self._rwp_lr0 = float(lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            network.backbone = self._build_incremental_lora()
            network.backbone.to(self._device)
            self._network = network
            self._prepare_network()

            params = [p for p in self._network.parameters() if p.requires_grad]
            optimizer = self._build_optimizer(params, stage="update")
            lr = optimizer.param_groups[0]["lr"] 
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=self.args.get("epochs", None),
                eta_min=self.args.get("min_lr", 0.1*lr),
            )
            if self._optimizer_type in {"arwp", "rwp"}:
                self._rwp_lr0 = float(lr)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        # Save new LoRA params for future tasks
        save_dir = self.args.get("filepath", "./")
        if self._cur_task > 0:
            net_obj = self._unwrap_network()
            net_obj.backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(self._network, "save_fc"):
            net_obj = self._unwrap_network()
            net_obj.save_fc(save_dir, self._cur_task)

    def _build_eval_backbone(self, task_idx):
        return self._build_incremental_lora(eval_mode=True)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]), disable=not self._is_main_process)
        for _, epoch in enumerate(prog_bar):
            
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
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
                        loss.backward()  # Required so p.grad is populated.
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
                elif self._optimizer_type == "rwp":
                    # True RWP for init: full logits
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else float(optimizer.param_groups[0]["lr"])  # type: ignore
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"])  # type: ignore
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr))
                        scale = (cur_lr / base_lr) if base_lr > 0 else 1.0
                        rwp_std = float(self._rwp_std) * float(scale)
                    else:
                        rwp_std = float(self._rwp_std)

                    enable_running_stats(self._network)
                    optimizer.zero_grad()
                    outputs = self._network(inputs)
                    logits_clean = outputs["logits"]
                    loss_clean = F.cross_entropy(logits_clean, targets)
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
                    loss_noisy = F.cross_entropy(logits_noisy, targets)
                    loss_noisy.backward()

                    with torch.no_grad():
                        if "fisher" in self.rwp_noise_type:
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

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["init_epoch"], losses / len(train_loader), train_acc, test_acc
                )
            elif self._is_main_process:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["init_epoch"], losses / len(train_loader), train_acc
                )
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        lamda_1 = float(self.args.get("olora_lamda_1", 1.0))
        lamda_2 = float(self.args.get("olora_lamda_2", 0.0))

        prog_bar = tqdm(range(self.args["epochs"]), disable=not self._is_main_process)
        for _, epoch in enumerate(prog_bar):
            # self._set_epoch(train_loader, epoch)
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
                        backbone = self._get_backbone()
                        loss_clf = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                        orthogonal_loss = backbone.compute_ortho_loss()
                        l2_loss = 0.0
                        for w in list(backbone.w_As) + list(backbone.w_Bs):
                            l2_loss = l2_loss + torch.norm(w.weight, p=2)
                        total_loss = loss_clf + lamda_1 * orthogonal_loss + lamda_2 * l2_loss
                        total_loss.backward()
                        return outputs, [total_loss]

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
                        loss.backward()  # Required so p.grad is populated.
                        return outputs, loss_value
                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()
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
                elif self._optimizer_type == "rwp":
                    # True RWP with oLoRA regularization
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else float(optimizer.param_groups[0]["lr"])  # type: ignore
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"])  # type: ignore
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr))
                        scale = (cur_lr / base_lr) if base_lr > 0 else 1.0
                        rwp_std = float(self._rwp_std) * float(scale)
                    else:
                        rwp_std = float(self._rwp_std)

                    lamda_1 = float(self.args.get("olora_lamda_1", 1.0))
                    lamda_2 = float(self.args.get("olora_lamda_2", 0.0))

                    enable_running_stats(self._network)
                    optimizer.zero_grad()
                    outputs = self._network(inputs)
                    logits_clean = outputs["logits"]
                    loss_clf = F.cross_entropy(logits_clean[:, self._known_classes :], fake_targets)
                    backbone = self._get_backbone()
                    orthogonal_loss = backbone.compute_ortho_loss()
                    l2_loss = 0.0
                    for w in list(backbone.w_As) + list(backbone.w_Bs):
                        l2_loss = l2_loss + torch.norm(w.weight, p=2)
                    loss_clean = loss_clf + lamda_1 * orthogonal_loss + lamda_2 * l2_loss
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
                    loss_clf_noisy = F.cross_entropy(logits_noisy[:, self._known_classes :], fake_targets)
                    backbone = self._get_backbone()
                    orthogonal_loss = backbone.compute_ortho_loss()
                    l2_loss = 0.0
                    for w in list(backbone.w_As) + list(backbone.w_Bs):
                        l2_loss = l2_loss + torch.norm(w.weight, p=2)
                    loss_noisy = loss_clf_noisy + lamda_1 * orthogonal_loss + lamda_2 * l2_loss
                    loss_noisy.backward()

                    with torch.no_grad():
                        if "fisher" in self.rwp_noise_type :
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
                else:
                    optimizer.zero_grad()
                    logits = self._network(inputs)["logits"]

                    loss_clf = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    backbone = self._get_backbone()
                    orthogonal_loss = backbone.compute_ortho_loss()
                    l2_loss = 0.0
                    for w in list(backbone.w_As) + list(backbone.w_Bs):
                        l2_loss = l2_loss + torch.norm(w.weight, p=2)

                    loss = loss_clf + lamda_1 * orthogonal_loss + lamda_2 * l2_loss

                    if self._optimizer_type == "sam":
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        logits = self._network(inputs)["logits"]
                        loss_clf = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                        backbone = self._get_backbone()
                        orthogonal_loss = backbone.compute_ortho_loss()
                        l2_loss = 0.0
                        for w in list(backbone.w_As) + list(backbone.w_Bs):
                            l2_loss = l2_loss + torch.norm(w.weight, p=2)
                        second_loss = loss_clf + lamda_1 * orthogonal_loss + lamda_2 * l2_loss
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
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["epochs"], losses / len(train_loader), train_acc, test_acc
                )
            elif self._is_main_process:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["epochs"], losses / len(train_loader), train_acc
                )
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)

    def _get_backbone(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.backbone
        return self._network.backbone

    # Use base optimizer for true RWP (mixing in-loop), otherwise defer
    def _build_optimizer(self, params, stage):
        if getattr(self, "_optimizer_type", "").lower() == "rwp":
            lr = float(self.args.get("lr", self.args.get("init_lr", 0.01))) if stage == "init" else float(self.args.get("lr", self.args.get("lr", 0.01)))
            momentum = float(self.args.get("momentum", 0.0))
            weight_decay = float(self.args.get("weight_decay", 0.0))
            base_name = str(self.args.get("optimizer", "sgd")).lower()
            if base_name == "adam":
                return optim.Adam(params, lr=lr, weight_decay=weight_decay)
            if base_name == "adamw":
                return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        return super()._build_optimizer(params, stage)

   
