import numpy as np
import torch
from tqdm import tqdm

from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from utils.toolkit import tensor2numpy
from optimer_PerturabtionType.util import (
    enable_running_stats,
    disable_running_stats,
    generate_pertubation,
)

import timm
from backbone.lora import LoRA_ViT_timm, MultiTaskLoRA_ResNet
from types import SimpleNamespace


num_workers = 8


class Learner(LoraBaseLearner):
    """
    SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning.

    Paper: SD-LoRA (ICLR 2025)
    Forward composition (Eq. 4):
        h' = (W₀ + α₁A₁B₁ + α₂A₂B₂ + ... + αₜAₜBₜ) x
    where:
      - W₀ is the frozen pre-trained weight
      - {AₖBₖ}_{k<t} are frozen previous-task directions (loaded from disk)
      - AₜBₜ is the current-task direction (trainable)
      - {αₖ}_{k≤t} are learnable scalar magnitudes (all trainable each task)

    Per-task training:
      - Create a fresh LoRA_ViT_timm with cur_task_index=t; it auto-loads
        saved LoRA for tasks 0..t-1 from disk and allocates new A/B for task t.
      - Train AₜBₜ + all αₖ on new-task CE loss.
      - Save AₜBₜ and αₖ to disk; next task loads them.
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # Optimizer-specific hyperparameters
        if self._optimizer_type == "sam":
            self._sam_rho = float(args.get("sam_rho", 0.05))
            self._sam_adaptive = bool(args.get("sam_adaptive", False))
        elif self._is_flatlora_optimizer():
            self._init_flatlora_state(args)
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

            self._gam_args = SimpleNamespace(
                grad_beta_1=self._gam_beta1,
                grad_beta_2=self._gam_beta2,
                grad_beta_3=self._gam_beta3,
                grad_gamma=self._gam_gamma,
                grad_rho=self._gam_grad_rho,
                grad_norm_rho=self._gam_grad_norm_rho,
                adaptive=self._gam_adaptive,
                perturb_eps=self._gam_perturb_eps,
                grad_reduce=str(self._gam_grad_reduce),
            )
        elif self._optimizer_type == "arwp":
            self._rwp_std = float(args.get("rwp_std", 0.01))
            self._rwp_eta = float(args.get("rwp_eta", 1.0))
            self._rwp_beta = float(args.get("rwp_beta", 0.9))
            self._rwp_std_follow_lr = bool(args.get("rwp_std_follow_lr", False))
        elif self._optimizer_type == "rwp":
            self._rwp_std = float(args.get("rwp_std", 0.01))
            self._rwp_eta = float(args.get("rwp_eta", 1.0))
            self._rwp_beta = float(args.get("rwp_beta", 0.9))
            self._rwp_lambda = float(args.get("rwp_lambda", 0.5))
            self._rwp_std_follow_lr = bool(args.get("rwp_std_follow_lr", False))
            self._rwp_range = str(args.get("rwp_range", "lora"))

            self.rwp_noise_type = str(self.args.get("rwp_noise_type", "Gauss_standard"))
            if "fisher" in self.rwp_noise_type:
                self._rwp_fisher = {}

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._refresh_distributed_context()

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        self._network.update_fc(self._total_classes)
        self._log("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
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

        self._train(self.train_loader, self.test_loader)
        self._network = self._unwrap_network()

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as _nme_exc:
            self._log(f"[SDLoRA][NME] Failed to compute class means: {_nme_exc}")

    def _create_backbone(self, task_idx, eval_mode=False):
        """Create a task-aware LoRA backbone for task `task_idx`."""
        backbone_type = self.args.get("backbone_type", "vit_base_patch16_224")
        rank = int(self.args.get("lora_rank", 10))
        if rank <= 0:
            raise ValueError(f"lora_rank must be > 0, got {rank}")

        if "resnet" in backbone_type.lower():
            from backbone.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

            resnet_map = {
                "resnet18": resnet18,
                "resnet34": resnet34,
                "resnet50": resnet50,
                "resnet101": resnet101,
                "resnet152": resnet152,
            }
            fn = resnet_map.get(backbone_type.lower(), resnet50)
            model = fn(pretrained=True, args=self.args)
            return MultiTaskLoRA_ResNet(
                model,
                r=rank,
                lora_layers=self.args.get("lora_layers", None),
                task_id=int(task_idx),
                save_dir=self.args["filepath"],
                eval_mode=eval_mode,
                learn_alpha=True,
            )

        model = timm.create_model(backbone_type, pretrained=True, num_classes=0)
        backbone = LoRA_ViT_timm(
            vit_model=model.eval(),
            r=rank,
            num_classes=0,
            index=True,
            increment=self.args["increment"],
            filepath=self.args["filepath"],
            cur_task_index=task_idx,
            learn_alpha=True,
            eval=eval_mode,
        )
        backbone.out_dim = 768
        return backbone

    def _build_eval_backbone(self, task_idx):
        """Build backbone for evaluation after training task `task_idx`.

        Creates backbone with cur_task_index=task_idx+1 so that saved LoRA for
        tasks 0..task_idx are loaded and composed at eval time (eval mode uses
        _LoRA_qkv_timm_eval which reads all saved tasks from disk).
        """
        backbone_type = self.args.get("backbone_type", "vit_base_patch16_224").lower()
        if "resnet" in backbone_type:
            return self._create_backbone(task_idx=task_idx, eval_mode=True)
        return self._create_backbone(task_idx=task_idx + 1, eval_mode=True)

    def _train(self, train_loader, test_loader):
        """Train SD-LoRA for the current task.

        Core SD-LoRA step: create a fresh backbone at every task.
        The backbone auto-loads previous tasks' frozen LoRA and allocates
        new trainable A/B for the current task.
        """
        network = self._unwrap_network()

        # --- SD-LoRA paper requirement: fresh backbone per task ---
        # LoRA_ViT_timm(cur_task_index=t) loads tasks 0..t-1 from disk (frozen)
        # and creates new w_As/w_Bs for task t (trainable).
        network.backbone = self._create_backbone(task_idx=self._cur_task, eval_mode=False)
        network.backbone.to(self._device)
        self._network = network
        self._prepare_network()

        if self._cur_task == 0:
            params = [p for p in self._network.parameters() if p.requires_grad]
            optimizer = self._build_optimizer(params, stage="init")
            lr0 = self.args.get("init_lr", 0.1)
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=self.args.get("epochs", None),
                eta_min=self.args.get("min_lr", 0.1 * float(lr0)),
            )
            if self._optimizer_type in {"arwp", "rwp"}:
                self._rwp_lr0 = float(lr0)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            params = [p for p in self._network.parameters() if p.requires_grad]
            optimizer = self._build_optimizer(params, stage="update")
            lr0 = self.args.get("lrate", 0.1)
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=self.args.get("epochs", None),
                eta_min=self.args.get("min_lr", 0.1 * float(lr0)),
            )
            if self._optimizer_type in {"arwp", "rwp"}:
                self._rwp_lr0 = float(lr0)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        # Save current task's LoRA (A/B) and scaling factors (α) to disk.
        # Next task's backbone will load them automatically.
        save_lora_name = self.args["filepath"]
        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_lora_name, self._cur_task)
            if getattr(backbone, "learn_alpha", False) and hasattr(backbone, "save_wrap_param"):
                try:
                    backbone.save_wrap_param(save_lora_name)
                except Exception:
                    pass
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_lora_name, self._cur_task)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if self._is_flatlora_optimizer():
            self._reset_flatlora_schedule(len(train_loader) * max(int(self.args["init_epoch"]), 1))
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
                        loss.backward()
                        return outputs, loss_value
                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()
                elif self._is_flatlora_optimizer():
                    def flatlora_loss():
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
                        loss = F.cross_entropy(logits, targets)
                        return logits, loss

                    logits, loss_value = self._flatlora_step(optimizer, flatlora_loss)
                    losses += loss_value

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
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else float(optimizer.param_groups[0]["lr"])
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"])
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr) or cur_lr)
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
                    base_model = self._unwrap_network()
                    for name, p in base_model.named_parameters():
                        if p.requires_grad and (p.grad is not None):
                            g0[name] = p.grad.detach().clone()

                    disable_running_stats(self._network)
                    noise_dict = {}
                    with torch.no_grad():
                        std_for_noise = float(self.args.get("noise_std", rwp_std))
                        for name, p in base_model.named_parameters():
                            if self._rwp_range == "lora":
                                if not p.requires_grad or p.numel() == 0:
                                    continue
                            fisher_param = self._rwp_fisher.get(name, None) if hasattr(self, "_rwp_fisher") else None
                            e = generate_pertubation(
                                p,
                                pertubation_mode=self.rwp_noise_type,
                                std=std_for_noise,
                                fisher_param=fisher_param,
                                fisher_scaler=float(self._rwp_eta),
                            )
                            p.data.add_(e)
                            noise_dict[name] = e

                    optimizer.zero_grad()
                    outputs_noisy = self._network(inputs)
                    logits_noisy = outputs_noisy["logits"]
                    loss_noisy = F.cross_entropy(logits_noisy, targets)
                    loss_noisy.backward()

                    if hasattr(self, "_rwp_fisher"):
                        with torch.no_grad():
                            for name, p in base_model.named_parameters():
                                if not p.requires_grad or (p.grad is None):
                                    continue
                                g2 = p.grad.detach() ** 2
                                if name not in self._rwp_fisher:
                                    self._rwp_fisher[name] = g2
                                else:
                                    self._rwp_fisher[name] = float(self._rwp_beta) * self._rwp_fisher[name] + g2

                    with torch.no_grad():
                        for name, p in base_model.named_parameters():
                            if name in noise_dict:
                                p.data.sub_(noise_dict[name])

                    lam = float(self._rwp_lambda)
                    for name, p in base_model.named_parameters():
                        if not p.requires_grad:
                            continue
                        if p.grad is not None:
                            g1 = p.grad.detach()
                            g0_n = g0.get(name, torch.zeros_like(g1))
                            p.grad.data.copy_(lam * g1 + (1.0 - lam) * g0_n)
                    optimizer.step()
                    logits = logits_clean.detach()
                    losses += lam * float(loss_noisy.detach().item()) + (1.0 - lam) * float(loss_clean.detach().item())

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
                    self._cur_task, epoch + 1, self.args["init_epoch"],
                    losses / len(train_loader), train_acc, test_acc,
                )
            elif self._is_main_process:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["init_epoch"],
                    losses / len(train_loader), train_acc,
                )
            if self._is_main_process:
                prog_bar.set_description(info)

        if self._is_main_process:
            self._log(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        if self._is_flatlora_optimizer():
            self._reset_flatlora_schedule(len(train_loader) * max(int(self.args["epochs"]), 1))
        prog_bar = tqdm(range(self.args["epochs"]), disable=not self._is_main_process)
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
                        loss = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
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
                        loss = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                        loss_value = loss.detach()
                        loss.backward()
                        return outputs, loss_value
                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()
                elif self._is_flatlora_optimizer():
                    def flatlora_loss():
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
                        loss = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                        return logits, loss

                    logits, loss_value = self._flatlora_step(optimizer, flatlora_loss)
                    losses += loss_value

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
                        loss = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                        loss_value = loss.detach()
                        loss.backward()
                        return outputs, loss_value
                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()

                elif self._optimizer_type == "rwp":
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else float(optimizer.param_groups[0]["lr"])
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"])
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr) or cur_lr)
                        scale = (cur_lr / base_lr) if base_lr > 0 else 1.0
                        rwp_std = float(self._rwp_std) * float(scale)
                    else:
                        rwp_std = float(self._rwp_std)

                    enable_running_stats(self._network)
                    optimizer.zero_grad()
                    outputs = self._network(inputs)
                    logits_clean = outputs["logits"]
                    loss_clean = F.cross_entropy(logits_clean[:, self._known_classes:], fake_targets)
                    loss_clean.backward()

                    g0 = {}
                    base_model = self._unwrap_network()
                    for name, p in base_model.named_parameters():
                        if p.requires_grad and (p.grad is not None):
                            g0[name] = p.grad.detach().clone()

                    disable_running_stats(self._network)
                    noise_dict = {}
                    with torch.no_grad():
                        std_for_noise = float(self.args.get("noise_std", rwp_std))
                        for name, p in base_model.named_parameters():
                            if self._rwp_range == "lora":
                                if not p.requires_grad or p.numel() == 0:
                                    continue
                            fisher_param = self._rwp_fisher.get(name, None) if hasattr(self, "_rwp_fisher") else None
                            e = generate_pertubation(
                                p,
                                pertubation_mode=self.rwp_noise_type,
                                std=std_for_noise,
                                fisher_param=fisher_param,
                                fisher_scaler=float(self._rwp_eta),
                            )
                            p.data.add_(e)
                            noise_dict[name] = e

                    optimizer.zero_grad()
                    outputs_noisy = self._network(inputs)
                    logits_noisy = outputs_noisy["logits"]
                    loss_noisy = F.cross_entropy(logits_noisy[:, self._known_classes:], fake_targets)
                    loss_noisy.backward()

                    if hasattr(self, "_rwp_fisher"):
                        with torch.no_grad():
                            for name, p in base_model.named_parameters():
                                if not p.requires_grad or (p.grad is None):
                                    continue
                                g2 = p.grad.detach() ** 2
                                if name not in self._rwp_fisher:
                                    self._rwp_fisher[name] = g2
                                else:
                                    self._rwp_fisher[name] = float(self._rwp_beta) * self._rwp_fisher[name] + g2

                    with torch.no_grad():
                        for name, p in base_model.named_parameters():
                            if name in noise_dict:
                                p.data.sub_(noise_dict[name])

                    lam = float(self._rwp_lambda)
                    for name, p in base_model.named_parameters():
                        if not p.requires_grad:
                            continue
                        if p.grad is not None:
                            g1 = p.grad.detach()
                            g0_n = g0.get(name, torch.zeros_like(g1))
                            p.grad.data.copy_(lam * g1 + (1.0 - lam) * g0_n)
                    optimizer.step()
                    logits = logits_clean.detach()
                    losses += lam * float(loss_noisy.detach().item()) + (1.0 - lam) * float(loss_clean.detach().item())

                else:
                    optimizer.zero_grad()
                    logits = self._network(inputs)["logits"]
                    loss = F.cross_entropy(logits[:, self._known_classes:], fake_targets)

                    if self._optimizer_type == "sam":
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        logits = self._network(inputs)["logits"]
                        second_loss = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
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
                    self._cur_task, epoch + 1, self.args["epochs"],
                    losses / len(train_loader), train_acc, test_acc,
                )
            elif self._is_main_process:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["epochs"],
                    losses / len(train_loader), train_acc,
                )
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)
