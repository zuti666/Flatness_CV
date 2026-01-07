import numpy as np
import torch
from tqdm import tqdm

from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from loraCL.baseLoRA import LoraBaseLearner
from utils.toolkit import tensor2numpy

import timm
from backbone.lora import LoRA_ViT_timm
from types import SimpleNamespace


num_workers = 8




class Learner(LoraBaseLearner):
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
        
                # --- Approximate RWP hyperparams (ARWP) ---
        elif self._optimizer_type == "arwp":
            self._rwp_std = float(args.get("rwp_std", 0.01))
            self._rwp_eta = float(args.get("rwp_eta", 1.0))
            self._rwp_beta = float(args.get("rwp_beta", 0.9))
            # optional: dynamically scale std by current LR from scheduler
            self._rwp_std_follow_lr = bool(args.get("rwp_std_follow_lr", False))
        # --- True RWP hyperparams/state ---
        elif self._optimizer_type == "rwp":
            self._rwp_std = float(args.get("rwp_std", 0.01))
            self._rwp_eta = float(args.get("rwp_eta", 1.0))
            self._rwp_beta = float(args.get("rwp_beta", 0.9))
            self._rwp_lambda = float(args.get("rwp_lambda", 0.5))  # mix g = λ g1 + (1-λ) g0
            self._rwp_std_follow_lr = bool(args.get("rwp_std_follow_lr", False))
            # fisher-like EMA storage of grad^2 per-parameter
            self._rwp_range = str(args.get("rwp_range", "lora"))
            
            self.rwp_noise_type = str(self.args.get("rwp_noise_type", "Gauss_standard"))
            if "fisher" in  self.rwp_noise_type:
                self._rwp_fisher = {}

    def after_task(self):
        # After each task, advance known classes to the current total.
        # This makes C_old equal to this task's C_total for the next task.
        self._known_classes = self._total_classes
        
        # Release CUDA cache after each task to mitigate fragmentation/OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        

    def incremental_train(self, data_manager):
        self._refresh_distributed_context()

       

        # Known classes (old) + new classes = total classes after training.
        # Note: _known_classes is updated in after_task().
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        # Expand the classifier to total classes; keep old weights in first C_old columns.
        self._network.update_fc(self._total_classes)
        self._log("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # Build this task's train set: global class range [C_old, C_total).
        # Many datasets keep global labels, so training shifts targets (see fake_targets).
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

        self._train(self.train_loader, self.test_loader)
        # self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._unwrap_network()

        # Compute class-means over all seen classes for NME evaluation (Class-IL)
        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as _nme_exc:  # pylint: disable=broad-except
            self._log(f"[LoRA][NME] Failed to compute class means: {_nme_exc}")

    def update_network(self, index=True, eval_mode=False):
        # if use VIT-B-16
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        print(model.pretrained_cfg.get('hf_hub_id'), model.pretrained_cfg.get('tag'))
        # if use DINO
        
        # LoRA hyperparams from args
        rank = int(self.args.get("lora_rank", 10))
        if rank <= 0:
            raise ValueError(f"lora_rank must be > 0, got {rank}")
        # lora_num_classes = int(self.args.get("lora_num_classes", 0))  # 0 -> use external head
        model = LoRA_ViT_timm(
            vit_model=model.eval(),
            r=rank,
            num_classes=0,
            index=index,
            increment=self.args['increment'],
            filepath=self.args['filepath'],
            cur_task_index=self._cur_task,
            learn_alpha=True,
            eval=eval_mode,
        )
        model.out_dim = 768
        return model

    def _train(self, train_loader, test_loader):
        network = self._unwrap_network()

        if self._cur_task == 0:
            if not hasattr(network.backbone, "save_lora_parameters"):
                network.backbone = self.update_network(index=True, eval_mode=False)
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
            self._init_train(train_loader, test_loader, optimizer, scheduler)

        else:
            network.backbone = self.update_network(index=False, eval_mode=False)
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
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        save_lora_name = self.args["filepath"]
        base_net = self._unwrap_network()

        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_lora_name, self._cur_task)
            # Persist additional trainable scaling parameters if enabled
            if getattr(backbone, "learn_alpha", False) and hasattr(backbone, "save_wrap_param"):
                try:
                    backbone.save_wrap_param(save_lora_name)
                except Exception:
                    pass
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_lora_name, self._cur_task)

    def _build_eval_backbone(self, task_idx):
        return self.update_network(index=False, eval_mode=True)

    
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
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            elif self._is_main_process:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )
            if self._is_main_process:
                prog_bar.set_description(info)

        if self._is_main_process:
            self._log(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]), disable=not self._is_main_process)
        for _, epoch in enumerate(prog_bar):
            
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # Shift global labels to local indices for new classes.
                # If labels are in [C_old, C_total), then fake_targets = targets - C_old.
                # PyTorch supports LongTensor - int broadcasting; dtype stays long.
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
                else:
                    optimizer.zero_grad()
                    logits = self._network(inputs)["logits"]  # Shape [B, C_total].

                    # Compute CE only on the new-class subspace.
                    # Slice columns from C_old: shape [B, C_inc], aligned with fake_targets.
                    loss = F.cross_entropy(
                        logits[:, self._known_classes :], fake_targets
                    )

                    if self._optimizer_type == "sam":
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        logits = self._network(inputs)["logits"]
                        second_loss = F.cross_entropy(
                            logits[:, self._known_classes :], fake_targets
                        )
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
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            elif self._is_main_process:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)

    
