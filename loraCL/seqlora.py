import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.inc_net import IncrementalNet
from loraCL.base import LoraBaseLearner
from utils.toolkit import tensor2numpy

import timm
from backbone.lora import LoRA_ViT_timm

num_workers = 8


class Learner(LoraBaseLearner):
    """
    SeqLoRA: Train a single, fixed-size LoRA module sequentially over tasks
    without replay or explicit regularizers. Base backbone remains frozen.
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._lora_initialized = False
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()
        self._use_sam = self._optimizer_type == "sam"
        self._use_cflat = self._optimizer_type == "cflat"
        self._sam_rho = float(args.get("sam_rho", 0.05))
        self._sam_adaptive = bool(args.get("sam_adaptive", False))
        self._cflat_rho = float(args.get("cflat_rho", 0.2))
        self._cflat_lambda = float(args.get("cflat_lambda", 0.2))
        self._cflat_adaptive = bool(args.get("cflat_adaptive", False))
        self._cflat_perturb_eps = float(args.get("cflat_perturb_eps", 1e-12))
        self._cflat_grad_reduce = args.get("cflat_grad_reduce", "mean")

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._refresh_distributed_context()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._log("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        train_sampler = self._build_sampler(train_dataset, shuffle=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
        )

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        self._train(self.train_loader, self.test_loader)

        self._network = self._unwrap_network()

    def build_lora_backbone(self):
        # Build LoRA-wrapped timm ViT once and reuse across tasks
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        rank = self.args.get("lora_rank", 8)
        model = LoRA_ViT_timm(
            vit_model=model.eval(),
            r=rank,
            num_classes=0,
            index=True,
            increment=self.args["increment"],
            filepath=self.args.get("filepath", "./"),
            cur_task_index=0,
            learn_alpha=False,
        )
        model.out_dim = 768
        return model

    def _train(self, train_loader, test_loader):
        network = self._unwrap_network()

        # Ensure LoRA is attached only at the first task (t=0), one-time guard
        

        if self._cur_task == 0:
            if not self._lora_initialized:
                network.backbone = self.build_lora_backbone()
                network.backbone.to(self._device)
                self._lora_initialized = True
            self._network = network
            self._prepare_network()

            params = [p for p in self._network.parameters() if p.requires_grad]
            optimizer = self._build_optimizer(params, stage="init")
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._network = network
            self._prepare_network()

            params = [p for p in self._network.parameters() if p.requires_grad]
            optimizer = self._build_optimizer(params, stage="update")
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        # Persist LoRA parameters and classifier head for this task
        save_dir = self.args.get("filepath", "./")
        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_dir, self._cur_task)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]), disable=not self._is_main_process)
        for _, epoch in enumerate(prog_bar):
            self._set_epoch(train_loader, epoch)
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                if self._use_cflat:
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
                        loss = F.cross_entropy(logits, targets)
                        return outputs, [loss]

                    _, loss_list = optimizer.step(closure=closure)
                    loss_value = torch.stack([loss_term.detach() for loss_term in loss_list]).sum()
                    losses += loss_value.item()

                    with torch.no_grad():
                        logits = self._network(inputs)["logits"]
                else:
                    optimizer.zero_grad()
                    logits = self._network(inputs)["logits"]
                    loss = F.cross_entropy(logits, targets)

                    if self._use_sam:
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

            if epoch % 5 == 0 and self._is_main_process:
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
        prog_bar = tqdm(range(self.args["epochs"]), disable=not self._is_main_process)
        for _, epoch in enumerate(prog_bar):
            self._set_epoch(train_loader, epoch)
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                fake_targets = targets - self._known_classes

                if self._use_cflat:
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
                        loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                        return outputs, [loss]

                    _, loss_list = optimizer.step(closure=closure)
                    loss_value = torch.stack([loss_term.detach() for loss_term in loss_list]).sum()
                    losses += loss_value.item()

                    with torch.no_grad():
                        logits = self._network(inputs)["logits"]
                else:
                    optimizer.zero_grad()
                    logits = self._network(inputs)["logits"]
                    loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)

                    if self._use_sam:
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
            if epoch % 5 == 0 and self._is_main_process:
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
    
    def _build_eval_backbone(self, task_idx):
        vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        rank = self.args.get("lora_rank", 8)
        lora_backbone = LoRA_ViT_timm(
            vit_model=vit.eval(),
            r=rank,
            num_classes=0,
            index=False,
            increment=self.args["increment"],
            filepath=self.args.get("filepath", "./"),
            cur_task_index=self._cur_task,
            learn_alpha=False,
            eval=True,
        )
        lora_backbone.out_dim = 768

        # SeqLoRA keeps only the current task's adaptation at evaluation time
        last_idx = max(0, self._cur_task - 1)
        try:
            for blk in lora_backbone.lora_vit.blocks:
                qkv = getattr(getattr(blk, "attn", blk), "qkv", None)
                if qkv is None:
                    continue
                if hasattr(qkv, "saved_A") and hasattr(qkv, "saved_B"):
                    keep_a = f"saved_A_{last_idx}"
                    keep_b = f"saved_B_{last_idx}"
                    qkv.saved_A = {k: v for k, v in qkv.saved_A.items() if k == keep_a and v is not None}
                    qkv.saved_B = {k: v for k, v in qkv.saved_B.items() if k == keep_b and v is not None}
        except Exception:
            pass

        return lora_backbone
    def _build_optimizer(self, params, stage):
        if self._use_sam:
            from optimer.optimer_sam import SAM

            lr = self.args["init_lr"] if stage == "init" else self.args["lrate"]
            return SAM(
                params,
                optim.SGD,
                rho=self._sam_rho,
                adaptive=self._sam_adaptive,
                lr=lr,
                momentum=0.9,
            )

        if self._use_cflat:
            from optimer.c_flat import C_Flat

            lr = self.args["init_lr"] if stage == "init" else self.args["lrate"]
            base_opt = optim.SGD(params, lr=lr, momentum=0.9)
            return C_Flat(
                params,
                base_optimizer=base_opt,
                model=self._network,
                cflat=True,
                rho=self._cflat_rho,
                lamb=self._cflat_lambda,
                adaptive=self._cflat_adaptive,
                perturb_eps=self._cflat_perturb_eps,
                grad_reduce=self._cflat_grad_reduce,
            )

        if stage == "init":
            lr = self.args["init_lr"]
        else:
            lr = self.args["lrate"]

        return optim.SGD(params, lr=lr, momentum=0.9)
