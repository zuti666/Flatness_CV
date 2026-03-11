"""EWC in ΔW-space for LoRA-tuned ViT (Yaoyue variant).

Key differences vs. `EWC_LoRA_AB`:
 - Fisher is estimated w.r.t. effective full weights W (equivalently ΔW),
   **not** separately for A/B. We temporarily enable grad on frozen base
   qkv weights to collect ∂L/∂W, then use their diagonals.
 - Regularization is applied on ΔW tensors (B@A projected into the full
   qkv weight shape), so curvature matches the paper's Eq.(4) setting.
 - Classifier head (fc) still uses standard EWC on its own parameters.

Notation:
    ΔW (for a qkv) = concat([B_q@A_q, 0, B_v@A_v]) with shape [3d, d]
    Penalty = (λ/2) * Σ_i F_i * (ΔW_i - ΔW_i^{prev})²  (plus fc terms)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.lora import LoRA_ViT_timm, _LoRA_qkv_timm_train
from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(LoraBaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._lora_initialized = False
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # EWC hyperparameters
        self._ewc_lambda = float(args.get("ewc_lambda", 20.0))
        self._ewc_gamma = float(args.get("ewc_gamma", 1.0))
        self._ewc_max_batches = int(args.get("ewc_max_batches", 100))
        self._ewc_eps = float(args.get("ewc_eps", 1e-5))

        # Buffers (CPU)
        self._fisher_lora: Dict[str, torch.Tensor] | None = None
        self._checkpoint_lora: Dict[str, torch.Tensor] | None = None
        self._fisher_head: Dict[str, torch.Tensor] | None = None
        self._checkpoint_head: Dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------#
    # LoRA backbone builder                                             #
    # ------------------------------------------------------------------#
    def build_lora_backbone(self, index=True, eval_mode=False):
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        rank = int(self.args.get("lora_rank", 10))
        if rank <= 0:
            raise ValueError(f"lora_rank must be > 0, got {rank}")
        model = LoRA_ViT_timm(
            vit_model=model.eval(),
            r=rank,
            num_classes=0,
            index=index,
            increment=self.args["increment"],
            filepath=self.args.get("filepath", "./"),
            cur_task_index=0,
            learn_alpha=False,
        )
        model.out_dim = 768
        return model

    # ------------------------------------------------------------------#
    # Lifecycle                                                         #
    # ------------------------------------------------------------------#
    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._refresh_distributed_context()

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._log(f"Learning on {self._known_classes}-{self._total_classes}")

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

        network = self._unwrap_network()
        if not self._lora_initialized:
            network.backbone = self.build_lora_backbone()
            network.backbone.to(self._device)
            self._lora_initialized = True
        self._network = network
        self._prepare_network()
        self._align_head_ewc_buffers(self._unwrap_network())

        params = [p for p in self._network.parameters() if p.requires_grad]
        stage = "init" if self._cur_task == 0 else "update"
        optimizer = self._build_optimizer(params, stage=stage)
        if self._optimizer_type not in {"sgd", "adam", "adamw"}:
            raise ValueError(f"EWC_Yaoyue_LoRA supports sgd/adam/adamw only, got {self._optimizer_type}")

        if stage == "init":
            epochs = int(self.args.get("init_epoch", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )
        else:
            epochs = int(self.args.get("epochs", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )

        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        for epoch in prog_bar:
            self._network.train()
            total_loss = 0.0
            correct, total = 0, 0

            for batch in self.train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                optimizer.zero_grad()
                outputs = self._network(inputs)
                logits = outputs["logits"]

                if stage == "init":
                    loss_task = F.cross_entropy(logits, targets)
                    eval_logits = logits
                    eval_targets = targets
                else:
                    fake_targets = targets - self._known_classes
                    loss_task = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits = logits[:, self._known_classes :]
                    eval_targets = fake_targets

                loss = loss_task + self._ewc_penalty(self._unwrap_network())
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)
                    total_loss += float(loss.detach().item())

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => Loss {total_loss/len(self.train_loader):.3f}, Train_accy {train_acc:.2f}"
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, self.test_loader)
                info += f", Test_accy {test_acc:.2f}"
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)

        # Save LoRA + head
        save_dir = self.args.get("filepath", "./")
        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_dir, self._cur_task)

        # After-task Fisher + checkpoints (ΔW-space)
        self._checkpoint_lora = self._capture_delta_checkpoints(self._unwrap_network())
        self._checkpoint_head = self._capture_head_checkpoints(self._unwrap_network())
        self._fisher_lora, self._fisher_head = self._compute_fisher(
            self.train_loader, self._unwrap_network()
        )

        # Rehearsal if any
        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._unwrap_network()

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as exc:  # pylint: disable=broad-except
            self._log(f"[EWC_Yaoyue_LoRA][NME] Failed to compute class means: {exc}")

    # ------------------------------------------------------------------#
    # ΔW helpers                                                        #
    # ------------------------------------------------------------------#
    def _collect_lora_targets(self, model: nn.Module) -> List[Dict]:
        """Find all LoRA-wrapped qkv modules and their base weights.

        Returns a list of dict with keys: name, param, module.
        """
        targets: List[Dict] = []
        for mod_name, mod in model.named_modules():
            if isinstance(mod, _LoRA_qkv_timm_train):
                weight = getattr(mod.qkv, "weight", None)
                if weight is None:
                    continue
                targets.append({"name": f"{mod_name}.qkv.weight", "param": weight, "module": mod})
        return targets

    def _delta_from_qkv(self, mod: _LoRA_qkv_timm_train) -> torch.Tensor:
        """Construct ΔW with shape matching the underlying qkv weight [3d, d]."""
        dim = mod.dim
        device = mod.linear_a_q.weight.device
        dtype = mod.linear_a_q.weight.dtype
        delta_q = mod.linear_b_q.weight @ mod.linear_a_q.weight  # [d, d]
        delta_v = mod.linear_b_v.weight @ mod.linear_a_v.weight  # [d, d]
        delta = torch.zeros((dim * 3, dim), device=device, dtype=dtype)
        delta[:dim] = delta_q
        delta[-dim:] = delta_v
        return delta

    # ------------------------------------------------------------------#
    # EWC components                                                    #
    # ------------------------------------------------------------------#
    def _ewc_penalty(self, model: nn.Module):
        pen = None

        # LoRA ΔW term
        if self._fisher_lora is not None and self._checkpoint_lora is not None:
            for target in self._collect_lora_targets(model):
                name = target["name"]
                fisher = self._fisher_lora.get(name)
                ref = self._checkpoint_lora.get(name)
                if fisher is None or ref is None:
                    continue
                d_w = self._delta_from_qkv(target["module"])
                fisher_d = fisher.to(device=d_w.device, dtype=d_w.dtype)
                ref_d = ref.to(device=d_w.device, dtype=d_w.dtype)
                term = (fisher_d * (d_w - ref_d).pow(2)).sum()
                pen = term if pen is None else pen + term

        # Head term (standard EWC)
        if self._fisher_head is not None and self._checkpoint_head is not None:
            param_map = dict(model.named_parameters())
            for name, fisher in self._fisher_head.items():
                if name not in param_map:
                    continue
                p = param_map[name]
                ref = self._checkpoint_head.get(name)
                if ref is None:
                    continue
                ref_m = self._match_tensor(ref, p).to(device=p.device, dtype=p.dtype)
                fisher_m = self._match_tensor(fisher, p).to(device=p.device, dtype=p.dtype)
                diff = p - ref_m
                term = fisher_m * diff.pow(2)
                term = term.sum()
                pen = term if pen is None else pen + term

        if pen is None:
            return torch.tensor(0.0, device=self._device)
        return 0.5 * self._ewc_lambda * pen

    # ------------------------------------------------------------------#
    # Fisher estimation                                                 #
    # ------------------------------------------------------------------#
    def _compute_fisher(self, loader, model: nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        model.eval()

        lora_targets = self._collect_lora_targets(model)
        lora_params = [(t["name"], t["param"]) for t in lora_targets]
        head_params = [(n, p) for n, p in model.named_parameters() if self._is_head_param(n, p)]
        selected = lora_params + head_params
        if len(selected) == 0:
            return {}, {}

        # Temporarily enable grad for base qkv weights
        orig_req = {n: p.requires_grad for n, p in lora_params}
        for _, p in lora_params:
            p.requires_grad_(True)

        fisher_lora: Dict[str, torch.Tensor] = {}
        fisher_head: Dict[str, torch.Tensor] = {}
        total_samples = 0

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= self._ewc_max_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = model(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                grads = torch.autograd.grad(
                    loss,
                    [p for _, p in selected],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

            bs = inputs.shape[0]
            total_samples += bs

            for (name, _), g in zip(selected, grads):
                if g is None:
                    continue
                g2 = (g.detach().cpu() ** 2) * bs
                if name in dict(lora_params):
                    fisher_lora[name] = g2 if name not in fisher_lora else fisher_lora[name] + g2
                else:
                    fisher_head[name] = g2 if name not in fisher_head else fisher_head[name] + g2

        if total_samples > 0:
            for name in fisher_lora:
                fisher_lora[name] = (fisher_lora[name] / float(total_samples)).clamp_min(self._ewc_eps)
            for name in fisher_head:
                fisher_head[name] = (fisher_head[name] / float(total_samples)).clamp_min(self._ewc_eps)

        # Online accumulation with decay gamma
        fisher_lora = self._accumulate_fisher(self._fisher_lora, fisher_lora)
        fisher_head = self._accumulate_fisher(self._fisher_head, fisher_head)

        # Restore requires_grad flags
        for name, p in lora_params:
            p.requires_grad_(orig_req[name])

        if self._is_main_process:
            self._log(f"[EWC_Yaoyue_LoRA] Fisher estimated over {min(len(loader), self._ewc_max_batches)} batches")

        return fisher_lora, fisher_head

    def _accumulate_fisher(self, old: Dict[str, torch.Tensor] | None, new: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if old is None:
            return {k: v.clone() for k, v in new.items()}
        out: Dict[str, torch.Tensor] = {}
        keys = set(old.keys()) | set(new.keys())
        for k in keys:
            a = old.get(k)
            b = new.get(k)
            if a is None:
                out[k] = b.clone()
            elif b is None:
                out[k] = a * self._ewc_gamma
            else:
                out[k] = self._match_tensor(a, b) * self._ewc_gamma + b
        return out

    # ------------------------------------------------------------------#
    # Checkpoints                                                       #
    # ------------------------------------------------------------------#
    def _capture_delta_checkpoints(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        ckpt: Dict[str, torch.Tensor] = {}
        for target in self._collect_lora_targets(model):
            delta = self._delta_from_qkv(target["module"]).detach().cpu().clone()
            ckpt[target["name"]] = delta
        return ckpt

    def _capture_head_checkpoints(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        ckpt: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if self._is_head_param(name, p):
                ckpt[name] = p.detach().cpu().clone()
        return ckpt

    # ------------------------------------------------------------------#
    # Helpers                                                           #
    # ------------------------------------------------------------------#
    def _match_tensor(self, stored: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_cpu = target.detach().cpu()
        if stored.shape == target_cpu.shape:
            return stored
        new = torch.zeros_like(target_cpu)
        if stored.ndim != target_cpu.ndim:
            return new
        slices = tuple(slice(0, min(a, b)) for a, b in zip(stored.shape, target_cpu.shape))
        new[slices] = stored[slices]
        return new

    def _align_head_ewc_buffers(self, model: nn.Module):
        if self._fisher_head is None and self._checkpoint_head is None:
            return
        for name, p in model.named_parameters():
            if not self._is_head_param(name, p):
                continue
            if self._fisher_head is not None and name in self._fisher_head:
                self._fisher_head[name] = self._match_tensor(self._fisher_head[name], p)
            if self._checkpoint_head is not None and name in self._checkpoint_head:
                self._checkpoint_head[name] = self._match_tensor(self._checkpoint_head[name], p)

    def _is_head_param(self, name: str, param: nn.Parameter) -> bool:
        return name.startswith("fc") and param.requires_grad
