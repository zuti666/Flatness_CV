import numpy as np
import torch
import logging
import json
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.inc_net import IncrementalNet
from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from utils.toolkit import tensor2numpy
from optimer_PerturabtionType.util import enable_running_stats, disable_running_stats, generate_pertubation

import timm
from backbone.lora import LoRA_ViT_timm, _LoRA_qkv_timm_train
from types import SimpleNamespace

num_workers = 8
logger = logging.getLogger(__name__)


class Learner(LoraBaseLearner):
    """
    SeqLoRA: Train a single, fixed-size LoRA module sequentially over tasks
    without replay or explicit regularizers. Base backbone remains frozen.
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._lora_initialized = True
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()
        self._base_optimizer_type = self._optimizer_type
        self._optimizer_type_by_task = self._parse_optimizer_type_schedule(
            args.get("optimizer_type_by_task", None)
        )


        # hyperparamter for SAM optimizer
        if self._optimizer_type == "sam":
            self._sam_rho = float(args.get("sam_rho", 0.05))
            self._sam_adaptive = bool(args.get("sam_adaptive", False))

        # Scoped SAM variants for controlled perturbation-support experiments.
        # The base optimizer still updates the ordinary trainable parameters
        # (LoRA A/B and classifier head); only the ascent perturbation support
        # changes.
        elif self._optimizer_type in {"sam_factor", "sam_full", "sam_delta", "sam_all", "sam_random", "sam_frozen"}:
            self._sam_scope_rho = float(args.get("sam_rho", args.get("rho", 0.05)))
            self._sam_scope_adaptive = bool(args.get("sam_adaptive", False))
            self._sam_scope_eps = float(args.get("sam_eps", 1e-12))
            self._sam_delta_basis_eps = float(args.get("sam_delta_basis_eps", 1e-6))
            self._sam_random_seed = int(args.get("sam_random_seed", args.get("seed", [42])[0] if isinstance(args.get("seed", 42), list) else args.get("seed", 42)))
            self._sam_random_basis_cache = {}

        # Scoped random perturbation variants. These use the same support
        # locations as scoped SAM, but replace gradient-ascent directions by
        # normalized Gaussian directions.
        elif self._optimizer_type in {"random_factor", "random_full", "random_delta", "random_all", "random_frozen"}:
            self._random_scope_rho = float(args.get("random_rho", args.get("sam_rho", args.get("rho", 0.05))))
            self._random_scope_eps = float(args.get("random_eps", args.get("sam_eps", 1e-12)))
            self._sam_delta_basis_eps = float(args.get("sam_delta_basis_eps", 1e-6))

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
        elif self._optimizer_type in {"flatlora", "flatlora_full", "faltlora", "faltlora_full"}:
            # Flat-LoRA: optimize a single noisy loss around the effective merged weight.
            self._flatlora_rho = float(args.get("flatlora_rho", args.get("rho", 0.05)))
            self._flatlora_noise_type = str(args.get("flatlora_noise_type", "flatLoRA_Gauss"))
            self._flatlora_use_cosine_schedule = bool(args.get("flatlora_use_cosine_schedule", True))
            self._flatlora_total_steps = 1
            self._flatlora_step_idx = 0
            self._flatlora_warned_no_modules = False
        elif self._optimizer_type in {"mergegam", "mergegam_lora"}:
            # MergeGAM-LoRA: use GAM-style multi-pass optimization, but compute the
            # perturbation geometry in the effective merged q/v weight space.
            self._mergegam_grad_rho = float(args.get("mergegam_grad_rho", args.get("gam_grad_rho", 0.2)))
            self._mergegam_grad_norm_rho = float(
                args.get("mergegam_grad_norm_rho", args.get("gam_grad_norm_rho", 0.2))
            )
            self._mergegam_beta1 = float(
                args.get("mergegam_grad_beta_1", args.get("mergegam_beta1", args.get("gam_grad_beta_1", 1.0)))
            )
            self._mergegam_beta2 = float(
                args.get("mergegam_grad_beta_2", args.get("mergegam_beta2", args.get("gam_grad_beta_2", 1.0)))
            )
            self._mergegam_beta3 = float(
                args.get("mergegam_grad_beta_3", args.get("mergegam_beta3", args.get("gam_grad_beta_3", 1.0)))
            )
            self._mergegam_gamma = float(
                args.get("mergegam_grad_gamma", args.get("mergegam_gamma", args.get("gam_grad_gamma", 0.1)))
            )
            self._mergegam_eps = float(args.get("mergegam_eps", args.get("gam_perturb_eps", 1e-12)))
            self._mergegam_adaptive = bool(args.get("mergegam_adaptive", args.get("gam_adaptive", False)))
            self._mergegam_mask_mode = str(args.get("mergegam_mask_mode", "qv_only")).lower()
            self._mergegam_warned_no_modules = False

        for opt_type in set(self._optimizer_type_by_task or []):
            self._ensure_optimizer_state(opt_type)

    def _parse_optimizer_type_schedule(self, spec):
        if spec is None:
            return None
        if isinstance(spec, str):
            text = spec.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = [item.strip() for item in text.split(",") if item.strip()]
            spec = parsed
        elif isinstance(spec, (tuple, list)):
            spec = list(spec)
        else:
            spec = [spec]
        schedule = [str(item).strip().lower() for item in spec if str(item).strip()]
        return schedule or None

    def _ensure_optimizer_state(self, opt_type: str) -> None:
        if opt_type == "sam":
            self._sam_rho = float(self.args.get("sam_rho", 0.05))
            self._sam_adaptive = bool(self.args.get("sam_adaptive", False))
        elif opt_type in {"sam_factor", "sam_full", "sam_delta", "sam_all", "sam_random", "sam_frozen"}:
            self._sam_scope_rho = float(self.args.get("sam_rho", self.args.get("rho", 0.05)))
            self._sam_scope_adaptive = bool(self.args.get("sam_adaptive", False))
            self._sam_scope_eps = float(self.args.get("sam_eps", 1e-12))
            self._sam_delta_basis_eps = float(self.args.get("sam_delta_basis_eps", 1e-6))
            seed = self.args.get("seed", 42)
            if isinstance(seed, list):
                seed = seed[0] if seed else 42
            self._sam_random_seed = int(self.args.get("sam_random_seed", seed))
            if not hasattr(self, "_sam_random_basis_cache"):
                self._sam_random_basis_cache = {}
        elif opt_type in {"random_factor", "random_full", "random_delta", "random_all", "random_frozen"}:
            self._random_scope_rho = float(
                self.args.get("random_rho", self.args.get("sam_rho", self.args.get("rho", 0.05)))
            )
            self._random_scope_eps = float(self.args.get("random_eps", self.args.get("sam_eps", 1e-12)))
            self._sam_delta_basis_eps = float(self.args.get("sam_delta_basis_eps", 1e-6))
        elif opt_type in {"flatlora", "flatlora_full", "faltlora", "faltlora_full"}:
            self._flatlora_rho = float(self.args.get("flatlora_rho", self.args.get("rho", 0.05)))
            self._flatlora_noise_type = str(self.args.get("flatlora_noise_type", "flatLoRA_Gauss"))
            self._flatlora_use_cosine_schedule = bool(self.args.get("flatlora_use_cosine_schedule", True))
            self._flatlora_total_steps = 1
            self._flatlora_step_idx = 0
            self._flatlora_warned_no_modules = False

    def _set_optimizer_type_for_task(self, task_idx: int) -> None:
        if not self._optimizer_type_by_task:
            return
        schedule = self._optimizer_type_by_task
        opt_type = schedule[task_idx] if task_idx < len(schedule) else schedule[-1]
        self._ensure_optimizer_state(opt_type)
        if opt_type != self._optimizer_type:
            logger.info(
                "[SeqLoRA] task-wise optimizer switch at task %d: %s -> %s",
                task_idx,
                self._optimizer_type,
                opt_type,
            )
        self._optimizer_type = opt_type

    def after_task(self):
        self._known_classes = self._total_classes

        # Release CUDA cache after each task to mitigate fragmentation/OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def incremental_train(self, data_manager):
        self._refresh_distributed_context()
        # Keep a handle for downstream components (e.g., InfoBudget old-train loader)
        self.data_manager = data_manager

        self._cur_task += 1
        self._set_optimizer_type_for_task(self._cur_task)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._log("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        
        self.train_loader = data_manager.build_dataloader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args.get("train_num_workers", 8),
        )

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = data_manager.build_dataloader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args.get("train_num_workers", 8),
        )

        self._train(self.train_loader, self.test_loader)
        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._unwrap_network()

        # Compute class-means over all seen classes for NME evaluation (Class-IL).
        if bool(self.args.get("compute_nme_class_means", True)):
            try:
                self.compute_all_seen_class_means(data_manager)
            except Exception as _nme_exc:  # pylint: disable=broad-except
                self._log(f"[LoRA-Seq][NME] Failed to compute class means: {_nme_exc}")

    
    # def incremental_train2(self, data_manager):
    #     self._refresh_distributed_context()

    #     # self._cur_task += 1
    #     self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
    #     self._network.update_fc(self._total_classes)
    #     self._log("Learning on {}-{}".format(self._known_classes, self._total_classes))

    #     train_dataset = data_manager.get_dataset(
    #         np.arange(self._known_classes, self._total_classes), source="train", mode="train"
    #     )
        
    #     self.train_loader = DataLoader(
    #         train_dataset,
    #         batch_size=self.args["batch_size"],
    #         shuffle=True,
    #         num_workers=self.args.get("train_num_workers", 8)
    #     )

    #     test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
    #     self.test_loader = DataLoader(
    #         test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args.get("train_num_workers", 8)
    #     )

    #     self._train(self.train_loader, self.test_loader)
    #     # self.build_rehearsal_memory(data_manager, self.samples_per_class)
    #     self._network = self._unwrap_network()

    #     # Compute class-means over all seen classes for NME evaluation (Class-IL)
    #     try:
    #         self.compute_all_seen_class_means(data_manager)
    #     except Exception as _nme_exc:  # pylint: disable=broad-except
    #         self._log(f"[LoRA-Seq][NME] Failed to compute class means: {_nme_exc}")

    def finetune_all_data(self, data_manager):
        """Train a single LoRA module on the full dataset (all classes at once)."""
        self._refresh_distributed_context()
        self._cur_task = 0
        self._known_classes = 0
        self._total_classes = data_manager.nb_classes
        self._network.update_fc(self._total_classes)

        network = self._unwrap_network()
        if not self._lora_initialized:
            network.backbone = self.build_lora_backbone()
            network.backbone.to(self._device)
            self._lora_initialized = True
        self._network = network
        # Prepare network for parallelism.
        self._prepare_network()

        full_class_indices = np.arange(0, self._total_classes)
        train_dataset = data_manager.get_dataset(full_class_indices, source="train", mode="train")
        self.train_loader = data_manager.build_dataloader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args.get("train_num_workers", 8),
        )

        test_dataset = data_manager.get_dataset(full_class_indices, source="test", mode="test")
        self.test_loader = data_manager.build_dataloader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args.get("train_num_workers", 8),
        )

        params = [p for p in self._network.parameters() if p.requires_grad]
        
        optimizer = self._build_optimizer(params, stage="full")
        
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

        epochs = int(self.args.get("full_epochs", self.args.get("init_epoch", 1)))
        self._full_finetune(self.train_loader, self.test_loader, optimizer, scheduler, epochs)

        self._network = self._unwrap_network()
        self._known_classes = self._total_classes

    def build_lora_backbone(self, index=True, eval_mode=False):
        rank = int(self.args.get("lora_rank", 10))
        if rank <= 0:
            raise ValueError(f"lora_rank must be > 0, got {rank}")

        backbone_type = self.args.get("backbone_type", "vit_base_patch16_224").lower()

        if "resnet" in backbone_type:
            from backbone.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
            from backbone.lora import LoRA_ResNet
            _resnet_map = {
                "resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,
                "resnet101": resnet101, "resnet152": resnet152,
            }
            fn = _resnet_map.get(backbone_type, resnet50)
            base = fn(pretrained=True, args=self.args)
            lora_layers = self.args.get("lora_layers", None)
            model = LoRA_ResNet(base, r=rank, lora_layers=lora_layers)
            # LoRA A/B are already requires_grad=True from _LoRAConv init
        else:
            # ViT (default)
            model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
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

        for name, _param in model.named_parameters():
            logger.info("[LoRA] param name: %s", name)
        return model

    def _is_flatlora_optimizer(self) -> bool:
        return self._optimizer_type in {"flatlora", "flatlora_full", "faltlora", "faltlora_full"}

    def _is_mergegam_optimizer(self) -> bool:
        return self._optimizer_type in {"mergegam", "mergegam_lora"}

    def _is_scoped_sam_optimizer(self) -> bool:
        return self._optimizer_type in {"sam_factor", "sam_full", "sam_delta", "sam_all", "sam_random", "sam_frozen"}

    def _is_scoped_random_optimizer(self) -> bool:
        return self._optimizer_type in {"random_factor", "random_full", "random_delta", "random_all", "random_frozen"}

    def _reset_flatlora_schedule(self, total_steps: int) -> None:
        self._flatlora_total_steps = max(int(total_steps), 1)
        self._flatlora_step_idx = 0

    def _get_scale_value(self, scale_module) -> float:
        if hasattr(scale_module, "param"):
            param = getattr(scale_module, "param")
            if torch.is_tensor(param):
                return float(param.detach().view(-1)[0].item())
        return 1.0

    def _collect_flatlora_modules(self):
        base_model = self._unwrap_network()
        return [module for module in base_model.modules() if isinstance(module, _LoRA_qkv_timm_train)]

    def _build_effective_qkv_weight(self, module: _LoRA_qkv_timm_train) -> torch.Tensor:
        merged_weight = module.qkv.weight.detach().clone()
        device = merged_weight.device
        dtype = merged_weight.dtype

        for task_idx in range(int(module.task_id)):
            key_a = f"saved_A_{task_idx}"
            key_b = f"saved_B_{task_idx}"
            if key_a not in module.saved_A or key_b not in module.saved_B:
                continue

            saved_A = module.saved_A[key_a]
            saved_B = module.saved_B[key_b]
            layer_pairs = list(zip(saved_A, saved_B))[module.t_layer_i * 2 : module.t_layer_i * 2 + 2]
            if len(layer_pairs) < 2:
                continue
            q_entry, v_entry = layer_pairs
            A_q, B_q = q_entry
            A_v, B_v = v_entry
            hist_scale = self._get_scale_value(module.scaling_factor_prev[task_idx]) if task_idx < len(module.scaling_factor_prev) else 1.0

            delta_q = B_q.weight.detach().to(device=device, dtype=dtype) @ A_q.weight.detach().to(device=device, dtype=dtype)
            delta_v = B_v.weight.detach().to(device=device, dtype=dtype) @ A_v.weight.detach().to(device=device, dtype=dtype)
            merged_weight[: module.dim, :].add_(hist_scale * delta_q)
            merged_weight[-module.dim :, :].add_(hist_scale * delta_v)

        cur_scale = self._get_scale_value(module.scaling_factor[0])
        delta_q_cur = module.linear_b_q.weight.detach().to(device=device, dtype=dtype) @ module.linear_a_q.weight.detach().to(device=device, dtype=dtype)
        delta_v_cur = module.linear_b_v.weight.detach().to(device=device, dtype=dtype) @ module.linear_a_v.weight.detach().to(device=device, dtype=dtype)
        merged_weight[: module.dim, :].add_(cur_scale * delta_q_cur)
        merged_weight[-module.dim :, :].add_(cur_scale * delta_v_cur)
        return merged_weight

    def _current_flatlora_std(self) -> float:
        base_rho = float(self._flatlora_rho)
        if (not self._flatlora_use_cosine_schedule) or self._flatlora_total_steps <= 1:
            return base_rho
        progress = min(max(float(self._flatlora_step_idx) / float(self._flatlora_total_steps), 0.0), 1.0)
        factor = 0.5 * (1.0 - np.cos(progress * np.pi))
        return base_rho * float(factor)

    def _apply_flatlora_noise(self, std: float):
        flatlora_modules = self._collect_flatlora_modules()
        if not flatlora_modules:
            if not self._flatlora_warned_no_modules:
                self._log("[FlatLoRA] No mergeable LoRA-qkv modules found; falling back to the clean objective.")
                self._flatlora_warned_no_modules = True
            return []

        injected = []
        with torch.no_grad():
            for module in flatlora_modules:
                effective_weight = self._build_effective_qkv_weight(module)
                noise = generate_pertubation(
                    effective_weight,
                    pertubation_mode=self._flatlora_noise_type,
                    std=float(std),
                )
                module.qkv.weight.data.add_(noise)
                injected.append((module.qkv.weight, noise))
        return injected

    def _revert_flatlora_noise(self, injected) -> None:
        if not injected:
            return
        with torch.no_grad():
            for weight, noise in injected:
                weight.data.sub_(noise)

    def _flatlora_step(self, optimizer, inputs, targets, class_offset: int = 0):
        optimizer.zero_grad()
        std = self._current_flatlora_std()
        injected = self._apply_flatlora_noise(std) if std > 0 else []
        try:
            outputs = self._network(inputs)
            logits = outputs["logits"]
            if class_offset > 0:
                loss = F.cross_entropy(logits[:, class_offset:], targets)
            else:
                loss = F.cross_entropy(logits, targets)
            loss.backward()
        finally:
            self._revert_flatlora_noise(injected)
        optimizer.step()
        self._flatlora_step_idx += 1
        return logits.detach(), float(loss.detach().item())

    def _forward_loss_for_scope(self, inputs, targets, class_offset: int = 0):
        outputs = self._network(inputs)
        logits = outputs["logits"]
        if class_offset > 0:
            loss = F.cross_entropy(logits[:, class_offset:], targets)
        else:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def _collect_lora_factor_params(self):
        base_model = self._unwrap_network()
        out = []
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            if ("linear_a" in name) or ("linear_b" in name):
                out.append((name, param))
        return out

    def _collect_all_named_params_with_grad_enabled(self):
        base_model = self._unwrap_network()
        states = []
        named_params = []
        for name, param in base_model.named_parameters():
            states.append((param, bool(param.requires_grad)))
            if not param.requires_grad:
                param.requires_grad_(True)
            named_params.append((name, param))
        return named_params, states

    def _collect_frozen_named_params_with_grad_enabled(self):
        base_model = self._unwrap_network()
        states = []
        named_params = []
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                continue
            states.append((param, False))
            param.requires_grad_(True)
            named_params.append((name, param))
        return named_params, states

    def _restore_requires_grad_states(self, states) -> None:
        for param, requires_grad in states:
            if bool(param.requires_grad) != bool(requires_grad):
                param.requires_grad_(requires_grad)

    def _collect_all_named_params_no_grad_change(self):
        base_model = self._unwrap_network()
        return [(name, param) for name, param in base_model.named_parameters()]

    def _collect_frozen_named_params_no_grad_change(self):
        base_model = self._unwrap_network()
        return [(name, param) for name, param in base_model.named_parameters() if not param.requires_grad]

    def _sam_named_param_norm(self, named_params) -> torch.Tensor:
        terms = []
        device = None
        for _, param in named_params:
            if param.grad is None:
                continue
            device = param.grad.device
            if self._sam_scope_adaptive:
                terms.append((torch.abs(param) * param.grad).norm(p=2))
            else:
                terms.append(param.grad.norm(p=2))
        if not terms:
            return torch.zeros((), device=device or self._device)
        return torch.norm(torch.stack([term.to(terms[0].device) for term in terms]), p=2)

    def _apply_sam_named_param_perturb(self, named_params):
        grad_norm = self._sam_named_param_norm(named_params)
        scale = float(self._sam_scope_rho) / (float(grad_norm.detach().item()) + float(self._sam_scope_eps))
        injected = []
        with torch.no_grad():
            for _, param in named_params:
                if param.grad is None:
                    continue
                rescale = torch.pow(param, 2) if self._sam_scope_adaptive else 1.0
                perturb = rescale * param.grad * scale
                param.add_(perturb)
                injected.append((param, perturb))
        return injected

    def _revert_sam_param_perturb(self, injected) -> None:
        if not injected:
            return
        with torch.no_grad():
            for param, perturb in injected:
                param.sub_(perturb)

    @staticmethod
    def _orthonormal_columns_for_sam(mat: torch.Tensor, eps: float) -> torch.Tensor:
        if mat.ndim != 2 or mat.numel() == 0:
            rows = int(mat.shape[0]) if mat.ndim >= 1 else 0
            return mat.new_zeros((rows, 0))
        mat_f = mat.float()
        try:
            u, s, _ = torch.linalg.svd(mat_f, full_matrices=False)
        except Exception:
            try:
                u, _ = torch.linalg.qr(mat_f, mode="reduced")
                return u.to(dtype=mat.dtype)
            except Exception:
                return mat.new_zeros((int(mat.shape[0]), 0))
        if s.numel() == 0:
            return mat.new_zeros((int(mat.shape[0]), 0))
        tol = float(eps) * max(float(s.max().item()), 1.0)
        keep = s > tol
        if not bool(keep.any()):
            return mat.new_zeros((int(mat.shape[0]), 0))
        return u[:, keep].to(dtype=mat.dtype)

    def _project_grad_to_current_lora_tangent(
        self,
        grad_w: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        U = self._orthonormal_columns_for_sam(B.to(device=grad_w.device, dtype=grad_w.dtype), self._sam_delta_basis_eps)
        V = self._orthonormal_columns_for_sam(A.t().to(device=grad_w.device, dtype=grad_w.dtype), self._sam_delta_basis_eps)
        if U.shape[1] == 0 and V.shape[1] == 0:
            return torch.zeros_like(grad_w)
        if U.shape[1] == 0:
            return (grad_w @ V) @ V.t()
        if V.shape[1] == 0:
            return U @ (U.t() @ grad_w)
        p_b_g = U @ (U.t() @ grad_w)
        g_p_a = (grad_w @ V) @ V.t()
        p_b_g_p_a = U @ (U.t() @ grad_w @ V) @ V.t()
        return p_b_g + g_p_a - p_b_g_p_a

    def _get_random_matched_bases(
        self,
        cache_key,
        out_dim: int,
        in_dim: int,
        rank: int,
        device,
        dtype,
    ):
        rank = max(0, min(int(rank), int(out_dim), int(in_dim)))
        if rank <= 0:
            return (
                torch.zeros((out_dim, 0), device=device, dtype=dtype),
                torch.zeros((in_dim, 0), device=device, dtype=dtype),
            )

        if cache_key not in self._sam_random_basis_cache:
            seed = int(self._sam_random_seed)
            for part in cache_key:
                if isinstance(part, str):
                    part_val = sum((idx + 1) * ord(ch) for idx, ch in enumerate(part))
                elif isinstance(part, (tuple, list)):
                    part_val = 0
                    for idx, item in enumerate(part):
                        part_val += (idx + 1) * int(item)
                else:
                    part_val = int(part)
                seed = (seed * 1315423911 + part_val) & 0x7FFFFFFF
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            rand_u = torch.randn(out_dim, rank, generator=generator, dtype=torch.float32)
            rand_v = torch.randn(in_dim, rank, generator=generator, dtype=torch.float32)
            q_u, _ = torch.linalg.qr(rand_u, mode="reduced")
            q_v, _ = torch.linalg.qr(rand_v, mode="reduced")
            self._sam_random_basis_cache[cache_key] = (q_u.cpu(), q_v.cpu())

        basis_u, basis_v = self._sam_random_basis_cache[cache_key]
        return basis_u.to(device=device, dtype=dtype), basis_v.to(device=device, dtype=dtype)

    def _project_grad_to_random_matched_tangent(
        self,
        grad_w: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        cache_key,
    ) -> torch.Tensor:
        out_dim, in_dim = int(grad_w.shape[0]), int(grad_w.shape[1])
        rank = int(A.shape[0]) if A.ndim == 2 else 0
        U, V = self._get_random_matched_bases(
            cache_key,
            out_dim,
            in_dim,
            rank,
            grad_w.device,
            grad_w.dtype,
        )
        if U.shape[1] == 0 and V.shape[1] == 0:
            return torch.zeros_like(grad_w)
        if U.shape[1] == 0:
            return (grad_w @ V) @ V.t()
        if V.shape[1] == 0:
            return U @ (U.t() @ grad_w)
        p_u_g = U @ (U.t() @ grad_w)
        g_p_v = (grad_w @ V) @ V.t()
        p_u_g_p_v = U @ (U.t() @ grad_w @ V) @ V.t()
        return p_u_g + g_p_v - p_u_g_p_v

    def _collect_effective_qkv_grad_records(self, modules, scope: str):
        records = []
        norm2 = None
        for module in modules:
            grad = module.qkv.weight.grad
            if grad is None:
                continue
            if scope == "sam_full":
                projected = grad.detach().clone()
                records.append((module.qkv.weight, 0, grad.shape[0], projected))
                term = torch.sum(projected * projected)
                norm2 = term if norm2 is None else norm2 + term
                continue

            dim = int(getattr(module, "dim", grad.shape[0] // 3))
            specs = [
                ("q", 0, dim, module.linear_a_q.weight, module.linear_b_q.weight),
                ("v", grad.shape[0] - dim, grad.shape[0], module.linear_a_v.weight, module.linear_b_v.weight),
            ]
            for branch_name, row_start, row_end, A_param, B_param in specs:
                grad_block = grad[row_start:row_end, :].detach()
                if scope == "sam_random":
                    projected = self._project_grad_to_random_matched_tangent(
                        grad_block,
                        A_param.detach(),
                        B_param.detach(),
                        cache_key=(int(getattr(module, "t_layer_i", 0)), branch_name, tuple(grad_block.shape), int(A_param.shape[0])),
                    )
                else:
                    projected = self._project_grad_to_current_lora_tangent(
                        grad_block,
                        A_param.detach(),
                        B_param.detach(),
                    )
                if not torch.isfinite(projected).all():
                    continue
                block_norm2 = torch.sum(projected * projected)
                norm2 = block_norm2 if norm2 is None else norm2 + block_norm2
                records.append((module.qkv.weight, row_start, row_end, projected))
        if norm2 is None:
            return [], torch.zeros((), device=self._device)
        return records, torch.sqrt(norm2)

    def _apply_effective_qkv_sam_perturb(self, records, grad_norm: torch.Tensor):
        scale = float(self._sam_scope_rho) / (float(grad_norm.detach().item()) + float(self._sam_scope_eps))
        injected = []
        with torch.no_grad():
            for weight, row_start, row_end, projected in records:
                perturb = projected * scale
                weight.data[row_start:row_end, :].add_(perturb)
                injected.append((weight, row_start, row_end, perturb))
        return injected

    def _revert_effective_qkv_sam_perturb(self, injected) -> None:
        if not injected:
            return
        with torch.no_grad():
            for weight, row_start, row_end, perturb in injected:
                weight.data[row_start:row_end, :].sub_(perturb)

    def _apply_random_named_param_perturb(self, named_params):
        records = []
        norm2 = None
        for _, param in named_params:
            if param.numel() == 0 or not torch.is_floating_point(param):
                continue
            noise = torch.randn_like(param)
            term = torch.sum(noise * noise)
            norm2 = term if norm2 is None else norm2 + term
            records.append((param, noise))
        if norm2 is None:
            return []
        scale = float(self._random_scope_rho) / (float(torch.sqrt(norm2).detach().item()) + float(self._random_scope_eps))
        injected = []
        with torch.no_grad():
            for param, noise in records:
                perturb = noise * scale
                param.add_(perturb)
                injected.append((param, perturb))
        return injected

    def _collect_effective_qkv_random_records(self, modules, scope: str):
        records = []
        norm2 = None
        for module in modules:
            weight = module.qkv.weight
            if scope == "random_full":
                noise = torch.randn_like(weight)
                records.append((weight, 0, noise.shape[0], noise))
                term = torch.sum(noise * noise)
                norm2 = term if norm2 is None else norm2 + term
                continue

            dim = int(getattr(module, "dim", weight.shape[0] // 3))
            specs = [
                (0, dim, module.linear_a_q.weight, module.linear_b_q.weight),
                (weight.shape[0] - dim, weight.shape[0], module.linear_a_v.weight, module.linear_b_v.weight),
            ]
            for row_start, row_end, A_param, B_param in specs:
                raw_noise = torch.randn(
                    (row_end - row_start, weight.shape[1]),
                    device=weight.device,
                    dtype=weight.dtype,
                )
                projected = self._project_grad_to_current_lora_tangent(
                    raw_noise,
                    A_param.detach(),
                    B_param.detach(),
                )
                if not torch.isfinite(projected).all():
                    continue
                term = torch.sum(projected * projected)
                norm2 = term if norm2 is None else norm2 + term
                records.append((weight, row_start, row_end, projected))
        if norm2 is None:
            return [], torch.zeros((), device=self._device)
        return records, torch.sqrt(norm2)

    def _apply_effective_qkv_random_perturb(self, records, noise_norm: torch.Tensor):
        scale = float(self._random_scope_rho) / (float(noise_norm.detach().item()) + float(self._random_scope_eps))
        injected = []
        with torch.no_grad():
            for weight, row_start, row_end, noise in records:
                perturb = noise * scale
                weight.data[row_start:row_end, :].add_(perturb)
                injected.append((weight, row_start, row_end, perturb))
        return injected

    def _scoped_sam_step(self, optimizer, inputs, targets, class_offset: int = 0):
        scope = self._optimizer_type
        optimizer.zero_grad()
        req_states = []
        module_state = []
        perturb = []
        effective_perturb = []
        modules = []
        all_named_for_cleanup = []
        try:
            if scope == "sam_all":
                named_params, req_states = self._collect_all_named_params_with_grad_enabled()
                all_named_for_cleanup = named_params
            elif scope == "sam_frozen":
                named_params, req_states = self._collect_frozen_named_params_with_grad_enabled()
                all_named_for_cleanup = named_params
            elif scope == "sam_factor":
                named_params = self._collect_lora_factor_params()
                all_named_for_cleanup = named_params
            else:
                named_params = []
                modules = self._collect_flatlora_modules()
                module_state = self._prepare_mergegam_modules(modules)

            for _, param in all_named_for_cleanup:
                param.grad = None

            enable_running_stats(self._network)
            logits, loss = self._forward_loss_for_scope(inputs, targets, class_offset=class_offset)
            loss.backward()

            if scope in {"sam_factor", "sam_all", "sam_frozen"}:
                perturb = self._apply_sam_named_param_perturb(named_params)
            else:
                records, grad_norm = self._collect_effective_qkv_grad_records(modules, scope)
                effective_perturb = self._apply_effective_qkv_sam_perturb(records, grad_norm)

            optimizer.zero_grad()
            for _, param in all_named_for_cleanup:
                param.grad = None
            for module in modules:
                module.qkv.weight.grad = None
            if scope in {"sam_all", "sam_frozen"}:
                self._restore_requires_grad_states(req_states)
                req_states = []
            disable_running_stats(self._network)
            perturbed_logits, second_loss = self._forward_loss_for_scope(inputs, targets, class_offset=class_offset)
            second_loss.backward()

            self._revert_sam_param_perturb(perturb)
            perturb = []
            self._revert_effective_qkv_sam_perturb(effective_perturb)
            effective_perturb = []
            self._restore_requires_grad_states(req_states)
            req_states = []
            self._restore_mergegam_modules(module_state)
            module_state = []
            optimizer.step()
            return perturbed_logits.detach(), float(second_loss.detach().item())
        finally:
            self._revert_sam_param_perturb(perturb)
            self._revert_effective_qkv_sam_perturb(effective_perturb)
            self._restore_requires_grad_states(req_states)
            self._restore_mergegam_modules(module_state)
            optimizer.zero_grad()
            for _, param in all_named_for_cleanup:
                param.grad = None
            for module in modules:
                module.qkv.weight.grad = None
            enable_running_stats(self._network)

    def _scoped_random_step(self, optimizer, inputs, targets, class_offset: int = 0):
        scope = self._optimizer_type
        optimizer.zero_grad()
        perturb = []
        effective_perturb = []
        modules = []
        try:
            if scope == "random_all":
                named_params = self._collect_all_named_params_no_grad_change()
            elif scope == "random_frozen":
                named_params = self._collect_frozen_named_params_no_grad_change()
            elif scope == "random_factor":
                named_params = self._collect_lora_factor_params()
            else:
                named_params = []
                modules = self._collect_flatlora_modules()

            if scope in {"random_factor", "random_all", "random_frozen"}:
                perturb = self._apply_random_named_param_perturb(named_params)
            else:
                records, noise_norm = self._collect_effective_qkv_random_records(modules, scope)
                effective_perturb = self._apply_effective_qkv_random_perturb(records, noise_norm)

            disable_running_stats(self._network)
            logits, loss = self._forward_loss_for_scope(inputs, targets, class_offset=class_offset)
            loss.backward()

            self._revert_sam_param_perturb(perturb)
            perturb = []
            self._revert_effective_qkv_sam_perturb(effective_perturb)
            effective_perturb = []
            optimizer.step()
            return logits.detach(), float(loss.detach().item())
        finally:
            self._revert_sam_param_perturb(perturb)
            self._revert_effective_qkv_sam_perturb(effective_perturb)
            optimizer.zero_grad()
            enable_running_stats(self._network)

    def _collect_optimizer_params(self, optimizer):
        params = []
        seen = set()
        for group in optimizer.param_groups:
            for param in group["params"]:
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                params.append(param)
        return params

    def _prepare_mergegam_modules(self, modules):
        state = []
        for module in modules:
            weight = module.qkv.weight
            state.append((weight, bool(weight.requires_grad)))
            weight.requires_grad_(True)
            weight.grad = None
        return state

    def _restore_mergegam_modules(self, state) -> None:
        for weight, requires_grad in state:
            weight.grad = None
            weight.requires_grad_(requires_grad)

    def _build_mergegam_mask(self, module: _LoRA_qkv_timm_train, like_tensor: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(like_tensor)
        if self._mergegam_mask_mode in {"qv_only", "qv", "lora", "delta"}:
            mask[: module.dim, :] = 1
            mask[-module.dim :, :] = 1
            return mask
        if self._mergegam_mask_mode in {"full", "all", "merged"}:
            mask.fill_(1)
            return mask
        raise ValueError(f"Unsupported mergegam_mask_mode: {self._mergegam_mask_mode}")

    def _zero_mergegam_grads(self, optimizer, modules) -> None:
        optimizer.zero_grad()
        for module in modules:
            module.qkv.weight.grad = None

    def _capture_param_grads(self, optimizer_params):
        grads = {}
        for param in optimizer_params:
            if param.grad is not None:
                grads[id(param)] = param.grad.detach().clone()
        return grads

    def _merge_linear_combo(self, modules, left, right, left_w: float, right_w: float):
        combined = {}
        for module in modules:
            left_grad = left.get(module, None)
            right_grad = right.get(module, None)
            if left_grad is None and right_grad is None:
                continue
            if left_grad is None:
                combined[module] = right_grad * right_w
            elif right_grad is None:
                combined[module] = left_grad * left_w
            else:
                combined[module] = left_grad * left_w + right_grad * right_w
        return combined

    def _merge_diff(self, modules, left, right):
        return self._merge_linear_combo(modules, left, right, 1.0, -1.0)

    def _merge_inner_product(self, modules, left, right) -> torch.Tensor:
        inner = None
        device = None
        dtype = None
        for module in modules:
            left_grad = left.get(module, None)
            right_grad = right.get(module, None)
            if left_grad is None or right_grad is None:
                continue
            if device is None:
                device = left_grad.device
                dtype = left_grad.dtype
            term = torch.sum(left_grad * right_grad)
            inner = term if inner is None else inner + term
        if inner is None:
            inner = torch.zeros((), device=device or "cpu", dtype=dtype or torch.float32)
        return inner

    def _merge_grad_norm(self, modules, grads, weight_adaptive: bool = False) -> torch.Tensor:
        norm = None
        device = None
        dtype = None
        for module in modules:
            grad = grads.get(module, None)
            if grad is None:
                continue
            if device is None:
                device = grad.device
                dtype = grad.dtype

            if weight_adaptive:
                effective_weight = self._build_effective_qkv_weight(module).to(device=grad.device, dtype=grad.dtype)
                effective_weight = effective_weight * self._build_mergegam_mask(module, effective_weight)
                term = torch.sum((grad * torch.abs(effective_weight)) ** 2)
            else:
                term = torch.sum(grad ** 2)
            norm = term if norm is None else norm + term

        if norm is None:
            norm = torch.zeros((), device=device or "cpu", dtype=dtype or torch.float32)
        return torch.sqrt(norm)

    def _apply_mergegam_perturb(self, modules, grads, rho: float):
        if float(rho) <= 0.0:
            return []

        grad_norm = self._merge_grad_norm(modules, grads, weight_adaptive=self._mergegam_adaptive)
        scale = float(rho) / (float(grad_norm.detach().item()) + float(self._mergegam_eps))
        perturbations = []

        with torch.no_grad():
            for module in modules:
                grad = grads.get(module, None)
                if grad is None:
                    continue

                perturb = grad.detach().clone() * scale
                if self._mergegam_adaptive:
                    effective_weight = self._build_effective_qkv_weight(module).to(device=grad.device, dtype=grad.dtype)
                    effective_weight = effective_weight * self._build_mergegam_mask(module, effective_weight)
                    perturb = perturb * (effective_weight ** 2)

                module.qkv.weight.data.add_(perturb)
                perturbations.append((module.qkv.weight, perturb))
        return perturbations

    def _revert_mergegam_perturb(self, perturbations) -> None:
        if not perturbations:
            return
        with torch.no_grad():
            for weight, perturb in perturbations:
                weight.data.sub_(perturb)

    def _forward_backward_capture_merge_and_param_grads(
        self,
        optimizer,
        modules,
        optimizer_params,
        inputs,
        targets,
        class_offset: int = 0,
    ):
        self._zero_mergegam_grads(optimizer, modules)
        outputs = self._network(inputs)
        logits = outputs["logits"]
        if class_offset > 0:
            loss = F.cross_entropy(logits[:, class_offset:], targets)
        else:
            loss = F.cross_entropy(logits, targets)
        loss.backward()

        merge_grads = {}
        for module in modules:
            grad = module.qkv.weight.grad
            if grad is None:
                grad = torch.zeros_like(module.qkv.weight)
            grad = grad.detach().clone()
            grad.mul_(self._build_mergegam_mask(module, grad))
            merge_grads[module] = grad

        param_grads = self._capture_param_grads(optimizer_params)
        return logits.detach(), float(loss.detach().item()), merge_grads, param_grads

    def _param_linear_combo(self, optimizer_params, left, right, left_w, right_w):
        combined = {}
        for param in optimizer_params:
            param_id = id(param)
            left_grad = left.get(param_id, None)
            right_grad = right.get(param_id, None)
            if left_grad is None and right_grad is None:
                continue
            if left_grad is None:
                combined[param_id] = right_grad * right_w
            elif right_grad is None:
                combined[param_id] = left_grad * left_w
            else:
                combined[param_id] = left_grad * left_w + right_grad * right_w
        return combined

    def _param_scale(self, optimizer_params, grads, scale):
        scaled = {}
        for param in optimizer_params:
            param_id = id(param)
            grad = grads.get(param_id, None)
            if grad is None:
                continue
            scaled[param_id] = grad * scale
        return scaled

    def _param_sub(self, optimizer_params, left, right):
        return self._param_linear_combo(optimizer_params, left, right, 1.0, -1.0)

    def _write_back_param_grads(self, optimizer_params, grads) -> None:
        for param in optimizer_params:
            param_id = id(param)
            grad = grads.get(param_id, None)
            if grad is None:
                param.grad = None
                continue
            if param.grad is None:
                param.grad = grad.detach().clone()
            else:
                param.grad.data.copy_(grad)

    def _mergegam_fallback_step(self, optimizer, inputs, targets, class_offset: int = 0):
        optimizer.zero_grad()
        outputs = self._network(inputs)
        logits = outputs["logits"]
        if class_offset > 0:
            loss = F.cross_entropy(logits[:, class_offset:], targets)
        else:
            loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        return logits.detach(), float(loss.detach().item())

    def _mergegam_step(self, optimizer, inputs, targets, class_offset: int = 0):
        modules = self._collect_flatlora_modules()
        if not modules:
            if not self._mergegam_warned_no_modules:
                self._log("[MergeGAM] No LoRA qkv modules found; falling back to the clean objective.")
                self._mergegam_warned_no_modules = True
            return self._mergegam_fallback_step(optimizer, inputs, targets, class_offset=class_offset)

        optimizer_params = self._collect_optimizer_params(optimizer)
        module_state = self._prepare_mergegam_modules(modules)
        e0, e12, e2 = None, None, None

        enable_running_stats(self._network)
        try:
            logits0, loss0, g_merge_0, g_param_0 = self._forward_backward_capture_merge_and_param_grads(
                optimizer,
                modules,
                optimizer_params,
                inputs,
                targets,
                class_offset=class_offset,
            )

            e0 = self._apply_mergegam_perturb(modules, g_merge_0, self._mergegam_grad_rho)
            disable_running_stats(self._network)
            _, _, g_merge_1, g_param_1 = self._forward_backward_capture_merge_and_param_grads(
                optimizer,
                modules,
                optimizer_params,
                inputs,
                targets,
                class_offset=class_offset,
            )
            self._revert_mergegam_perturb(e0)
            e0 = None

            g_merge_diff = self._merge_diff(modules, g_merge_1, g_merge_0)
            e12 = self._apply_mergegam_perturb(modules, g_merge_diff, self._mergegam_grad_norm_rho)
            _, _, g_merge_2, g_param_2 = self._forward_backward_capture_merge_and_param_grads(
                optimizer,
                modules,
                optimizer_params,
                inputs,
                targets,
                class_offset=class_offset,
            )

            e2 = self._apply_mergegam_perturb(modules, g_merge_2, self._mergegam_grad_rho)
            _, _, g_merge_3, g_param_3 = self._forward_backward_capture_merge_and_param_grads(
                optimizer,
                modules,
                optimizer_params,
                inputs,
                targets,
                class_offset=class_offset,
            )
            self._revert_mergegam_perturb(e2)
            e2 = None
            self._revert_mergegam_perturb(e12)
            e12 = None

            pro_merge = self._merge_linear_combo(modules, g_merge_0, g_merge_2, 1.0, abs(self._mergegam_beta2))
            upd_merge = self._merge_linear_combo(
                modules,
                g_merge_1,
                g_merge_3,
                self._mergegam_beta1,
                self._mergegam_beta3,
            )
            inner_prod = self._merge_inner_product(modules, pro_merge, upd_merge)
            old_norm = self._merge_grad_norm(modules, pro_merge, weight_adaptive=False)
            new_norm = self._merge_grad_norm(modules, upd_merge, weight_adaptive=False)
            cosine = inner_prod / (old_norm * new_norm + self._mergegam_eps)
            alpha = cosine * old_norm / (new_norm + self._mergegam_eps)

            pro_param = self._param_linear_combo(
                optimizer_params,
                g_param_0,
                g_param_2,
                1.0,
                abs(self._mergegam_beta2),
            )
            upd_param = self._param_linear_combo(
                optimizer_params,
                g_param_1,
                g_param_3,
                self._mergegam_beta1,
                self._mergegam_beta3,
            )
            vertical_param = self._param_sub(
                optimizer_params,
                pro_param,
                self._param_scale(optimizer_params, upd_param, alpha),
            )
            final_param = self._param_sub(
                optimizer_params,
                upd_param,
                self._param_scale(optimizer_params, vertical_param, self._mergegam_gamma),
            )

            self._zero_mergegam_grads(optimizer, modules)
            self._write_back_param_grads(optimizer_params, final_param)
            optimizer.step()
        finally:
            self._revert_mergegam_perturb(e2)
            self._revert_mergegam_perturb(e12)
            self._revert_mergegam_perturb(e0)
            self._zero_mergegam_grads(optimizer, modules)
            self._restore_mergegam_modules(module_state)
            enable_running_stats(self._network)

        return logits0, float(loss0)

    def _split_param_groups(self, stage: str):
        """Return param groups, optionally splitting FC head vs LoRA backbone.

        When ``fc_lrate`` (or ``init_fc_lrate`` for stage='init') is present in
        args, returns a list of two param-group dicts so the FC head trains at a
        higher learning rate than the LoRA adapters.  Otherwise returns a plain
        list of tensors (existing behaviour, fully backward-compatible).

        The LoRA group uses the default lr set by ``_build_optimizer``
        (``init_lr`` / ``lrate``).  The FC group overrides with ``fc_lrate``
        (or ``init_fc_lrate`` for task-0).
        """
        fc_lr_key = "init_fc_lrate" if stage == "init" else "fc_lrate"
        fc_lr = self.args.get(fc_lr_key, self.args.get("fc_lrate", None))

        all_trainable = [(name, p) for name, p in self._network.named_parameters()
                         if p.requires_grad]

        if fc_lr is None:
            return [p for _, p in all_trainable]

        fc_params   = [p for name, p in all_trainable
                       if name.startswith("fc.") or ".fc." in name]
        lora_params = [p for name, p in all_trainable
                       if not (name.startswith("fc.") or ".fc." in name)]

        param_groups = [{"params": lora_params}]
        if fc_params:
            param_groups.append({"params": fc_params, "lr": float(fc_lr)})
        return param_groups

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

            params = self._split_param_groups(stage="init")
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
            # remember base LR for potential LR-following std
            if self._optimizer_type in {"arwp", "rwp"}:
                self._rwp_lr0 = float(lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._network = network
            self._prepare_network()

            params = self._split_param_groups(stage="update")
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

        # Persist LoRA parameters and classifier head for this task
        save_dir = self.args.get("filepath", "./")
        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        if hasattr(backbone, "save_lora_parameters"):
            backbone.save_lora_parameters(save_dir, self._cur_task)
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(save_dir, self._cur_task)

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
                        loss.backward()  # Required so p.grad is populated.
                        return outputs, loss_value
                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()
                elif self._is_scoped_sam_optimizer():
                    logits, loss_value = self._scoped_sam_step(optimizer, inputs, targets)
                    losses += loss_value
                elif self._is_scoped_random_optimizer():
                    logits, loss_value = self._scoped_random_step(optimizer, inputs, targets)
                    losses += loss_value
                elif self._is_mergegam_optimizer():
                    logits, loss_value = self._mergegam_step(optimizer, inputs, targets)
                    losses += loss_value
                elif self._is_flatlora_optimizer():
                    logits, loss_value = self._flatlora_step(optimizer, inputs, targets)
                    losses += loss_value
                elif self._optimizer_type == "rwp":
                    # True RWP during full finetune
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

                    # clean pass
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

                    # noisy pass
                    disable_running_stats(self._network)
                    noise_dict = {}
                    with torch.no_grad():
                        
                        std_for_noise = float(self.args.get("noise_std", rwp_std))
                        for name, p in base_model.named_parameters():
                            if self._rwp_range == "lora":
                                if not p.requires_grad:
                                    continue
                                if p.numel() == 0:
                                    continue
                            fisher_param = None
                            if hasattr(self, "_rwp_fisher") and (name in self._rwp_fisher):
                                fisher_param = self._rwp_fisher[name]
                            e = generate_pertubation(p, pertubation_mode=self.rwp_noise_type, std=std_for_noise, fisher_param=fisher_param, fisher_scaler=float(self._rwp_eta))
                            p.data.add_(e)
                            noise_dict[name] = e

                    optimizer.zero_grad()
                    outputs_noisy = self._network(inputs)
                    logits_noisy = outputs_noisy["logits"]
                    loss_noisy = F.cross_entropy(logits_noisy, targets)
                    loss_noisy.backward()

                    # update fisher
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

                    # revert noise
                    with torch.no_grad():
                        for name, p in base_model.named_parameters():
                            if name in noise_dict:
                                p.data.sub_(noise_dict[name])

                    # mix and step
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
                    losses += (lam * float(loss_noisy.detach().item()) + (1.0 - lam) * float(loss_clean.detach().item()))
                elif self._optimizer_type == "arwp":
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else float(optimizer.param_groups[0]["lr"])
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"]) 
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr))
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

    def _full_finetune(self, train_loader, test_loader, optimizer, scheduler, epochs):
        if self._is_flatlora_optimizer():
            self._reset_flatlora_schedule(len(train_loader) * max(int(epochs), 1))
        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
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
                        loss.backward()  # Required so p.grad is populated.
                        return outputs, loss_value
                    outputs, loss_value = optimizer.step(closure=closure)
                    losses += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                    logits = outputs["logits"].detach()
                elif self._is_scoped_sam_optimizer():
                    logits, loss_value = self._scoped_sam_step(optimizer, inputs, targets)
                    losses += loss_value
                elif self._is_scoped_random_optimizer():
                    logits, loss_value = self._scoped_random_step(optimizer, inputs, targets)
                    losses += loss_value
                elif self._is_mergegam_optimizer():
                    logits, loss_value = self._mergegam_step(optimizer, inputs, targets)
                    losses += loss_value
                elif self._is_flatlora_optimizer():
                    logits, loss_value = self._flatlora_step(optimizer, inputs, targets)
                    losses += loss_value
                elif self._optimizer_type == "rwp":
                    # True RWP: clean/noisy passes and gradient mix for full finetune
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

                    # Clean pass
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

                    # Noisy pass
                    disable_running_stats(self._network)
                    noise_dict = {}
                    with torch.no_grad():
                       
                        std_for_noise = float(self.args.get("noise_std", rwp_std))
                        for name, p in base_model.named_parameters():
                            if self._rwp_range == "lora":
                                if not p.requires_grad or p.numel() == 0:
                                    continue
                            fisher_param = self._rwp_fisher.get(name, None) if hasattr(self, "_rwp_fisher") else None
                            e = generate_pertubation(p, pertubation_mode=self.rwp_noise_type, std=std_for_noise, fisher_param=fisher_param, fisher_scaler=float(self._rwp_eta))
                            p.data.add_(e)
                            noise_dict[name] = e

                    optimizer.zero_grad()
                    outputs_noisy = self._network(inputs)
                    logits_noisy = outputs_noisy["logits"]
                    loss_noisy = F.cross_entropy(logits_noisy, targets)
                    loss_noisy.backward()

                    # Fisher update
                    if hasattr(self, "_rwp_fisher"):
                        # self._rwp_fisher = {}
                        with torch.no_grad():
                            for name, p in base_model.named_parameters():
                                if not p.requires_grad or (p.grad is None):
                                    continue
                                g2 = p.grad.detach() ** 2
                                self._rwp_fisher[name] = g2 if name not in self._rwp_fisher else float(self._rwp_beta) * self._rwp_fisher[name] + g2

                    # Revert noise
                    with torch.no_grad():
                        for name, p in base_model.named_parameters():
                            if name in noise_dict:
                                p.data.sub_(noise_dict[name])

                    # Gradient mix and step
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

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if ((epoch % 5 == 4) or epoch == epochs - 1) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "FullFinetune Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    epoch + 1, epochs, losses / len(train_loader), train_acc, test_acc
                )
            elif self._is_main_process:
                info = "FullFinetune Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    epoch + 1, epochs, losses / len(train_loader), train_acc
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
                elif self._is_scoped_sam_optimizer():
                    logits, loss_value = self._scoped_sam_step(
                        optimizer,
                        inputs,
                        fake_targets,
                        class_offset=int(self._known_classes),
                    )
                    losses += loss_value
                elif self._is_scoped_random_optimizer():
                    logits, loss_value = self._scoped_random_step(
                        optimizer,
                        inputs,
                        fake_targets,
                        class_offset=int(self._known_classes),
                    )
                    losses += loss_value
                elif self._is_mergegam_optimizer():
                    logits, loss_value = self._mergegam_step(
                        optimizer,
                        inputs,
                        fake_targets,
                        class_offset=int(self._known_classes),
                    )
                    losses += loss_value
                elif self._is_flatlora_optimizer():
                    logits, loss_value = self._flatlora_step(
                        optimizer,
                        inputs,
                        fake_targets,
                        class_offset=int(self._known_classes),
                    )
                    losses += loss_value
                elif self._optimizer_type == "rwp":
                    # True RWP branch for incremental training
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

                    # clean pass
                    enable_running_stats(self._network)
                    optimizer.zero_grad()
                    outputs = self._network(inputs)
                    logits_clean = outputs["logits"]
                    loss_clean = F.cross_entropy(logits_clean[:, self._known_classes :], fake_targets)
                    loss_clean.backward()
                    g0 = {}
                    base_model = self._unwrap_network()
                    for name, p in base_model.named_parameters():
                        if p.requires_grad and (p.grad is not None):
                            g0[name] = p.grad.detach().clone()

                    # noisy pass
                    disable_running_stats(self._network)
                    noise_dict = {}
                    with torch.no_grad():
                        
                        std_for_noise = float(self.args.get("noise_std", rwp_std))
                        for name, p in base_model.named_parameters():
                            if self._rwp_range == "lora":
                                if not p.requires_grad or p.numel() == 0:
                                    continue
                            fisher_param = None
                            if hasattr(self, "_rwp_fisher") and (name in self._rwp_fisher):
                                fisher_param = self._rwp_fisher[name]
                            e = generate_pertubation(p, pertubation_mode=self.rwp_noise_type, std=std_for_noise, fisher_param=fisher_param, fisher_scaler=float(self._rwp_eta))
                            p.data.add_(e)
                            noise_dict[name] = e

                    optimizer.zero_grad()
                    outputs_noisy = self._network(inputs)
                    logits_noisy = outputs_noisy["logits"]
                    loss_noisy = F.cross_entropy(logits_noisy[:, self._known_classes :], fake_targets)
                    loss_noisy.backward()

                    # fisher update
                    if hasattr(self, "_rwp_fisher"):
                        # self._rwp_fisher = {}
                        with torch.no_grad():
                            for name, p in base_model.named_parameters():
                                if not p.requires_grad or (p.grad is None):
                                    continue
                                g2 = p.grad.detach() ** 2
                                if name not in self._rwp_fisher:
                                    self._rwp_fisher[name] = g2
                                else:
                                    self._rwp_fisher[name] = float(self._rwp_beta) * self._rwp_fisher[name] + g2

                    # revert noise
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
                    losses += (lam * float(loss_noisy.detach().item()) + (1.0 - lam) * float(loss_clean.detach().item()))
                elif self._optimizer_type == "arwp":
                    if getattr(self, "_rwp_std_follow_lr", False):
                        try:
                            cur_lr = float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else float(optimizer.param_groups[0]["lr"])
                        except Exception:
                            cur_lr = float(optimizer.param_groups[0]["lr"]) 
                        base_lr = float(getattr(self, "_rwp_lr0", cur_lr))
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
                else:
                    optimizer.zero_grad()
                    logits = self._network(inputs)["logits"]
                    loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)

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
    
    

    def _build_eval_backbone(self, task_idx):
        rank = self.args.get("lora_rank", 8)
        backbone_type = self.args.get("backbone_type", "vit_base_patch16_224").lower()
        # SeqLoRA: at eval for task T, use LoRA saved after task T-1
        last_idx = max(0, self._cur_task - 1)
        save_dir = self.args.get("filepath", "./")

        if "resnet" in backbone_type:
            from backbone.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
            from backbone.lora import LoRA_ResNet
            _resnet_map = {
                "resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,
                "resnet101": resnet101, "resnet152": resnet152,
            }
            fn = _resnet_map.get(backbone_type, resnet50)
            base = fn(pretrained=True, args=self.args)
            lora_layers = self.args.get("lora_layers", None)
            lora_backbone = LoRA_ResNet(base, r=rank, lora_layers=lora_layers)
            try:
                lora_backbone.load_lora_parameters(save_dir, last_idx)
            except FileNotFoundError:
                logger.warning(
                    "[SeqLoRA] ResNet LoRA checkpoint not found for task %d, using init weights",
                    last_idx,
                )
            return lora_backbone

        # --- ViT (default) ---
        vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        lora_backbone = LoRA_ViT_timm(
            vit_model=vit.eval(),
            r=rank,
            num_classes=0,
            index=False,
            increment=self.args["increment"],
            filepath=save_dir,
            cur_task_index=self._cur_task,
            learn_alpha=False,
            eval=True,
        )
        lora_backbone.out_dim = 768

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
    
    
