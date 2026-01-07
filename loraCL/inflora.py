import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from optimer.util import enable_running_stats, disable_running_stats, generate_pertubation
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from loraCL.baseLoRA import LoraBaseLearner
from utils.toolkit import tensor2numpy
import math
from copy import deepcopy

from utils.inc_net import SiNet
from backbone.lora import LoRA_ViT_timm
import torch.nn as nn



# class Attention_LoRA(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.dim = dim
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.attn_gradients = None
#         self.attention_map = None
#         self.rank = r

#         self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
#         self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
#         self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
#         self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
#         self.rank = r

#         self.matrix = torch.zeros(dim ,dim)
#         self.n_matrix = 0
#         self.cur_matrix = torch.zeros(dim ,dim)
#         self.n_cur_matrix = 0

#     def init_param(self):
#         for t in range(len(self.lora_A_k)):
#             nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
#             nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B_k[t].weight)
#             nn.init.zeros_(self.lora_B_v[t].weight)

#     def init_param_ada(self, t, r):
#         self.lora_A_k[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
#         self.lora_B_k[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)
#         self.lora_A_v[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
#         self.lora_B_v[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)

#         nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_B_k[t].weight)
#         nn.init.zeros_(self.lora_B_v[t].weight)

#     def save_attn_gradients(self, attn_gradients):
#         self.attn_gradients = attn_gradients
        
#     def get_attn_gradients(self):
#         return self.attn_gradients
    
#     def save_attention_map(self, attention_map):
#         self.attention_map = attention_map
        
#     def get_attention_map(self):
#         return self.attention_map
    
#     def forward(self, x, task, register_hook=False, get_feat=False,get_cur_feat=False):
#         if get_feat:
#             self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
#             self.n_matrix += x.shape[0]*x.shape[1]
#         if get_cur_feat:
#             self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
#             self.n_cur_matrix += x.shape[0]*x.shape[1]

#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

#         # insert lora
#         if task > -0.5:
#             weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
#             weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
#             k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#             v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
                
#         if register_hook:
#             self.save_attention_map(attn)
#             attn.register_hook(self.save_attn_gradients)        

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

#     def get_matrix(self, task):
#         matrix_k = torch.mm(self.lora_B_k[task].weight, self.lora_A_k[task].weight)
#         matrix_v = torch.mm(self.lora_B_v[task].weight, self.lora_A_v[task].weight)
#         return matrix_k, matrix_v
    
#     def get_pre_matrix(self, task):
#         with torch.no_grad():
#             weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task)], dim=0).sum(dim=0)
#             weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task)], dim=0).sum(dim=0)
#         return weight_k, weight_v



class Learner(LoraBaseLearner):
    """
    InfLoRA (interface-aligned): information-guided LoRA training in our CL framework.

    - Interface matches existing LoRA learners (inclora/seqlora):
      incremental_train, _train with _init_train/_update_representation, and _build_eval_backbone.
    - Optimizers supported via BaseLearner factory: sgd | sam | cflat | gam | rwp
    - Optional RWP std-follow-LR scaling (rwp_std_follow_lr) kept consistent with others.

    Note: The original external InfLoRA implementation (with custom modules and SVD-based
    initialization on cached feature matrices) is not directly portable here because those
    modules are not present in this repository. This class preserves the training contract
    and hooks where information-guided initialization could be added in the future using our
    LoRA_ViT_timm backbone.
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = SiNet(args)
        # initialize LoRA params in Attention_LoRA blocks (match original InfLoRA)
        # for module in self._network.modules():
        #         if isinstance(module, Attention_LoRA):
        #             module.init_param()

        self.total_sessions = args.get("total_sessions",None)
        if self.total_sessions is None:
            self._total_classes =  int(200/int(args.get("increment",10)))
        
        
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        # Optimizer-specific hyperparameters (kept aligned with other learners)
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

        # InfLoRA-specific subspace tracking (DualGPM)
        self.all_keys = []
        self.feature_list: list[np.ndarray] = []    # per-layer basis (D x k)
        self.project_type: list[str] = []           # 'remove' or 'retain'
        self.feature_mat: list[torch.Tensor] = []   # per-layer projection matrix (D x D)
        self.lamb = float(args.get("lamb", 0.5))
        self.lame = float(args.get("lame", 0.9))
        
       

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_eval_backbone(self, task_idx):
        # SiNet handles eval within itself
        return None

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
            num_workers=self.args.get("train_num_workers", 8),
        )

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args.get("train_num_workers", 8)
        )

        self._train(self.train_loader, self.test_loader)
        self._network = self._unwrap_network()
        # optional clustering step (as in original InfLoRA)
        
        self.clustering(self.train_loader)
        

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as _nme_exc:  # pylint: disable=broad-except
            self._log(f"[InfLoRA][NME] Failed to compute class means: {_nme_exc}")

    def clustering(self, dataloader: DataLoader):
        """KMeans cluster centers as keys (optional utility from original code)."""
        
        features = []
        for _, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            if mask.numel() == 0:
                continue
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if hasattr(self._network, 'extract_vector'):
                    vec = self._network.extract_vector(inputs)
                else:
                    out = self._network(inputs)
                    vec = out.get('features', None)
                    if vec is None:
                        continue
            vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-12)
            features.append(vec)
        if not features:
            return
        feats = torch.cat(features, 0).detach().cpu().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(feats)
        # store as tensor on same device
        centers = torch.tensor(clustering.cluster_centers_, dtype=torch.float32, device=self._device)
        if not hasattr(self, 'all_keys'):
            self.all_keys = []
        self.all_keys.append(centers)

    def _train(self, train_loader, test_loader):
        # Move model to device
        self._prepare_network()

        # Freeze all params first; we will reopen specific ones below
        try:
            cur_idx = max(0, int(getattr(self._network, 'numtask', 1)) - 1)
        except Exception:
            cur_idx = 0
        for _, p in self._network.named_parameters():
            p.requires_grad_(False)

        # Choose path by backbone type
        lora_backbone = getattr(self._network, 'backbone', None)
        if isinstance(lora_backbone, LoRA_ViT_timm):
            # 1) collect covariance per LoRA-attached layer via qkv pre-hooks
            covs = self._collect_cov_via_hooks(lora_backbone, train_loader)

            # 2) information-guided A init (with optional DualGPM projection for t>0)
            rank = int(getattr(lora_backbone, 'rank', self._rank))
            for li, cov in enumerate(covs):
                if cov is None:
                    continue
                cur = cov.to(self._device)
                if cur_idx > 0 and li < len(self.feature_mat):
                    P = self.feature_mat[li].to(cur.device)
                    if li < len(self.project_type) and self.project_type[li] == 'remove':
                        cur = cur - torch.mm(P, cur)
                    else:
                        cur = torch.mm(P, cur)
                try:
                    U, S, V = torch.linalg.svd(cur, full_matrices=False)
                    U_top = U[:, :rank]
                except Exception:
                    Csym = 0.5 * (cur + cur.t())
                    evals, evecs = torch.linalg.eigh(Csym)
                    U_top = evecs[:, -rank:]
                lora_backbone.init_current_task_A(layer_idx=li, A_q=U_top, A_v=U_top, scale=1/math.sqrt(3))
            # Freeze A to only train B
            lora_backbone.freeze_current_task_A()

            # 3) enable current task classifier head + LoRA B
            enabled_head = False
            for name, param in self._network.named_parameters():
                if name.startswith("fc."):
                    param.requires_grad_(True)
                    enabled_head = True
                elif f"classifier_pool.{cur_idx}" in name:
                    param.requires_grad_(True)
                    enabled_head = True
            for B in getattr(lora_backbone, 'w_Bs', []):
                B.weight.requires_grad_(True)

        params = [p for p in self._network.parameters() if p.requires_grad]

        if self._cur_task == 0:
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

        # === Update DualGPM after training: collect again cov and build projection matrices ===
        with torch.no_grad():
            mat_list = []
            if isinstance(lora_backbone, LoRA_ViT_timm):
                covs = self._collect_cov_via_hooks(lora_backbone, train_loader)
                for c in covs:
                    if c is not None:
                        mat_list.append(deepcopy(c.cpu().numpy()))
            # else:
            #     for _, (_, inputs, targets) in enumerate(train_loader):
            #         inputs, targets = inputs.to(self._device), targets.to(self._device)
            #         _ = self._network(inputs, get_cur_feat=True)
            #     for module in self._network.modules():
            #         if isinstance(module, Attention_LoRA):
            #             mat_list.append(deepcopy(module.cur_matrix.cpu().numpy()))
            #             module.cur_matrix.zero_(); module.n_cur_matrix = 0

            if mat_list:
                self.update_DualGPM(mat_list)
                self.feature_mat = []
                for p in range(len(self.feature_list)):
                    Uf = torch.tensor(self.feature_list[p] @ self.feature_list[p].T, dtype=torch.float32)
                    self.feature_mat.append(Uf)

    # NOTE: Deprecated duplicate; use the typed version above.
    # def clustering(self, dataloader):
    #     pass


    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]), disable=not self._is_main_process)
        for _, epoch in enumerate(prog_bar):
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
                    loss_value = torch.stack([l.detach() for l in loss_list]).sum()
                    losses += float(loss_value.item())
                    with torch.no_grad():
                        logits = self._network(inputs)["logits"]
                elif self._optimizer_type == "gam":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
                        loss = F.cross_entropy(logits, targets)
                        lv = loss.detach(); loss.backward(); return outputs, lv
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
                        lv = loss.detach(); loss.backward(); return outputs, lv
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
                        rwp_std = float(self._rwp_std) * scale
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
                        
                        std_for_noise = float(self.args.get("noise_std", rwp_std))
                        for name, p in self._network.named_parameters():
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
                        if hasattr(self, "_rwp_fisher"):
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
                        loss.backward(); optimizer.first_step(zero_grad=True)
                        logits = self._network(inputs)["logits"]
                        second = F.cross_entropy(logits, targets)
                        second.backward(); optimizer.second_step(zero_grad=True)
                        losses += float(second.item())
                    else:
                        loss.backward(); optimizer.step(); losses += float(loss.item())

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                desc = f"Task {self._cur_task}, Epoch {epoch+1}/{self.args['init_epoch']} => Loss {losses/len(train_loader):.3f}, Train {train_acc:.2f}, Test {test_acc:.2f}"
            else:
                desc = f"Task {self._cur_task}, Epoch {epoch+1}/{self.args['init_epoch']} => Loss {losses/len(train_loader):.3f}, Train {train_acc:.2f}"
            if self._is_main_process:
                prog_bar.set_description(desc)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]), disable=not self._is_main_process)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                fake_targets = targets - self._known_classes
                new_class_count = int(self._total_classes - self._known_classes)

                if self._optimizer_type == "cflat":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits_all = outputs["logits"]
                        cur_logits = logits_all if logits_all.size(1) == new_class_count else logits_all[:, self._known_classes :]
                        loss = F.cross_entropy(cur_logits, fake_targets)
                        loss.backward()
                        return outputs, [loss]
                    _, loss_list = optimizer.step(closure=closure)
                    loss_value = torch.stack([l.detach() for l in loss_list]).sum()
                    losses += float(loss_value.item())
                    with torch.no_grad():
                        logits = self._network(inputs)["logits"]
                elif self._optimizer_type == "gam":
                    def closure():
                        optimizer.zero_grad()
                        outputs = self._network(inputs)
                        logits_all = outputs["logits"]
                        cur_logits = logits_all if logits_all.size(1) == new_class_count else logits_all[:, self._known_classes :]
                        loss = F.cross_entropy(cur_logits, fake_targets)
                        lv = loss.detach(); loss.backward(); return outputs, lv
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
                        logits_all = outputs["logits"]
                        cur_logits = logits_all if logits_all.size(1) == new_class_count else logits_all[:, self._known_classes :]
                        loss = F.cross_entropy(cur_logits, fake_targets)
                        lv = loss.detach(); loss.backward(); return outputs, lv
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
                        rwp_std = float(self._rwp_std) * ((cur_lr / base_lr) if base_lr > 0 else 1.0)
                    else:
                        rwp_std = float(self._rwp_std)

                    enable_running_stats(self._network)
                    optimizer.zero_grad()
                    outputs = self._network(inputs)
                    logits_clean_all = outputs["logits"]
                    logits_clean = logits_clean_all if logits_clean_all.size(1) == new_class_count else logits_clean_all[:, self._known_classes :]
                    loss_clean = F.cross_entropy(logits_clean, fake_targets)
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
                            if not p.requires_grad or p.numel() == 0:
                                continue
                            fisher_param = getattr(self, "_rwp_fisher", {}).get(name, None)
                            e = generate_pertubation(p, pertubation_mode=self.rwp_noise_type, std=std_for_noise, fisher_param=fisher_param, fisher_scaler=float(self._rwp_eta))
                            p.data.add_(e)
                            noise_dict[name] = e

                    optimizer.zero_grad()
                    outputs_noisy = self._network(inputs)
                    logits_noisy_all = outputs_noisy["logits"]
                    logits_noisy = logits_noisy_all if logits_noisy_all.size(1) == new_class_count else logits_noisy_all[:, self._known_classes :]
                    loss_noisy = F.cross_entropy(logits_noisy, fake_targets)
                    loss_noisy.backward()

                    with torch.no_grad():
                        if hasattr(self, "_rwp_fisher"):
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
                    logits = logits_clean_all.detach()
                    losses += (lam * float(loss_noisy.detach().item()) + (1.0 - lam) * float(loss_clean.detach().item()))
                else:
                    optimizer.zero_grad()
                    logits_all = self._network(inputs)["logits"]
                    cur_logits = logits_all if logits_all.size(1) == new_class_count else logits_all[:, self._known_classes :]
                    loss = F.cross_entropy(cur_logits, fake_targets)
                    if self._optimizer_type == "sam":
                        loss.backward(); optimizer.first_step(zero_grad=True)
                        logits_all = self._network(inputs)["logits"]
                        cur_logits = logits_all if logits_all.size(1) == new_class_count else logits_all[:, self._known_classes :]
                        second = F.cross_entropy(cur_logits, fake_targets)
                        second.backward(); optimizer.second_step(zero_grad=True)
                        losses += float(second.item())
                        logits = logits_all
                    else:
                        loss.backward(); optimizer.step(); losses += float(loss.item())
                        logits = logits_all

                # compute train accuracy with correct label space
                logits_all_for_acc = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
                if logits_all_for_acc.size(1) == new_class_count:
                    preds_local = torch.argmax(logits_all_for_acc, dim=1)
                    preds_global = preds_local + int(self._known_classes)
                else:
                    preds_global = torch.argmax(logits_all_for_acc, dim=1)
                correct += preds_global.eq(targets).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                desc = f"Task {self._cur_task}, Epoch {epoch+1}/{self.args['epochs']} => Loss {losses/len(train_loader):.3f}, Train {train_acc:.2f}, Test {test_acc:.2f}"
            else:
                desc = f"Task {self._cur_task}, Epoch {epoch+1}/{self.args['epochs']} => Loss {losses/len(train_loader):.3f}, Train {train_acc:.2f}"
            if self._is_main_process:
                prog_bar.set_description(desc)

    def _collect_cov_via_hooks(self, lora_backbone: LoRA_ViT_timm, loader: DataLoader):
        """Collect per-layer covariance X^T X at the qkv input using pre-forward hooks."""
        device = self._device
        blocks = list(getattr(lora_backbone.lora_vit, 'blocks', []))
        L = len(blocks)
        covs = [None for _ in range(L)]
        counts = [0 for _ in range(L)]

        hooks = []
        def make_hook(li):
            def pre_hook(module, inputs):
                x = inputs[0]
                B, N, C = x.shape
                X = x.reshape(B*N, C)
                cov = X.t().matmul(X).detach().to('cpu')
                if covs[li] is None:
                    covs[li] = cov
                else:
                    covs[li].add_(cov)
                counts[li] += B*N
            return pre_hook

        for li, blk in enumerate(blocks):
            try:
                h = blk.attn.qkv.register_forward_pre_hook(make_hook(li))
                hooks.append(h)
            except Exception:
                covs[li] = None

        # one pass through loader
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            _ = lora_backbone(inputs)

        # remove hooks and normalize
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        # normalize and return per-layer covariance matrices
        for li in range(L):
            if covs[li] is not None and counts[li] > 0:
                covs[li] = covs[li] / float(counts[li])
        return covs

    # ===== DualGPM (ported from original implementation) =====
    def update_DualGPM(self, mat_list: list[np.ndarray]):
        threshold = (self.lame - self.lamb) * (self._cur_task / max(1, self.total_sessions)) + self.lamb
        print('Threshold: ', threshold)
        if len(self.feature_list) == 0:
            # After First Task
            for activation in mat_list:
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S**2).sum()
                sval_ratio = (S**2) / (sval_total + 1e-12)
                r = int(np.sum(np.cumsum(sval_ratio) < threshold))
                r = max(r, 1)
                # decide policy per layer
                if r < (activation.shape[0] / 2):
                    self.feature_list.append(U[:, :r])
                    self.project_type.append('remove')
                else:
                    self.feature_list.append(U[:, :r])
                    self.project_type.append('retain')
        else:
            for i, activation in enumerate(mat_list):
                if self.project_type[i] == 'remove':
                    U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    act_hat = activation - self.feature_list[i] @ (self.feature_list[i].T @ activation)
                    U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2) / (sval_total + 1e-12)
                    accumulated_sval = (sval_total - sval_hat) / (sval_total + 1e-12)
                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print(f'Skip Updating DualGPM for layer: {i+1}')
                        continue
                    Ui = np.hstack((self.feature_list[i], U[:, :r]))
                    if Ui.shape[1] > Ui.shape[0]:
                        self.feature_list[i] = Ui[:, 0:Ui.shape[0]]
                    else:
                        self.feature_list[i] = Ui
                else:
                    # retain
                    activation = activation
                    U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    act_hat = self.feature_list[i] @ (self.feature_list[i].T @ activation)
                    U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2) / (sval_total + 1e-12)
                    accumulated_sval = sval_hat / (sval_total + 1e-12)
                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval >= (1 - threshold):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print(f'Skip Updating DualGPM for layer: {i+1}')
                        continue
                    act_feature = self.feature_list[i] - U[:, :r] @ (U[:, :r].T @ self.feature_list[i])
                    Ui, Si, Vi = np.linalg.svd(act_feature, full_matrices=False)
                    k = max(1, self.feature_list[i].shape[1] - r)
                    self.feature_list[i] = Ui[:, :k]

        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            if self.project_type[i] == 'remove' and (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0] / 2)):
                feature = self.feature_list[i]
                U, S, V = np.linalg.svd(feature, full_matrices=False)
                new_feature = U[:, feature.shape[1]:]
                self.feature_list[i] = new_feature
                self.project_type[i] = 'retain'
            elif self.project_type[i] == 'retain':
                assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0] / 2)
            print('Layer {} : {}/{} type {}'.format(i+1, self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
        print('-'*40)
