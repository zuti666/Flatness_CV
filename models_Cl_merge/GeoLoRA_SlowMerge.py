"""GeoLoRA with projection-based slow merge + incremental task-wise private branches.

Core behavior:
    - Each task creates a new private GeoLoRA branch (IncLoRA-style organization).
    - Previous task private branches are loaded and frozen.
    - Shared branch is initialized from cross-task shared subspaces (alternating updates on
      projected energy); only its middle coefficients are trainable in the current task.
    - New private branch is fully trainable in the current task.
    - After each task, run slow merge to refresh shared dictionary atoms.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


# ---------- GeoLoRA adapter ----------
def _build_householder_orth_linear(out_dim: int, rank: int, init_std: float = 0.02) -> nn.Linear:
    """Create a linear layer whose weight has orthonormal columns (Householder map)."""
    if rank > out_dim:
        raise ValueError(f"rank ({rank}) must be <= out_dim ({out_dim}) for Stiefel columns.")

    layer = nn.Linear(rank, out_dim, bias=False)
    parametrizations.orthogonal(
        layer,
        name="weight",
        orthogonal_map="householder",
        use_trivialization=False,
    )
    nn.init.normal_(layer.parametrizations.weight.original, mean=0.0, std=init_std)
    return layer


class GeoLoRA_QKV(nn.Module):
    class _GeoPrivateBranch(nn.Module):
        def __init__(
            self,
            dim: int,
            rank: int,
            eps: float,
            s_init: float = -1.0,
            init_std: float = 0.02,
            state: Dict[str, torch.Tensor] | None = None,
            trainable: bool = True,
        ):
            super().__init__()
            self.theta_a = _build_householder_orth_linear(dim, rank, init_std=init_std)
            self.theta_b = _build_householder_orth_linear(dim, rank, init_std=init_std)
            self.s = nn.Parameter(torch.full((rank,), float(s_init)))
            self.eps = float(eps)
            if state is not None:
                with torch.no_grad():
                    self.theta_a.parametrizations.weight.original.copy_(state["theta_a_original"])
                    self.theta_b.parametrizations.weight.original.copy_(state["theta_b_original"])
                    self.s.copy_(state["s"])
            if not trainable:
                self.freeze()

        def freeze(self):
            for p in self.parameters():
                p.requires_grad = False

        def sigma(self) -> torch.Tensor:
            return F.softplus(self.s) + self.eps

        def delta(self, x: torch.Tensor, scale: float) -> torch.Tensor:
            u = torch.matmul(x, self.theta_a.weight) * self.sigma()
            return torch.matmul(u, self.theta_b.weight.t()) * scale

        @torch.no_grad()
        def export_state(self) -> Dict[str, torch.Tensor]:
            return {
                "theta_a_original": self.theta_a.parametrizations.weight.original.detach().cpu(),
                "theta_b_original": self.theta_b.parametrizations.weight.original.detach().cpu(),
                "s": self.s.detach().cpu(),
            }

    def __init__(
        self,
        qkv: nn.Linear,
        r_private: int,
        alpha: float = 1.0,
        eps: float = 1e-6,
        s_init: float = -1.0,
        old_task_states: List[Dict[str, Dict[str, torch.Tensor]]] | None = None,
        add_current_branch: bool = True,
    ):
        super().__init__()
        self.qkv = qkv
        dim = qkv.in_features
        self.r_private = r_private
        self.alpha = alpha
        self.eps = eps
        self.s_init = float(s_init)

        # Historical private branches (frozen): one per previous task.
        self.old_q = nn.ModuleList()
        self.old_v = nn.ModuleList()
        for st in (old_task_states or []):
            if "q" not in st or "v" not in st:
                continue
            self.old_q.append(
                self._GeoPrivateBranch(dim, r_private, eps, s_init=self.s_init, state=st["q"], trainable=False)
            )
            self.old_v.append(
                self._GeoPrivateBranch(dim, r_private, eps, s_init=self.s_init, state=st["v"], trainable=False)
            )

        # Current task private branch (fully trainable).
        self.cur_q: GeoLoRA_QKV._GeoPrivateBranch | None = None
        self.cur_v: GeoLoRA_QKV._GeoPrivateBranch | None = None
        if add_current_branch:
            self.cur_q = self._GeoPrivateBranch(
                dim, r_private, eps, s_init=self.s_init, init_std=0.02, trainable=True
            )
            self.cur_v = self._GeoPrivateBranch(
                dim, r_private, eps, s_init=self.s_init, init_std=0.02, trainable=True
            )

        # shared atoms and trainable coefficients (set per task)
        self.register_buffer("A_sh_q", torch.empty(dim, 0))
        self.register_buffer("B_sh_q", torch.empty(dim, 0))
        self.register_buffer("A_sh_v", torch.empty(dim, 0))
        self.register_buffer("B_sh_v", torch.empty(dim, 0))
        self.S_q: nn.Parameter | None = None
        self.S_v: nn.Parameter | None = None

        for p in self.qkv.parameters():
            p.requires_grad = False

    def set_shared_atoms(
        self,
        atoms_q: List[Tuple[torch.Tensor, torch.Tensor]],
        atoms_v: List[Tuple[torch.Tensor, torch.Tensor]],
        init_S_q: torch.Tensor | None = None,
        init_S_v: torch.Tensor | None = None,
    ):
        # atoms are canonicalized, stored on CPU; move to module device
        device = next(self.parameters()).device
        if atoms_q:
            A = torch.stack([a.to(device) for (a, _) in atoms_q], dim=1)  # [dim, k]
            B = torch.stack([b.to(device) for (_, b) in atoms_q], dim=1)
        else:
            A = torch.empty(self.qkv.in_features, 0, device=device)
            B = torch.empty(self.qkv.in_features, 0, device=device)
        if atoms_v:
            A_v = torch.stack([a.to(device) for (a, _) in atoms_v], dim=1)
            B_v = torch.stack([b.to(device) for (_, b) in atoms_v], dim=1)
        else:
            A_v = torch.empty(self.qkv.in_features, 0, device=device)
            B_v = torch.empty(self.qkv.in_features, 0, device=device)

        self.A_sh_q = A
        self.B_sh_q = B
        self.A_sh_v = A_v
        self.B_sh_v = B_v

        # (re)create trainable shared core matrices
        if A.shape[1] > 0:
            S0 = torch.zeros(A.shape[1], A.shape[1], device=device)
            if (
                init_S_q is not None
                and init_S_q.dim() == 2
                and init_S_q.shape[0] == A.shape[1]
                and init_S_q.shape[1] == A.shape[1]
            ):
                S0 = init_S_q.to(device)
            self.S_q = nn.Parameter(S0)
        else:
            self.S_q = None
        if A_v.shape[1] > 0:
            S0v = torch.zeros(A_v.shape[1], A_v.shape[1], device=device)
            if (
                init_S_v is not None
                and init_S_v.dim() == 2
                and init_S_v.shape[0] == A_v.shape[1]
                and init_S_v.shape[1] == A_v.shape[1]
            ):
                S0v = init_S_v.to(device)
            self.S_v = nn.Parameter(S0v)
        else:
            self.S_v = None

    def _delta_private_sum(
        self,
        x: torch.Tensor,
        old_branches: nn.ModuleList,
        cur_branch: _GeoPrivateBranch | None,
        shared_rank: int,
    ) -> torch.Tensor:
        scale = self.alpha / float(max(1, self.r_private + shared_rank))
        delta = torch.zeros_like(x)
        for br in old_branches:
            delta = delta + br.delta(x, scale)
        if cur_branch is not None:
            delta = delta + cur_branch.delta(x, scale)
        return delta

    def _delta_shared(
        self,
        x: torch.Tensor,
        A_sh: torch.Tensor,
        B_sh: torch.Tensor,
        S: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        if S is None or A_sh.numel() == 0:
            return torch.zeros_like(x), []
        scale = self.alpha / float(max(1, self.r_private + A_sh.shape[1]))
        u = torch.matmul(x, A_sh)  # [B,N,k]
        u = torch.matmul(u, S)     # [B,N,k]
        delta = torch.matmul(u, B_sh.t()) * scale
        atoms = []
        for i in range(S.shape[0]):
            atoms.append((A_sh[:, i].detach(), B_sh[:, i].detach(), S[i].detach()))
        return delta, atoms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)  # [B,N,3*dim]
        _, _, threeD = qkv.shape
        q, k, v = qkv.chunk(3, dim=2)

        sh_q = self.A_sh_q.shape[1] if hasattr(self, "A_sh_q") else 0
        sh_v = self.A_sh_v.shape[1] if hasattr(self, "A_sh_v") else 0

        dq_pr = self._delta_private_sum(x, self.old_q, self.cur_q, sh_q)
        dv_pr = self._delta_private_sum(x, self.old_v, self.cur_v, sh_v)
        dq_sh, atoms_q_shared = self._delta_shared(x, self.A_sh_q, self.B_sh_q, self.S_q)
        dv_sh, atoms_v_shared = self._delta_shared(x, self.A_sh_v, self.B_sh_v, self.S_v)
        self._last_atoms = {"q_shared": atoms_q_shared, "v_shared": atoms_v_shared}
        if self.cur_q is not None:
            self._last_atoms["q"] = (
                self.cur_q.theta_a.weight.detach(),
                self.cur_q.theta_b.weight.detach(),
                self.cur_q.sigma().detach(),
            )
        if self.cur_v is not None:
            self._last_atoms["v"] = (
                self.cur_v.theta_a.weight.detach(),
                self.cur_v.theta_b.weight.detach(),
                self.cur_v.sigma().detach(),
            )

        q = q + dq_pr + dq_sh
        v = v + dv_pr + dv_sh
        return torch.cat([q, k, v], dim=2)

    @torch.no_grad()
    def export_current_private_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self.cur_q is None or self.cur_v is None:
            raise RuntimeError("Current private branch not available in this adapter.")
        return {"q": self.cur_q.export_state(), "v": self.cur_v.export_state()}

    @torch.no_grad()
    def extract_current_task_updates(self) -> Dict[str, torch.Tensor]:
        """Return current-task update matrices for q/v:
        ΔW_t = ΔW_private(current) + ΔW_shared(current-coeff).
        """
        dim = int(self.qkv.in_features)
        device = next(self.parameters()).device

        def _private_delta(branch: GeoLoRA_QKV._GeoPrivateBranch | None, shared_rank: int) -> torch.Tensor:
            if branch is None:
                return torch.zeros(dim, dim, device=device)
            scale = self.alpha / float(max(1, self.r_private + shared_rank))
            A = branch.theta_a.weight.detach()
            B = branch.theta_b.weight.detach()
            sigma = branch.sigma().detach()
            return torch.matmul(B * sigma.unsqueeze(0), A.t()) * scale

        def _shared_delta(A_sh: torch.Tensor, B_sh: torch.Tensor, S: torch.Tensor | None) -> torch.Tensor:
            if S is None or A_sh.numel() == 0:
                return torch.zeros(dim, dim, device=device)
            scale = self.alpha / float(max(1, self.r_private + A_sh.shape[1]))
            core = S.detach()
            return torch.matmul(torch.matmul(B_sh.detach(), core.t()), A_sh.detach().t()) * scale

        sh_q = int(self.A_sh_q.shape[1])
        sh_v = int(self.A_sh_v.shape[1])
        delta_q = _private_delta(self.cur_q, sh_q) + _shared_delta(self.A_sh_q, self.B_sh_q, self.S_q)
        delta_v = _private_delta(self.cur_v, sh_v) + _shared_delta(self.A_sh_v, self.B_sh_v, self.S_v)
        return {"q": delta_q.cpu(), "v": delta_v.cpu()}


class GeoLoRA_ViT_timm(nn.Module):
    def __init__(
        self,
        vit_model: nn.Module,
        r_private: int,
        alpha: float = 1.0,
        eps: float = 1e-6,
        s_init: float = -1.0,
        filepath: str = "./",
        task_id: int = 0,
        eval_mode: bool = False,
    ):
        super().__init__()
        self.vit = vit_model
        self.r_private = r_private
        self.alpha = alpha
        self.eps = eps
        self.s_init = float(s_init)
        self.filepath = filepath
        self.task_id = int(task_id)
        self.eval_mode = bool(eval_mode)
        for p in self.vit.parameters():
            p.requires_grad = False
        self._adapters: List[GeoLoRA_QKV] = []
        self._inject(self._load_old_layer_states())
        self.out_dim = getattr(self.vit, "num_features", 768)

    def _task_ckpt(self, tid: int) -> str:
        return os.path.join(self.filepath, f"geolora_task_{tid}.pt")

    def _load_old_layer_states(self) -> List[Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
        # Training task t: load [0..t-1]. Eval task t: load [0..t].
        num = self.task_id + 1 if self.eval_mode else self.task_id
        out: List[Dict[str, Dict[str, Dict[str, torch.Tensor]]]] = []
        for tid in range(max(0, int(num))):
            f = self._task_ckpt(tid)
            if os.path.exists(f):
                payload = torch.load(f, map_location="cpu")
                out.append(payload.get("layer_states", {}))
            else:
                out.append({})
        return out

    def _inject(self, old_layer_states: List[Dict[str, Dict[str, Dict[str, torch.Tensor]]]]):
        for blk in self.vit.blocks:
            layer_idx = len(self._adapters)
            qkv = blk.attn.qkv
            old_task_states = []
            for task_dict in old_layer_states:
                state = task_dict.get(str(layer_idx), None)
                if state is not None:
                    old_task_states.append(state)
            adapter = GeoLoRA_QKV(
                qkv,
                r_private=self.r_private,
                alpha=self.alpha,
                eps=self.eps,
                s_init=self.s_init,
                old_task_states=old_task_states,
                add_current_branch=(not self.eval_mode),
            )
            blk.attn.qkv = adapter
            self._adapters.append(adapter)

    @torch.no_grad()
    def save_current_task_private(self, task_id: int):
        os.makedirs(self.filepath, exist_ok=True)
        layer_states: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        for layer_idx, adapter in enumerate(self._adapters):
            layer_states[str(layer_idx)] = adapter.export_current_private_state()
        torch.save({"layer_states": layer_states}, self._task_ckpt(task_id))

    def forward(self, x):
        return self.vit(x)

    def adapters(self) -> List[GeoLoRA_QKV]:
        return self._adapters


# ---------- Learner with slow merge ----------
class Learner(LoraBaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._lora_initialized = False
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()

        self._rank = int(args.get("geolora_rank", args.get("lora_rank", 8)))  # legacy total rank
        self._r_private = int(args.get("geolora_r_private", self._rank))
        self._r_shared = int(args.get("geolora_r_shared", min(self._rank, 4)))
        self._alpha = float(args.get("geolora_alpha", 1.0))
        self._eps = float(args.get("geolora_eps", 1e-6))
        self._s_init = float(args.get("geolora_s_init", -1.0))
        self._dict_size = int(args.get("geolora_dict_size", 32))
        self._sim_thresh = float(args.get("geolora_sim_thresh", 0.95))
        self._merge_iters = int(args.get("geolora_merge_iters", 3))
        self._merge_gamma = float(args.get("geolora_merge_gamma", 2.0))
        self._shared_eta_min = float(args.get("geolora_shared_eta_min", 0.0))
        self._core_init_lambda = float(args.get("geolora_core_init_lambda", 0.6))
        self._core_init_rho = float(args.get("geolora_core_init_rho", 0.01))

        # Per-layer shared bases (q/v kept separate), represented as atom pairs [(a_k, b_k)].
        self._shared_q: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
        self._shared_v: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
        # Per-layer per-task updates (dense matrices on CPU): ΔW_t^q / ΔW_t^v.
        self._task_delta_q: List[Dict[int, torch.Tensor]] = []
        self._task_delta_v: List[Dict[int, torch.Tensor]] = []
        # Per-layer per-task shared core matrix in current shared basis.
        self._task_core_q: List[Dict[int, torch.Tensor]] = []
        self._task_core_v: List[Dict[int, torch.Tensor]] = []

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def build_lora_backbone(self, index=True, eval_mode=False, task_id: int | None = None):
        tid = self._cur_task if task_id is None else int(task_id)
        vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        model = GeoLoRA_ViT_timm(
            vit_model=vit.eval(),
            r_private=self._r_private,
            alpha=self._alpha,
            eps=self._eps,
            s_init=self._s_init,
            filepath=self.args.get("filepath", "./"),
            task_id=tid,
            eval_mode=eval_mode,
        )
        model.out_dim = 768
        return model

    def _build_eval_backbone(self, task_idx):
        return self.build_lora_backbone(eval_mode=True, task_id=task_idx)

    def incremental_train(self, data_manager):
        self._refresh_distributed_context()
        self.data_manager = data_manager

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
        self._save_current_task_private(self._cur_task)
        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._unwrap_network()

        # slow merge
        self._slow_merge(self._cur_task)
        # prepare shared atoms for next task (initialization)
        self._prepare_shared_for_next_task()

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as _nme_exc:  # pylint: disable=broad-except
            self._log(f"[GeoLoRA_SlowMerge][NME] Failed to compute class means: {_nme_exc}")

    def _train(self, train_loader, test_loader):
        network = self._unwrap_network()
        # IncLoRA-style: each task rebuilds with frozen historical branches + a new branch.
        network.backbone = self.build_lora_backbone(eval_mode=False, task_id=self._cur_task)
        network.backbone.to(self._device)
        self._lora_initialized = True
        self._network = network
        self._prepare_network()
        # inject shared atoms for current task before building optimizer
        self._set_shared_for_current_task()

        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage="init" if self._cur_task == 0 else "update")
        lr = optimizer.param_groups[0]["lr"]
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("milestones", []),
            gamma=float(self.args.get("lrate_decay", 1.0)),
            T_max=self.args.get("epochs", None),
            eta_min=self.args.get("min_lr", 0.1 * lr),
        )
        epochs = int(self.args.get("init_epoch" if self._cur_task == 0 else "epochs", 1))

        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                optimizer.zero_grad()
                logits = self._network(inputs)["logits"]
                if self._cur_task == 0:
                    loss = F.cross_entropy(logits, targets)
                    eval_logits, eval_targets = logits, targets
                else:
                    fake_targets = targets - self._known_classes
                    loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    eval_logits, eval_targets = logits[:, self._known_classes :], fake_targets

                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(eval_logits, dim=1)
                correct += preds.eq(eval_targets).cpu().sum()
                total += len(eval_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            if (epoch % 5 == 4) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            if self._is_main_process:
                prog_bar.set_description(info)
        if self._is_main_process:
            self._log(info)

    def _save_current_task_private(self, task_id: int):
        net = self._unwrap_network()
        bb = getattr(net, "backbone", None)
        if bb is None or not hasattr(bb, "save_current_task_private"):
            return
        try:
            bb.save_current_task_private(task_id)
            self._log(f"[GeoLoRA_SlowMerge] Saved task-{task_id} private GeoLoRA.")
        except Exception as exc:  # pylint: disable=broad-except
            self._log(f"[GeoLoRA_SlowMerge] Failed to save task-{task_id} private GeoLoRA: {exc}")

    # ---------------- slow merge ---------------- #
    @staticmethod
    def _top_eig_basis(cov: torch.Tensor, k: int) -> torch.Tensor:
        dim = int(cov.shape[0])
        k = int(min(max(0, k), dim))
        if k <= 0:
            return cov.new_zeros(dim, 0)
        vals, vecs = torch.linalg.eigh(cov)
        idx = torch.argsort(vals, descending=True)[:k]
        return vecs[:, idx]

    @staticmethod
    def _core_from_basis(
        delta_w: torch.Tensor,
        atoms: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if not atoms:
            return torch.empty(0, 0, dtype=delta_w.dtype)
        A = torch.stack([a for (a, _) in atoms], dim=1).to(device=delta_w.device, dtype=delta_w.dtype)  # [din, k]
        B = torch.stack([b for (_, b) in atoms], dim=1).to(device=delta_w.device, dtype=delta_w.dtype)  # [dout, k]
        return B.t().matmul(delta_w).matmul(A)  # [k, k]

    @staticmethod
    def _shared_energy_ratio(
        delta_w: torch.Tensor,
        atoms: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> float:
        if not atoms:
            return 0.0
        core = Learner._core_from_basis(delta_w, atoms)
        if core.numel() == 0:
            return 0.0
        A = torch.stack([a for (a, _) in atoms], dim=1).to(device=delta_w.device, dtype=delta_w.dtype)
        B = torch.stack([b for (_, b) in atoms], dim=1).to(device=delta_w.device, dtype=delta_w.dtype)
        shared = torch.matmul(torch.matmul(B, core), A.t())
        num = float(torch.sum(shared * shared))
        den = float(torch.sum(delta_w * delta_w) + 1e-12)
        return num / den

    def _update_shared_basis(
        self,
        task_deltas: Dict[int, torch.Tensor],
        prev_atoms: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Dict[int, float]]:
        if not task_deltas:
            return [], {}

        first = next(iter(task_deltas.values()))
        dout, din = int(first.shape[0]), int(first.shape[1])
        k = int(min(self._r_shared, dout, din))
        if k <= 0:
            return [], {}

        etas: Dict[int, float] = {}
        if prev_atoms:
            for tid, w in task_deltas.items():
                etas[int(tid)] = float(self._shared_energy_ratio(w.float(), prev_atoms))
            mean_eta = float(np.mean(list(etas.values()))) if etas else 0.0
            if mean_eta < float(self._shared_eta_min):
                return prev_atoms[:k], etas
            logits = np.array([-self._merge_gamma * etas[int(tid)] for tid in task_deltas.keys()], dtype=np.float64)
            logits = logits - np.max(logits)
            w_np = np.exp(logits)
            w_np = w_np / np.sum(w_np)
            task_weights = {int(tid): float(wv) for tid, wv in zip(task_deltas.keys(), w_np)}
        else:
            uniform = 1.0 / float(max(1, len(task_deltas)))
            task_weights = {int(tid): uniform for tid in task_deltas.keys()}

        if prev_atoms and len(prev_atoms) >= k:
            A = torch.stack([a for (a, _) in prev_atoms[:k]], dim=1).float()  # [din, k]
            B = torch.stack([b for (_, b) in prev_atoms[:k]], dim=1).float()  # [dout, k]
        else:
            mean_w = torch.zeros(dout, din, dtype=torch.float32)
            for w in task_deltas.values():
                mean_w = mean_w + w.float()
            mean_w = mean_w / float(max(1, len(task_deltas)))
            U, _, Vh = torch.linalg.svd(mean_w, full_matrices=False)
            B = U[:, :k].contiguous()
            A = Vh.t()[:, :k].contiguous()

        for _ in range(max(1, self._merge_iters)):
            cov_b = torch.zeros(dout, dout, dtype=torch.float32)
            for tid, w in task_deltas.items():
                wa = w.float().matmul(A)  # [dout, k]
                cov_b = cov_b + float(task_weights[int(tid)]) * wa.matmul(wa.t())
            B = self._top_eig_basis(cov_b, k)

            cov_a = torch.zeros(din, din, dtype=torch.float32)
            for tid, w in task_deltas.items():
                wtb = w.float().t().matmul(B)  # [din, k]
                cov_a = cov_a + float(task_weights[int(tid)]) * wtb.matmul(wtb.t())
            A = self._top_eig_basis(cov_a, k)

        out = []
        for i in range(k):
            out.append((A[:, i].detach().cpu(), B[:, i].detach().cpu()))
        return out, etas

    @staticmethod
    def _task_cores_from_basis(
        task_deltas: Dict[int, torch.Tensor],
        atoms: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[int, torch.Tensor]:
        if not atoms:
            return {}
        A = torch.stack([a for (a, _) in atoms], dim=1).float()  # [din, k]
        B = torch.stack([b for (_, b) in atoms], dim=1).float()  # [dout, k]
        out: Dict[int, torch.Tensor] = {}
        for tid, w in task_deltas.items():
            core = B.t().matmul(w.float()).matmul(A)  # [k, k]
            out[int(tid)] = core.detach().cpu()
        return out

    def _mean_core_init(
        self,
        task_cores: Dict[int, torch.Tensor],
        atoms: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor | None:
        if not atoms:
            return None
        k = len(atoms)
        eye = torch.eye(k, dtype=torch.float32)
        if not task_cores:
            return float(self._core_init_rho) * eye
        acc = torch.zeros(k, k, dtype=torch.float32)
        cnt = 0
        for m in task_cores.values():
            if m.dim() == 2 and m.shape[0] == k and m.shape[1] == k:
                acc = acc + m.float()
                cnt += 1
        if cnt <= 0:
            return float(self._core_init_rho) * eye
        mean_core = acc / float(cnt)
        lam = float(max(0.0, min(1.0, self._core_init_lambda)))
        rho = float(self._core_init_rho)
        return lam * mean_core + (1.0 - lam) * (rho * eye)

    def _slow_merge(self, task_id: int):
        adapters: List[GeoLoRA_QKV] = self._unwrap_network().backbone.adapters()
        if not self._shared_q:
            self._shared_q = [[] for _ in adapters]
            self._shared_v = [[] for _ in adapters]
            self._task_delta_q = [dict() for _ in adapters]
            self._task_delta_v = [dict() for _ in adapters]
            self._task_core_q = [dict() for _ in adapters]
            self._task_core_v = [dict() for _ in adapters]

        for lidx, adapter in enumerate(adapters):
            updates = adapter.extract_current_task_updates()
            self._task_delta_q[lidx][task_id] = updates["q"].float()
            self._task_delta_v[lidx][task_id] = updates["v"].float()

            self._shared_q[lidx], etas_q = self._update_shared_basis(self._task_delta_q[lidx], self._shared_q[lidx])
            self._shared_v[lidx], etas_v = self._update_shared_basis(self._task_delta_v[lidx], self._shared_v[lidx])
            self._task_core_q[lidx] = self._task_cores_from_basis(self._task_delta_q[lidx], self._shared_q[lidx])
            self._task_core_v[lidx] = self._task_cores_from_basis(self._task_delta_v[lidx], self._shared_v[lidx])

            eta_q = np.mean(
                [self._shared_energy_ratio(w, self._shared_q[lidx]) for w in self._task_delta_q[lidx].values()]
            )
            eta_v = np.mean(
                [self._shared_energy_ratio(w, self._shared_v[lidx]) for w in self._task_delta_v[lidx].values()]
            )
            if self._is_main_process:
                if etas_q:
                    eta_q_min, eta_q_max = min(etas_q.values()), max(etas_q.values())
                else:
                    eta_q_min, eta_q_max = 0.0, 0.0
                if etas_v:
                    eta_v_min, eta_v_max = min(etas_v.values()), max(etas_v.values())
                else:
                    eta_v_min, eta_v_max = 0.0, 0.0
                self._log(
                    f"[GeoLoRA_SlowMerge] Layer {lidx}: k_q={len(self._shared_q[lidx])}, "
                    f"k_v={len(self._shared_v[lidx])}, eta_q={eta_q:.4f}, eta_v={eta_v:.4f}, "
                    f"eta_q[min,max]=({eta_q_min:.4f},{eta_q_max:.4f}), "
                    f"eta_v[min,max]=({eta_v_min:.4f},{eta_v_max:.4f})"
                )

    def _set_shared_for_current_task(self):
        adapters: List[GeoLoRA_QKV] = self._unwrap_network().backbone.adapters()
        if not self._shared_q:
            return
        for lidx, adapter in enumerate(adapters):
            atoms_q = self._shared_q[lidx] if lidx < len(self._shared_q) else []
            atoms_v = self._shared_v[lidx] if lidx < len(self._shared_v) else []
            init_q = self._mean_core_init(self._task_core_q[lidx], atoms_q) if lidx < len(self._task_core_q) else None
            init_v = self._mean_core_init(self._task_core_v[lidx], atoms_v) if lidx < len(self._task_core_v) else None
            adapter.set_shared_atoms(atoms_q=atoms_q, atoms_v=atoms_v, init_S_q=init_q, init_S_v=init_v)

    def _prepare_shared_for_next_task(self):
        # Shared atoms are re-injected at the start of the next task via _set_shared_for_current_task.
        return

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
