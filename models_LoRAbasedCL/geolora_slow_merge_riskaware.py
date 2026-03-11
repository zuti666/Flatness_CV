"""GeoLoRA with Slow Atomic Merge (non-CL fast path, slow consolidation after each task).

Fast path:
    - GeoLoRA adapters on ViT attention qkv (Q and V only).
    - ΔW = B diag(softplus(s)+eps) A^T with A,B column-orthonormal (QR retraction).
Slow path (after each task):
    - Extract rank-1 atoms (a_i, b_i, σ_i) per adapter, canonicalize (sort by σ, fix sign).
    - Merge with a shared atom dictionary per adapter using similarity-thresholded greedy
      selection under budget K.
    - Store coefficients c_k = <ΔW, u_k>_F for bookkeeping (not used in forward yet).

Notes:
    - This is a lightweight, single-pass consolidation aimed at keeping atoms comparable
      across tasks. It does not yet re-initialize the next task from the dictionary; it
      only maintains the dictionary for analysis/inspection.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


# ---------- GeoLoRA adapter ----------
def _orthonormal(mat: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q


class GeoLoRA_QKV(nn.Module):
    """
    QKV projection with a two-tier (shared + private) low-rank update.

    Shared atoms store consolidated cross-task directions (frozen directions, trainable coefficients).
    Private atoms are re-initialized each task to absorb task-specific updates.
    """
    def __init__(
        self,
        base: nn.Linear,
        rank_total: int = 8,
        rank_shared: int = 0,
        alpha: float = 1.0,
        merge_weights: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert isinstance(base, nn.Linear)
        assert 0 <= rank_shared <= rank_total
        self.base = base
        self.rank_total = int(rank_total)
        self.rank_shared = int(rank_shared)
        self.rank_private = int(rank_total - rank_shared)
        self.alpha = float(alpha)
        self.merge_weights = bool(merge_weights)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None

        in_dim = base.in_features
        out_dim = base.out_features // 3  # timm ViT qkv projection: dim -> 3*dim

        # Shared atoms (directions + coefficients)
        if self.rank_shared > 0:
            self.theta_a_q_sh = nn.Parameter(torch.randn(in_dim, self.rank_shared) * 0.02)
            self.theta_b_q_sh = nn.Parameter(torch.randn(out_dim, self.rank_shared) * 0.02)
            self.sigma_q_sh = nn.Parameter(torch.zeros(self.rank_shared))

            self.theta_a_v_sh = nn.Parameter(torch.randn(in_dim, self.rank_shared) * 0.02)
            self.theta_b_v_sh = nn.Parameter(torch.randn(out_dim, self.rank_shared) * 0.02)
            self.sigma_v_sh = nn.Parameter(torch.zeros(self.rank_shared))
        else:
            self.theta_a_q_sh = None
            self.theta_b_q_sh = None
            self.sigma_q_sh = None
            self.theta_a_v_sh = None
            self.theta_b_v_sh = None
            self.sigma_v_sh = None

        # Private atoms (directions + coefficients)
        if self.rank_private > 0:
            self.theta_a_q_pr = nn.Parameter(torch.randn(in_dim, self.rank_private) * 0.02)
            self.theta_b_q_pr = nn.Parameter(torch.randn(out_dim, self.rank_private) * 0.02)
            self.sigma_q_pr = nn.Parameter(torch.zeros(self.rank_private))

            self.theta_a_v_pr = nn.Parameter(torch.randn(in_dim, self.rank_private) * 0.02)
            self.theta_b_v_pr = nn.Parameter(torch.randn(out_dim, self.rank_private) * 0.02)
            self.sigma_v_pr = nn.Parameter(torch.zeros(self.rank_private))
        else:
            self.theta_a_q_pr = None
            self.theta_b_q_pr = None
            self.sigma_q_pr = None
            self.theta_a_v_pr = None
            self.theta_b_v_pr = None
            self.sigma_v_pr = None

        # Fisher collection (KFAC-style factors) for risk-aware slow merge
        self._collect_fisher = False
        self._fisher_xtx = None
        self._fisher_gtg_q = None
        self._fisher_gtg_v = None
        self._fisher_count = 0

        # cache atoms for task-end extraction
        self._last_atoms = {"q": (None, None, None), "v": (None, None, None)}

    @staticmethod
    def _orthonormal(M: torch.Tensor | None):
        if M is None:
            return None
        Q, _ = torch.linalg.qr(M, mode="reduced")
        return Q

    @staticmethod
    def _compose_delta(B: torch.Tensor | None, A: torch.Tensor | None, sigma: torch.Tensor | None):
        if A is None or B is None or sigma is None or sigma.numel() == 0:
            return None
        return (B * sigma.view(1, -1)) @ A.t()

    def set_shared_atoms(self, atoms_q, atoms_v):
        """
        atoms_q / atoms_v: list of (a, b, sigma_scale), where a,b are 1D tensors.
        This fills the shared tier and resets the private tier.
        """
        if self.rank_shared <= 0:
            # no shared tier: just reset private
            if self.rank_private > 0:
                self.theta_a_q_pr.data.normal_(0, 0.02)
                self.theta_b_q_pr.data.normal_(0, 0.02)
                self.sigma_q_pr.data.zero_()
                self.theta_a_v_pr.data.normal_(0, 0.02)
                self.theta_b_v_pr.data.normal_(0, 0.02)
                self.sigma_v_pr.data.zero_()
            return

        def _fill(A_sh, B_sh, s_sh, atoms):
            A_sh.data.zero_()
            B_sh.data.zero_()
            s_sh.data.zero_()
            k = min(len(atoms), A_sh.shape[1])
            if k > 0:
                a_mat = torch.stack([atoms[i][0] for i in range(k)], dim=1).to(A_sh.device)
                b_mat = torch.stack([atoms[i][1] for i in range(k)], dim=1).to(B_sh.device)
                s_vec = torch.tensor([float(atoms[i][2]) for i in range(k)], device=s_sh.device, dtype=s_sh.dtype)
                A_sh.data[:, :k].copy_(a_mat)
                B_sh.data[:, :k].copy_(b_mat)
                s_sh.data[:k].copy_(s_vec)

        _fill(self.theta_a_q_sh, self.theta_b_q_sh, self.sigma_q_sh, atoms_q)
        _fill(self.theta_a_v_sh, self.theta_b_v_sh, self.sigma_v_sh, atoms_v)

        # freeze shared directions, keep coefficients trainable
        self.theta_a_q_sh.requires_grad_(False)
        self.theta_b_q_sh.requires_grad_(False)
        self.theta_a_v_sh.requires_grad_(False)
        self.theta_b_v_sh.requires_grad_(False)
        self.sigma_q_sh.requires_grad_(True)
        self.sigma_v_sh.requires_grad_(True)

        # reset private tier
        if self.rank_private > 0:
            self.theta_a_q_pr.data.normal_(0, 0.02)
            self.theta_b_q_pr.data.normal_(0, 0.02)
            self.sigma_q_pr.data.zero_()
            self.theta_a_v_pr.data.normal_(0, 0.02)
            self.theta_b_v_pr.data.normal_(0, 0.02)
            self.sigma_v_pr.data.zero_()

    def fisher_begin(self):
        """
        Enable Fisher factor collection for this module.
        Factors are accumulated on CPU.
        """
        self._collect_fisher = True
        in_dim = self.base.in_features
        out_dim = self.base.out_features // 3
        self._fisher_xtx = torch.zeros(in_dim, in_dim, dtype=torch.float32, device="cpu")
        self._fisher_gtg_q = torch.zeros(out_dim, out_dim, dtype=torch.float32, device="cpu")
        self._fisher_gtg_v = torch.zeros(out_dim, out_dim, dtype=torch.float32, device="cpu")
        self._fisher_count = 0

    def fisher_end(self):
        self._collect_fisher = False

    def fisher_get_stats(self):
        return self._fisher_xtx, self._fisher_gtg_q, self._fisher_gtg_v, int(self._fisher_count)

    def _accumulate_fisher(self, x_det: torch.Tensor, grad_out: torch.Tensor):
        if self._fisher_xtx is None:
            return
        x2d = x_det.reshape(-1, x_det.shape[-1])  # [M, in_dim]
        g2d = grad_out.reshape(-1, grad_out.shape[-1])  # [M, 3*out_dim]
        out_dim = self.base.out_features // 3
        gq = g2d[:, :out_dim]
        gv = g2d[:, 2*out_dim:3*out_dim]

        self._fisher_xtx.add_((x2d.t() @ x2d).detach().to("cpu", dtype=torch.float32))
        self._fisher_gtg_q.add_((gq.t() @ gq).detach().to("cpu", dtype=torch.float32))
        self._fisher_gtg_v.add_((gv.t() @ gv).detach().to("cpu", dtype=torch.float32))
        self._fisher_count += x2d.shape[0]

    def extract_atoms(self):
        """
        Export current atoms as a dict: {"q": (A,B,sigma), "v": (A,B,sigma)},
        where columns correspond to atoms and sigma is non-negative.
        """
        def _pack(A_sh, B_sh, s_sh, A_pr, B_pr, s_pr):
            mats_a = []
            mats_b = []
            vec_s = []
            if A_sh is not None:
                mats_a.append(A_sh)
                mats_b.append(B_sh)
                vec_s.append(s_sh.abs())
            if A_pr is not None:
                mats_a.append(A_pr)
                mats_b.append(B_pr)
                vec_s.append(s_pr.abs())
            if not mats_a:
                return None, None, None
            A = torch.cat(mats_a, dim=1)
            B = torch.cat(mats_b, dim=1)
            s = torch.cat(vec_s, dim=0)
            return A.detach().cpu(), B.detach().cpu(), s.detach().cpu()

        Aq, Bq, sq = _pack(self.theta_a_q_sh, self.theta_b_q_sh, self.sigma_q_sh,
                           self.theta_a_q_pr, self.theta_b_q_pr, self.sigma_q_pr)
        Av, Bv, sv = _pack(self.theta_a_v_sh, self.theta_b_v_sh, self.sigma_v_sh,
                           self.theta_a_v_pr, self.theta_b_v_pr, self.sigma_v_pr)
        return {"q": (Aq, Bq, sq), "v": (Av, Bv, sv)}

    def forward(self, x):
        if self.dropout is not None:
            x_in = self.dropout(x)
        else:
            x_in = x

        # Orthonormalize directions
        Aq_sh = self._orthonormal(self.theta_a_q_sh)
        Bq_sh = self._orthonormal(self.theta_b_q_sh)
        Av_sh = self._orthonormal(self.theta_a_v_sh)
        Bv_sh = self._orthonormal(self.theta_b_v_sh)

        Aq_pr = self._orthonormal(self.theta_a_q_pr)
        Bq_pr = self._orthonormal(self.theta_b_q_pr)
        Av_pr = self._orthonormal(self.theta_a_v_pr)
        Bv_pr = self._orthonormal(self.theta_b_v_pr)

        dW_q = None
        dW_v = None
        if self.rank_shared > 0:
            dW_q = self._compose_delta(Bq_sh, Aq_sh, self.sigma_q_sh)
            dW_v = self._compose_delta(Bv_sh, Av_sh, self.sigma_v_sh)
        if self.rank_private > 0:
            dq_pr = self._compose_delta(Bq_pr, Aq_pr, self.sigma_q_pr)
            dv_pr = self._compose_delta(Bv_pr, Av_pr, self.sigma_v_pr)
            dW_q = dq_pr if dW_q is None else (dW_q + dq_pr)
            dW_v = dv_pr if dW_v is None else (dW_v + dv_pr)

        if dW_q is None:
            # no atoms at all
            return self.base(x_in)

        out_dim = self.base.out_features // 3
        dW_k = torch.zeros_like(dW_q)
        dW_qkv = torch.cat([dW_q, dW_k, dW_v], dim=0)
        out = self.base(x_in) + F.linear(x_in, dW_qkv) * self.alpha

        # cache atoms for extraction
        self._last_atoms = self.extract_atoms()

        if self._collect_fisher and out.requires_grad:
            x_det = x_in.detach()
            def _hook(grad):
                self._accumulate_fisher(x_det, grad.detach())
            out.register_hook(_hook)

        return out

class GeoLoRA_ViT_timm(nn.Module):
    def __init__(
        self,
        vit_model: nn.Module,
        rank_total: int,
        rank_shared: int = 0,
        alpha: float = 1.0,
        merge_weights: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vit = vit_model
        self.rank_total = int(rank_total)
        self.rank_shared = int(rank_shared)
        self.alpha = float(alpha)
        self.merge_weights = bool(merge_weights)
        self.dropout = float(dropout)

        for p in self.vit.parameters():
            p.requires_grad = False

        self._adapters: List[GeoLoRA_QKV] = []
        self._inject()
        self.out_dim = getattr(self.vit, "num_features", 768)

    def _inject(self):
        for blk in self.vit.blocks:
            qkv = blk.attn.qkv
            adapter = GeoLoRA_QKV(
                base=qkv,
                rank_total=self.rank_total,
                rank_shared=self.rank_shared,
                alpha=self.alpha,
                merge_weights=self.merge_weights,
                dropout=self.dropout,
            )
            blk.attn.qkv = adapter
            self._adapters.append(adapter)

    def forward(self, x):
        return self.vit(x)

    def adapters(self) -> List[GeoLoRA_QKV]:
        return self._adapters

# ---------- Learner with slow merge ----------
class Learner(LoraBaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._backbone_type = "vit_base_patch16_224"
        self._rank = int(args.get("geolora_rank", 8))
        self._rank_shared = int(args.get("geolora_rank_shared", 0))
        self._alpha = float(args.get("geolora_alpha", 1.0))

        # dictionary / slow-merge hyperparameters
        self._dict_size = int(args.get("geolora_dict_size", 32))
        self._merge_sim_thresh = float(args.get("geolora_merge_sim_thresh", args.get("geolora_sim_thresh", 0.95)))
        self._risk_lambda = float(args.get("geolora_risk_lambda", 0.0))

        # fisher estimation hyperparameters
        self._fisher_gamma = float(args.get("geolora_fisher_gamma", 0.9))
        self._fisher_topk_in = int(args.get("geolora_fisher_topk_in", 32))
        self._fisher_topk_out = int(args.get("geolora_fisher_topk_out", 32))
        self._fisher_max_batches = int(args.get("geolora_fisher_max_batches", 10))

        # per-layer dictionaries and per-task atoms, separated by branch ("q"/"v")
        # dict atom record: (a, b, sigma_scale)
        self._dicts: List[Dict[str, List[Tuple[torch.Tensor, torch.Tensor, float]]]] = []
        self._task_atoms: List[Dict[str, Dict[int, List[Tuple[torch.Tensor, torch.Tensor, float]]]]] = []

        # fisher KFAC factors (cumulative) and their leading eigenspaces for risk-aware merge
        self._fisher_cov_in: List[torch.Tensor | None] = []
        self._fisher_cov_out_q: List[torch.Tensor | None] = []
        self._fisher_cov_out_v: List[torch.Tensor | None] = []
        self._fisher_subspace: List[Dict[str, torch.Tensor] | None] = []

    def after_task(self):
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def build_lora_backbone(self):
        vit = timm.create_model(self._backbone_type, pretrained=True, num_classes=0)
        vit.reset_classifier(0)
        model = GeoLoRA_ViT_timm(
            vit_model=vit.eval(),
            rank_total=int(self._rank),
            rank_shared=int(self._rank_shared),
            alpha=float(self._alpha),
            merge_weights=bool(self.args.get("geolora_merge_weights", False)),
            dropout=float(self.args.get("geolora_dropout", 0.0)),
        )
        return model

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

        # Fast-tier initialization: load shared atoms from the current dictionary and reset private atoms
        self._refresh_shared_from_dict()

        self._train(self.train_loader, self.test_loader)
        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._network = self._unwrap_network()

        # slow merge
        self._slow_merge(self._cur_task)

        # Commit merged dictionary back to the shared tier (and reset private tier)
        self._refresh_shared_from_dict()

        # Update old-task Fisher principal subspaces (used by the next slow-merge step)
        self._update_fisher_subspace(self.train_loader)

        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as _nme_exc:  # pylint: disable=broad-except
            self._log(f"[GeoLoRA_SlowMerge][NME] Failed to compute class means: {_nme_exc}")

    def _train(self, train_loader, test_loader):
        network = self._unwrap_network()
        if not self._lora_initialized:
            network.backbone = self.build_lora_backbone()
            network.backbone.to(self._device)
            self._lora_initialized = True
        self._network = network
        self._prepare_network()

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

    # ---------------- slow merge ---------------- #
    @staticmethod
    def _canonicalize_atoms(A: torch.Tensor, B: torch.Tensor, sigma: torch.Tensor):
        idx = torch.argsort(sigma, descending=True)
        A, B, sigma = A[:, idx], B[:, idx], sigma[idx]
        for i in range(sigma.numel()):
            a = A[:, i]
            b = B[:, i]
            pivot = torch.argmax(torch.abs(a))
            if a[pivot] < 0:
                A[:, i] = -a
                B[:, i] = -b
        return A, B, sigma

    @staticmethod
    def _sim(a1, b1, a2, b2):
        return float(torch.abs(torch.dot(a1, a2)) * torch.abs(torch.dot(b1, b2)))

    @staticmethod
    def _coeff_to_dict(task_atoms, dict_atoms):
        c = []
        for ak, bk in dict_atoms:
            s = 0.0
            for (ai, bi, si) in task_atoms:
                s += float(si) * float(torch.dot(ai, ak)) * float(torch.dot(bi, bk))
            c.append(s)
        return c

    def _refresh_shared_from_dict(self):
        r_sh = int(self.args.get("geolora_rank_shared", 0))
        if r_sh <= 0:
            return
        adapters = self._unwrap_network().backbone.adapters()
        for lidx, qkv in enumerate(adapters):
            if not isinstance(qkv, GeoLoRA_QKV):
                continue
            atoms_q = self._dicts[lidx]["q"][:r_sh] if lidx < len(self._dicts) else []
            atoms_v = self._dicts[lidx]["v"][:r_sh] if lidx < len(self._dicts) else []
            qkv.set_shared_atoms(atoms_q, atoms_v)

    @staticmethod
    def _coeff_to_atom(task_atoms: List[Tuple[torch.Tensor, torch.Tensor, float]], a: torch.Tensor, b: torch.Tensor) -> float:
        c = 0.0
        for ai, bi, si in task_atoms:
            c += float(si) * float(torch.dot(ai, a)) * float(torch.dot(bi, b))
        return c

    def _fisher_risk(self, layer_idx: int, branch: str, a: torch.Tensor, b: torch.Tensor) -> float:
        if layer_idx >= len(self._fisher_subspace) or self._fisher_subspace[layer_idx] is None:
            return 0.0
        sub = self._fisher_subspace[layer_idx]
        Uin = sub.get("U_in", None)
        Sin = sub.get("S_in", None)
        tr_in = float(sub.get("tr_in", 0.0))
        if Uin is None or Sin is None or tr_in <= 0:
            return 0.0

        if branch == "q":
            Uout = sub.get("U_out_q", None)
            Sout = sub.get("S_out_q", None)
            tr_out = float(sub.get("tr_out_q", 0.0))
        else:
            Uout = sub.get("U_out_v", None)
            Sout = sub.get("S_out_v", None)
            tr_out = float(sub.get("tr_out_v", 0.0))

        if Uout is None or Sout is None or tr_out <= 0:
            return 0.0

        ain = (Uin.t() @ a).pow(2)
        a_energy = float((ain * Sin).sum().item())
        bout = (Uout.t() @ b).pow(2)
        b_energy = float((bout * Sout).sum().item())

        return (a_energy / tr_in) * (b_energy / tr_out)

    def _update_fisher_subspace(self, dataloader):
        if dataloader is None:
            return

        net = self._unwrap_network()
        net.eval()

        adapters = net.backbone.adapters()
        for qkv in adapters:
            if isinstance(qkv, GeoLoRA_QKV):
                qkv.fisher_begin()

        max_batches = int(self._fisher_max_batches)
        n_batches = 0

        for batch in dataloader:
            if n_batches >= max_batches:
                break
            n_batches += 1

            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch.get("inputs", batch.get("input", batch.get("x", None)))
                targets = batch.get("targets", batch.get("target", batch.get("label", batch.get("y", None))))
            else:
                continue

            if inputs is None:
                continue

            inputs = inputs.to(self._device, non_blocking=True)
            if targets is not None:
                targets = targets.to(self._device, non_blocking=True)

            net.zero_grad(set_to_none=True)

            out = net(inputs)
            if isinstance(out, dict):
                logits = out.get("logits", None)
                if logits is None:
                    for v in out.values():
                        if torch.is_tensor(v):
                            logits = v
                            break
            elif isinstance(out, (list, tuple)):
                logits = out[0]
            else:
                logits = out

            if logits is None or not torch.is_tensor(logits):
                continue

            if targets is None:
                targets = torch.argmax(logits.detach(), dim=-1)

            loss = F.cross_entropy(logits, targets)
            loss.backward()

        for lidx, qkv in enumerate(adapters):
            if not isinstance(qkv, GeoLoRA_QKV):
                continue
            xtx, gtg_q, gtg_v, count = qkv.fisher_get_stats()
            qkv.fisher_end()
            if count <= 0:
                continue

            cov_in = xtx / float(count)
            cov_out_q = gtg_q / float(count)
            cov_out_v = gtg_v / float(count)

            if self._fisher_cov_in[lidx] is None:
                self._fisher_cov_in[lidx] = cov_in
                self._fisher_cov_out_q[lidx] = cov_out_q
                self._fisher_cov_out_v[lidx] = cov_out_v
            else:
                self._fisher_cov_in[lidx] = self._fisher_cov_in[lidx] * self._fisher_gamma + cov_in
                self._fisher_cov_out_q[lidx] = self._fisher_cov_out_q[lidx] * self._fisher_gamma + cov_out_q
                self._fisher_cov_out_v[lidx] = self._fisher_cov_out_v[lidx] * self._fisher_gamma + cov_out_v

            def _topk_eigh(C: torch.Tensor, k: int):
                C = 0.5 * (C + C.t())
                evals, evecs = torch.linalg.eigh(C)
                evals = torch.clamp(evals, min=0.0)
                k = min(k, evals.numel())
                if k <= 0:
                    return None, None, 0.0
                idx = torch.argsort(evals, descending=True)[:k]
                return evecs[:, idx], evals[idx], float(evals.sum().item())

            U_in, S_in, tr_in = _topk_eigh(self._fisher_cov_in[lidx], self._fisher_topk_in)
            U_q, S_q, tr_q = _topk_eigh(self._fisher_cov_out_q[lidx], self._fisher_topk_out)
            U_v, S_v, tr_v = _topk_eigh(self._fisher_cov_out_v[lidx], self._fisher_topk_out)

            self._fisher_subspace[lidx] = {
                "U_in": U_in,
                "S_in": S_in,
                "tr_in": tr_in,
                "U_out_q": U_q,
                "S_out_q": S_q,
                "tr_out_q": tr_q,
                "U_out_v": U_v,
                "S_out_v": S_v,
                "tr_out_v": tr_v,
            }

    def _slow_merge(self, task_id: int):
        """
        Risk-aware slow merge.

        For each layer and branch (q/v):
        1) Extract current task atoms (shared+private) and store in self._task_atoms.
        2) Build candidate directions from existing dictionary atoms and new task atoms.
        3) Score each candidate by (utility - λ * risk), where:
           - utility is the accumulated explained energy across seen tasks (via projection coefficients),
           - risk is the occupancy of old-task Fisher principal directions (KFAC top-eigenspaces).
        4) Select top-K candidates with redundancy pruning to form the new dictionary.
        """
        adapters = self._unwrap_network().backbone.adapters()

        # 1) extract atoms for this task
        for lidx, qkv in enumerate(adapters):
            if not isinstance(qkv, GeoLoRA_QKV):
                continue
            atoms = qkv.extract_atoms()
            for branch in ["q", "v"]:
                A, B, s = atoms[branch]
                if A is None:
                    self._task_atoms[lidx][branch][task_id] = []
                    continue
                A, B, s = self._canonicalize_atoms(A, B, s)
                task_list = []
                for i in range(A.shape[1]):
                    a = A[:, i].contiguous()
                    b = B[:, i].contiguous()
                    sigma = float(s[i].item())
                    task_list.append((a, b, sigma))
                self._task_atoms[lidx][branch][task_id] = task_list

        # 2) merge into dictionary per layer/branch
        for lidx in range(len(adapters)):
            for branch in ["q", "v"]:
                existing = self._dicts[lidx][branch]

                # collect candidate directions (a,b) from existing dict + new task atoms
                cand_dirs: List[Tuple[torch.Tensor, torch.Tensor]] = []
                for a, b, _ in existing:
                    cand_dirs.append((a, b))
                for a, b, _ in self._task_atoms[lidx][branch].get(task_id, []):
                    cand_dirs.append((a, b))

                unique: List[Tuple[torch.Tensor, torch.Tensor]] = []
                for a, b in cand_dirs:
                    redundant = False
                    for ua, ub in unique:
                        sim = abs(float(torch.dot(a, ua)) * float(torch.dot(b, ub)))
                        if sim >= self._merge_sim_thresh:
                            redundant = True
                            break
                    if not redundant:
                        unique.append((a, b))

                if not unique:
                    self._dicts[lidx][branch] = []
                    continue

                scored = []
                for a, b in unique:
                    utility = 0.0
                    for t, atoms_t in self._task_atoms[lidx][branch].items():
                        if t > task_id:
                            continue
                        c = self._coeff_to_atom(atoms_t, a, b)
                        utility += c * c
                    risk = self._fisher_risk(lidx, branch, a, b) if self._risk_lambda > 0 else 0.0
                    score = utility - self._risk_lambda * risk
                    scored.append((score, utility, risk, a, b))

                scored.sort(key=lambda x: x[0], reverse=True)

                selected: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
                for score, utility, risk, a, b in scored:
                    if len(selected) >= self._dict_size:
                        break
                    is_red = False
                    for sa, sb, _ in selected:
                        sim = abs(float(torch.dot(a, sa)) * float(torch.dot(b, sb)))
                        if sim >= self._merge_sim_thresh:
                            is_red = True
                            break
                    if is_red:
                        continue
                    sigma_scale = math.sqrt(max(utility, 0.0))
                    selected.append((a, b, sigma_scale))

                self._dicts[lidx][branch] = selected

        if self._cur_task > 0:
            print(f"[SlowMerge] task={task_id} dict_size={self._dict_size} risk_lambda={self._risk_lambda}")


    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
