# === NEW FILE: efm_trust.py ===
import copy
import torch
import torch.nn.functional as F

class EFMTrustRegionHelper:
    """
    统一接口的 EFM 附件：
    - capture_teacher(model): 在阶段/任务开始前冻结 teacher 快照（含已扩展 fc）
    - select_slice(known, total, loss_on_new, subset): 选择类子空间 [start,end)
    - penalty(inputs, outputs, known, total, loss_on_new): 计算特征域 trust-region 项
    依赖：模型 forward 返回 dict，至少包含 {"logits":..., "features":...}
    """
    def __init__(self, args: dict):
        self.enable = bool(args.get("efm_enable", False))
        self.lam    = float(args.get("efm_lambda", 0.1))
        self.eta    = float(args.get("efm_eta", 0.01))
        self.tau    = float(args.get("efm_tau", 1.0))
        # "auto"（与 CE 一致）|"new"|"all"
        self.subset = str(args.get("efm_subset", "auto")).lower()
        self.teacher = None

    def capture_teacher(self, model):
        """冻结一份 teacher（含当前已扩展的 fc）。在阶段/任务开始前调用。"""
        if not self.enable:
            self.teacher = None
            return
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def select_slice(self, known: int, total: int, loss_on_new: bool):
        """与 CE 切片语义对齐。"""
        if self.subset == "all":
            return 0, total
        if self.subset == "new":
            return known, total
        # auto: 与 CE 的选择一致
        return (known, total) if loss_on_new else (0, total)

    @torch.no_grad()
    def _teacher_forward(self, inputs):
        return self.teacher(inputs)

    def penalty(self, inputs, outputs: dict, known: int, total: int, loss_on_new: bool):
        """
        计算 batch 的 EFM trust-region 二次项均值（未乘 self.lam）。
        要求 outputs, teacher_outputs 均含 "features","logits"。
        """
        if (not self.enable) or (self.teacher is None):
            return None

        f_cur = outputs.get("features", None)
        if f_cur is None:
            return None

        with torch.no_grad():
            t_out = self._teacher_forward(inputs)
            f_prev = t_out.get("features", None)
            if f_prev is None:
                return None

        delta_f = f_cur - f_prev  # [B, D]
        start, end = self.select_slice(known, total, loss_on_new)
        if end <= start:
            return None

        # Teacher 头与 softmax 概率（与切片一致）
        W_prev = self.teacher.fc.weight[start:end]  # [C_sub, D]
        with torch.no_grad():
            logits_prev = t_out["logits"][:, start:end] / max(self.tau, 1e-8)
            p_prev = F.softmax(logits_prev, dim=1)     # [B, C_sub]

        # Fisher/GN 二次型：Δeᵀ Wᵀ(Diag(p)-ppᵀ)W Δe + η||Δe||²
        q = torch.matmul(delta_f, W_prev.t())         # [B, C_sub]
        term1 = (p_prev * (q ** 2)).sum(dim=1)        # [B]
        term2 = ((p_prev * q).sum(dim=1)) ** 2        # [B]
        quad  = term1 - term2
        if self.eta > 0:
            quad = quad + self.eta * (delta_f ** 2).sum(dim=1)
        return quad.mean()
