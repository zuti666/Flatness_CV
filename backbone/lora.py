# Sheng Wang at Feb 22 2023

import math
import json

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from safetensors import safe_open
# from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter

from backbone.base_vit import ViT
import os
from backbone.linears import SimpleLinear
import gc
import torch.nn.utils as utils
import copy

class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class LoRA_ViT(nn.Module):
    """Applies low-rank adaptation to a vision transformer.
    Args:
        vit_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.
    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """
    def __init__(self, vit_model: ViT, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT, self).__init__()

        assert r > 0
        base_vit_dim = vit_model.transformer.blocks[0].attn.proj_q.in_features
        dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.transformer.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.transformer.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_q_linear = blk.attn.proj_q
            w_v_linear = blk.attn.proj_v
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)


class _LoRA_qkv_timm(nn.Module):
    """
    In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x)) #* self.scaling_factor
        new_v = self.linear_b_v(self.linear_a_v(x)) #* self.scaling_factor
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv
    
class _LoRA_qkv_timm_train(nn.Module):
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v, #linear_a_q1, linear_b_q1, linear_a_v1, linear_b_v1,
        task_id, saved_A, saved_B, t_layer_i, rank, scaling_factor, scaling_factor_prev, eval1=False):
        super().__init__()
        # Align modules to the same device as qkv parameters
        self.qkv = qkv
        device = getattr(getattr(qkv, "weight", None), "device", torch.device("cpu"))
        self.linear_a_q = linear_a_q.to(device)
        self.linear_b_q = linear_b_q.to(device)
        self.linear_a_v = linear_a_v.to(device)
        self.linear_b_v = linear_b_v.to(device)

        # scaling wrappers (ModuleList)
        try:
            self.scaling_factor = scaling_factor.to(device)
        except Exception:
            self.scaling_factor = scaling_factor
        try:
            self.scaling_factor_prev = scaling_factor_prev.to(device)
        except Exception:
            self.scaling_factor_prev = scaling_factor_prev

        self.task_id = task_id
        self.dim = qkv.in_features
        self.saved_A = saved_A
        self.saved_B = saved_B
        self.t_layer_i = t_layer_i
        self.rank = rank
        self.eval = eval1

    def forward(self, x):

        w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
        w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)
         
        new_q, new_v = 0, 0
        for i in range(self.task_id):
        # for i in range(0):
            key_a, key_b = 'saved_A_'+str(i), 'saved_B_'+str(i)
            if key_a not in self.saved_A or key_b not in self.saved_B:
                continue
            saved_A_i, saved_B_i = self.saved_A[key_a], self.saved_B[key_b]
            Q, V = list(enumerate(zip(saved_A_i,saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
            _, (A_q, B_q) = Q
            _, (A_v, B_v) = V

            w_a_linear_q.weight = Parameter(A_q.weight)
            w_a_linear_q.weight.requires_grad = False 
            w_a_linear_q.to(x.device)
            w_b_linear_q.weight = Parameter(B_q.weight)
            w_b_linear_q.weight.requires_grad = False 
            w_b_linear_q.to(x.device)
            w_a_linear_v.weight = Parameter(A_v.weight)
            w_a_linear_v.weight.requires_grad = False 
            w_a_linear_v.to(x.device)
            w_b_linear_v.weight = Parameter(B_v.weight)
            w_b_linear_v.weight.requires_grad = False  
            w_b_linear_v.to(x.device)


            if i ==0 :
                new_q = self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x))/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
                new_v = self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x))/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )
            else:

                new_q += self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x))/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
                new_v += self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x))/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )

        new_q += self.scaling_factor[0]( self.linear_b_q(self.linear_a_q(x)) )
        new_v += self.scaling_factor[0]( self.linear_b_v(self.linear_a_v(x)) )
        qkv = self.qkv(x) 
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv

class _LoRA_qkv_timm_eval(nn.Module):
    """
    In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """
    def __init__(
        self,
        task_id: int,
        qkv: nn.Module,
        saved_A: dict,
        saved_B: dict,
        t_layer_i: int,
        rank: int,
        scaling_factor: nn.Module,        # ModuleList[ParameterWrapper 或 IdentityScale]
        scaling_factor_prev: nn.Module,   # ModuleList 同上
        save_file: str,
        learn_alpha: bool = False,
    ):
        super().__init__()
        self.task_id = task_id
        self.qkv = qkv
        self.dim = qkv.in_features
        self.saved_A = saved_A # 这里通常仅含 [0 .. task_id-1] 的快照
        self.saved_B = saved_B
        self.t_layer_i = t_layer_i
        self.rank = rank

        self.save_file = save_file
        self.learn_alpha = learn_alpha
        # Move scaling modules to the same device as qkv parameters
        device = getattr(getattr(qkv, "weight", None), "device", torch.device("cpu"))
        try:
            self.scaling_factor = scaling_factor.to(device)
        except Exception:
            self.scaling_factor = scaling_factor
        try:
            self.scaling_factor_prev = scaling_factor_prev.to(device)
        except Exception:
            self.scaling_factor_prev = scaling_factor_prev

        # 说明：learn_alpha=True 时，建议在 LoRA_ViT_timm.__init__ (eval=True) 里
        # 已经把磁盘上的 scaling_factor 矩阵恢复进 wrapped_param / wrapped_param_prev。
        # 因此本类不再重复读取 scaling 参数；避免二次 I/O 与状态不一致。

    @torch.no_grad()  # 评测期：不建立计算图
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device

        # 临时 Linear：只作为权重载体，不注册可训练参数
        w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False).to(device)
        w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False).to(device)
        w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False).to(device)
        w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False).to(device)

        new_q, new_v = None, None

        # -------- 1) 历史任务累加：i ∈ [0, task_id-1] --------
        for i in range(self.task_id):
            key_a, key_b = f"saved_A_{i}", f"saved_B_{i}"
            if key_a not in self.saved_A or key_b not in self.saved_B:
                continue
            saved_A_i, saved_B_i = self.saved_A[key_a], self.saved_B[key_b]
            # 取本 block 的 (Q, V) 两组 LoRA（顺序与训练分支保持一致）
            Q, V = list(enumerate(zip(saved_A_i, saved_B_i)))[self.t_layer_i * 2 : self.t_layer_i * 2 + 2]
            _, (A_q, B_q) = Q
            _, (A_v, B_v) = V

            # 权重写入临时 Linear，并禁止梯度
            w_a_linear_q.weight = Parameter(A_q.weight.detach(), requires_grad=False)
            w_b_linear_q.weight = Parameter(B_q.weight.detach(), requires_grad=False)
            w_a_linear_v.weight = Parameter(A_v.weight.detach(), requires_grad=False)
            w_b_linear_v.weight = Parameter(B_v.weight.detach(), requires_grad=False)

            # 归一化防止尺度漂移；加入 clamping 保证数值稳定
            norm_q = (w_b_linear_q.weight.norm() * w_a_linear_q.weight.norm()).clamp_min(1e-12)
            norm_v = (w_b_linear_v.weight.norm() * w_a_linear_v.weight.norm()).clamp_min(1e-12)

            contrib_q = w_b_linear_q(w_a_linear_q(x)) / norm_q
            contrib_v = w_b_linear_v(w_a_linear_v(x)) / norm_v

            # per-task 缩放后累加
            contrib_q = self.scaling_factor_prev[i](contrib_q)
            contrib_v = self.scaling_factor_prev[i](contrib_v)

            if new_q is None:
                new_q, new_v = contrib_q, contrib_v
            else:
                new_q = new_q + contrib_q
                new_v = new_v + contrib_v

        # -------- 2) 当前任务项：i = task_id（显式加载） --------
        cur_idx = self.task_id
        cur_A_list, cur_B_list = None, None

        # 优先从传入的 saved_A/B 读；否则尝试磁盘快照
        cur_key_a, cur_key_b = f"saved_A_{cur_idx}", f"saved_B_{cur_idx}"
        if cur_key_a in self.saved_A and cur_key_b in self.saved_B:
            cur_A_list, cur_B_list = self.saved_A[cur_key_a], self.saved_B[cur_key_b]
        else:
            file_a = os.path.join(self.save_file, f"lora_w_a_{cur_idx}.pt")
            file_b = os.path.join(self.save_file, f"lora_w_b_{cur_idx}.pt")
            if os.path.exists(file_a) and os.path.exists(file_b):
                cur_A_list = torch.load(file_a, map_location="cpu")
                cur_B_list = torch.load(file_b, map_location="cpu")

        if cur_A_list is not None and cur_B_list is not None:
            Qc, Vc = list(enumerate(zip(cur_A_list, cur_B_list)))[self.t_layer_i * 2 : self.t_layer_i * 2 + 2]
            _, (Aq_cur, Bq_cur) = Qc
            _, (Av_cur, Bv_cur) = Vc

            # 将“当前任务”权重装入临时 Linear
            w_a_linear_q.weight = Parameter(Aq_cur.weight.detach(), requires_grad=False)
            w_b_linear_q.weight = Parameter(Bq_cur.weight.detach(), requires_grad=False)
            w_a_linear_v.weight = Parameter(Av_cur.weight.detach(), requires_grad=False)
            w_b_linear_v.weight = Parameter(Bv_cur.weight.detach(), requires_grad=False)

            # 与训练分支保持一致：当前项默认不做 ‖B‖·‖A‖ 归一化
            contrib_q_cur = w_b_linear_q(w_a_linear_q(x))
            contrib_v_cur = w_b_linear_v(w_a_linear_v(x))

            contrib_q_cur = self.scaling_factor[0](contrib_q_cur)
            contrib_v_cur = self.scaling_factor[0](contrib_v_cur)

            if new_q is None:
                new_q, new_v = contrib_q_cur, contrib_v_cur
            else:
                new_q = new_q + contrib_q_cur
                new_v = new_v + contrib_v_cur
        # 若当前任务权重不可用：仅历史项生效（与之前逻辑等价的安全降级）

        # -------- 3) 注入到 qkv 对应通道并返回 --------
        qkv = self.qkv(x)
        if new_q is not None and new_v is not None:
            qkv[:, :, : self.dim]  = qkv[:, :, : self.dim]  + new_q
            qkv[:, :, -self.dim :] = qkv[:, :, -self.dim :] + new_v
        return qkv


    # def forward(self, x): # 推理态：避免对临时 Linear 的权重建立计算图
    #     new_q, new_v = 0, 0

    #     # 为本层构建临时 Linear（仅承载权重，不注册为参数）
    #     w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
    #     w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
    #     w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
    #     w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)


    #     scaling_param = None
    #     if self.learn_alpha and self.task_id > 0:
    #         file_path = self.save_file + 'scaling_factor' + str(self.task_id - 1) + '.pt'
    #         if os.path.exists(file_path):
    #             scaling_param = torch.load(file_path)

    #     # ---------- 1) 历史任务残差累加：i ∈ [0, task_id-1] ----------
    #     for i in range(self.task_id):
    #         key_a, key_b = 'saved_A_'+str(i), 'saved_B_'+str(i)
    #         if key_a not in self.saved_A or key_b not in self.saved_B:
    #             continue
    #         saved_A_i, saved_B_i = self.saved_A[key_a], self.saved_B[key_b]
    #         # 对应本 block 的 (Q, V) 两组 A/B
    #         Q, V = list(enumerate(zip(saved_A_i,saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
    #         _, (A_q, B_q) = Q
    #         _, (A_v, B_v) = V

    #         # w_a_linear_q.weight = Parameter(A_q.weight)
    #         # w_b_linear_q.weight = Parameter(B_q.weight)
    #         # w_a_linear_v.weight = Parameter(A_v.weight)
    #         # w_b_linear_v.weight = Parameter(B_v.weight)

    #         # 把快照权重装入临时 Linear，并且禁止梯度
    #         w_a_linear_q.weight = Parameter(A_q.weight.detach(), requires_grad=False)
    #         w_b_linear_q.weight = Parameter(B_q.weight.detach(), requires_grad=False)
    #         w_a_linear_v.weight = Parameter(A_v.weight.detach(), requires_grad=False)
    #         w_b_linear_v.weight = Parameter(B_v.weight.detach(), requires_grad=False)

            
    #         if i ==0 :
    #             new_q = self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x))/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
    #             new_v = self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x))/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )
    #         else:
    #             new_q += self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x))/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
    #             new_v += self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x))/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )

    #     new_q = self.scaling_factor[0]( w_b_linear_q(w_a_linear_q(x)) )
    #     new_v = self.scaling_factor[0]( w_b_linear_v(w_a_linear_v(x)) )
 
    #     qkv = self.qkv(x) 
    #     qkv[:, :, : self.dim] += new_q
    #     qkv[:, :, -self.dim :] += new_v
    #     return qkv
    


class IdentityScale(nn.Module):
    """Placeholder scaling module that leaves inputs unchanged."""

    def forward(self, x):
        return x


class ParameterWrapper(nn.Module):
    def __init__(self, param: nn.Parameter):
        super(ParameterWrapper, self).__init__()
        self.param = param
    
    def forward(self, x):
        return x * self.param
    
class MyLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyLinear, self).__init__()
        self.linear_b_q = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_b_q = utils.weight_norm(self.linear_b_q)

    def forward(self, x):
        return self.linear_b_q(x)


class LoRA_ViT_timm(nn.Module):
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        num_classes: int = 0,
        increment: int = 10,
        filepath: str = './',
        lora_layer=None,
        eval: bool = False,
        index: bool = True,
        cur_task_index=None,
        *,
        learn_alpha: bool = False,
        max_prev_tasks: int = 200,
    ):
        super(LoRA_ViT_timm, self).__init__()

        assert r > 0
        self.rank = r
        self.learn_alpha = learn_alpha
        self.max_prev_tasks = max_prev_tasks
        self.base_vit = copy.deepcopy(vit_model)
        # set path and step size for both train & eval
        self.save_file = filepath
        self.increment = increment
        if not eval:
            print('save_file', self.save_file)


        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))


        # Trainable A/B for the current task with explicit metadata
        self.w_As, self.w_Bs = [], []
        self.w_meta = []  # per-entry: {'layer': int, 'channel': 'q'|'k'|'v', 'task_id': int}
        self.idx_by_layer_channel = {}

        
        if index:
            print('Initialize task-id and curtask id')
            self.task_id, self.cur_id = 0,0
        
        if cur_task_index != None:
            # print('Update the network!!!', cur_task_index)
            self.task_id = cur_task_index

        # freeze the saved part
        for param in self.base_vit.parameters():
            param.requires_grad = False


        for param in vit_model.parameters():
            param.requires_grad = False

        saved_lora_A, saved_lora_B = {}, {}
        saved_lora_meta = {}
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            if os.path.exists(file_path):
                saved_lora_A['saved_A_'+str(i)] = torch.load(file_path)
            else:
                continue
            file_path = self.save_file+'lora_w_b_'+str(i)+'.pt'
            if os.path.exists(file_path):
                saved_lora_B['saved_B_'+str(i)] = torch.load(file_path)
            # try to read optional metadata for this task
            meta_path = self.save_file + f'lora_meta_{i}.json'
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        saved_lora_meta[f'saved_meta_{i}'] = json.load(f)
                except Exception:
                    pass

        # Expose saved A/B lists for compatibility and analysis
        # Keys: 'saved_A_{t}' / 'saved_B_{t}' -> list[nn.Linear] ordered as [Q0, V0, Q1, V1, ...]
        self.saved_A = saved_lora_A
        self.saved_B = saved_lora_B

        if self.learn_alpha:
            scaling_factor = nn.Parameter(torch.tensor([0.8]))
            self.wrapped_param = nn.ModuleList([ParameterWrapper(scaling_factor)])
            self.wrapped_param_prev = nn.ModuleList(
                [ParameterWrapper(nn.Parameter(torch.tensor([0.8]))) for _ in range(self.max_prev_tasks)]
            )
        else:
            self.wrapped_param = nn.ModuleList([IdentityScale()])
            self.wrapped_param_prev = nn.ModuleList(
                [IdentityScale() for _ in range(self.max_prev_tasks)]
            )

        # Attempt to restore learned scaling parameters for SDLoRA when available
        # NOTE: Guarded by `eval` to avoid changing training-time behavior.
        if self.learn_alpha and self.task_id > 0 and eval:
            try:
                sf_idx = self.task_id - 1
                sf_path = self.save_file + f'scaling_factor{sf_idx}.pt'
                if os.path.exists(sf_path):
                    scaling_param = torch.load(sf_path)
                    # Expect a 2D matrix [max_tasks, max_tasks]; use row sf_idx
                    row = scaling_param[sf_idx]
                    for j in range(sf_idx + 1):
                        if j == sf_idx:
                            if isinstance(self.wrapped_param[0], ParameterWrapper):
                                self.wrapped_param[0].param.data = row[j].detach().clone().to(
                                    self.wrapped_param[0].param.device
                                )
                        else:
                            if isinstance(self.wrapped_param_prev[j], ParameterWrapper):
                                self.wrapped_param_prev[j].param.data = row[j].detach().clone().to(
                                    self.wrapped_param_prev[j].param.device
                                )
            except Exception:
                pass

        # Do the surgery 
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)

            # Append with metadata mapping (no reliance on even/odd)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_meta.append({'layer': t_layer_i, 'channel': 'q', 'task_id': int(self.task_id)})
            self.idx_by_layer_channel[(t_layer_i, 'q')] = len(self.w_As) - 1

            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            self.w_meta.append({'layer': t_layer_i, 'channel': 'v', 'task_id': int(self.task_id)})
            self.idx_by_layer_channel[(t_layer_i, 'v')] = len(self.w_As) - 1

            if not eval:
                blk.attn.qkv = _LoRA_qkv_timm_train(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                    self.task_id,
                    saved_lora_A,
                    saved_lora_B,
                    t_layer_i,
                    self.rank,
                    self.wrapped_param,
                    self.wrapped_param_prev,
                    eval1=False,
                )
            else:
                blk.attn.qkv = _LoRA_qkv_timm_eval(
                    self.task_id,
                    w_qkv_linear,
                    saved_lora_A,
                    saved_lora_B,
                    t_layer_i,
                    self.rank,
                    self.wrapped_param,
                    self.wrapped_param_prev,
                    self.save_file,
                    learn_alpha=self.learn_alpha,
                )

        self.reset_parameters()
        self.lora_vit = vit_model
        # Always output features; classification head is handled externally
        self.lora_vit.head = torch.nn.Identity()
        # Expose saved lists/meta for compatibility and analysis
        self.saved_A = saved_lora_A
        self.saved_B = saved_lora_B
        self.saved_meta = saved_lora_meta



    def reset_lora_vit_head(self):
        task_incremental = self.increment
        device = next(self.lora_vit.parameters()).device
        self.lora_vit.head = self.generate_fc(768, (self.task_id)*task_incremental).to(device)
        temp_weights = torch.load(self.save_file+'CLs_weight'+str(self.task_id-1)+'.pt') 
        temp_bias = torch.load(self.save_file+'CLs_bias'+str(self.task_id-1)+'.pt') 

        self.lora_vit.head.weight.data = temp_weights.data.to(device)
        self.lora_vit.head.bias.data = temp_bias.data.to(device)


    # This part is only used during the evaluation
    def reset(self, eval=False):
        self.__init__(
            self.base_vit,
            self.rank,
            lora_layer=None,
            eval=eval,
            index=False,
            increment=self.increment,
            filepath=self.save_file,
            cur_task_index=self.task_id,
            learn_alpha=self.learn_alpha,
            max_prev_tasks=self.max_prev_tasks,
        )

    def reset_parameters(self) -> None:
        # if self.task_id ==0: 
            for w_A in self.w_As:
                nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
                # nn.init.kaiming_uniform_(w_A.linear_b_q.weight, a=math.sqrt(5) )
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)


    def save_wrap_param(self, filename):
        if not self.learn_alpha:
            return
        if self.task_id ==1:   
            scaling_param = torch.zeros(self.max_prev_tasks, self.max_prev_tasks)
        else:
            scaling_param = torch.load(filename + 'scaling_factor'+str(self.task_id-2)+'.pt')
        i = self.task_id-1
        # print('save i', i)
        for j in range(i+1):
            if j == i:
                scaling_param[i][j] = self.wrapped_param[0].param.clone()
            else:
                scaling_param[i][j] = self.wrapped_param_prev[j].param.clone()  
        torch.save(scaling_param, filename + 'scaling_factor'+str(self.task_id-1)+'.pt')
        
    def save_lora_parameters(self, filename: str, task_id) -> None:
        self.task_id += 1
        if not os.path.exists(filename):
           os.makedirs(filename)
        torch.save(self.w_As, filename + 'lora_w_a_'+str(task_id)+'.pt')
        torch.save(self.w_Bs, filename + 'lora_w_b_'+str(task_id)+'.pt')
        # write side-car metadata for explicit layer/channel/task mapping
        try:
            with open(filename + f'lora_meta_{task_id}.json', 'w') as f:
                json.dump(self.w_meta, f)
        except Exception:
            pass

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def load_eval_vit(self):
        self.lora_vit = copy.deepcopy(self.base_vit)
        saved_lora_A, saved_lora_B = {}, {}
        saved_lora_meta = {}
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            if os.path.exists(file_path):
                saved_lora_A['saved_A_'+str(i)] = torch.load(file_path)
            file_path = self.save_file+'lora_w_b_'+str(i)+'.pt'
            if os.path.exists(file_path):
                saved_lora_B['saved_B_'+str(i)] = torch.load(file_path)
            meta_path = self.save_file + f'lora_meta_{i}.json'
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        saved_lora_meta[f'saved_meta_{i}'] = json.load(f)
                except Exception:
                    pass

        # 同步到公开属性，便于外部查询历史任务 LoRA
        self.saved_A = saved_lora_A
        self.saved_B = saved_lora_B
        self.saved_meta = saved_lora_meta

        # for param in self.eval_vit.parameters():
        for param in self.lora_vit.parameters():
            param.requires_grad = False
        
        # for t_layer_i, blk in enumerate(self.eval_vit.blocks):
        for t_layer_i, blk in enumerate(self.lora_vit.blocks):
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            blk.attn.qkv = _LoRA_qkv_timm_eval(
                self.task_id,
                w_qkv_linear,
                saved_lora_A,
                saved_lora_B,
                t_layer_i,
                self.rank,
                self.wrapped_param,
                self.wrapped_param_prev,
                self.save_file,
                learn_alpha=self.learn_alpha,
            )
        self.reset_lora_vit_head()

    def compute_ortho_loss(self):
        loss = torch.tensor(0.0, dtype=torch.float32)
        # print('task_id', self.task_id)
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            if os.path.exists(file_path):
                w_As = torch.load(file_path)
                num_layer = len(self.w_As)
                for j in range(num_layer):
                    temp = torch.matmul(w_As[j].weight.to(self.w_As[j].weight.device), self.w_As[j].weight.t())
                    temp = torch.sum(torch.square(temp))
                    loss = loss.to(self.w_As[j].weight.device)
                    loss += temp
        return loss
    
    def forward(self, x: Tensor, loss= False, eval=False) -> Tensor:
        if eval:
            self.reset(eval=True)
            return self.lora_vit(x)
        elif loss:
            loss = self.compute_ortho_loss()
            return self.lora_vit(x), loss
        else:
            return self.lora_vit(x)

    # ===== Compatibility helpers for Attention_LoRA-style inspection/initialization =====
    def _qv_index(self, layer_idx: int):
        base = int(layer_idx) * 2
        return base, base + 1  # (q_idx, v_idx)

    def get_matrix(self, layer_idx: int, task_idx: int = None, device=None):
        """Return B@A for (q, v) of a given layer and task.

        - layer_idx: transformer block index where LoRA is attached
        - task_idx: 0..(self.task_id) where self.task_id denotes the CURRENT task being trained.
                    If None, use current task.
        - device: optional device to place returned tensors.

        Returns: (matrix_q, matrix_v) each of shape [dim, dim]
        """
        if task_idx is None:
            task_idx = int(self.task_id)

        q_idx, v_idx = self._qv_index(layer_idx)
        if task_idx == self.task_id:
            A_q = self.w_As[q_idx].weight
            B_q = self.w_Bs[q_idx].weight
            A_v = self.w_As[v_idx].weight
            B_v = self.w_Bs[v_idx].weight
        else:
            key_a, key_b = f'saved_A_{task_idx}', f'saved_B_{task_idx}'
            if key_a not in getattr(self, 'saved_A', {}) or key_b not in getattr(self, 'saved_B', {}):
                raise ValueError(f'LoRA for task {task_idx} not available')
            A_list = self.saved_A[key_a]
            B_list = self.saved_B[key_b]
            A_q = A_list[q_idx].weight
            B_q = B_list[q_idx].weight
            A_v = A_list[v_idx].weight
            B_v = B_list[v_idx].weight

        mat_q = B_q @ A_q
        mat_v = B_v @ A_v
        if device is not None:
            mat_q = mat_q.to(device)
            mat_v = mat_v.to(device)
        return mat_q, mat_v

    def get_pre_matrix(self, layer_idx: int, upto_task_idx: int, device=None):
        """Sum of B@A over tasks [0, upto_task_idx), for (q, v) at a given layer."""
        if upto_task_idx <= 0:
            dim = getattr(self, 'dim', None) or (self.w_As[0].in_features if len(self.w_As) > 0 else None)
            if dim is None:
                raise ValueError('Cannot infer dimension for empty LoRA lists')
            zero = torch.zeros((dim, dim))
            return (zero.to(device) if device else zero.clone(), zero.to(device) if device else zero.clone())

        mats_q, mats_v = None, None
        for t in range(int(upto_task_idx)):
            try:
                mq, mv = self.get_matrix(layer_idx, t, device=device)
            except Exception:
                continue
            mats_q = mq if mats_q is None else (mats_q + mq)
            mats_v = mv if mats_v is None else (mats_v + mv)
        if mats_q is None:
            dim = getattr(self, 'dim', None) or (self.w_As[0].in_features if len(self.w_As) > 0 else None)
            mq = torch.zeros((dim, dim))
            mv = torch.zeros((dim, dim))
            if device is not None:
                mq, mv = mq.to(device), mv.to(device)
            return mq, mv
        return mats_q, mats_v

    def init_current_task_A(self, layer_idx: int, A_q=None, A_v=None, scale: float = 1.0):
        """Initialize current task's A (q,v) by copying basis vectors.

        - A_q/A_v: [dim, r] basis (columns as directions). If None, no-op for that channel.
        - scale: multiply copied weights by this factor (e.g., 1/sqrt(3) as in Attention_LoRA).
        """
        q_idx, v_idx = self._qv_index(layer_idx)
        if A_q is not None:
            self.w_As[q_idx].weight.data.copy_((A_q.t() * float(scale)).to(self.w_As[q_idx].weight.device))
        if A_v is not None:
            self.w_As[v_idx].weight.data.copy_((A_v.t() * float(scale)).to(self.w_As[v_idx].weight.device))

    def freeze_current_task_A(self):
        """Freeze all A parameters for the current task (both q and v)."""
        for A in self.w_As:
            A.weight.requires_grad_(False)
