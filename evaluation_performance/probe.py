import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from backbone.linears import SimpleLinear
LOGGER = logging.getLogger(__name__)
from collections.abc import Mapping

def _ensure_eval_mode(module):
    was_training = module.training
    module.eval()
    return was_training


def _restore_mode(module, was_training):
    if was_training:
        module.train()


def _extract_features(network, loader, device, max_batches=None, *, store: str = "auto"):
    """
    Extract features for a dataloader.
    - store="auto": keep previous behavior (store on CUDA if device is CUDA)
    - store="cpu":  always move and keep extracted features/labels on CPU
    - store="device": always keep on the provided device
    Returned tensors are concatenated and kept on the chosen storage location.
    """
    features = []
    labels = []

    module = network.module if isinstance(network, torch.nn.DataParallel) else network
    was_training = _ensure_eval_mode(module)

    # Resolve storage policy
    if isinstance(device, str):
        device = torch.device(device)
    if store not in ("auto", "cpu", "device"):
        store = "auto"

    device_is_cuda = isinstance(device, torch.device) and device.type == 'cuda'
    if store == "cpu":
        store_on_cuda = False
    elif store == "device":
        store_on_cuda = device_is_cuda
    else:  # auto
        store_on_cuda = device_is_cuda

    with torch.no_grad():
        for batch_idx, (_, inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
            outputs = module.extract_vector(inputs)
            # Coerce to [B, D]: prefer CLS token for [B, N, D], otherwise flatten >2D
            
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            if hasattr(outputs, 'dim'):
                if outputs.dim() == 3:
                    outputs = outputs[:, 0, ...]
                elif outputs.dim() > 2:
                    outputs = outputs.view(outputs.size(0), -1)
            

            if store_on_cuda:
                features.append(outputs.detach())  # already on CUDA
                labels.append(targets.to(device, non_blocking=True).detach())
            else:
                features.append(outputs.detach().cpu())
                labels.append(targets.detach().cpu())

    _restore_mode(module, was_training)

    if not features:
        return None, None

    feats = torch.cat(features, dim=0).float()
    lbls = torch.cat(labels, dim=0).long()
    if store_on_cuda:
        feats = feats.to(device)
        lbls = lbls.to(device)
    return feats, lbls


def _fit_ridge_classifier(features, labels, num_classes, l2_reg=1e-3):
    """Closed-form ridge solve performed on the same device as features."""
    device = features.device
    # Ensure features are 2D [N, D]
    if hasattr(features, 'dim') and features.dim() > 2:
        features = features.view(features.size(0), -1)
    n_samples, feat_dim = features.shape
    eye = torch.eye(feat_dim, device=device, dtype=features.dtype)

    one_hot = torch.zeros(n_samples, num_classes, device=device, dtype=features.dtype)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    XtX = features.T @ features + l2_reg * eye
    XtY = features.T @ one_hot

    try:
        weights = torch.linalg.solve(XtX, XtY)
    except RuntimeError:
        # Fallback least-squares
        weights = torch.linalg.lstsq(XtX, XtY).solution

    return weights


def evaluate_linear_probe(
    network,
    train_loader,
    test_loader,
    class_offset,
    num_classes,
    device,
    l2_reg=1e-3,
    max_train_batches=None,
    max_test_batches=None,
):
    train_feats, train_lbls = _extract_features(
        network, train_loader, device, max_batches=max_train_batches
    )
    test_feats, test_lbls = _extract_features(
        network, test_loader, device, max_batches=max_test_batches
    )

    if train_feats is None or test_feats is None:
        return float("nan")

    train_lbls_local = (train_lbls - class_offset).to(device=device, dtype=torch.long)
    test_lbls_local = (test_lbls - class_offset).to(device=device, dtype=torch.long)

    weights = _fit_ridge_classifier(train_feats, train_lbls_local, num_classes, l2_reg=l2_reg)

    logits = test_feats @ weights
    preds = torch.argmax(logits, dim=1)
    correct = (preds == test_lbls_local).float().mean().item()

    return correct * 100.0


# Added: clearer aliases/variants for linear probes
def evaluate_linear_probe_ridge(
    network,
    train_loader,
    test_loader,
    class_offset,
    num_classes,
    device,
    l2_reg=1e-3,
    max_train_batches=None,
    max_test_batches=None,
):
    """
    Linear probe using closed-form ridge solution (alias of evaluate_linear_probe).
    Keeps semantics identical while providing a method-specific name.
    """
    return evaluate_linear_probe(
        network=network,
        train_loader=train_loader,
        test_loader=test_loader,
        class_offset=class_offset,
        num_classes=num_classes,
        device=device,
        l2_reg=l2_reg,
        max_train_batches=max_train_batches,
        max_test_batches=max_test_batches,
    )

@torch.no_grad()
def _epoch_eval_acc(head: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    head.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).long()
         
        output = head(xb)
        if isinstance(output, Mapping):
            logits = output["logits"]
        else: 
            logits = output
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    head.train()
    return correct / max(total, 1)

def _evaluate_on_features(head: torch.nn.Module, features: torch.Tensor, labels: torch.Tensor, device: torch.device):
    if features is None or labels is None:
        return None, None
    head.eval()
    with torch.no_grad():
        feats = features.to(device) if features.device != device else features
        lbls = labels.to(device) if labels.device != device else labels
        # logits = head(feats.float())

        output = head(feats.float())
        if isinstance(output, Mapping):
            logits = output["logits"]
        else: 
            logits = output

        loss = torch.nn.functional.cross_entropy(logits, lbls)
        acc = (logits.argmax(dim=1) == lbls).float().mean()
    head.train()
    return float(loss.item()), float(acc.item() * 100.0)

def _fit_softmax_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    *,
    epochs: int = 20,
    lr: float = 1e-2,
    momentum: float = 0.0,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 0,
    pin_memory: bool = False,
    log_interval: Optional[int] = None,
    log_prefix: str = "[Probe-Softmax]",
    eval_features: Optional[torch.Tensor] = None,
    eval_labels: Optional[torch.Tensor] = None,
    eval_interval: int = 10,
) -> torch.nn.Module:
    """Fit a linear softmax classifier and optionally monitor eval features."""
    assert features.ndim == 2, "features 必须是 [N, D]"
    assert labels.ndim == 1 and labels.shape[0] == features.shape[0], "labels 需为长度 N 的一维张量"

    ds = TensorDataset(features.detach().cpu(), labels.detach().cpu())
    dl = DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    _, feat_dim = features.shape
    # head = torch.nn.Linear(feat_dim, num_classes, bias=True).to(device)
    head = SimpleLinear(feat_dim, num_classes).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    eval_feats = eval_features.detach().cpu() if eval_features is not None else None
    eval_lbls = eval_labels.detach().cpu() if eval_labels is not None else None

    num_epochs = max(1, int(epochs))
    if log_interval is None or log_interval <= 0:
        log_interval = max(1, num_epochs // 5)
    if eval_interval is None or eval_interval <= 0:
        eval_interval = max(1, num_epochs // 5)

    head.train()
    for epoch_idx in range(num_epochs):
        running_loss, seen = 0.0, 0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()

            opt.zero_grad(set_to_none=True)
            # logits = head(xb)["logits"]
            output = head(xb)
            if isinstance(output, Mapping):
                logits = output["logits"]
            else: 
                logits = output
            loss = torch.nn.functional.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            batch_size_curr = yb.size(0)
            running_loss += loss.item() * batch_size_curr
            seen += batch_size_curr

        epoch_loss = running_loss / max(seen, 1)
        epoch_acc = _epoch_eval_acc(head, dl, device)
        if (epoch_idx + 1) % log_interval == 0 or epoch_idx == 0 or (epoch_idx + 1) == num_epochs:
            LOGGER.info(
                "%s epoch %d/%d loss=%.4f acc=%.2f lr=%.4g momentum=%.3f wd=%.4g",
                log_prefix,
                epoch_idx + 1,
                num_epochs,
                epoch_loss,
                epoch_acc * 100.0,
                lr,
                momentum,
                weight_decay,
            )
        if eval_feats is not None and (epoch_idx + 1) % eval_interval == 0:
            eval_loss, eval_acc = _evaluate_on_features(head, eval_feats, eval_lbls, device)
            if eval_loss is not None:
                LOGGER.info(
                    "%s epoch %d/%d [eval] loss=%.4f acc=%.2f",
                    log_prefix,
                    epoch_idx + 1,
                    num_epochs,
                    eval_loss,
                    eval_acc if eval_acc is not None else float("nan"),
                )

    head.eval()
    return head





def evaluate_linear_probe_softmax(
    network,
    train_loader,
    test_loader,
    class_offset,
    num_classes,
    device,
    *,
    epochs=20,
    lr=1e-2,
    momentum=0.0,
    weight_decay=1e-4,
    batch_size=128,
    max_train_batches=None,
    max_test_batches=None,
    log_interval: Optional[int] = None,
    eval_interval: int = 10,
    log_prefix: str = "[Probe-Softmax]",
):
    """
    Linear probe using a learned softmax linear head optimized with cross-entropy.
    """
    train_feats, train_lbls = _extract_features(
        network, train_loader, device, max_batches=max_train_batches, store="cpu"
    )
    test_feats, test_lbls = _extract_features(
        network, test_loader, device, max_batches=max_test_batches, store="cpu"
    )
    if train_feats is None or test_feats is None:
        return float("nan")

    train_local = (train_lbls - class_offset).to(device=device, dtype=torch.long)
    test_local = (test_lbls - class_offset).to(device=device, dtype=torch.long)

    head = _fit_softmax_classifier(
        features=train_feats,
        labels=train_local,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
        log_interval=log_interval,
        log_prefix=log_prefix,
        eval_features=test_feats,
        eval_labels=test_local,
        eval_interval=eval_interval,
    )

    with torch.no_grad():
        logits = head(test_feats.to(device))
        preds = torch.argmax(logits, dim=1)
        acc = (preds == test_local).float().mean().item()
    return acc * 100.0


# ===== New: expose fit/evaluate helpers for reusing a single head across many tests =====
def fit_linear_probe_softmax_head(
    network,
    train_loader,
    class_offset: int,
    num_classes: int,
    device,
    *,
    epochs: int = 20,
    lr: float = 1e-2,
    momentum: float = 0.0,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    max_train_batches=None,
    monitor_loader=None,
    monitor_max_batches=None,
    log_interval: Optional[int] = None,
    eval_interval: int = 10,
    log_prefix: str = "[Probe-Softmax]",
):
    """Train a linear softmax head and optionally monitor another loader."""
    train_feats, train_lbls = _extract_features(
        network, train_loader, device, max_batches=max_train_batches, store="cpu"
    )
    if train_feats is None:
        return None
    train_local = (train_lbls - class_offset).to(device=device, dtype=torch.long)

    monitor_feats = None
    monitor_local = None
    if monitor_loader is not None:
        monitor_feats, monitor_lbls = _extract_features(
            network, monitor_loader, device, max_batches=monitor_max_batches, store="cpu"
        )
        if monitor_feats is not None and monitor_lbls is not None:
            monitor_local = (monitor_lbls - class_offset).to(device=device, dtype=torch.long)

    head = _fit_softmax_classifier(
        features=train_feats,
        labels=train_local,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
        log_interval=log_interval,
        log_prefix=log_prefix,
        eval_features=monitor_feats,
        eval_labels=monitor_local,
        eval_interval=eval_interval,
    )
    return head





@torch.no_grad()
def evaluate_linear_probe_softmax_with_head(
    head: torch.nn.Module,
    network,
    test_loader,
    class_offset: int,
    device,
    *,
    max_test_batches=None,
):
    """Evaluate a prefit softmax head on a test loader (labels shifted by offset)."""
    test_feats, test_lbls = _extract_features(
        network, test_loader, device, max_batches=max_test_batches, store="cpu"
    )
    if test_feats is None:
        return float("nan")
    test_local = (test_lbls - class_offset).to(device=device, dtype=torch.long)
    # logits = head(test_feats.to(device))
    output = head(test_feats.to(device))
    if isinstance(output, Mapping):
        logits = output["logits"]
    else: 
        logits = output
    preds = torch.argmax(logits, dim=1)
    acc = (preds == test_local).float().mean().item()
    return acc * 100.0




@dataclass
class LinearProbeConfig:
    """
    统一的 Linear Probe 配置入口。
    - 通过 methods 控制启用哪些评估：可选
      ["ridge", "softmax", "ridge_base", "softmax_base",
       "softmax_joint_seen", "softmax_joint_future"]
    - 其余训练/数据参数集中管理，trainer 不再逐项判断。
    """
    enabled: bool = False
    methods: List[str] = field(default_factory=lambda: ["ridge", "softmax"])

    # dataloader
    eval_batch_size: int = 64
    eval_num_workers: int = 0
    shuffle_eval: bool = False   # 通常评估不打乱

    # dataset/view
    train_mode: str = "train"
    test_mode: str = "test"

    # ridge
    ridge_l2: float = 1e-3
    ridge_train_max_batches: Optional[int] = 10
    ridge_test_max_batches: Optional[int] = 5

    # softmax (per-task heads)
    softmax_epochs: int = 50
    softmax_lr: float = 5e-3
    softmax_wd: float = 1e-4
    softmax_batch_size: int = 512
    softmax_train_max_batches: Optional[int] = None
    softmax_test_max_batches: Optional[int] = None
    softmax_eval_interval: int = 10

    # softmax (joint heads)
    joint_epochs: int = 50
    joint_lr: float = 5e-3
    joint_wd: float = 1e-4
    joint_batch_size: int = 512
    joint_train_max_batches: Optional[int] = None
    joint_test_max_batches: Optional[int] = None
    joint_eval_interval: int = 10

    @staticmethod
    def from_args(args: dict) -> "LinearProbeConfig":
        """
        兼容你当前的 args：如果用户只给了 linear_probe_eval=True，
        默认 methods=["ridge","softmax"]；若给了更细开关，则按开关汇总。
        """
        cfg = LinearProbeConfig()
        cfg.enabled = bool(args.get("linear_probe_eval", False))
        cfg.eval_batch_size = int(args.get("eval_batch_size", 64))
        cfg.eval_num_workers = int(args.get("eval_num_workers", 0))
        cfg.shuffle_eval = bool(args.get("shuffle", False))
        cfg.train_mode = str(args.get("probe_train_mode", "train"))
        cfg.test_mode = str(args.get("probe_test_mode", "test"))

        cfg.ridge_l2 = float(args.get("probe_ridge_lambda", 1e-3))
        cfg.ridge_train_max_batches = args.get("probe_train_max_batches", 10)
        cfg.ridge_test_max_batches = args.get("probe_test_max_batches", 5)

        cfg.softmax_epochs = int(args.get("probe_fit_epochs", 50))
        cfg.softmax_lr = float(args.get("probe_fit_lr", 5e-3))
        cfg.softmax_wd = float(args.get("probe_fit_wd", 1e-4))
        cfg.softmax_batch_size = int(args.get("probe_fit_batch_size", 512))
        cfg.softmax_train_max_batches = args.get("probe_train_max_batches", None)
        cfg.softmax_test_max_batches = args.get("probe_test_max_batches", None)
        cfg.softmax_eval_interval = int(args.get("probe_eval_interval", cfg.softmax_eval_interval))

        cfg.joint_epochs = int(args.get("probe_fit_epochs", cfg.softmax_epochs))
        cfg.joint_lr = float(args.get("probe_fit_lr", cfg.softmax_lr))
        cfg.joint_wd = float(args.get("probe_fit_wd", cfg.softmax_wd))
        cfg.joint_batch_size = int(args.get("probe_fit_batch_size", cfg.softmax_batch_size))
        cfg.joint_train_max_batches = args.get("probe_train_max_batches", None)
        cfg.joint_test_max_batches = args.get("probe_test_max_batches", None)
        cfg.joint_eval_interval = int(args.get("probe_joint_eval_interval", cfg.joint_eval_interval))

        # 统一在此解析 methods（若用户提供更细开关）
        methods: List[str] = []
        if bool(args.get("linear_probe_ridge_eval", False)): methods.append("ridge")
        if bool(args.get("linear_probe_softmax_eval", False)): methods.append("softmax")
        if bool(args.get("linear_probe_ridge_base_eval", False)): methods.append("ridge_base")
        if bool(args.get("linear_probe_softmax_base_eval", False)): methods.append("softmax_base")
        if bool(args.get("linear_probe_softmax_joint_seen_eval", False)): methods.append("softmax_joint_seen")
        if bool(args.get("linear_probe_softmax_joint_future_eval", False)): methods.append("softmax_joint_future")

        # 若用户没给细开关但打开了 linear_probe_eval，则使用默认
        if cfg.enabled and len(methods) == 0:
            methods = ["ridge", "softmax"]

        # 也支持字符串/列表显式指定
        user_methods = args.get("linear_probe_methods", None)
        if user_methods:
            if isinstance(user_methods, str):
                methods = [m.strip() for m in user_methods.split(",") if m.strip()]
            elif isinstance(user_methods, (list, tuple)):
                methods = list(user_methods)

        # 去重并保存
        cfg.methods = sorted(set(methods))
        return cfg


class _FeatureView:
    """
    与 trainer.py 相同：包装 backbone，暴露 BASE/LORA 两种特征提取视图。
    """
    def __init__(self, backbone, which: str = "lora"):
        self._bb = backbone
        self._which = which
        self.training = False

    def _set_mode(self, mode: bool) -> None:
        self.training = bool(mode)
        try:
            self._bb.train(mode)
        except Exception:
            pass
        for name in ("lora_vit", "base_vit"):
            m = getattr(self._bb, name, None)
            if hasattr(m, "train"):
                try:
                    m.train(mode)
                except Exception:
                    pass

    def train(self, mode: bool = True):
        self._set_mode(mode)

    def eval(self):
        self._set_mode(False)


    def extract_vector(self, x):
        if self._which == "lora":
            m = getattr(self._bb, "lora_vit", self._bb)
            # 确保 m.head = Identity()（构造时已设；这里可再保险）
            if hasattr(m, "forward_features"):
                return m.forward_features(x)
            return m(x)
        else:  # "base"
            m = getattr(self._bb, "base_vit", self._bb)
            # 确保 m.head = Identity()
            if hasattr(m, "forward_features"):
                return m.forward_features(x)
            return m(x)


class LinearProbeRunner:
    """
    用一个 Runner 管理 linear probe 的所有评估路径：
    - 初始化一次，内含所有方法的 R 矩阵（nb_tasks × nb_tasks）
    - 每学完一个任务，调用 evaluate_step(t, model_net) 返回 {tag: row}
    - 训练结束，可通过 get_matrices() 取回所有方法的完整矩阵
    """
    # 与 JSON/日志 key 对齐，保持向后兼容
    TAGS = {
        "ridge": "probe_ridge",
        "softmax": "probe_softmax",
        "ridge_base": "probe_ridge_base",
        "softmax_base": "probe_softmax_base",
        "softmax_joint_seen": "probe_softmax_joint_seen",
        "softmax_joint_future": "probe_softmax_joint_future",
    }

    def __init__(
        self,
        nb_tasks: int,
        class_ranges: List[Tuple[int, int]],
        data_manager,  # utils.data_manager.DataManager
        config: LinearProbeConfig,
        device: torch.device,
    ):
        self.nb_tasks = int(nb_tasks)
        self.class_ranges = list(class_ranges)
        self.dm = data_manager
        self.cfg = config
        self.device = device

        # 每个方法一张矩阵
        self._mats: Dict[str, np.ndarray] = {
            self.TAGS[m]: np.full((self.nb_tasks, self.nb_tasks), np.nan, dtype=float)
            for m in self.cfg.methods
            if m in self.TAGS
        }

    # 统一 loader 构造：与 trainer.py 的 _build_loader 等价
    def _build_loader(self, class_range: Tuple[int,int], *, source="train", mode="test", shuffle: bool=True):
        import numpy as np

        dataset = self.dm.get_dataset(
            np.arange(class_range[0], class_range[1]),
            source=source,
            mode=mode,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.eval_batch_size,
            shuffle=bool(shuffle),
            num_workers=self.cfg.eval_num_workers,
            
        )

    def method_tags(self) -> List[str]:
        """返回内部使用的标签名（与 JSON key 一致）。"""
        return list(self._mats.keys())

    def get_matrices(self) -> Dict[str, np.ndarray]:
        """返回每种方法的完整矩阵（nb_tasks × nb_tasks）。"""
        return {k: v.copy() for k, v in self._mats.items()}

    @torch.no_grad()
    def evaluate_step(self, task_idx: int, model_net) -> Dict[str, np.ndarray]:
        """
        在第 task_idx 步（已学习至第 t 个任务后）进行所有启用方法的评估。
        返回 {tag: row}，其中 row.shape = (nb_tasks,)。
        """
        rows: Dict[str, np.ndarray] = {}
        start_seen = self.class_ranges[0][0]
        end_seen = self.class_ranges[task_idx][1]

        # 1) ridge
        if "ridge" in self.cfg.methods:
            row = np.full(self.nb_tasks, np.nan, dtype=float)
            for j, (start, end) in enumerate(self.class_ranges):
                train_loader = self._build_loader((start, end), source="train", mode=self.cfg.train_mode, shuffle=True)
                test_loader  = self._build_loader((start, end), source="test",  mode=self.cfg.test_mode,  shuffle=False)
                acc = evaluate_linear_probe_ridge(
                    model_net,
                    train_loader,
                    test_loader,
                    class_offset=start,
                    num_classes=end - start,
                    device=self.device,
                    l2_reg=self.cfg.ridge_l2,
                    max_train_batches=self.cfg.ridge_train_max_batches,
                    max_test_batches=self.cfg.ridge_test_max_batches,
                )
                row[j] = float(acc)
            tag = self.TAGS["ridge"]
            self._mats[tag][task_idx, :] = row
            rows[tag] = row

        # 2) softmax (per-task)
        if "softmax" in self.cfg.methods:
            row = np.full(self.nb_tasks, np.nan, dtype=float)
            for j, (start, end) in enumerate(self.class_ranges):
                train_loader = self._build_loader((start, end), source="train", mode=self.cfg.train_mode, shuffle=True)
                test_loader  = self._build_loader((start, end), source="test",  mode=self.cfg.test_mode,  shuffle=False)
                acc = evaluate_linear_probe_softmax(
                    model_net,
                    train_loader,
                    test_loader,
                    class_offset=start,
                    num_classes=end - start,
                    device=self.device,
                    epochs=self.cfg.softmax_epochs,
                    lr=self.cfg.softmax_lr,
                    weight_decay=self.cfg.softmax_wd,
                    batch_size=self.cfg.softmax_batch_size,
                    max_train_batches=self.cfg.softmax_train_max_batches,
                    max_test_batches=self.cfg.softmax_test_max_batches,
                )
                row[j] = float(acc)
            tag = self.TAGS["softmax"]
            self._mats[tag][task_idx, :] = row
            rows[tag] = row

        # 3) BASE 视图（ridge / softmax）
        bb = getattr(model_net, "backbone", None)

        if bb is not None and ("ridge_base" in self.cfg.methods or "softmax_base" in self.cfg.methods):
            base_view = _FeatureView(bb, which="base")

            if "ridge_base" in self.cfg.methods:
                row = np.full(self.nb_tasks, np.nan, dtype=float)
                for j, (start, end) in enumerate(self.class_ranges):
                    train_loader = self._build_loader((start, end), source="train", mode=self.cfg.train_mode, shuffle=True)
                    test_loader  = self._build_loader((start, end), source="test",  mode=self.cfg.test_mode,  shuffle=False)
                    acc = evaluate_linear_probe_ridge(
                        base_view,
                        train_loader,
                        test_loader,
                        class_offset=start,
                        num_classes=end - start,
                        device=self.device,
                        l2_reg=self.cfg.ridge_l2,
                        max_train_batches=self.cfg.ridge_train_max_batches,
                        max_test_batches=self.cfg.ridge_test_max_batches,
                    )
                    row[j] = float(acc)
                tag = self.TAGS["ridge_base"]
                self._mats[tag][task_idx, :] = row
                rows[tag] = row

            if "softmax_base" in self.cfg.methods:
                row = np.full(self.nb_tasks, np.nan, dtype=float)
                for j, (start, end) in enumerate(self.class_ranges):
                    train_loader = self._build_loader((start, end), source="train", mode=self.cfg.train_mode, shuffle=True)
                    test_loader  = self._build_loader((start, end), source="test",  mode=self.cfg.test_mode,  shuffle=False)
                    acc = evaluate_linear_probe_softmax(
                        base_view,
                        train_loader,
                        test_loader,
                        class_offset=start,
                        num_classes=end - start,
                        device=self.device,
                        epochs=self.cfg.softmax_epochs,
                        lr=self.cfg.softmax_lr,
                        weight_decay=self.cfg.softmax_wd,
                        batch_size=self.cfg.softmax_batch_size,
                        max_train_batches=self.cfg.softmax_train_max_batches,
                        max_test_batches=self.cfg.softmax_test_max_batches,
                        log_interval=self.cfg.softmax_eval_interval,
                        eval_interval=self.cfg.softmax_eval_interval,
                        log_prefix=(f"[Probe-Softmax][BASE][task={j}]" if self.nb_tasks > 1 else "[Probe-Softmax][BASE]"),
                    )
                    row[j] = float(acc)
                tag = self.TAGS["softmax_base"]
                self._mats[tag][task_idx, :] = row
                rows[tag] = row

        # 4) joint seen (softmax)：1..t 的并集训练一个头，再按已见任务评估
        if "softmax_joint_seen" in self.cfg.methods:
            row = np.full(self.nb_tasks, np.nan, dtype=float)
            union_train = self._build_loader((start_seen, end_seen), source="train", mode=self.cfg.train_mode, shuffle=True)
            union_test = self._build_loader((start_seen, end_seen), source="test", mode=self.cfg.test_mode, shuffle=False)
            head = fit_linear_probe_softmax_head(
                model_net,
                union_train,
                class_offset=start_seen,
                num_classes=end_seen - start_seen,
                device=self.device,
                epochs=self.cfg.joint_epochs,
                lr=self.cfg.joint_lr,
                weight_decay=self.cfg.joint_wd,
                batch_size=self.cfg.joint_batch_size,
                max_train_batches=self.cfg.joint_train_max_batches,
                monitor_loader=union_test,
                monitor_max_batches=self.cfg.joint_test_max_batches,
                log_interval=self.cfg.joint_eval_interval,
                eval_interval=self.cfg.joint_eval_interval,
                log_prefix="[Probe-Softmax][JOINT-SEEN]",
            )
            if head is not None:
                for j in range(task_idx + 1):
                    start, end = self.class_ranges[j]
                    test_loader = self._build_loader((start, end), source="test", mode=self.cfg.test_mode, shuffle=False)
                    acc = evaluate_linear_probe_softmax_with_head(
                        head,
                        model_net,
                        test_loader,
                        class_offset=start_seen,
                        device=self.device,
                        max_test_batches=self.cfg.joint_test_max_batches,
                    )
                    row[j] = float(acc)
            tag = self.TAGS["softmax_joint_seen"]
            self._mats[tag][task_idx, :] = row
            rows[tag] = row

        # 5) joint future (softmax)：t+1..T 的并集
        if "softmax_joint_future" in self.cfg.methods and (task_idx < self.nb_tasks - 1):
            row = np.full(self.nb_tasks, np.nan, dtype=float)
            start_future = self.class_ranges[task_idx + 1][0]
            end_future = self.class_ranges[-1][1]
            union_train = self._build_loader((start_future, end_future), source="train", mode=self.cfg.train_mode, shuffle=True)
            union_test = self._build_loader((start_future, end_future), source="test", mode=self.cfg.test_mode, shuffle=False)
            head = fit_linear_probe_softmax_head(
                model_net,
                union_train,
                class_offset=start_future,
                num_classes=end_future - start_future,
                device=self.device,
                epochs=self.cfg.joint_epochs,
                lr=self.cfg.joint_lr,
                weight_decay=self.cfg.joint_wd,
                batch_size=self.cfg.joint_batch_size,
                max_train_batches=self.cfg.joint_train_max_batches,
                monitor_loader=union_test,
                monitor_max_batches=self.cfg.joint_test_max_batches,
                log_interval=self.cfg.joint_eval_interval,
                eval_interval=self.cfg.joint_eval_interval,
                log_prefix="[Probe-Softmax][JOINT-FUTURE]",
            )
            if head is not None:
                for j in range(task_idx + 1, self.nb_tasks):
                    start, end = self.class_ranges[j]
                    test_loader = self._build_loader((start, end), source="test", mode=self.cfg.test_mode, shuffle=False)
                    acc = evaluate_linear_probe_softmax_with_head(
                        head,
                        model_net,
                        test_loader,
                        class_offset=start_future,
                        device=self.device,
                        max_test_batches=self.cfg.joint_test_max_batches,
                    )
                    row[j] = float(acc)
            tag = self.TAGS["softmax_joint_future"]
            self._mats[tag][task_idx, :] = row
            rows[tag] = row

        return rows


import torch
import torch.nn as nn
import torch.nn.functional as F

class _TunaFcView(nn.Module):
    """只读权重视图：将 TunaLinear 的所有 head 的 Linear.weight 按行拼接并做 L2 归一化。
    仅提供 .weight 属性，供 eval_flat_feature._classifier_weights() 读取。
    """
    def __init__(self, tuna_net: nn.Module):
        super().__init__()
        self._tuna = tuna_net
        # 不注册为 Parameter，避免被优化器/选择器当作真实可训练权重
        self.register_buffer("_weight_cache", torch.empty(0), persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        heads = getattr(self._tuna, "fc", None)
        if heads is None or not hasattr(heads, "heads"):
            raise AttributeError("TUNA fc.heads not found")
        rows = []
        for seq in heads.heads:
            # seq 形如 [LayerNorm? , Linear]
            lin = None
            if isinstance(seq, nn.Sequential) and len(seq) > 0:
                # 取最后一个 Linear
                for m in reversed(seq):
                    if isinstance(m, nn.Linear):
                        lin = m
                        break
            elif isinstance(seq, nn.Linear):
                lin = seq
            if lin is None or not hasattr(lin, "weight"):
                continue
            w = lin.weight  # [C_i, D]
            # TunaLinear.forward 用 F.normalize(feature, dim=1) 和 F.normalize(weight, dim=1)
            w = F.normalize(w, p=2, dim=1)
            rows.append(w)
        if not rows:
            raise RuntimeError("No linear weights collected from TunaLinear.heads")
        W = torch.cat(rows, dim=0).detach()
        # 缓存到和上一版相同的 device/dtype
        if self._weight_cache.numel() != W.numel() or self._weight_cache.shape != W.shape or self._weight_cache.dtype != W.dtype or self._weight_cache.device != W.device:
            self._weight_cache = W
        else:
            self._weight_cache.copy_(W)
        return self._weight_cache

class TunaEvalWrapper(nn.Module):
    """用于 EFM/Hessian 的轻量包装：
    - 前向显式指定 adapter_id=fused_id（‘融合 adapter’）
    - logits 用 TUNA 自己的 TunaLinear 计算（和 EFM 的 fc.weight 一致）
    - 暴露 .fc = _TunaFcView(tuna_net)，供 EFM 工具读取线性权重
    - 保留对底层 TUNANet 的参数访问（Hessian 可覆盖全模型）
    """
    def __init__(self, tuna_net: nn.Module, fused_id: int):
        super().__init__()
        self.net = tuna_net  # 保留为子模块，named_parameters() 能遍历到全模型参数
        self.fused_id = int(fused_id)
        self.fc = _TunaFcView(tuna_net)

    def train(self, mode: bool = True):
        self.net.train(mode)
        super().train(mode)
        return self

    def eval(self):
        self.net.eval()
        super().eval()
        return self

    @torch.no_grad()
    def extract_vector(self, x: torch.Tensor) -> torch.Tensor:
        # 若需要给 probe 统一接口（可选）
        # 使用融合 adapter 抽特征
        out = self.net.backbone(x, adapter_id=self.fused_id, train=False)
        feat = out["features"]
        if feat.dim() == 3:  # 取 CLS
            feat = feat[:, 0, ...]
        elif feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        return feat

    def forward(self, x: torch.Tensor):
        # 1) 抽融合 adapter 特征
        out = self.net.backbone(x, adapter_id=self.fused_id, train=False)
        feat = out["features"]
        # 2) 用 TUNA 的 TunaLinear 头计算 logits（保持与训练一致）
        #    注意 TunaLinear.forward 内部做了归一化；我们可以直接复用：
        logits = self.net.fc(feat)["logits"]
        # 3) 返回 dict（EFM/Hessian 都要求有 logits/features）
        return {"features": feat, "logits": logits}
