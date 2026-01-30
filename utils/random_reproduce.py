import torch
import numpy as np
import random
from typing import Any
from pathlib import Path
import enum
import os

def set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def set_random(seed=1):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False







def json_safe(obj: Any) -> Any:
    """
    将常见科研/训练对象递归地转换为 JSON 可序列化的 Python 基本类型。
    - dict/list/tuple/set：递归处理；set 转为排序后的 list（避免 JSON 不支持 set）
    - numpy 标量/数组：转为 .item() / .tolist()
    - torch.Tensor：转为 .detach().cpu().tolist()
    - torch.device / torch.dtype：转为 str
    - Path：转为 str
    - Enum：转为其值（.value）
    - 其他不可序列化对象：回退为 str(obj)，保证不会抛出 TypeError
    """
    # 基本标量与 None：原样返回
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # 路径与枚举
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return json_safe(obj.value)

    # 容器类型
    if isinstance(obj, dict):
        return {json_safe(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, set):
        # set 无序：为稳定性，转为排序后的字符串表示
        return sorted(json_safe(v) for v in obj)

    # numpy 支持
    if np is not None:
        if isinstance(obj, (np.generic,)):  # 标量，如 np.float32(1.0)
            try:
                return obj.item()
            except Exception:
                return str(obj)
        if isinstance(obj, (np.ndarray,)):  # 数组
            try:
                return obj.tolist()
            except Exception:
                return str(obj)

    # torch 支持
    if torch is not None:
        # Tensor
        if isinstance(obj, getattr(torch, "Tensor", ())):
            try:
                return obj.detach().cpu().tolist()
            except Exception:
                return str(obj)
        # 设备 / dtype
        if isinstance(obj, getattr(torch, "device", ())):
            return str(obj)
        if isinstance(obj, getattr(torch, "dtype", ())):
            return str(obj)

    # 其他不可 JSON 序列化对象：回退为字符串（保证不抛错）
    return str(obj)