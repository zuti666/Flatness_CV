from pathlib import Path
from huggingface_hub import snapshot_download
import timm, os

# 1) 下载到你指定目录（不是默认cache）
target_dir = Path("/disk0/users/liying/Flatness_CV/cache/hf_models/timm_vit_b16_augreg2")
target_dir.mkdir(parents=True, exist_ok=True)

local_dir = snapshot_download(
    repo_id="timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
    local_dir=str(target_dir),
    local_dir_use_symlinks=False,   # True=指向cache的软链接；False=实际拷贝文件（NFS上更稳）
    allow_patterns=["*.safetensors", "*.json"]
)
print("saved to:", local_dir)

# 2) 从本地目录加载（完全离线）
os.environ["HF_HUB_OFFLINE"] = "1"   # 可选：强制只走本地
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    pretrained_cfg="augreg2_in21k_ft_in1k",
    pretrained_cfg_overlay={"hf_hub_id": str(local_dir)},
    num_classes=0
)
