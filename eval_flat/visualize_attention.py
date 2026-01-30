# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import colorsys
import os
import random
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import timm
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from torchvision import transforms as pth_transforms

from eval_flat.attn_utils import enable_last_attn


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """Generate random colors."""
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    colors = random_colors(N)

    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis("off")
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect="auto")
    fig.savefig(fname)
    print(f"{fname} saved.")
    plt.close(fig)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _extract_state_dict(blob, preferred_key: Optional[str] = None):
    if not isinstance(blob, dict):
        return blob
    if preferred_key and preferred_key in blob and isinstance(blob[preferred_key], dict):
        return blob[preferred_key]
    for key in ("model_state_dict", "state_dict", "model", "net", "backbone"):
        if key in blob and isinstance(blob[key], dict):
            return blob[key]
    return blob


def _build_transform(image_size: Optional[Tuple[int, int]]):
    transforms = []
    if image_size is not None:
        transforms.append(pth_transforms.Resize(image_size))
    transforms.extend(
        [
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return pth_transforms.Compose(transforms)


def _load_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image path {path} is not valid.")
    img = Image.open(path)
    return img.convert("RGB")


def _resolve_patch_size(model, fallback: Optional[int]) -> int:
    patch_size = fallback
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None:
        size = getattr(patch_embed, "patch_size", None)
        if isinstance(size, tuple):
            patch_size = size[0]
        elif isinstance(size, int):
            patch_size = size
    if patch_size is None:
        raise ValueError("Unable to infer patch size; please provide --patch_size explicitly.")
    return int(patch_size)


def _normalize_map(t: torch.Tensor) -> torch.Tensor:
    t_min = torch.min(t)
    t_max = torch.max(t)
    denom = t_max - t_min
    if torch.isfinite(denom) and denom > 1e-8:
        return (t - t_min) / denom
    return torch.zeros_like(t)


def main():
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument("--arch", default="vit_base_patch16_224", type=str, help="timm model name")
    parser.add_argument(
        "--timm_pretrained",
        action="store_true",
        help="Load timm's default pretrained weights before applying checkpoints.",
    )
    parser.add_argument("--pretrained_weights", default=None, type=str, help="Path to checkpoint to load.")
    parser.add_argument(
        "--checkpoint_key",
        default=None,
        type=str,
        help="Optional nested key inside the checkpoint that contains the state dict.",
    )
    parser.add_argument(
        "--strict_load",
        action="store_true",
        help="Enable strict state_dict loading (raises on missing/unexpected keys).",
    )
    parser.add_argument("--image_path", required=True, type=str, help="Path of the image to load.")
    parser.add_argument(
        "--image_size",
        default=None,
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="Optional resize (height width) before feeding to the model.",
    )
    parser.add_argument("--output_dir", default=".", help="Path where to save visualizations.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Visualize masks obtained by thresholding the attention maps to keep xx%% of the mass.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        type=str,
        help="Device to run on (e.g. 'cuda', 'cpu', or 'auto' for best effort).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="Override patch size if it cannot be inferred from the model.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)

    model = timm.create_model(args.arch, pretrained=args.timm_pretrained, num_classes=0)
    model.to(device)
    model.eval()

    if args.pretrained_weights:
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint, args.checkpoint_key)
        if not isinstance(state_dict, dict):
            raise TypeError("The loaded checkpoint does not contain a state_dict dictionary.")
        cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        load_result = model.load_state_dict(cleaned_state, strict=args.strict_load)
        if args.strict_load:
            print("Checkpoint loaded strictly.")
        else:
            print("Checkpoint loaded (strict=False)."); print("Missing keys:", load_result.missing_keys); print("Unexpected keys:", load_result.unexpected_keys)

    patch_size = _resolve_patch_size(model, args.patch_size)

    transform = _build_transform(tuple(args.image_size) if args.image_size is not None else None)
    img = _load_image(args.image_path)
    img_tensor = transform(img)

    _, H, W = img_tensor.shape
    H_aligned = H - (H % patch_size)
    W_aligned = W - (W % patch_size)
    img_tensor = img_tensor[:, :H_aligned, :W_aligned].unsqueeze(0)

    feat_h = H_aligned // patch_size
    feat_w = W_aligned // patch_size

    cache, restore = enable_last_attn(model)
    with torch.no_grad():
        _ = model(img_tensor.to(device))
    attentions = cache.get("last_attn", None)
    restore()

    if attentions is None:
        raise RuntimeError("No attention maps were captured. Make sure the model is a ViT from timm.")

    attentions = attentions[:, :, 0, 1:]  # CLS -> patch attention
    B, num_heads, tokens = attentions.shape
    if B != 1:
        print("Warning: multiple images detected, using only the first instance.")
    attentions = attentions[0].reshape(num_heads, feat_h, feat_w)

    attn_heads_up = (
        F.interpolate(attentions.unsqueeze(0), size=(H_aligned, W_aligned), mode="bilinear", align_corners=False)[0]
        .cpu()
        .contiguous()
    )

    avg_map = attentions.mean(dim=0, keepdim=True)
    avg_map_up = (
        F.interpolate(avg_map.unsqueeze(0), size=(H_aligned, W_aligned), mode="bilinear", align_corners=False)[0, 0]
        .cpu()
        .contiguous()
    )
    avg_map_up = _normalize_map(avg_map_up)

    os.makedirs(args.output_dir, exist_ok=True)
    img_path = os.path.join(args.output_dir, "img.png")
    torchvision.utils.save_image(
        torchvision.utils.make_grid(img_tensor, normalize=True, scale_each=True), img_path
    )

    avg_path = os.path.join(args.output_dir, "attn-avg.png")
    plt.imsave(fname=avg_path, arr=avg_map_up.numpy(), format="png")
    print(f"{avg_path} saved.")

    for j in range(num_heads):
        head_map = _normalize_map(attn_heads_up[j])
        fname = os.path.join(args.output_dir, f"attn-head{j}.png")
        plt.imsave(fname=fname, arr=head_map.numpy(), format="png")
        print(f"{fname} saved.")

    if args.threshold is not None:
        flat_attn = attentions.reshape(num_heads, -1)
        val, idx = torch.sort(flat_attn, dim=1)
        val = val / torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx, dim=1)
        th_attn = torch.gather(th_attn, dim=1, index=idx2)
        th_attn = th_attn.reshape(num_heads, feat_h, feat_w).float()
        th_attn = (
            F.interpolate(th_attn.unsqueeze(0), size=(H_aligned, W_aligned), mode="nearest")[0]
            .cpu()
            .numpy()
        )

        image = skimage.io.imread(img_path)
        for j in range(num_heads):
            display_instances(
                image,
                th_attn[j],
                fname=os.path.join(
                    args.output_dir, f"mask_th{args.threshold}_head{j}.png"
                ),
                blur=False,
            )


if __name__ == "__main__":
    main()
