"""
Per-step feature drift utilities: compare current-task features/prototypes against
the first task and save per-step JSON artifacts.
"""
from __future__ import annotations

import os
import json
import torch
import numpy as np
import logging
from typing import Any, Sequence

from eval_flat.eval_flat_feature import extract_features_and_labels, linear_cka
from utils.data_manager import fractional_loader


def _save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _extract_xy(*args, **kwargs):
    """Compatibility wrapper for extract_features_and_labels return variants."""
    out = extract_features_and_labels(*args, **kwargs)
    if isinstance(out, (list, tuple)):
        if len(out) >= 2:
            return out[0], out[1]
    raise ValueError("extract_features_and_labels must return at least (features, labels)")


def _build_fractional_seen_loader(data_manager, start_seen: int, end_seen: int, args: dict):
    data_source = str(args.get("flat_eval_data_source", "train")).lower()
    if data_source not in {"train", "test"}:
        logging.warning(
            "[FeatureDrift] Unknown flat_eval_data_source=%s, fallback to train", data_source
        )
        data_source = "train"
    data_mode = "train"
    dataset_seen = data_manager.get_dataset(
        np.arange(start_seen, end_seen),
        source=data_source,
        mode=data_mode,
    )
    loader_seen = torch.utils.data.DataLoader(
        dataset_seen,
        batch_size=args.get("flat_eval_batch_size", 32),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    return fractional_loader(
        loader=loader_seen,
        fraction=args.get("flat_eval_dataset_fraction", 0.1),
        seed=args.get("flat_eval_dataset_fraction_seed", args.get("seed", 42)),
        balanced=True,
        batch_size=args.get("flat_eval_batch_size", 32),
    )


def compute_first_vs_current_feature_drift(
    model,
    data_manager,
    class_ranges: Sequence[Sequence[int]],
    args: dict,
    task: int,
    log_dir: str,
    logfilename: str,
):
    """
    Compare current-step features/prototypes against first-task references.
    Saves per-step JSONs under <log_dir>/feature_flatness/.
    """
    if task < 1:
        return

    feature_dir = os.path.join(log_dir, "feature_flatness")
    os.makedirs(feature_dir, exist_ok=True)
    base_stub = os.path.basename(logfilename)
    step_tag_cur = f"t{task:02d}"

    anchor_first_path = os.path.join(feature_dir, f"{base_stub}_t00_anchors_seen.pt")
    proto_first_path = os.path.join(feature_dir, f"{base_stub}_t00_prototypes.pt")
    anchor_cur_path = os.path.join(feature_dir, f"{base_stub}_{step_tag_cur}_anchors_seen.pt")
    proto_cur_path = os.path.join(feature_dir, f"{base_stub}_{step_tag_cur}_prototypes.pt")

    start0, end0 = class_ranges[0][0], class_ranges[0][1]

    device_override = getattr(model, "_device", None)
    if isinstance(device_override, str):
        device_override = torch.device(device_override)

    # ----- CKA drift (first vs current) -----
    if args.get("feature_cka_eval", False):
        X0 = None
        y0 = None
        if os.path.exists(anchor_first_path):
            first = torch.load(anchor_first_path, map_location="cpu")
            X0 = first.get("features", None)
            y0 = first.get("labels", None)
            classes0 = first.get("classes", (start0, end0))
            start0, end0 = int(classes0[0]), int(classes0[1])
        else:
            logging.info("[FIRST-vs-CUR][CKA] first-step anchors not found: %s", anchor_first_path)

        XT = None
        yT = None
        if X0 is not None and X0.numel() > 0:
            if os.path.exists(anchor_cur_path):
                cur = torch.load(anchor_cur_path, map_location="cpu")
                XT_all = cur.get("features", None)
                yT_all = cur.get("labels", None)
                if (XT_all is not None) and (yT_all is not None):
                    mask = (yT_all >= start0) & (yT_all < end0)
                    if mask.any():
                        XT = XT_all[mask]
                        yT = yT_all[mask]
            if XT is None or XT.numel() == 0:
                seen_loader_first = _build_fractional_seen_loader(
                    data_manager, start0, end0, args
                )
                anchor_max_batches = int(args.get("feature_cka_max_batches", 8))
                anchor_max_samples = int(args.get("feature_cka_max_samples", 2048))
                XT, yT = _extract_xy(
                    model._network,
                    seen_loader_first,
                    device=device_override,
                    max_batches=anchor_max_batches,
                    max_samples=anchor_max_samples,
                )
            if XT is not None and XT.numel() > 0:
                cka = linear_cka(X0, XT)
                out = {
                    "task": int(task),
                    "classes": [int(start0), int(end0)],
                    "cka_linear": float(cka),
                    "n_first": int(X0.shape[0]),
                    "n_cur": int(XT.shape[0]),
                }
                _save_json(os.path.join(feature_dir, f"{base_stub}_{step_tag_cur}_cka_first_vs_cur.json"), out)

    # ----- Prototype drift (first vs current) -----
    if args.get("feature_proto_eval", False):
        proto0 = None
        if os.path.exists(proto_first_path):
            proto0 = torch.load(proto_first_path, map_location="cpu")
        else:
            logging.info("[FIRST-vs-CUR][PROTO] first-step prototypes not found: %s", proto_first_path)

        protoT = None
        if os.path.exists(proto_cur_path):
            protoT = torch.load(proto_cur_path, map_location="cpu")

        if proto0 is None or protoT is None:
            seen_loader_first = _build_fractional_seen_loader(
                data_manager, start0, end0, args
            )
            proto_max_batches = int(args.get("feature_proto_max_batches", 8))
            proto_max_samples = int(args.get("feature_proto_max_samples", 2048))
            features, labels = _extract_xy(
                model._network,
                seen_loader_first,
                device=device_override,
                max_batches=proto_max_batches,
                max_samples=proto_max_samples,
            )
            if features is not None and labels is not None:
                proto0 = []
                for cls in range(start0, end0):
                    mask = (labels == cls)
                    if mask.any():
                        proto0.append(features[mask].mean(dim=0, keepdim=True))
                proto0 = torch.cat(proto0, dim=0) if proto0 else None
                torch.save(
                    {"prototypes": proto0, "classes": (start0, end0)},
                    proto_first_path,
                )
                protoT = proto0

        if proto0 is not None and protoT is not None:
            drift = (protoT - proto0).norm(dim=1).mean().item()
            out = {
                "task": int(task),
                "classes": [int(start0), int(end0)],
                "proto_l2_drift_mean": float(drift),
                "n_classes": int(proto0.shape[0]),
            }
            _save_json(os.path.join(feature_dir, f"{base_stub}_{step_tag_cur}_proto_first_vs_cur.json"), out)
