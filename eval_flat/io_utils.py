import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from eval_flat.param_utils import _param_names_and_shapes, _unflatten_to_param_like


def _save_eigvecs(save_path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_path = f"{save_path}.tmp"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, save_path)
    except Exception:
        logging.exception("[FlatEval] torch.save failed, retry with legacy serialization")
        try:
            torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
            os.replace(tmp_path, save_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    logging.exception("[FlatEval] Failed to remove temp eigvec file: %s", tmp_path)


def _load_eigvecs_for_params(path: str, module: nn.Module, params: List[torch.nn.Parameter]):
    """Load flat eigenvectors and align to current params by names/shapes.

    Returns (vals, vecs_flat, dirs_list) on success, else None.
    dirs_list is a list of list[Tensor] shaped like params: [v1_list, v2_list, ...].
    """
    if not os.path.isfile(path):
        return None
    payload = torch.load(path, map_location="cpu")

    vals = payload.get("vals", None)
    vlist = []
    for key in ("v1", "v2", "v3"):
        if key in payload and isinstance(payload[key], torch.Tensor):
            vlist.append(payload[key].float().view(-1))
    names_saved = payload.get("names", None)
    shapes_saved = payload.get("shapes", None)
    splits_saved = payload.get("splits", None)
    if vals is None or not vlist or names_saved is None or shapes_saved is None or splits_saved is None:
        return None

    names_cur, shapes_cur, splits_cur = _param_names_and_shapes(module, params)
    if names_saved != names_cur or splits_saved != splits_cur:
        logging.info("[FlatEval] Saved eigvecs do not match current param ordering; fallback.")
        return None

    dirs = []
    for v in vlist:
        dirs.append(_unflatten_to_param_like(v, params))
    return vals, vlist, dirs


def _load_args(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
