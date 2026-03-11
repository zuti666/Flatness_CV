"""OOD evaluation helpers (Tiny-ImageNet-* variants)."""
from __future__ import annotations

import copy
import gc
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_manager import DataManager
from utils.metrics_book import write_final_metrics
from evaluation_performance.probe import (
    fit_linear_probe_softmax_head,
    evaluate_linear_probe_softmax_with_head,
    _FeatureView,
)


# write_final_metrics is imported from utils.metrics_book


def _build_ood_loader(dm: DataManager, batch_size: int, num_workers: int):
    ds = dm.get_dataset(np.arange(0, dm.nb_classes), source="test", mode="test")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def _ood_linear_probe(model, dm: DataManager, args: dict, *, tag: str) -> Tuple[float, float]:
    try:
        net = getattr(model, "_network", model)
        if hasattr(net, "module"):
            net = net.module
        net.eval()

        # Adapter handling for special models (e.g., TUNA)
        if str(args.get("model_name", "")).lower() == "tuna":
            module = net
            bb = getattr(module, "backbone", None)
            if bb is not None and hasattr(bb, "adapter_list"):
                fused_id = (len(bb.adapter_list) + 1) if (len(bb.adapter_list) > 0 and getattr(bb, "merged_adapter", None) is not None) else len(bb.adapter_list)
                if hasattr(module, "set_active_adapter"):
                    module.set_active_adapter(fused_id)
                else:
                    try:
                        setattr(bb, "active_adapter", fused_id)
                    except Exception:
                        pass

        start_seen = 0
        end_seen = int(dm.nb_classes)
        num_classes = end_seen - start_seen

        train_ds = dm.get_dataset(np.arange(start_seen, end_seen), source="train", mode=args.get("probe_train_mode", "train"))
        test_ds = dm.get_dataset(np.arange(start_seen, end_seen), source="test", mode=args.get("probe_test_mode", "test"))

        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.get("probe_fit_train_batch_size", 128)),
            shuffle=True,
            num_workers=int(args.get("linear_probe_eval_num_workers", 0)),
            persistent_workers=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=int(args.get("probe_fit_test_batch_size", 128)),
            shuffle=False,
            num_workers=int(args.get("linear_probe_eval_num_workers", 0)),
            persistent_workers=False,
        )

        probe_log_interval = args.get("probe_log_interval", None)
        probe_eval_interval = max(1, int(args.get("probe_fit_epochs", 50)) // 5)

        head_final = fit_linear_probe_softmax_head(
            net, train_loader,
            class_offset=start_seen, num_classes=num_classes, device=model._device,
            epochs=int(args.get("probe_fit_epochs", 50)),
            lr=float(args.get("probe_fit_lr", 5e-3)),
            weight_decay=float(args.get("probe_fit_wd", 0.0)),
            batch_size=int(args.get("probe_fit_train_batch_size", 128)),
            max_train_batches=args.get("probe_train_max_batches", None),
            monitor_loader=test_loader,
            monitor_max_batches=args.get("probe_test_max_batches", None),
            log_interval=probe_log_interval,
            eval_interval=probe_eval_interval,
            log_prefix=f"[LP-Softmax][OOD:{tag}][Final]",
        )
        acc_final = evaluate_linear_probe_softmax_with_head(
            head_final, net, test_loader,
            class_offset=start_seen, device=model._device,
            max_test_batches=args.get("probe_test_max_batches", None)
        )

        acc_base = float("nan")
        if str(args.get("model_name", "")).lower() != "tuna":
            base_view = _FeatureView(getattr(net, "backbone", net), which="base")
            base_view.eval()
            head_base = fit_linear_probe_softmax_head(
                base_view, train_loader,
                class_offset=start_seen, num_classes=num_classes, device=model._device,
                epochs=int(args.get("probe_fit_epochs", 50)),
                lr=float(args.get("probe_fit_lr", 5e-3)),
                weight_decay=float(args.get("probe_fit_wd", 0.0)),
                batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                max_train_batches=args.get("probe_train_max_batches", None),
                monitor_loader=test_loader,
                monitor_max_batches=args.get("probe_test_max_batches", None),
                log_interval=probe_log_interval,
                eval_interval=probe_eval_interval,
                log_prefix=f"[LP-Softmax][OOD:{tag}][Base]",
            )
            acc_base = evaluate_linear_probe_softmax_with_head(
                head_base, base_view, test_loader,
                class_offset=start_seen, device=model._device,
                max_test_batches=args.get("probe_test_max_batches", None)
            )
        else:
            acc_base = 0.0

        logging.info("[OOD-LP][%s] final-model=%.2f | base-model=%.2f", tag, acc_final, acc_base)
        return float(acc_final), float(acc_base)
    except Exception as _lp_exc:
        logging.exception("[OOD-LP][%s] Linear probe failed: %s", tag, _lp_exc)
        return float("nan"), float("nan")
    finally:
        try:
            for _m in ["head_final", "head_base", "base_view"]:
                if _m in locals() and locals()[_m] is not None:
                    mod = locals()[_m]
                    if isinstance(mod, torch.nn.Module):
                        with torch.inference_mode():
                            mod.to("cpu")
                        for p in mod.parameters():
                            p.grad = None
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def run_ood_evaluation(model, args: dict, *, metrics_path: Optional[str] = None) -> Dict[str, Any]:
    """Run OOD evaluation and optionally persist results to the metrics JSON."""
    ood_results: Dict[str, Any] = {}
    try:
        ood_bs = int(args.get("ood_eval_batch_size", 128))
        ood_workers = int(args.get("ood_eval_num_workers", 0))
        class_shuffle_ood = bool(args.get("class_shuffle", False))

        # Tiny-ImageNet-R
        if bool(args.get("ood_imagener_r", False)):
            ood_args_r = copy.deepcopy(args)
            ood_args_r["dataset"] = "imagenetr"
            dm_r = DataManager("imagenetr", class_shuffle_ood, args["seed"], 200, 0, ood_args_r)
            loader_r = _build_ood_loader(dm_r, ood_bs, ood_workers)
            acc_r = model.evaluate_full_dataset(loader_r)
            lp_final, lp_base = _ood_linear_probe(model, dm_r, args, tag="Tiny-ImageNet-R")
            ood_results["imagenetr"] = {
                "top1": float(acc_r.get("top1", 0.0)),
                "top5": float(acc_r.get("top5", 0.0)),
                "lp_softmax_final": float(lp_final) if lp_final == lp_final else 0.0,
                "lp_softmax_base": float(lp_base) if lp_base == lp_base else 0.0,
            }
            logging.info(
                "[OOD][Tiny-ImageNet-R] top1=%.2f | top5=%.2f | LP(final)=%.2f | LP(base)=%.2f",
                ood_results["imagenetr"]["top1"], ood_results["imagenetr"]["top5"],
                ood_results["imagenetr"].get("lp_softmax_final", float("nan")),
                ood_results["imagenetr"].get("lp_softmax_base", float("nan"))
            )
            if metrics_path:
                write_final_metrics(metrics_path, "imagenetr", final_metrics=ood_results["imagenetr"], final_matrix=None)

        # Tiny-ImageNet-C
        if bool(args.get("ood_imagener_c", False)):
            ood_args_c = copy.deepcopy(args)
            ood_args_c["dataset"] = "tiny_imagenetc"
            dm_c = DataManager("tiny_imagenetc", class_shuffle_ood, args["seed"], 200, 0, ood_args_c)
            loader_c = _build_ood_loader(dm_c, ood_bs, ood_workers)
            acc_c = model.evaluate_full_dataset(loader_c)
            lp_final, lp_base = _ood_linear_probe(model, dm_c, args, tag="Tiny-ImageNet-C")
            ood_results["tiny_imagenetc"] = {
                "top1": float(acc_c.get("top1", 0.0)),
                "top5": float(acc_c.get("top5", 0.0)),
                "lp_softmax_final": float(lp_final) if lp_final == lp_final else 0.0,
                "lp_softmax_base": float(lp_base) if lp_base == lp_base else 0.0,
            }
            logging.info(
                "[OOD][Tiny-ImageNet-C] top1=%.2f | top5=%.2f | LP(final)=%.2f | LP(base)=%.2f",
                ood_results["tiny_imagenetc"]["top1"], ood_results["tiny_imagenetc"]["top5"],
                ood_results["tiny_imagenetc"].get("lp_softmax_final", float("nan")),
                ood_results["tiny_imagenetc"].get("lp_softmax_base", float("nan"))
            )
            if metrics_path:
                write_final_metrics(metrics_path, "tiny_imagenetc", final_metrics=ood_results["tiny_imagenetc"], final_matrix=None)

        # Tiny-ImageNet-A
        if bool(args.get("ood_imagener_a", False)):
            ood_args_a = copy.deepcopy(args)
            ood_args_a["dataset"] = "imageneta"
            dm_a = DataManager("imageneta", class_shuffle_ood, args["seed"], 200, 0, ood_args_a)
            loader_a = _build_ood_loader(dm_a, ood_bs, ood_workers)
            acc_a = model.evaluate_full_dataset(loader_a)
            lp_final, lp_base = _ood_linear_probe(model, dm_a, args, tag="Tiny-ImageNet-A")
            ood_results["imageneta"] = {
                "top1": float(acc_a.get("top1", 0.0)),
                "lp_softmax_final": float(lp_final) if lp_final == lp_final else 0.0,
                "lp_softmax_base": float(lp_base) if lp_base == lp_base else 0.0,
            }
            logging.info(
                "[OOD][Tiny-ImageNet-A] top1=%.2f | LP(final)=%.2f | LP(base)=%.2f",
                ood_results["imageneta"]["top1"],
                ood_results["imageneta"].get("lp_softmax_final", float("nan")),
                ood_results["imageneta"].get("lp_softmax_base", float("nan"))
            )
            if metrics_path:
                write_final_metrics(metrics_path, "imageneta", final_metrics=ood_results["imageneta"], final_matrix=None)

        # Tiny-ImageNet-P
        if bool(args.get("ood_imagenet_p", False)):
            ood_args_p = copy.deepcopy(args)
            ood_args_p["dataset"] = "tiny_imagenetp"
            dm_p = DataManager("tiny_imagenetp", class_shuffle_ood, args["seed"], 200, 0, ood_args_p)
            loader_p = _build_ood_loader(dm_p, ood_bs, ood_workers)
            acc_p = model.evaluate_full_dataset(loader_p)
            lp_final, lp_base = _ood_linear_probe(model, dm_p, args, tag="Tiny-ImageNet-P")
            ood_results["tiny_imagenetp"] = {
                "top1": float(acc_p.get("top1", 0.0)),
                "top5": float(acc_p.get("top5", 0.0)),
                "lp_softmax_final": float(lp_final) if lp_final == lp_final else 0.0,
                "lp_softmax_base": float(lp_base) if lp_base == lp_base else 0.0,
            }
            logging.info(
                "[OOD][Tiny-ImageNet-P] top1=%.2f | top5=%.2f | LP(final)=%.2f | LP(base)=%.2f",
                ood_results["tiny_imagenetp"]["top1"], ood_results["tiny_imagenetp"]["top5"],
                ood_results["tiny_imagenetp"].get("lp_softmax_final", float("nan")),
                ood_results["tiny_imagenetp"].get("lp_softmax_base", float("nan"))
            )
            if metrics_path:
                write_final_metrics(metrics_path, "ood_tiny_tiny_imagenetp", final_metrics=ood_results["tiny_imagenetp"], final_matrix=None)

    except Exception as _ood_exc:
        logging.exception("[OOD] Evaluation failed: %s", _ood_exc)
    return ood_results
