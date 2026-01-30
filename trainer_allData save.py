# === Added: finetune-all train entrypoints to align with trainer.py ===
import os
import sys
import logging
import copy
from typing import Any, Dict
import torch
import json
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from evaluation.probe import fit_linear_probe_softmax_head, evaluate_linear_probe_softmax_with_head
from torch.utils.data import DataLoader
import gc
from utils.data_manager import DataManager, fractional_loader  # ← fractional_loader
from eval_flat.eval_flatness_weight_Loss import FlatnessConfig, evaluate_flatness_metrics
from eval_flat.eval_flat_feature import FeatureFlatnessConfig, evaluate_feature_metrics
from backbone.lora import LoRA_ViT_timm
from utils.random_reproduce import json_safe

def _to_float_metric(x):
    import numpy as np
    import torch
    # tuple: 取第 1 个值作为 accuracy
    if isinstance(x, tuple) and len(x) > 0:
        x = x[0]
    # dict: 优先取常见命名
    if isinstance(x, dict):
        for k in ("acc", "accuracy", "top1", "final_acc"):
            if k in x:
                x = x[k]
                break
    # torch / numpy 标量
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.numel() == 1:
            x = x.item()
    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.item()
    return float(x)

def train_all(args: Dict[str, Any]):
    """
    Mirror trainer.train(): iterate over seeds and dispatch to _train.
    'args' is already populated by main.py (config + CLI overrides).
    """
    # Normalize seeds/devices
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args: Dict[str, Any]):
    """
    Finetune on the full training set and evaluate once.
    Keep the logging layout close to trainer._train but simplified for all-data FT.
    """
    # ---------------------------
    # 0) I/O & logging
    # ---------------------------
    # ——— root: logs/<model>/<dataset> ———
    logs_root = os.path.join("logs_all", str(args["model_name"]),str(args["optimizer_type"]), str(args["dataset"]),str(args["seed"]),str(args["prefix"]))
    os.makedirs(logs_root, exist_ok=True)
    # Store checkpoints and auxiliary artifacts alongside logs for easier tracking
    # ——— checkpoints 放在: logs/<model>/<dataset>/<optimizer_type>/checkpoints ———
    checkpoint_dir = os.path.join(logs_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Downstream components expect a trailing separator when concatenating filenames
    args["filepath"] = checkpoint_dir if checkpoint_dir.endswith(os.sep) else (checkpoint_dir + os.sep)
    args.setdefault("feature_flat_save_path", checkpoint_dir)

    # ——— 日志放在: logs/<model>/<dataset>/<optimizer_type>/<increment>/ ———
    log_dir = os.path.join(logs_root, str(args["increment"]))
    os.makedirs(log_dir, exist_ok=True)

    logfilename = os.path.join(
        log_dir,
        f'{args["prefix"]}_{args["backbone_type"]}'
    )

    # === 统一评估结果保存（与 trainer.py 对齐） ===
    def _metrics_book_path(_log_dir: str, _logfilename: str) -> str:
        return os.path.join(_log_dir, f"{os.path.basename(_logfilename)}_cl_metrics.json")

    def _write_final_metrics(json_path: str, section: str, final_metrics: dict, final_matrix=None):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        J = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    J = json.load(f)
            except Exception:
                J = {}
        S = J.setdefault(section, {})
        S["final"] = final_metrics
        if final_matrix is not None:
            S.setdefault("matrices", {})["final"] = np.asarray(final_matrix, dtype=float).tolist()
        tmp = json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(J, f, indent=2, ensure_ascii=False, default=json_safe)
        os.replace(tmp, json_path)

    metrics_book_path = _metrics_book_path(log_dir, logfilename)

    # ---------------------------
    # 1) Env & data/model
    # ---------------------------
    # ——— 重新配置 logging（先移除旧的 handlers，避免重复） ———
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    # Build data manager
    data_manager = DataManager(
        args["dataset"],
        args["class_shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    args["nb_classes"] = getattr(data_manager, "nb_classes", None)
    args["nb_tasks"] = getattr(data_manager, "nb_tasks", None)

    # Model
    learner = factory.get_model(args["model_name"], args)

    # --- Save a snapshot of BASE backbone (pre-finetune) for true pre-LoRA probe ---
    # _base_pre_ckpt = None
    # _base_pre_ckpt_path = os.path.join(checkpoint_dir, f"_backbone_pre_ft.pth")

    # Ensure the network/backbone is on-device for any pre-training evals
    net_obj = getattr(learner, "_network", learner)
    if hasattr(net_obj, "module"):  # 兼容 DataParallel
        net_obj = net_obj.module

    # 将backbone 模型移动到设备上
    
    net_obj.to(learner._device)
    getattr(net_obj, "backbone", net_obj).to(learner._device)

    probe_enabled = bool(args.get("linear_probe_softmax_joint_seen_eval", False))
    probe_metrics = {}
    probe_context = None
    probe_heads_dir = os.path.join(checkpoint_dir, "probe_heads")
    probe_log_interval = args.get("probe_log_interval", None)
    try:
        probe_log_interval = int(probe_log_interval) if probe_log_interval is not None else None
    except (TypeError, ValueError):
        probe_log_interval = None

    probe_eval_interval = args.get("probe_eval_interval", 5)
    head = None
    if probe_enabled:
        os.makedirs(probe_heads_dir, exist_ok=True)

        def _prepare_joint_probe_context():
            start_all = 0
            total_classes = int(args.get("nb_classes", getattr(data_manager, "nb_classes", 0) or 0))
            end_all = total_classes
            if end_all <= start_all:
                logging.warning("[PROBE] Skip context build: total classes=%s", total_classes)
                return None
            indices = np.arange(start_all, end_all)
            train_dataset = data_manager.get_dataset(
                indices, source="train", mode=args.get("probe_train_mode", "train")
            )
            test_dataset = data_manager.get_dataset(
                indices, source="test", mode=args.get("probe_test_mode", "test")
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                shuffle=True,
                num_workers=int(args.get("eval_num_workers", 0)),
              
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=int(args.get("batch_size", 128)),
                shuffle=False,
                num_workers=int(args.get("eval_num_workers", 0)),
                
            )
            return {
                "start": start_all,
                "end": end_all,
                "num_classes": end_all - start_all,
                "train_loader": train_loader,
                "test_loader": test_loader,
            }

        def _save_probe_head(head_module, filepath, stage_name, accuracy):
            if head_module is None:
                return
            head_cpu = head_module.to("cpu")
            meta = {
                "stage": stage_name,
                "accuracy": float(accuracy),
                "class_offset": probe_context.get("start") if probe_context else 0,
                "num_classes": probe_context.get("num_classes") if probe_context else 0,
                "epochs": int(args.get("probe_fit_epochs", 20)),
                "lr": float(args.get("probe_fit_lr", 1e-2)),
                "weight_decay": float(args.get("probe_fit_wd", 0.0)),
                "batch_size": int(args.get("probe_fit_train_batch_size", 128)),
            }
            torch.save({"state_dict": head_cpu.state_dict(), "meta": meta}, filepath)
            logging.info("[PROBE] Saved %s head to %s (acc=%.2f)", stage_name, filepath, float(accuracy))

        def _run_softmax_probe_stage(network, stage_name, head_filename, log_prefix):
            nonlocal probe_context
            if probe_context is None:
                probe_context = _prepare_joint_probe_context()
            if probe_context is None:
                return float("nan")
            
            # 日志输出的频次设置
            eval_interval = probe_eval_interval
            if eval_interval is None or eval_interval <= 0:
                try:
                    eval_interval = max(1, int(args.get("probe_fit_epochs", 20)) // 5)
                except Exception:
                    eval_interval = 5
            was_training = bool(getattr(network, "training", False))
            
            network.eval()
            
            head = fit_linear_probe_softmax_head(
                network,
                probe_context["train_loader"],
                class_offset=probe_context["start"],
                num_classes=probe_context["num_classes"],
                device=learner._device,
                epochs=int(args.get("probe_fit_epochs", 20)),
                lr=float(args.get("probe_fit_lr", 1e-2)),
                weight_decay=float(args.get("probe_fit_wd", 0.0)),
                batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                max_train_batches=args.get("probe_train_max_batches", None),
                monitor_loader=probe_context["test_loader"],
                monitor_max_batches=args.get("probe_test_max_batches", None),
                log_interval=probe_log_interval,
                eval_interval=eval_interval,
                log_prefix=log_prefix,
            )
            if head is None:
                logging.warning("[PROBE] %s head fitting failed (None)", stage_name)
                if was_training:
                    try:
                        network.train(True)
                    except Exception:
                        pass
                return float("nan")
            acc = evaluate_linear_probe_softmax_with_head(
                head,
                network,
                probe_context["test_loader"],
                class_offset=probe_context["start"],
                device=learner._device,
                max_test_batches=args.get("probe_test_max_batches", None),
            )
            head_path = os.path.join(probe_heads_dir, head_filename)
            _save_probe_head(head, head_path, stage_name, acc)
            # if was_training:
            #     try:
            #         network.train(True)
            #     except Exception:
            #         pass
            return head,float(acc)

        probe_context = _prepare_joint_probe_context()
        
        # 当是lora插入的时候，net_obj的backbone为LoRA_ViT_timm其有属性base_vit
        # 当时finetune的时候，net_obj的backbone为VisionTransformer少了LoRA_VI的包装，没有属性base_vit
        # base_backbone = getattr(net_obj, "backbone", None)
        # if isinstance(base_backbone, LoRA_ViT_timm):
        #     backbone_base_vit = backbone.base_vit  # 
        #     backbone_lora_vit = backbone.lora_vit
        #     base_backbone = backbone_base_vit

            
        head,base_acc = _run_softmax_probe_stage(
            net_obj,
            stage_name="base_pre_ft",
            head_filename="linear_probe_base_pre_ft.pth",
            log_prefix="[Probe-Softmax][BASE]",
        )
        probe_metrics["base_model_pre_ft"] = _to_float_metric(base_acc)
        logging.info("[PROBE] BASE(pre-ft) joint softmax acc=%.2f", float(base_acc))

        # ---- Pre-train flatness (BASE + trained probe head) ----
        if bool(args.get("flat_eval", False)) and head is not None:
            net_flat = net_obj
            if hasattr(net_flat, "module"):
                net_flat = net_flat.module
            old_fc = getattr(net_flat, "fc", None)
            was_training = bool(getattr(net_flat, "training", False))
            try:
                net_flat.fc = head.to(learner._device)
                net_flat.eval()
                if probe_context is None:
                    probe_context = _prepare_joint_probe_context()
                if probe_context is None:
                    raise RuntimeError("Probe context unavailable for base flatness evaluation")

                flat_loader = fractional_loader(
                    loader=probe_context["train_loader"],
                    fraction=args.get("flat_eval_dataset_fraction", 0.01),
                    seed=args.get("flat_eval_dataset_fraction_seed", args.get("seed", 42)),
                    balanced=True,
                    batch_size=args.get("flat_eval_batch_size", 64),
                )

                flat_cfg = FlatnessConfig(
                    args=args,
                    save_metrics_path=os.path.join(logs_root, "flatness"),
                    save_prefix=f"{os.path.basename(logfilename)}_base_pre_ft",
                    param_name_substrings=None,
                )
                _saved_requires = [(p, bool(p.requires_grad)) for _, p in net_flat.named_parameters()]
                for _, _p in net_flat.named_parameters():
                    if not _p.requires_grad:
                        _p.requires_grad_(True)
                try:
                    flat_metrics = evaluate_flatness_metrics(
                        net_flat,
                        flat_loader,
                        device=learner._device,
                        config=flat_cfg,
                    )
                    logging.info("[FlatEval][BASE] metrics: %s", flat_metrics)
                finally:
                    for _p, _old in _saved_requires:
                        _p.requires_grad_(_old)
            except Exception:
                logging.exception("[FlatEval][BASE] Failed to evaluate pre-ft flatness")
            finally:
                try:
                    net_flat.fc = old_fc
                except Exception:
                    pass
                try:
                    net_flat.train(was_training)
                except Exception:
                    pass
    


    # ---------------------------
    # 2) Finetune full data
    # ---------------------------
    if not hasattr(learner, "finetune_all_data"):
        logging.error("Learner '%s' has no method 'finetune_all_data(data_manager)'.", type(learner).__name__)
        raise AttributeError("finetune_all_data not implemented in learner")

    logging.info("==> Start finetune_all_data on full training set.")
    learner.finetune_all_data(data_manager)

    # # 替换掉分类头，只有进行评估原始模型的时候注释掉上面使用下面
    # net = learner._network.module if hasattr(learner._network, 'module') else learner._network
    # dev = next(net.parameters()).device
    # dtype = next(net.parameters()).dtype
    # net.fc = head.to(device=dev, dtype=dtype).eval()


    #先进行保存模型
    # Save the final trained model once (backbone + head)
    try:
        if  bool(args.get("save_final_model", True)) :
            net_to_save = getattr(learner, "_network", None)
            if net_to_save is not None:
                if hasattr(net_to_save, "module"):
                    net_to_save = net_to_save.module
                final_ckpt = {
                    "method": args.get("model_name","model_name"),
                    "dataset": args.get("dataset","dataset"),
                    "init_cls":args.get("init_cls","init_cls"),
                    "increment":args.get("init_cls","init_cls"),
                    "seed":args.get("seed","seed"),
                    "tasks": "all",
                    "model_state_dict": net_to_save.state_dict(),
                }
                final_path = os.path.join(checkpoint_dir, "final_model.pt")
                torch.save(final_ckpt, final_path)
                logging.info("Saved final model to %s", final_path)
    except Exception as _save_exc:
        logging.exception("[FinalSave] Failed to save final model: %s", _save_exc)


    # ---------------------------
    # 3) Evaluate & persist metrics
    # ---------------------------
    if hasattr(learner, "evaluate_full_dataset"):
        metrics = learner.evaluate_full_dataset()
    else:
        # Fallback: try standard evaluation API if available
        metrics = learner.eval_task()  # may return tuple; keep as-is
        

    logging.info("Full-dataset finetune metrics: %s", metrics)
    # Save a JSON snapshot for reproducibility
    
    with open(logfilename + "_finetune_all_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    logging.info("Saved metrics to %s", logfilename + "_finetune_all_metrics.json")
    

    # ---------------------------
    # 3.5) Linear-probe evaluation (pre-saved base, post-finetune final)
    # ---------------------------
    if probe_enabled:
        final_net = getattr(learner, "_network", learner)
        if hasattr(final_net, "module"):
            final_net = final_net.module
        head,final_acc = _run_softmax_probe_stage(
            final_net,
            stage_name="finetune_post_ft",
            head_filename="linear_probe_finetuned.pth",
            log_prefix="[Probe-Softmax][FINETUNE]",
        ) if probe_context is not None else float("nan")
        # probe_metrics["final_model"] = float(final_acc)
        probe_metrics["final_model"] = _to_float_metric(final_acc)
        logging.info("[PROBE] Finetune(post-ft) joint softmax acc=%.2f", float(final_acc))
        probe_json = f"{logfilename}_probe_softmax_joint_seen_all.json"
        with open(probe_json, "w", encoding="utf-8") as fh:
            json.dump(
                probe_metrics,
                fh,
                indent=2,
                ensure_ascii=False,
                default=lambda o: float(o) if hasattr(o, "__float__") else str(o),
            )
        logging.info("Saved probe metrics to %s", probe_json)
        gc.collect()
        if torch.cuda.is_available():
            
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            

    if args.get("attention_probe_eval", False):
        # Enhanced attention probe export (DINO-style + optional Grad-CAM)
        import time
        from torch.utils.data import Subset
        import torch.nn.functional as F

        probe_dir = os.path.join(log_dir, "attention_probe")
        os.makedirs(probe_dir, exist_ok=True)

        K = max(1, int(args.get("attention_probe_samples", 64)))
        seed = int(args.get("attention_probe_seed", 42))
        pick_mode = str(args.get("attention_probe_pick", "random")).lower()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        probe_path = os.path.join(probe_dir, f"attention_probe_{stamp}.pt")

        probe_dataset = None
        probe_loader = None
        attn_handle = None
        attn_bwd_handle = None
        last_attn_cache = {}
        last_attn_grad = {}

        try:
            # Build reproducible subset from test set
            total_classes = getattr(learner, "_total_classes", data_manager.nb_classes)
            probe_dataset = data_manager.get_dataset(
                np.arange(0, total_classes),
                source="test",
                mode="test",
            )
            N = len(probe_dataset)
            if N == 0:
                raise StopIteration

            rng = np.random.RandomState(seed)
            if pick_mode == "balanced" and hasattr(probe_dataset, "labels"):
                labels_np = np.asarray(getattr(probe_dataset, "labels"))
                from collections import defaultdict
                idxs_by_cls = defaultdict(list)
                for i, y in enumerate(labels_np.tolist()):
                    idxs_by_cls[int(y)].append(i)
                out = []
                ptr = {c: 0 for c in idxs_by_cls.keys()}
                classes = sorted(idxs_by_cls.keys())
                while len(out) < min(K, N) and classes:
                    for c in list(classes):
                        if ptr[c] < len(idxs_by_cls[c]):
                            out.append(idxs_by_cls[c][ptr[c]])
                            ptr[c] += 1
                            if len(out) >= min(K, N):
                                break
                        else:
                            classes.remove(c)
                sel_idx = np.array(out, dtype=int)
            elif pick_mode == "head":
                sel_idx = np.arange(0, min(K, N), dtype=int)
            else:
                sel_idx = rng.choice(N, size=min(K, N), replace=False)
            probe_subset = Subset(probe_dataset, sel_idx)

            # Fetch all selected samples in one batch for easier comparison
            probe_loader = DataLoader(
                probe_subset,
                batch_size=len(probe_subset),
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            probe_idx, probe_inputs, probe_labels = next(iter(probe_loader))
        except StopIteration:
            logging.warning("[AttentionProbe] No samples available for probe export.")
        except Exception as exc:
            logging.exception("[AttentionProbe] Failed to prepare probe data: %s", exc)
        else:
            # Forward and attention capture
            device_override = getattr(learner, "_device", None)
            if isinstance(device_override, str):
                device_override = torch.device(device_override)
            target_device = device_override or getattr(learner, "_device", None)
            if isinstance(target_device, str):
                target_device = torch.device(target_device)
            if target_device is None:
                target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 取底层 nn.Module
            net = getattr(learner, "_network", learner)
            if hasattr(net, "module"):
                net = net.module
            probe_inputs = probe_inputs.to(target_device, non_blocking=True)
            net.eval()

            backbone = getattr(net, "backbone", net)
            # Resolve inner ViT if wrapped (LoRA)
            inner = backbone
            for attr in ("base_vit", "lora_vit"):
                if hasattr(inner, attr) and getattr(inner, attr) is not None:
                    inner = getattr(inner, attr)
                    break
            blocks = getattr(inner, "blocks", None)

            def _hook_last_attn(module, inp, out):
                # inp[0]: softmaxed attention [B, heads, N, N]
                try:
                    last_attn_cache["attn"] = inp[0].detach()
                except Exception:
                    pass

            def _hook_last_attn_bwd(module, grad_input, grad_output):
                try:
                    if grad_input and grad_input[0] is not None:
                        last_attn_grad["grad"] = grad_input[0].detach()
                except Exception:
                    pass

            try:
                # Hook all blocks' attn_drop to capture attentions across layers
                handles = []
                captured = {}
                if blocks is not None:
                    for i, blk in enumerate(blocks):
                        if hasattr(blk, "attn") and hasattr(blk.attn, "attn_drop"):
                            def _make_hook(idx):
                                def _fn(module, inp, out):
                                    if isinstance(out, torch.Tensor) and out.dim() == 4:
                                        captured[idx] = out.detach()
                                        # keep last for Grad-CAM fallback
                                        last_attn_cache["attn"] = out.detach()
                                return _fn
                            handles.append(blk.attn.attn_drop.register_forward_hook(_make_hook(i)))
                            if bool(args.get("attention_probe_gradcam", False)):
                                handles.append(blk.attn.attn_drop.register_full_backward_hook(_hook_last_attn_bwd))

                # Disable SDPA/fused so hooks run
                try:
                    from torch.nn.attention import sdpa_kernel, SDPBackend  # type: ignore
                    cm = sdpa_kernel(SDPBackend.MATH)
                except Exception:
                    try:
                        from torch.backends.cuda import sdp_kernel  # type: ignore
                        cm = sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
                    except Exception:
                        from contextlib import nullcontext
                        cm = nullcontext()

                # Forward pass
                with cm:
                    outputs = net(probe_inputs)

                # Extract features (kept compatible with original logic)
                features = outputs.get("features") if isinstance(outputs, dict) else None
                if features is None and backbone is not None:
                    with torch.no_grad():
                        features = backbone(probe_inputs)

                # Attempt to locate logits for Grad-CAM
                def _find_logits(out):
                    if isinstance(out, torch.Tensor) and out.dim() == 2:
                        return out
                    if isinstance(out, dict):
                        for k in ["logits", "logit", "cls_logits", "output", "outputs"]:
                            v = out.get(k, None)
                            if isinstance(v, torch.Tensor) and v.dim() == 2:
                                return v
                        for v in out.values():
                            if isinstance(v, torch.Tensor) and v.dim() == 2:
                                return v
                    return None

                logits = _find_logits(outputs)

            except Exception as exc:
                logging.exception("[AttentionProbe] Failed to compute probe outputs: %s", exc)
                logits = None
                features = None

            # Build attention maps (last-layer + rollout)
            attn_heads_up = None
            attn_avg_up = None
            attn_rollout_up = None
            last_grid = None
            rollout_grid = None
            ps = None
            try:
                # prefer captured map; fallback to last_attn_cache
                attn = None
                try:
                    if 'captured' in locals() and captured:
                        attn = captured[max(captured.keys())]
                except Exception:
                    pass
                if attn is None:
                    attn = last_attn_cache.get("attn", None)
                if attn is not None and attn.ndim == 4:
                    def _resolve_patch_size(m):
                        pe = getattr(m, "patch_embed", None)
                        ps_ = getattr(pe, "patch_size", None)
                        return ps_[0] if isinstance(ps_, tuple) else (int(ps_) if ps_ is not None else None)

                    ps = _resolve_patch_size(inner)
                    B, C, H, W = probe_inputs.shape
                    if ps is None or ps <= 0:
                        # Fallback: infer from token count (assumes cls token + h*w)
                        Ntok = attn.shape[-1]
                        p = int(round((H * W / (Ntok - 1)) ** 0.5)) if Ntok > 1 else 16
                        ps = max(1, p)

                    H2, W2 = H - (H % ps), W - (W % ps)
                    h, w = max(1, H2 // ps), max(1, W2 // ps)

                    A = attn[:, :, 0, 1:].reshape(attn.shape[0], attn.shape[1], h, w)
                    A_avg = A.mean(dim=1, keepdim=True)
                    A_up = F.interpolate(A, size=(H2, W2), mode="bilinear", align_corners=False)
                    A_avg_up = F.interpolate(A_avg, size=(H2, W2), mode="bilinear", align_corners=False)

                    def _norm(x):
                        x_min = x.amin(dim=(-2, -1), keepdim=True)
                        x_max = x.amax(dim=(-2, -1), keepdim=True)
                        return (x - x_min) / (x_max - x_min + 1e-8)

                    attn_heads_up = _norm(A_up).cpu()
                    attn_avg_up = _norm(A_avg_up).cpu()
                    last_grid = A_avg.detach().cpu()

                    # rollout across layers if available
                    if 'captured' in locals() and captured:
                        eps = 1e-6
                        layers = sorted(captured.keys())
                        A_list = [captured[i].mean(dim=1) for i in layers]  # [B,N,N]
                        rollout = None
                        for A_l in A_list:
                            Bn, Nn, _ = A_l.shape
                            I = torch.eye(Nn, device=A_l.device, dtype=A_l.dtype).unsqueeze(0).expand(Bn, Nn, Nn)
                            A_bar = A_l + I
                            A_bar = A_bar / (A_bar.sum(dim=-1, keepdim=True) + eps)
                            rollout = A_bar if rollout is None else torch.bmm(rollout, A_bar)
                        rollout_cls = rollout[:, 0, 1:].reshape(B, 1, h, w)
                        rollout_grid = rollout_cls.detach().cpu()
                        attn_rollout_up = _norm(F.interpolate(rollout_cls, size=(H2, W2), mode="bilinear", align_corners=False)).cpu()
            except Exception as exc:
                logging.exception("[AttentionProbe] Failed to build attention maps: %s", exc)

            # Optional: class-discriminative Grad-CAM on attention heads
            gradcam_up = None
            if bool(args.get("attention_probe_gradcam", False)) and logits is not None and last_attn_cache.get("attn") is not None:
                try:
                    # Clear old grads
                    if hasattr(net, "zero_grad"):
                        net.zero_grad(set_to_none=True)
                    # Use provided labels if available; else top-1 predictions
                    with torch.no_grad():
                        top1 = logits.argmax(dim=1)
                    target_y = probe_labels.to(logits.device) if (probe_labels is not None and probe_labels.numel() == logits.size(0)) else top1
                    # Sum the target scores for backward
                    target_score = logits.gather(1, target_y.view(-1, 1)).sum()
                    target_score.backward(retain_graph=True)

                    G = last_attn_grad.get("grad", None)  # [B, heads, N, N]
                    A = last_attn_cache.get("attn", None)
                    if G is not None and A is not None and G.shape == A.shape:
                        # Heads weights by global-average of gradients on CLS->patch entries
                        B, Hh, Nq, Nk = G.shape
                        g = G[:, :, 0, 1:]  # [B, heads, H*W]
                        if ps is None:
                            # derive ps/h/w the same way as above
                            Himg, Wimg = probe_inputs.shape[-2], probe_inputs.shape[-1]
                            Ntok = Nk
                            p = int(round((Himg * Wimg / (Ntok - 1)) ** 0.5)) if Ntok > 1 else 16
                            ps_ = max(1, p)
                            H2, W2 = Himg - (Himg % ps_), Wimg - (Wimg % ps_)
                            h, w = max(1, H2 // ps_), max(1, W2 // ps_)
                        else:
                            Himg, Wimg = probe_inputs.shape[-2], probe_inputs.shape[-1]
                            H2, W2 = Himg - (Himg % ps), Wimg - (Wimg % ps)
                            h, w = max(1, H2 // ps), max(1, W2 // ps)

                        g = g.reshape(B, Hh, h, w)
                        w_k = g.mean(dim=(-2, -1), keepdim=True)  # [B, heads, 1, 1]

                        A_cls = A[:, :, 0, 1:].reshape(B, Hh, h, w)
                        cam = (w_k * A_cls).sum(dim=1, keepdim=True)  # [B,1,h,w]
                        cam = F.relu(cam)
                        cam_up = F.interpolate(cam, size=(H2, W2), mode="bilinear", align_corners=False)
                        # Normalize per-image
                        cam_min = cam_up.amin(dim=(-2, -1), keepdim=True)
                        cam_max = cam_up.amax(dim=(-2, -1), keepdim=True)
                        gradcam_up = ((cam_up - cam_min) / (cam_max - cam_min + 1e-8)).cpu()
                except Exception as exc:
                    logging.exception("[AttentionProbe] Grad-CAM computation failed: %s", exc)

            # Prepare payload and save
            meta = {
                "model_tag": getattr(net, "name", type(net).__name__),
                "input_size": (int(probe_inputs.shape[-2]), int(probe_inputs.shape[-1])),
                "patch_size": int(ps) if ps is not None else None,
                "seed": int(seed),
                "num_samples": int(len(probe_subset)) if "probe_subset" in locals() else int(probe_labels.shape[0]),
            }

            payload = {
                "indices": probe_idx.cpu(),
                "labels": probe_labels.cpu(),
                "features": features.detach().cpu() if (features is not None and torch.is_tensor(features)) else None,
                "meta": meta,
            }
            if args.get("attention_probe_save_inputs", False):
                payload["inputs"] = probe_inputs.detach().cpu().to(dtype=torch.float16)
            if attn_heads_up is not None:
                payload["attn_heads"] = attn_heads_up  # [B, heads, H', W']
                payload["attn_avg"] = attn_avg_up      # [B, 1,    H', W']
            if attn_rollout_up is not None:
                payload["attn_rollout_up"] = attn_rollout_up
            if last_grid is not None:
                payload["attn_last_grid"] = last_grid
            if rollout_grid is not None:
                payload["attn_rollout_grid"] = rollout_grid
            if gradcam_up is not None:
                payload["gradcam_cls"] = gradcam_up     # [B, 1,    H', W']

            try:
                torch.save(payload, probe_path)
                logging.info(
                    "[AttentionProbe] Saved %d samples (features%s%s%s%s) to %s",
                    int(probe_labels.shape[0]),
                    ", heads" if "attn_heads" in payload else "",
                    ", last/avg" if "attn_avg" in payload else "",
                    ", rollout" if "attn_rollout_up" in payload else "",
                    ", gradcam" if "gradcam_cls" in payload else "",
                    probe_path,
                )
            except Exception as exc:
                logging.exception("[AttentionProbe] Saving probe payload failed: %s", exc)
        finally:
            if attn_handle is not None:
                try:
                    attn_handle.remove()
                except Exception:
                    pass
            if attn_bwd_handle is not None:
                try:
                    attn_bwd_handle.remove()
                except Exception:
                    pass
            try:
                for h in handles:
                    try:
                        h.remove()
                    except Exception:
                        pass
            except Exception:
                pass
            probe_loader = None
            probe_dataset = None

    #  ---------------------------
    # 4) Flatness evaluations (weight & feature) on the SAME 10% subset
    # ---------------------------
    
    train_loader_full = getattr(learner, "train_loader", None)
    if train_loader_full is None:
        logging.warning("[FlatEval] learner has no train_loader, skip flatness.")
    else:
        # 采样同一份 10% 子集（balanced=true），供 weight & feature 共同使用
        flat_loader = fractional_loader(
            loader=train_loader_full,
            fraction=0.01,
            seed=args.get("seed", 42),
            balanced=True,
            batch_size=args.get("flat_eval_batch_size", 64)
        )

        # 取底层 nn.Module
        net = getattr(learner, "_network", learner)
        if hasattr(net, "module"):
            net = net.module

        # 记录并恢复 train/eval 模式（weight/feature 两段均使用）
        was_training = bool(getattr(net, "training", False))
        net.eval()
        

        # ---- (A) weight-space flatness：全参数评估 ----
        if bool(args.get("flat_eval", False)):
            flat_cfg = FlatnessConfig(
                args=args,                                  # ← 直接把 args 传进来
                save_metrics_path=os.path.join(logs_root, "flatness"),
                save_prefix=os.path.basename(logfilename),
                param_name_substrings=None,                 # finetune 情形：全参
            )

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            # 保存并临时将所有参数 requires_grad=True，用于全局二阶评估
            _saved_requires = [(p, bool(p.requires_grad)) for _, p in net.named_parameters()]
            for _, _p in net.named_parameters():
                if not _p.requires_grad:
                    _p.requires_grad_(True)
            try:
                flat_metrics = evaluate_flatness_metrics(
                    net,
                    flat_loader,
                    device=learner._device,
                    config=flat_cfg,
                )
            finally:
                # 恢复原 requires_grad 设置
                for _p, _old in _saved_requires:
                    _p.requires_grad_(_old)

            # logging.info("Flatness metrics (task %d): %s", task, flat_metrics)

            # 资源清理（但保留 flat_loader 给 feature 复用）
            if hasattr(net, "zero_grad"):
                net.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        # ---- (B) feature-space flatness：复用同一 flat_loader ----
        if bool(args.get("feature_flat_eval", False)):
            feature_dir = os.path.join(logs_root, "feature_flatness")
            os.makedirs(feature_dir, exist_ok=True)

            device_override = getattr(learner, "_device", None)
            if isinstance(device_override, str):
                device_override = torch.device(device_override)

            feature_cfg = FeatureFlatnessConfig(
                max_batches=args.get("feature_flat_max_batches", None),
                topk_eigen=int(args.get("feature_flat_topk", 20)),
                eps=float(args.get("feature_flat_eps", 1e-12)),
                rank_tol=float(args.get("feature_flat_rank_tol", 1e-6)),
                save_matrix_path=feature_dir,
                save_prefix=os.path.basename(logfilename),
                device_override=device_override,
            )

            # 直接复用上面的 flat_loader，确保两类指标基于同一子集
            feature_metrics = evaluate_feature_metrics(
                net, flat_loader, config=feature_cfg
            )
            logging.info("[FeatureFlat] metrics: %s", feature_metrics)

            # 资源清理
            if hasattr(net, "zero_grad"):
                net.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        # 用完再显式释放
        del flat_loader
        torch.cuda.empty_cache()

        # 恢复训练/评估模式
        
        net.train(was_training)
        

    
    # 4.5) OOD evaluation on Tiny-ImageNet-C/P (all classes)
    #      Placed after CL training loop and before final summaries.
    try:
        ood_results = {}
        # Common loader builder for OOD datasets
        def _build_ood_loader(dm, batch_size: int, num_workers: int):
            ds = dm.get_dataset(np.arange(0, dm.nb_classes), source="test", mode="test")
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=(num_workers > 0),
            )

        # Evaluation settings
        ood_bs = int(args.get("ood_eval_batch_size", 128))
        ood_workers = int(args.get("ood_eval_num_workers", 0))
        class_shuffle_ood = args.get("class_shuffle", True)

        # Tiny-ImageNet-R
        if bool(args.get("ood_imagener_r", True)):
            ood_args_c = copy.deepcopy(args)
            ood_args_c["dataset"] = "imagenetr"
            dm_c = DataManager("imagenetr", class_shuffle_ood, args["seed"], 200, 0, ood_args_c)
            loader_c = _build_ood_loader(dm_c, ood_bs, ood_workers)
            acc_c = learner.evaluate_full_dataset(loader_c)
            ood_results["imagenetr"] = {"top1": float(acc_c.get("top1", 0.0)), "top5": float(acc_c.get("top5", 0.0))}
            logging.info("[OOD][Tiny-ImageNet-R] top1=%.2f | top5=%.2f", ood_results["imagenetr"]["top1"], ood_results["imagenetr"]["top5"])
            # Persist into unified metrics json
            _write_final_metrics(metrics_book_path, "imagenetr", final_metrics=ood_results["imagenetr"], final_matrix=None)


        # Tiny-ImageNet-C
        if bool(args.get("ood_imagener_c", True)):
            ood_args_c = copy.deepcopy(args)
            ood_args_c["dataset"] = "tiny_imagenetc"
            dm_c = DataManager("tiny_imagenetc", class_shuffle_ood, args["seed"], 200, 0, ood_args_c)
            loader_c = _build_ood_loader(dm_c, ood_bs, ood_workers)
            acc_c = learner.evaluate_full_dataset(loader_c)
            ood_results["tiny_imagenetc"] = {"top1": float(acc_c.get("top1", 0.0)), "top5": float(acc_c.get("top5", 0.0))}
            logging.info("[OOD][Tiny-ImageNet-R] top1=%.2f | top5=%.2f", ood_results["tiny_imagenetc"]["top1"], ood_results["tiny_imagenetc"]["top5"])
            # Persist into unified metrics json
            _write_final_metrics(metrics_book_path, "tiny_imagenetc", final_metrics=ood_results["tiny_imagenetc"], final_matrix=None)

        # Tiny-ImageNet-P
        if bool(args.get("ood_imagenet_p", True)):
            ood_args_p = copy.deepcopy(args)
            ood_args_p["dataset"] = "tiny_imagenetp"
            dm_p = DataManager("tiny_imagenetp", class_shuffle_ood, args["seed"], 200, 0, ood_args_p)
            loader_p = _build_ood_loader(dm_p, ood_bs, ood_workers)
            acc_p = learner.evaluate_full_dataset(loader_p)
            ood_results["tiny_imagenetp"] = {"top1": float(acc_p.get("top1", 0.0)), "top5": float(acc_p.get("top5", 0.0))}
            logging.info("[OOD][Tiny-ImageNet-P-R] top1=%.2f | top5=%.2f", ood_results["tiny_imagenetp"]["top1"], ood_results["tiny_imagenetp"]["top5"])
            # Persist into unified metrics json
            _write_final_metrics(metrics_book_path, "ood_tiny_tiny_imagenetp", final_metrics=ood_results["tiny_imagenetp"], final_matrix=None)
    
    
    
    except Exception as _ood_exc:
        logging.exception("[OOD] Evaluation failed: %s", _ood_exc)

    #  Save the final trained model once (backbone + head)
    try:
        if bool(args.get("save_final_model", True)):
            net_to_save = getattr(learner, "_network", None)
            if net_to_save is not None:
                if hasattr(net_to_save, "module"):
                    net_to_save = net_to_save.module
                final_ckpt = {
                    "tasks": int(getattr(data_manager, "nb_tasks", 1) - 1),
                    "model_state_dict": net_to_save.state_dict(),
                }
                final_path = os.path.join(checkpoint_dir, "final_model.pt")
                torch.save(final_ckpt, final_path)
                logging.info("Saved final model to %s", final_path)
    except Exception as _save_exc:
        logging.exception("[FinalSave] Failed to save final model: %s", _save_exc)




def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
