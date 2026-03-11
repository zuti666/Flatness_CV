import sys
import logging
import copy
import torch
from torch.utils.data import DataLoader
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from evaluation.metrics import compute_sequence_metrics, save_metrics_and_vectors
from evaluation_feature.eval_flat_feature import (
    extract_features_and_labels,
    linear_cka,
    compare_first_last_features,
    compare_task_last_features,
)
import os
import numpy as np
from evaluation_weight_sharpness.eval_flatness_weight_Loss import FlatnessConfig, evaluate_flatness_metrics
from evaluation_weight_sharpness.param_utils import _select_params_by_name
import gc, torch
from evaluation_feature.eval_flat_feature import FeatureFlatnessConfig, evaluate_feature_metrics, save_task_feature_cache
from evaluation_attention.attention_probe import run_attention_probe
from evaluation_ood.ood_eval import run_ood_evaluation
from typing import Optional
import json
from utils.data_manager import fractional_loader
from evaluation_performance.probe import (
    LinearProbeConfig,
    LinearProbeRunner,
    fit_linear_probe_softmax_head,
    evaluate_linear_probe_softmax_with_head,
    _FeatureView,
)
from utils.random_reproduce import set_device, set_random, json_safe
from utils.metrics_book import (
    metrics_json_path,
    write_step_metrics,
    write_final_metrics,
    assemble_eval_matrix,
    log_eval_matrix,
    save_eval_matrix,
)

# ---------------------------
# Metrics helpers live in utils.metrics_book
# ---------------------------




def _parse_task_list(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        try:
            return [int(x) for x in val]
        except Exception:
            return None
    if isinstance(val, int):
        return [val]
    if isinstance(val, str):
        s = val.strip()
        try:
            if s.startswith("[") and s.endswith("]"):
                import json as _json
                return [int(x) for x in _json.loads(s)]
        except Exception:
            return None
        try:
            return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
        except Exception:
            return None
    return None
    logging.info("Saved %s to %s and %s", tag, npy_path, csv_path)


def _resolve_task_indices(spec, nb_tasks):
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = spec.strip().lower()
        if spec in {"", "all"}:
            return None
        if spec in {"first_last", "first-last", "firstlast"}:
            indices = [0, nb_tasks - 1]
        else:
            indices = [int(s) for s in spec.split(",") if s.strip() != ""]
    elif isinstance(spec, (list, tuple, set)):
        indices = [int(s) for s in spec]
    else:
        indices = [int(spec)]
    resolved = []
    for idx in indices:
        if idx < 0:
            idx = nb_tasks + idx
        if 0 <= idx < nb_tasks:
            resolved.append(idx)
    if not resolved:
        return []
    return sorted(set(resolved))


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)



def _train(args):

    # ---------------------------
    # 0) I/O & logging
    # ---------------------------

    # Root layout: <outputs_root>/logs_inc[_lora]/<model>/<opt>/<dataset>/<seed>/<prefix>/<mode>/
    # Allow configurable output root; default to outputs_logs for cleaner repo layout.
    outputs_root = args.get("outputs_root", "outputs_logs")

    is_lora = "lora" in str(args["model_name"])
    # optimizer tag
    if args["optimizer_type"] == "rwp":
        opt_tag = f"rwp_{args.get('rwp_range')}"
    else:
        opt_tag = str(args["optimizer_type"])
    # mode tag
    if args["optimizer_type"] == "rwp":
        mode = "rwp_full" if args.get("rwp_range") == "full" else ("rwp_lora" if args.get("rwp_range") == "lora" else "rwp")
    else:
        mode = "exp_run"

    logs_root = os.path.join(
        outputs_root,
        "logs_inc_lora" if is_lora else "logs_inc",
        str(args["model_name"]),
        opt_tag,
        str(args["dataset"]),
        str(args["seed"]),
        str(args["prefix"]),
        mode,
    )

    os.makedirs(logs_root, exist_ok=True)

    # Checkpoints and auxiliary artifacts live alongside logs for each run.
    checkpoint_dir = os.path.join(logs_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    args["filepath"] = checkpoint_dir if checkpoint_dir.endswith(os.sep) else (checkpoint_dir + os.sep)
    args.setdefault("feature_flat_save_path", checkpoint_dir)

    # Logs for a specific increment are nested under increment id.
    log_dir = os.path.join(logs_root, str(args["increment"]))
    os.makedirs(log_dir, exist_ok=True)

    logfilename = os.path.join(
        log_dir,
        f'{args["prefix"]}_{args["backbone_type"]}'
    )

    metrics_path = metrics_json_path(log_dir, logfilename)
    # ===  ===

    # ———  logging（ handlers，） ———
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

    # ---------------------------
    # 1) Env & data/model
    # ---------------------------

    set_random(args["seed"])
    set_device(args)
    print_args(args)

    class_shuffle = args.get("class_shuffle", args.get("shuffle", True))

    data_manager = DataManager(
        args["dataset"],
        class_shuffle,
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    
    args["nb_classes"] = data_manager.nb_classes # update args
    try:
        _max_tasks_req = args.get("max_train_tasks", data_manager.nb_tasks)
        _max_tasks_req = int(_max_tasks_req) if _max_tasks_req is not None else data_manager.nb_tasks
    except Exception:
        _max_tasks_req = data_manager.nb_tasks
    nb_tasks = min(data_manager.nb_tasks, max(1, _max_tasks_req))
    if nb_tasks < data_manager.nb_tasks:
        logging.info("Restrict training to %d tasks (of %d total)", nb_tasks, data_manager.nb_tasks)
    args["nb_tasks"] = nb_tasks
    
    model = factory.get_model(args["model_name"], args)
    

    # Ensure the network/backbone is on-device for any pre-training evals
    net_obj = getattr(model, "_network", None)
    if net_obj is None:
        raise AttributeError("Model has no _network attribute")
    bb = getattr(net_obj, "backbone", None)
    if bb is not None and hasattr(bb, "to"):
        bb.to(model._device)
    else:
        net_obj.to(model._device)

    for name, _param in net_obj.named_parameters():
        logging.info("[LoRA] net_obj param name: %s", name)

    class_ranges = [data_manager.get_task_class_range(task_idx) for task_idx in range(nb_tasks)]
    flat_eval_tasks = _resolve_task_indices(args.get("flat_eval_task_indices", None), nb_tasks)
    feature_eval_tasks = _resolve_task_indices(args.get("feature_flat_task_indices", None), nb_tasks)
    
    # ---------------------------



    # ---------------------------
    # 2) Switches for new metrics
    # ---------------------------
    # head_eval = args.get("head_eval", False)
    # head_R = np.full((nb_tasks, nb_tasks), np.nan, dtype=float)
    # def _evaluate_head_row(task_idx):
    #     row = np.full(nb_tasks, np.nan, dtype=float)
    #     for j in range(task_idx + 1):
    #         loader = _build_loader(class_ranges[j], source="test", mode="test")
    #         row[j] = model._compute_accuracy(model._network, loader)
    #     return row

    # 4) Main CL loop
    # ---------------------------
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    logging.info("Start traing CL")
    for task in range(nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()



        # ---- Curves & matrices (restore sorted keys for stable ordering) ----
        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            # cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            # nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            # logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            # cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

        # —— CNN/NME ：（） ——
        T_partial = task + 1
        if len(cnn_matrix) > 0:
            cnn_time_by_task_partial = assemble_eval_matrix(cnn_matrix, T_partial, orientation="time_by_task")
            write_step_metrics(
                metrics_path, "cnn", step=task,
                metrics=None,
                matrix=cnn_time_by_task_partial,
                json_safe=json_safe,
            )

        if nme_accy is not None and len(nme_matrix) > 0:
            nme_time_by_task_partial = assemble_eval_matrix(nme_matrix, T_partial, orientation="time_by_task")
            write_step_metrics(
                metrics_path, "nme", step=task,
                metrics=None,
                matrix=nme_time_by_task_partial,
                json_safe=json_safe,
            )

        model.after_task()
        gc.collect(); 
        torch.cuda.empty_cache(); 
        torch.cuda.ipc_collect()
        # ---------------------------
        #
        # Save the final trained model once (backbone + head)
        try:
            if (task == nb_tasks - 1) and bool(args.get("save_final_model", False)) :
                net_to_save = getattr(model, "_network", None)
                if net_to_save is not None:
                    if hasattr(net_to_save, "module"):
                        net_to_save = net_to_save.module
                    final_ckpt = {
                        "method": args.get("model_name","model_name"),
                        "dataset": args.get("dataset","dataset"),
                        "init_cls":args.get("init_cls","init_cls"),
                        "increment":args.get("init_cls","init_cls"),
                        "seed":args.get("seed","seed"),
                        "tasks": int(nb_tasks - 1),
                        "model_state_dict": net_to_save.state_dict(),
                    }
                    final_path = os.path.join(checkpoint_dir, "final_model.pt")
                    torch.save(final_ckpt, final_path)
                    logging.info("Saved final model to %s", final_path)
        except Exception as _save_exc:
            logging.exception("[FinalSave] Failed to save final model: %s", _save_exc)


        # Optional per-task linear probe at final step (disabled by default)
        try:
            if  (task == nb_tasks - 1) and bool(args.get("linear_probe_softmax_per_task_eval", False)):
                net = getattr(model, "_network", model)
                if hasattr(net, "module"):
                    net = net.module
                was_training = net.training
                net.eval()

                set_random(args["seed"])

                #  TUNA， adapter（）
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

                probe_log_interval = args.get("probe_log_interval", None)
                probe_eval_interval = max(1, int(args.get("probe_fit_epochs", 50)) // 5)

                # ： j（0..task），/
                acc_final_list = []
                acc_base_list = []
                for j in range(task + 1):
                    start_j, end_j = class_ranges[j][0], class_ranges[j][1]
                    num_classes_j = end_j - start_j
                    if num_classes_j <= 0:
                        acc_final_list.append(float("nan"))
                        acc_base_list.append(float("nan"))
                        continue

                    train_ds_j = data_manager.get_dataset(
                        np.arange(start_j, end_j), source="train", mode=args.get("probe_train_mode", "train")
                    )
                    test_ds_j = data_manager.get_dataset(
                        np.arange(start_j, end_j), source="test", mode=args.get("probe_test_mode", "test")
                    )

                    train_loader_j = DataLoader(
                        train_ds_j,
                        batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                        shuffle=True,
                        num_workers=int(args.get("linear_probe_eval_num_workers", 0)),
                        persistent_workers=False,
                    )
                    test_loader_j = DataLoader(
                        test_ds_j,
                        batch_size=int(args.get("probe_fit_test_batch_size", 128)),
                        shuffle=False,
                        num_workers=int(args.get("linear_probe_eval_num_workers", 0)),
                        persistent_workers=False,
                    )

                    # （）
                    head_final_j = fit_linear_probe_softmax_head(
                        net, train_loader_j,
                        class_offset=start_j, num_classes=num_classes_j, device=model._device,
                        epochs=int(args.get("probe_fit_epochs", 50)),
                        lr=float(args.get("probe_fit_lr", 5e-3)),
                        weight_decay=float(args.get("probe_fit_wd", 0.0)),
                        batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                        max_train_batches=args.get("probe_train_max_batches", None),
                        monitor_loader=test_loader_j,
                        monitor_max_batches=args.get("probe_test_max_batches", None),
                        log_interval=probe_log_interval,
                        eval_interval=probe_eval_interval,
                        log_prefix=f"[LP-Softmax][PerTask][Final][t{j:02d}]",
                    )
                    acc_final_j = evaluate_linear_probe_softmax_with_head(
                        head_final_j, net, test_loader_j,
                        class_offset=start_j, device=model._device,
                        max_test_batches=args.get("probe_test_max_batches", None)
                    )

                    # （base/backbone， TUNA）
                    if str(args.get("model_name", "")).lower() != "tuna":
                        base_view = _FeatureView(getattr(net, "backbone", net), which="base")
                        base_view.eval()
                        head_base_j = fit_linear_probe_softmax_head(
                            base_view, train_loader_j,
                            class_offset=start_j, num_classes=num_classes_j, device=model._device,
                            epochs=int(args.get("probe_fit_epochs", 50)),
                            lr=float(args.get("probe_fit_lr", 5e-3)),
                            weight_decay=float(args.get("probe_fit_wd", 0.0)),
                            batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                            max_train_batches=args.get("probe_train_max_batches", None),
                            monitor_loader=test_loader_j,
                            monitor_max_batches=args.get("probe_test_max_batches", None),
                            log_interval=probe_log_interval,
                            eval_interval=probe_eval_interval,
                            log_prefix=f"[LP-Softmax][PerTask][Base][t{j:02d}]",
                        )
                        acc_base_j = evaluate_linear_probe_softmax_with_head(
                            head_base_j, base_view, test_loader_j,
                            class_offset=start_j, device=model._device,
                            max_test_batches=args.get("probe_test_max_batches", None)
                        )
                    else:
                        acc_base_j = 0.0

                    logging.info("[LP-PerTask][t%02d] final-model Acc=%.2f | base-model Acc=%.2f", j, acc_final_j, acc_base_j)
                    acc_final_list.append(float(acc_final_j))
                    acc_base_list.append(float(acc_base_j))

                    # ，
                    try:
                        for _m in ["head_final_j", "head_base_j", "base_view"]:
                            if _m in locals() and locals()[_m] is not None:
                                mod = locals()[_m]
                                if isinstance(mod, torch.nn.Module):
                                    with torch.inference_mode():
                                        mod.to("cpu")
                                    for p in mod.parameters():
                                        p.grad = None
                    except Exception:
                        pass
                    head_final_j = None; head_base_j = None; base_view = None
                    train_loader_j = None; test_loader_j = None
                    train_ds_j = None; test_ds_j = None
                    gc.collect();
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache(); torch.cuda.ipc_collect()

                #  JSON（ + ）
                avg_final = float(np.nanmean(acc_final_list)) if len(acc_final_list) > 0 else float("nan")
                avg_base  = float(np.nanmean(acc_base_list)) if len(acc_base_list) > 0 else float("nan")
                write_step_metrics(
                    metrics_path,
                    section="probe_softmax_per_task",
                    step=task,
                    metrics={
                        "final_model_per_task": acc_final_list,
                        "base_model_per_task": acc_base_list,
                        "avg_final": avg_final,
                        "avg_base": avg_base,
                    },
                    matrix=None,
                    json_safe=json_safe,
                )
                write_final_metrics(
                    metrics_path,
                    section="probe_softmax_per_task",
                    final_metrics={
                        "final_model_per_task": acc_final_list,
                        "base_model_per_task": acc_base_list,
                        "avg_final": avg_final,
                        "avg_base": avg_base,
                    },
                    final_matrix=None,
                    json_safe=json_safe,
                )
        finally:
            pass

        # linear_probe_softmax_joint_seen_eval
        try:
            if (task == nb_tasks - 1) and args.get("linear_probe_softmax_joint_seen_eval", False):
                net = getattr(model, "_network", model)
                if hasattr(net, "module"):
                    net = net.module
                # ，
                was_training = net.training
                net.eval()


                set_random(args["seed"])

                # 1)～5) （/ head、）
                start_seen = class_ranges[0][0]
                end_seen = class_ranges[task][1]
                num_classes =  end_seen - start_seen 

                train_dataset_all = data_manager.get_dataset(
                    np.arange(start_seen, end_seen), source="train",
                    mode=args.get("probe_train_mode", "train")
                )
                test_dataset_all = data_manager.get_dataset(
                    np.arange(start_seen, end_seen), source="test",
                    mode=args.get("probe_test_mode", "test")
                )

             
                train_loader_all = DataLoader(
                    train_dataset_all,
                    batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                    shuffle=True,
                    num_workers=int(args.get("linear_probe_eval_num_workers", 0)),
                    persistent_workers=False,
                )
                test_loader_all = DataLoader(
                    test_dataset_all,
                    batch_size=int(args.get("probe_fit_test_batch_size", 128)),
                    shuffle=False,
                    num_workers=int(args.get("linear_probe_eval_num_workers", 0)),
                    persistent_workers=False,
                    
                )

                probe_log_interval = args.get("probe_log_interval", None)
                
                
                probe_eval_interval = max(1, int(args.get("probe_fit_epochs", 50)) // 5)

                # module = net.module if isinstance(net, torch.nn.DataParallel) else net
                # # ：TUNA “ adapter” = cur_task + 1；
                # #  merge（ adapter）， 0。
                # universal_id = (self._cur_task + 1) if (getattr(module.backbone, "merge", None) and self._cur_task > 0) else 0
                # module.set_active_adapter(universal_id)   # ★ ： adapter
                if str(args.get("model_name", "")).lower() == "tuna":
                    module = net.module if hasattr(net, "module") else net
                    bb = getattr(module, "backbone", None)
                    if bb is not None and hasattr(bb, "adapter_list"):
                        #  fused adapter id（eval  > len(adapter_list)  merged_adapter）
                        fused_id = (len(bb.adapter_list) + 1) if (len(bb.adapter_list) > 0 and getattr(bb, "merged_adapter", None) is not None) else len(bb.adapter_list)
                        if hasattr(module, "set_active_adapter"):
                            module.set_active_adapter(fused_id)
                        else:
                            setattr(bb, "active_adapter_id", int(fused_id))
                    
                    head_final = fit_linear_probe_softmax_head(
                        net, train_loader_all,
                        class_offset=start_seen, num_classes=num_classes, device=model._device,
                        epochs=int(args.get("probe_fit_epochs", 50)),
                        lr=float(args.get("probe_fit_lr", 5e-3)),
                        weight_decay=float(args.get("probe_fit_wd", 0.0)),
                        batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                        max_train_batches=args.get("probe_train_max_batches", None),
                        monitor_loader=test_loader_all,
                        monitor_max_batches=args.get("probe_test_max_batches", None),
                        log_interval=probe_log_interval,
                        eval_interval=probe_eval_interval,
                        log_prefix="[LP-Softmax][Final]",
                    )
                    acc_final = evaluate_linear_probe_softmax_with_head(
                        head_final, net, test_loader_all,
                        class_offset=start_seen, device=model._device,
                        max_test_batches=args.get("probe_test_max_batches", None)
                    )
                            
                else:
                    head_final = fit_linear_probe_softmax_head(
                        net, train_loader_all,
                        class_offset=start_seen, num_classes=num_classes, device=model._device,
                        epochs=int(args.get("probe_fit_epochs", 50)),
                        lr=float(args.get("probe_fit_lr", 5e-3)),
                        weight_decay=float(args.get("probe_fit_wd", 0.0)),
                        batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                        max_train_batches=args.get("probe_train_max_batches", None),
                        monitor_loader=test_loader_all,
                        monitor_max_batches=args.get("probe_test_max_batches", None),
                        log_interval=probe_log_interval,
                        eval_interval=probe_eval_interval,
                        log_prefix="[LP-Softmax][Final]",
                    )
                    acc_final = evaluate_linear_probe_softmax_with_head(
                        head_final, net, test_loader_all,
                        class_offset=start_seen, device=model._device,
                        max_test_batches=args.get("probe_test_max_batches", None)
                    )

                if str(args.get("model_name", "")).lower() != "tuna":
                    base_view = _FeatureView(getattr(net, "backbone", net), which="base")
                    base_view.eval()

                    head_base = fit_linear_probe_softmax_head(
                        base_view, train_loader_all,
                        class_offset=start_seen, num_classes=num_classes, device=model._device,
                        epochs=int(args.get("probe_fit_epochs", 50)),
                        lr=float(args.get("probe_fit_lr", 5e-3)),
                        weight_decay=float(args.get("probe_fit_wd", 0.0)),
                        batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                        max_train_batches=args.get("probe_train_max_batches", None),
                        monitor_loader=test_loader_all,
                        monitor_max_batches=args.get("probe_test_max_batches", None),
                        log_interval=probe_log_interval,
                        eval_interval=probe_eval_interval,
                        log_prefix="[LP-Softmax][Base]",
                    )
                    acc_base = evaluate_linear_probe_softmax_with_head(
                        head_base, base_view, test_loader_all,
                        class_offset=start_seen, device=model._device,
                        max_test_batches=args.get("probe_test_max_batches", None)
                    )
                else:
                    acc_base = 0.0


                logging.info("[LP-JointSeen@ALL] final-model Acc=%.2f | base-model Acc=%.2f", acc_final, acc_base)
                write_step_metrics(
                    json_path=metrics_path,
                    section="probe_softmax_joint_seen_all",
                    step=task,
                    metrics={"final_model": float(acc_final), "base_model": float(acc_base)},
                    matrix=None,
                    json_safe=json_safe,
                )
                write_final_metrics(
                    metrics_path, "probe_softmax_joint_seen_all",
                    final_metrics={"final_model": float(acc_final), "base_model": float(acc_base)},
                    final_matrix=None,
                    json_safe=json_safe,
                )

        finally:
            # 1)  Module  CPU ，
            for _m in ["head_final", "head_base", "base_view"]:
                if _m in locals() and locals()[_m] is not None:
                    try:
                        mod = locals()[_m]
                        if isinstance(mod, torch.nn.Module):
                            with torch.inference_mode():
                                mod.to("cpu")
                            for p in mod.parameters():
                                p.grad = None
                    except Exception:
                        pass
            head_final = None
            head_base = None
            base_view = None

            # 2)  DataLoader/Dataset （worker  persistent_workers=False ）
            train_loader_all = None
            test_loader_all = None
            train_dataset_all = None
            test_dataset_all = None

            # 3) /
            if 'net' in locals() and was_training:
                    net.train(True)
            

            # 4)  GC + CUDA 
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
 


        

        do_flat_eval = bool(args.get("flat_eval", False)) and (
            flat_eval_tasks is None or task in flat_eval_tasks
        )
        do_feature_eval = bool(args.get("feature_flat_eval", False)) and (
            feature_eval_tasks is None or task in feature_eval_tasks
        )
        # Evaluate flatness/feature metrics per task (optional)
        if do_flat_eval or do_feature_eval:

            # ---- shared temp handles (visible to finally) ----
            train_loader = getattr(model, "train_loader", None)
            flat_loader = None
            seen_test_loader = None
            dataset = None
            flat_cfg = None
            feature_cfg = None
            flat_metrics = None
            feature_metrics = None
            

            # unwrap to raw nn.Module and remember training flag
            net = getattr(model, "_network", model)
            if hasattr(net, "module"):
                net = net.module
            was_training = net.training

            # paths / tags for saving (as in your original code)
            feature_dir = os.path.join(log_dir, "feature_flatness")
            os.makedirs(feature_dir, exist_ok=True)
            base_stub = os.path.basename(logfilename)
            step_tag = f"t{task:02d}"
            save_prefix = f"{base_stub}_{step_tag}"
            is_main = bool(getattr(model, "_is_main_process", True))

            start_seen = class_ranges[task][0]
            end_seen = class_ranges[task][1]
            num_classes =  end_seen - start_seen 

            try:
                # ----------------- build the shared flat_loader once -----------------
                if train_loader is not None:
                    data_source = str(args.get("flat_eval_data_source", "train")).lower()
                    if data_source not in {"train", "test"}:
                        logging.warning(
                            "[FlatEval] Unknown flat_eval_data_source=%s, fallback to test", data_source
                        )
                        # data_source = "test"
                    data_mode = "train"  # if data_source == "train" else "test"
                    dataset_seen = data_manager.get_dataset(
                        np.arange(start_seen, end_seen),
                        source=data_source,
                        mode=data_mode,
                    )
                    loader_seen = DataLoader(
                        dataset_seen, 
                        batch_size=args.get("flat_eval_batch_size", 32), 
                        shuffle=True, num_workers=0)


                    flat_loader = fractional_loader(
                        loader=loader_seen,
                        fraction=args.get("flat_eval_dataset_fraction", 0.1),
                        seed=args.get("flat_eval_dataset_fraction_seed", args.get("seed", 42)),
                        balanced=True,
                        batch_size=args.get("flat_eval_batch_size", 32)
                    )

                # ================= weight-space flatness =================
                if do_flat_eval and (flat_loader is not None):
                    net.eval()  # switch to eval for metric extraction

                    flat_cfg = FlatnessConfig(
                        args=args,
                        save_metrics_path=os.path.join(log_dir, "flatness"),
                        save_prefix=f"{os.path.basename(logfilename)}_t{task:02d}",
                    )

                    # #  requires_grad=True，
                    _saved_requires = [(p, bool(p.requires_grad)) for _, p in net.named_parameters()]
                    # for _, _p in net.named_parameters():
                    #     if not _p.requires_grad:
                    #         _p.requires_grad_(True)
                    try:
                        # Explicitly pass the parameter list we want to probe (selected by substrings in config)
                        flat_params = _select_params_by_name(
                            net,
                            getattr(flat_cfg, "param_name_substrings", None),
                            include_frozen=bool(flat_cfg.include_frozen_params),
                        )
                        flat_metrics = evaluate_flatness_metrics(
                            net,
                            flat_loader,
                            device=model._device,
                            config=flat_cfg,
                            params_override=flat_params,
                        )
                    finally:
                        #  requires_grad 
                        for _p, _old in _saved_requires:
                            _p.requires_grad_(_old)

                    logging.info("Flatness metrics (task %d): %s", task, flat_metrics)

                # ================= feature-space flatness =================
                if do_feature_eval and (flat_loader is not None):
                    device_override = getattr(model, "_device", None)
                    if isinstance(device_override, str):
                        device_override = torch.device(device_override)

                    feature_cfg = FeatureFlatnessConfig(
                        args=args,
                        save_matrix_path=feature_dir,
                        save_prefix=save_prefix,
                        device_override=device_override,
                        max_examples_per_batch=args.get("feature_flat_max_examples_per_batch", None),
                    )

                    feature_metrics = evaluate_feature_metrics(
                        model._network,
                        flat_loader,                 #  flat_loader
                        config=feature_cfg,
                    )
                    logging.info("Feature flatness metrics (task %d): %s", task, feature_metrics)

                    if do_feature_eval and args.get("attention_probe_eval", False):
                        try:
                            run_attention_probe(
                                model=model,
                                net=net,
                                data_manager=data_manager,
                                args=args,
                                log_dir=log_dir,
                                save_prefix=save_prefix,
                                device_override=device_override,
                            )
                        except Exception:
                            logging.exception("[AttentionProbe] Failed to export probe payload")

                    # ---- atomic write of feature metrics ----
                    if do_feature_eval and is_main:
                        final_path = os.path.join(feature_dir, f"{save_prefix}_feature_flatness.json")
                        tmp_path = final_path + ".tmp"
                        with open(tmp_path, "w", encoding="utf-8") as fh:
                            json.dump(json_safe(feature_metrics), fh, ensure_ascii=False, indent=2)
                        os.replace(tmp_path, final_path)

            finally:
                # ===== single cleanup block (memory + handles) =====
                try:
                    # restore training state
                    if was_training:
                        net.train(True)
                    # drop big tensors/objects
                    X_seen = None; y_seen = None
                    prot = None; counts = None
                    flat_cfg = None; feature_cfg = None
                    flat_metrics = None; feature_metrics = None
                    # release loaders/datasets (do NOT touch model.train_loader)
                    seen_test_loader = None
                    dataset = None
                    flat_loader = None
                    # zero grads to free storage
                    if hasattr(net, "zero_grad"):
                        net.zero_grad(set_to_none=True)
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

       
        # ---- Cache per-task anchors/prototypes for later comparisons ----
        if args.get("feature_cka_eval", False) or args.get("feature_proto_eval", False):
            def _parse_task_list(val):
                if val is None:
                    return None
                if isinstance(val, (list, tuple)):
                    try:
                        return [int(x) for x in val]
                    except Exception:
                        return None
                if isinstance(val, int):
                    return [val]
                if isinstance(val, str):
                    s = val.strip()
                    try:
                        if s.startswith("[") and s.endswith("]"):
                            import json as _json
                            return [int(x) for x in _json.loads(s)]
                    except Exception:
                        return None
                    try:
                        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
                    except Exception:
                        return None
                return None

            # ---- unified feature cache + drift evaluation control ----
            drift_mode = str(
                args.get(
                    "feature_drift_mode",
                    "last_only",
                )
            ).lower()
            if bool(args.get("feature_drift_last_only", False)):
                drift_mode = "last_only"
            if drift_mode not in {"per_task", "last_only", "none"}:
                drift_mode = "per_task"

            cache_tasks = _parse_task_list(args.get("feature_cache_tasks", [0]))
            cache_enabled = bool(args.get("feature_cka_eval", False) or args.get("feature_proto_eval", False))

            if cache_enabled and (cache_tasks is None or task in cache_tasks):
                try:
                    device_override = getattr(model, "_device", None)
                    if isinstance(device_override, str):
                        device_override = torch.device(device_override)
                    base_stub = os.path.basename(logfilename)
                    overwrite = bool(args.get("feature_overwrite_first", False)) if task == 0 else False
                    save_task_feature_cache(
                        model._network,
                        data_manager,
                        class_ranges,
                        task_idx=task,
                        log_dir=log_dir,
                        base_stub=base_stub,
                        args=args,
                        device=device_override or model._device,
                        overwrite=overwrite,
                    )
                except Exception:
                    logging.exception("[FeatureFlat] Saving per-task cache failed")

            if drift_mode == "per_task" and task >= 1 and bool(args.get("feature_drift_eval", False)):
                try:
                    from evaluation_feature.feature_drift import compute_first_vs_current_feature_drift

                    compute_first_vs_current_feature_drift(
                        model=model,
                        data_manager=data_manager,
                        class_ranges=class_ranges,
                        args=args,
                        task=task,
                        log_dir=log_dir,
                        logfilename=logfilename,
                    )
                except Exception:
                    logging.exception("[FIRST-vs-CUR] feature drift failed")

            if drift_mode == "last_only" and task == nb_tasks - 1:
                try:
                    device_override = getattr(model, "_device", None)
                    if isinstance(device_override, str):
                        device_override = torch.device(device_override)
                    base_stub = os.path.basename(logfilename)

                    # Ensure last cache exists
                    save_task_feature_cache(
                        model._network,
                        data_manager,
                        class_ranges,
                        task_idx=task,
                        log_dir=log_dir,
                        base_stub=base_stub,
                        args=args,
                        device=device_override or model._device,
                        overwrite=bool(args.get("feature_overwrite_last", False)),
                    )

                    # Decide which refs to compare
                    ref_tasks = cache_tasks if cache_tasks is not None else [0]
                    for ref in ref_tasks:
                        if ref is None:
                            continue
                        try:
                            ref = int(ref)
                        except Exception:
                            continue
                        if ref < 0 or ref >= nb_tasks:
                            continue
                        if ref == task:
                            continue
                        save_task_feature_cache(
                            model._network,
                            data_manager,
                            class_ranges,
                            task_idx=ref,
                            log_dir=log_dir,
                            base_stub=base_stub,
                            args=args,
                            device=device_override or model._device,
                            overwrite=bool(args.get("feature_overwrite_first", False)) if ref == 0 else False,
                        )
                        compare_task_last_features(
                            model._network,
                            data_manager,
                            class_ranges,
                            ref_task_idx=ref,
                            last_task_idx=task,
                            log_dir=log_dir,
                            base_stub=base_stub,
                            args=args,
                            device=model._device,
                        )
                except Exception:
                    logging.exception("[FIRST-vs-LAST] feature comparison failed")


        # ===================== Last-step feature comparison handled above =====================
    
    # 4.5) OOD evaluation on Tiny-ImageNet-* (all classes)
    #      Placed after CL training loop and before final summaries.
    try:
        if False:
            run_ood_evaluation(model, args, metrics_path=metrics_path)
    except Exception as _ood_exc:
        logging.exception("[OOD] Evaluation failed: %s", _ood_exc)

    # 5) Summaries
    # ---------------------------
    # Save the final trained model once (backbone + head)
    try:
        if bool(args.get("save_final_model", False)):
            net_to_save = getattr(model, "_network", None)
            if net_to_save is not None:
                if hasattr(net_to_save, "module"):
                    net_to_save = net_to_save.module
                final_ckpt = {
                    "tasks": int(nb_tasks - 1),
                    "model_state_dict": net_to_save.state_dict(),
                }
                final_path = os.path.join(checkpoint_dir, "final_model.pt")
                torch.save(final_ckpt, final_path)
                logging.info("Saved final model to %s", final_path)
    except Exception as _save_exc:
        logging.exception("[FinalSave] Failed to save final model: %s", _save_exc)

    #  2) CNN / NME 
    #  CNN / NME 
    if True:
        T = task + 1
        run_dir = log_dir
        run_stub = os.path.basename(logfilename)

        if len(cnn_matrix) > 0:
            cnn_time_by_task = assemble_eval_matrix(cnn_matrix, T, orientation="time_by_task")
            # cnn_task_by_time = cnn_time_by_task.T
            log_eval_matrix(cnn_time_by_task, "CNN Evaluation", orientation="time_by_task")
            # log_matrix(cnn_task_by_time, "CNN Evaluation", orientation="task_by_time")
            forgetting = np.nanmean((np.nanmax(cnn_time_by_task, axis=0) - cnn_time_by_task[T-1, :])[:T-1])
            logging.info('Forgetting (CNN): %s', forgetting)
            if args.get("save_legacy_artifacts", False):
                save_eval_matrix(cnn_time_by_task, run_dir, run_stub, "cnn_R_time_by_task")
                # save_matrix(cnn_task_by_time, run_dir, run_stub, "cnn_R_task_by_time")

            #  JSON（）
            write_final_metrics(
                metrics_path, "cnn",
                final_metrics=compute_sequence_metrics(cnn_time_by_task),
                final_matrix=cnn_time_by_task,
                json_safe=json_safe,
            )

        if len(nme_matrix) > 0:
            nme_time_by_task = assemble_eval_matrix(nme_matrix, T, orientation="time_by_task")
            # nme_task_by_time = nme_time_by_task.T
            log_eval_matrix(nme_time_by_task, "NME Evaluation", orientation="time_by_task")
            # log_matrix(nme_task_by_time, "NME Evaluation", orientation="task_by_time")
            forgetting = np.nanmean((np.nanmax(nme_time_by_task, axis=0) - nme_time_by_task[T-1, :])[:T-1])
            logging.info('Forgetting (NME): %s', forgetting)
            if args.get("save_legacy_artifacts", False):
                save_eval_matrix(nme_time_by_task, run_dir, run_stub, "nme_R_time_by_task")
                # save_matrix(nme_task_by_time, run_dir, run_stub, "nme_R_task_by_time")

            #  JSON（）
            write_final_metrics(
                metrics_path, "nme",
                final_metrics=compute_sequence_metrics(nme_time_by_task),
                final_matrix=nme_time_by_task,
                json_safe=json_safe,
            )
