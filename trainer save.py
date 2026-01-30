import sys
import logging
import copy
import torch
from torch.utils.data import DataLoader
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from evaluation.metrics import compute_sequence_metrics, save_metrics_and_vectors
from eval_flat.eval_flat_feature import (
    extract_features_and_labels,
    linear_cka,
)
import os
import numpy as np
from eval_flat.eval_flatness_weight_Loss import FlatnessConfig, evaluate_flatness_metrics
import gc, torch
from eval_flat.eval_flat_feature import FeatureFlatnessConfig, evaluate_feature_metrics
from typing import  Optional
import json 
from utils.data_manager import fractional_loader
from evaluation.probe import  LinearProbeConfig, LinearProbeRunner,fit_linear_probe_softmax_head,evaluate_linear_probe_softmax_with_head,_FeatureView
from utils.random_reproduce import set_device,set_random,json_safe



def build_eval_matrix(seq_rows, T, orientation="time_by_task"): #, orientation="task_by_time"
    """
    将按训练步 i 累积的行列表 seq_rows（第 i 行长度 i+1）组装为方阵。
    orientation:
      - "time_by_task": R[i, j]  (行=训练步i, 列=任务j)  -> 下三角
    """
    M = np.full((T, T), np.nan, dtype=float)
    for i, row in enumerate(seq_rows):
        if len(row) > 0:
            M[i, :len(row)] = np.array(row, dtype=float)
    # if orientation == "task_by_time":
    #     return M.T
    return M

def log_matrix(M, name, orientation="time_by_task"):
    """
    打印并记录矩阵，明确行/列语义。
    name: "CNN Evaluation" | "NME Evaluation" | "Head Evaluation" | "Linear Probe Evaluation"
    orientation: "time_by_task" or "task_by_time"
    """
    print("\nAccuracy Matrix ({} | {}):".format(name, orientation))
    print("=" * 72)
    if orientation == "time_by_task":
        # 经典定义：R[i, j]，行=学到第 i 个任务后的模型，列=任务 j
        print("- 行 i：学习进度（学到第 i 个任务后，0-based）")
        print("- 列 j：任务编号（0-based）")
        print("- 含义：R[i, j] = 学到 i 后在任务 j 上的准确率（仅 j≤i 有效，为下三角）")
    # else:
    #     # 转置视图：R[j, i]，行=任务 j，列=学习进度 i
    #     print("- 行 j：任务编号（0-based）")
    #     print("- 列 i：学习进度（学到第 i 个任务后，0-based）")
    #     print("- 含义：R[j, i] = 学到 i 后在任务 j 上的准确率（仅 i≥j 有效，为上三角）")
    print("=" * 72)
    print(np.array2string(M, precision=2, suppress_small=True))
    logging.info("\nAccuracy Matrix (%s | %s):\n%s", name, orientation, M)

def save_matrix(M, run_dir, run_stub, tag):
    npy_path = os.path.join(run_dir, f"{run_stub}_{tag}.npy")
    csv_path = os.path.join(run_dir, f"{run_stub}_{tag}.csv")
    np.save(npy_path, M)
    np.savetxt(csv_path, M, delimiter=",", fmt="%.6f")
    logging.info("Saved %s to %s and %s", tag, npy_path, csv_path)



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
    
    # init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]

    # ——— root: logs/<model>/<dataset> ———
    if "lora" in str(args["model_name"]):
        opt_tag = f'{args["optimizer_type"]}_{args.get("rwp_range")}' if args["optimizer_type"]=="rwp" else str(args["optimizer_type"])
        if args["optimizer_type"]=="rwp" and args.get("rwp_range") =="full": 
            mode = "rwp_full" 
        elif args["optimizer_type"]=="rwp" and args.get("rwp_range") =="lora":
            mode = "rwp_redo"
        else: 
            mode = "exp_joost3"
        logs_root = os.path.join("logs_inc_lora", str(args["model_name"]), opt_tag, str(args["dataset"]), str(args["seed"]), str(args["prefix"]), mode)
    else:
        opt_tag = f'{args["optimizer_type"]}_{args.get("rwp_range")}' if args["optimizer_type"]=="rwp" else str(args["optimizer_type"])
        if args["optimizer_type"]=="rwp" and args.get("rwp_range") =="full": 
            mode = "rwp_full" 
        elif args["optimizer_type"]=="rwp" and args.get("rwp_range") =="lora":
            mode = "rwp_redo"
        else: 
            mode = "exp_ablation"
        
        logs_root = os.path.join("logs_inc", str(args["model_name"]), opt_tag, str(args["dataset"]), str(args["seed"]), str(args["prefix"]), mode)

    log_dir = os.path.join(logs_root, str(args["increment"]))
    logfilename = os.path.join(log_dir, f'{args["prefix"]}_{args["backbone_type"]}')

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

    # === 新增：阶段/最终统一 JSON 保存工具（其余逻辑不变） ===
    def _metrics_book_path(_log_dir: str, _logfilename: str) -> str:
        return os.path.join(_log_dir, f"{os.path.basename(_logfilename)}_cl_metrics.json")

    def _append_stage_metrics(json_path: str, section: str, step: int, metrics: Optional[dict] = None, R_time_by_task=None):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        J = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    J = json.load(f)
            except Exception:
                J = {}
        S = J.setdefault(section, {})
        if metrics is not None:
            steps = S.setdefault("steps", {})
            steps[str(step)] = metrics
        if R_time_by_task is not None:
            mats = S.setdefault("matrices", {})
            mats[f"t{step:02d}"] = np.asarray(R_time_by_task, dtype=float).tolist()
        tmp = json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(J, f, indent=2, ensure_ascii=False, default=json_safe)
        os.replace(tmp, json_path)

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
    # === 新增部分到此为止 ===

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

    class_ranges = [data_manager.get_task_class_range(task_idx) for task_idx in range(nb_tasks)]
    
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

        # —— CNN/NME 阶段性：仅保存原始矩阵（不保存阶段性指标） ——
        T_partial = task + 1
        if len(cnn_matrix) > 0:
            cnn_time_by_task_partial = build_eval_matrix(cnn_matrix, T_partial, orientation="time_by_task")
            _append_stage_metrics(
                metrics_book_path, "cnn", step=task,
                metrics=None,
                R_time_by_task=cnn_time_by_task_partial
            )

        if nme_accy is not None and len(nme_matrix) > 0:
            nme_time_by_task_partial = build_eval_matrix(nme_matrix, T_partial, orientation="time_by_task")
            _append_stage_metrics(
                metrics_book_path, "nme", step=task,
                metrics=None,
                R_time_by_task=nme_time_by_task_partial
            )

        # ============ Per-step FIRST-vs-CURRENT feature drift (CKA + prototype) ============
        # Compare current-step features/prototypes against FIRST-task references, and save per-step JSONs.
        if task >= 1 and (False):
            try:
                feature_dir = os.path.join(log_dir, "feature_flatness")
                os.makedirs(feature_dir, exist_ok=True)
                base_stub = os.path.basename(logfilename)
                step_tag_cur = f"t{task:02d}"

                # Saved FIRST-step anchors/prototypes
                anchor_first_path = os.path.join(feature_dir, f"{base_stub}_t00_anchors_seen.pt")
                proto_first_path  = os.path.join(feature_dir, f"{base_stub}_t00_prototypes.pt")

                # Optional cached CURRENT-step artifacts (if any)
                anchor_cur_path = os.path.join(feature_dir, f"{base_stub}_{step_tag_cur}_anchors_seen.pt")
                proto_cur_path  = os.path.join(feature_dir, f"{base_stub}_{step_tag_cur}_prototypes.pt")

                # FIRST-step class range (we always compare within these classes)
                start0, end0 = class_ranges[0][0], class_ranges[0][1]

                device_override = getattr(model, "_device", None)
                if isinstance(device_override, str):
                    device_override = torch.device(device_override)

                # ---------------- CKA: t00 vs t{task} ----------------
                if args.get("feature_cka_eval", False):
                    X0 = None; y0 = None
                    if os.path.exists(anchor_first_path):
                        first = torch.load(anchor_first_path, map_location="cpu")
                        X0 = first.get("features", None)
                        y0 = first.get("labels", None)
                        classes0 = first.get("classes", (start0, end0))
                        start0, end0 = int(classes0[0]), int(classes0[1])
                    else:
                        logging.info("[FIRST-vs-CUR][CKA] first-step anchors not found: %s", anchor_first_path)

                    XT = None; yT = None
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
                            # On-the-fly extract CURRENT-step anchors on FIRST classes (test split)
                            dataset_first = data_manager.get_dataset(
                                np.arange(start0, end0), source="test", mode="test"
                            )
                            seen_test_loader_first = DataLoader(dataset_first, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
                            anchor_max_batches = int(args.get("feature_cka_max_batches", 8))
                            anchor_max_samples = int(args.get("feature_cka_max_samples", 2048))
                            XT, yT = extract_features_and_labels(
                                model._network,
                                seen_test_loader_first,
                                device_override or model._device,
                                max_batches=anchor_max_batches,
                                max_samples=anchor_max_samples,
                            )

                        if XT is not None and XT.numel() > 0:
                            n_pairs = min(int(X0.shape[0]), int(XT.shape[0]))
                            if n_pairs > 0:
                                cka_val = linear_cka(
                                    X0[:n_pairs].to(model._device),
                                    XT[:n_pairs].to(model._device),
                                )
                                cka_json = os.path.join(feature_dir, f"{base_stub}_{step_tag_cur}_cka_seen_first.json")
                                with open(cka_json, "w", encoding="utf-8") as fh:
                                    json.dump(
                                        {
                                            "ref_step": 0,
                                            "cur_step": int(task),
                                            "classes": [start0, end0],
                                            "cka_stab_seen_first": float(cka_val),
                                            "n_pairs": int(n_pairs),
                                        },
                                        fh,
                                        indent=2,
                                    )
                                logging.info("[FIRST-vs-CUR][CKA] t00 vs %s: %.4f (pairs=%d)", step_tag_cur, float(cka_val), int(n_pairs))

                # ---------------- Prototype drift: t00 vs t{task} ----------------
                if args.get("feature_proto_eval", False):
                    prot0 = None
                    if os.path.exists(proto_first_path):
                        first = torch.load(proto_first_path, map_location="cpu")
                        prot0 = first.get("prototypes", None) or {}
                        classes0 = first.get("classes", (start0, end0))
                        start0, end0 = int(classes0[0]), int(classes0[1])
                    else:
                        logging.info("[FIRST-vs-CUR][PROTO] first-step prototypes not found: %s", proto_first_path)

                    protT = None
                    if prot0 is not None and len(prot0) > 0:
                        if os.path.exists(proto_cur_path):
                            cur = torch.load(proto_cur_path, map_location="cpu")
                            prot_all = cur.get("prototypes", None) or {}
                            protT = {int(k): v for k, v in prot_all.items() if (int(k) >= start0 and int(k) < end0)}
                        if protT is None or len(protT) == 0:
                            # On-the-fly compute CURRENT-step prototypes on FIRST class set
                            dataset_first = data_manager.get_dataset(
                                np.arange(start0, end0), source="test", mode="test"
                            )
                            seen_test_loader_first = DataLoader(dataset_first, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
                            proto_max_batches = int(args.get("feature_proto_max_batches", int(args.get("feature_cka_max_batches", 8))))
                            proto_max_samples = int(args.get("feature_proto_max_samples", int(args.get("feature_cka_max_samples", 2048))))
                            XTp, yTp = extract_features_and_labels(
                                model._network,
                                seen_test_loader_first,
                                device_override or model._device,
                                max_batches=proto_max_batches,
                                max_samples=proto_max_samples,
                            )
                            protT = {}
                            if XTp is not None and XTp.numel() > 0:
                                for cls in torch.unique(yTp).tolist():
                                    c = int(cls)
                                    if c >= start0 and c < end0:
                                        mask = yTp == c
                                        if int(mask.sum().item()) > 0:
                                            protT[c] = XTp[mask].mean(dim=0).cpu()

                        if protT is not None and len(protT) > 0:
                            l2_list, cos_list, cls_list = [], [], []
                            for cls, mu0 in prot0.items():
                                c = int(cls)
                                if c in protT:
                                    v0 = mu0.to(torch.float32)
                                    vT = protT[c].to(torch.float32)
                                    l2_list.append(torch.norm(v0 - vT, p=2).item())
                                    cos_list.append(1.0 - float(torch.nn.functional.cosine_similarity(v0.unsqueeze(0), vT.unsqueeze(0)).item()))
                                    cls_list.append(c)

                            drift_json = os.path.join(feature_dir, f"{base_stub}_{step_tag_cur}_prototype_drift_first.json")
                            mean_l2 = float(np.mean(l2_list)) if len(l2_list) > 0 else None
                            mean_cos = float(np.mean(cos_list)) if len(cos_list) > 0 else None
                            with open(drift_json, "w", encoding="utf-8") as fh:
                                json.dump(
                                    {
                                        "ref_step": 0,
                                        "cur_step": int(task),
                                        "classes": [start0, end0],
                                        "drift_l2": l2_list,
                                        "drift_cos": cos_list,
                                        "drift_l2_mean": mean_l2,
                                        "drift_cos_mean": mean_cos,
                                        "n_classes": len(cls_list),
                                    },
                                    fh,
                                    indent=2,
                                )
                            logging.info(
                                "[FIRST-vs-CUR][PROTO] t00 vs %s: L2_mean=%s, cos_mean=%s (classes=%d)",
                                step_tag_cur, str(mean_l2), str(mean_cos), len(cls_list)
                            )
            finally:
                try:
                    seen_test_loader_first = None
                    dataset_first = None
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        model.after_task()
        gc.collect(); 
        torch.cuda.empty_cache(); 
        torch.cuda.ipc_collect()
        # ---------------------------
        #先进行保存模型
        # Save the final trained model once (backbone + head)
        try:
            if  (task == nb_tasks - 1) and bool(args.get("save_final_model", True)) :
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

    #

        # 逐任务线性探针（Per-task LP at final step）
        # 在最后一个任务时，为每个历史任务分别训练一个线性探针，并评估“最终模型”在该任务上的表现。
        # 不影响既有功能，且放在联合探针代码块之前。
        try:
            if (task == nb_tasks - 1) and bool(args.get("linear_probe_softmax_per_task_eval", False)):
                net = getattr(model, "_network", model)
                if hasattr(net, "module"):
                    net = net.module
                was_training = net.training
                net.eval()

                set_random(args["seed"])

                # 若是 TUNA，切换到融合后的 adapter（与联合探针保持一致）
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

                # 逐任务进行：对每个任务 j（0..task），只用该任务的类集合训练/评估探针
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

                    # 最终模型（当前最终骨干的特征）
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

                    # 基础特征（base/backbone，不适用于 TUNA）
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

                    # 释放局部对象，避免累计显存
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

                # 记录 JSON（阶段性 + 最终一份）
                avg_final = float(np.nanmean(acc_final_list)) if len(acc_final_list) > 0 else float("nan")
                avg_base  = float(np.nanmean(acc_base_list)) if len(acc_base_list) > 0 else float("nan")
                _append_stage_metrics(
                    metrics_book_path,
                    section="probe_softmax_per_task",
                    step=task,
                    metrics={
                        "final_model_per_task": acc_final_list,
                        "base_model_per_task": acc_base_list,
                        "avg_final": avg_final,
                        "avg_base": avg_base,
                    },
                    R_time_by_task=None,
                )
                _write_final_metrics(
                    metrics_book_path,
                    section="probe_softmax_per_task",
                    final_metrics={
                        "final_model_per_task": acc_final_list,
                        "base_model_per_task": acc_base_list,
                        "avg_final": avg_final,
                        "avg_base": avg_base,
                    },
                    final_matrix=None,
                )
        finally:
            pass

        # # 最后一个任务的时候进行线性探针
        # —— 将原有主体放进 try/finally，保证异常也会清理 ——  
        # task == nb_tasks - 1 and
        try:
            if  (task == nb_tasks - 1) and args.get("linear_probe_softmax_joint_seen_eval", False):
                


                net = getattr(model, "_network", model)
                if hasattr(net, "module"):
                    net = net.module
                # 进入评估态，仅用于抽特征
                was_training = net.training
                net.eval()


                set_random(args["seed"])

                # 1)～5) 维持你的原有逻辑（训练/评估 head、写指标）
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
                # # 约定：TUNA 的“通用 adapter” = cur_task + 1；
                # # 若是首任务或还未 merge（不存在通用 adapter），就回落到 0。
                # universal_id = (self._cur_task + 1) if (getattr(module.backbone, "merge", None) and self._cur_task > 0) else 0
                # module.set_active_adapter(universal_id)   # ★ 关键：切到通用 adapter
                if str(args.get("model_name", "")).lower() == "tuna":
                    module = net.module if hasattr(net, "module") else net
                    bb = getattr(module, "backbone", None)
                    if bb is not None and hasattr(bb, "adapter_list"):
                        # 计算 fused adapter id（eval 时 > len(adapter_list) 触发 merged_adapter）
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
                _append_stage_metrics(
                    json_path=metrics_book_path,
                    section="probe_softmax_joint_seen_all",
                    step=task,
                    metrics={"final_model": float(acc_final), "base_model": float(acc_base)},
                    R_time_by_task=None
                )
                _write_final_metrics(
                    metrics_book_path, "probe_softmax_joint_seen_all",
                    final_metrics={"final_model": float(acc_final), "base_model": float(acc_base)},
                    final_matrix=None
                )

        finally:
            # 1) 将临时 Module 迁回 CPU 并清梯度，然后释放引用
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

            # 2) 释放 DataLoader/Dataset 引用（worker 在 persistent_workers=False 下会自动退出）
            train_loader_all = None
            test_loader_all = None
            train_dataset_all = None
            test_dataset_all = None

            # 3) 恢复主网原训练/评估态
            if 'net' in locals() and was_training:
                    net.train(True)
            

            # 4) 触发 GC + CUDA 显存整理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
 


        # ---- Save FIRST-step anchors & prototypes (for later comparisons) ----
        if task == 0 and (args.get("feature_cka_eval", False) or args.get("feature_proto_eval", False)):
            

            # 基本路径与标记
            feature_dir = os.path.join(log_dir, "feature_flatness")
            os.makedirs(feature_dir, exist_ok=True)
            base_stub  = os.path.basename(logfilename)
            step_tag   = "t00"  # 首步固定为 t00
            save_prefix = f"{base_stub}_{step_tag}"

            # 设备
            device_override = getattr(model, "_device", None)
            if isinstance(device_override, str):
                device_override = torch.device(device_override)

            # 首步“已见类”范围：仅第0任务
            start_seen = class_ranges[0][0]
            end_seen   = class_ranges[0][1]

            # Loader/缓存句柄（便于 finally 统一回收）
            dataset = None
            seen_test_loader = None
            X_seen = None
            y_seen = None
            prot = None
            counts = None

            


            # 可选：若不想反复覆盖首步文件，可通过开关控制
            overwrite_first = bool(args.get("feature_overwrite_first", False))
            anchor_path = os.path.join(feature_dir, f"{save_prefix}_anchors_seen.pt")
            proto_path  = os.path.join(feature_dir, f"{save_prefix}_prototypes.pt")

            try:
                # 只读测试集抽特征（与后续比较保持同分布）
                # 修复：传入完整的类别索引序列而不是区间元组，确保覆盖 [start_seen, end_seen) 的所有类
                dataset = data_manager.get_dataset(
                    np.arange(start_seen, end_seen), source="test", mode="test"
                )
                seen_test_loader = DataLoader(
                    dataset,
                    batch_size=64,
                    shuffle=False,
                    num_workers=0,    # 便于清理，无持久进程
                    pin_memory=False,
                )

                # --------- 保存首步 CKA 锚点 ---------
                if args.get("feature_cka_eval", False) and (overwrite_first or not os.path.exists(anchor_path)):
                    anchor_max_batches = int(args.get("feature_cka_max_batches", 8))
                    anchor_max_samples = int(args.get("feature_cka_max_samples", 2048))
                    X_seen, y_seen = extract_features_and_labels(
                        model._network,
                        seen_test_loader,
                        device_override or model._device,
                        max_batches=anchor_max_batches,
                        max_samples=anchor_max_samples,
                    )
                    torch.save(
                        {"features": X_seen.cpu(), "labels": y_seen.cpu(), "classes": (start_seen, end_seen)},
                        anchor_path,
                    )
                    logging.info("[FeatureFlat][CKA] first-step anchors saved: %s (n=%d)",
                                anchor_path, int(X_seen.size(0)) if X_seen is not None else 0)
                else:
                    logging.info("[FeatureFlat][CKA] first-step anchors skipped (exists=%s, overwrite=%s)",
                                str(os.path.exists(anchor_path)), str(overwrite_first))

                # --------- 保存首步 class prototypes ---------
                if args.get("feature_proto_eval", False) and (overwrite_first or not os.path.exists(proto_path)):
                    # 若未抽过或配额不同，则重抽
                    proto_max_batches = int(args.get("feature_proto_max_batches",
                                                    int(args.get("feature_cka_max_batches", 8))))
                    proto_max_samples = int(args.get("feature_proto_max_samples",
                                                    int(args.get("feature_cka_max_samples", 2048))))

                    need_resample = (
                        X_seen is None
                        or int(args.get("feature_cka_max_batches", 8)) != proto_max_batches
                        or int(args.get("feature_cka_max_samples", 2048)) != proto_max_samples
                    )
                    if need_resample:
                        X_seen, y_seen = extract_features_and_labels(
                            model._network,
                            seen_test_loader,
                            device_override or model._device,
                            max_batches=proto_max_batches,
                            max_samples=proto_max_samples,
                        )

                    if X_seen is not None and X_seen.numel() > 0:
                        prot, counts = {}, {}
                        for cls in torch.unique(y_seen).tolist():
                            mask = y_seen == cls
                            n = int(mask.sum().item())
                            if n > 0:
                                prot[int(cls)] = X_seen[mask].mean(dim=0).cpu()
                                counts[int(cls)] = n
                        torch.save(
                            {"prototypes": prot, "counts": counts, "classes": (start_seen, end_seen)},
                            proto_path,
                        )
                        logging.info("[FeatureFlat][PROTO] first-step prototypes saved: %s (classes=%d)",
                                    proto_path, len(prot))
                else:
                    logging.info("[FeatureFlat][PROTO] first-step prototypes skipped (exists=%s, overwrite=%s)",
                                str(os.path.exists(proto_path)), str(overwrite_first))

            finally:
                # —— 单一清理：释放本段临时对象（不动 model.train_loader）——
                seen_test_loader = None
                dataset = None
                X_seen = None; y_seen = None
                prot = None; counts = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()



        # 评估最后的结果 weight-loss feature-loss
        if task == nb_tasks - 1:

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

            start_seen = class_ranges[0][0]
            end_seen = class_ranges[task][1]
            num_classes =  end_seen - start_seen 

            try:
                # ----------------- build the shared flat_loader once -----------------
                if train_loader is not None:
                    dataset_seen = data_manager.get_dataset(np.arange(start_seen, end_seen), source="test", mode="test")
                    loader_seen = DataLoader(
                        dataset_seen, 
                        batch_size=args.get("flat_eval_batch_size", 32), 
                        shuffle=True, num_workers=0)


                    flat_loader = fractional_loader(
                        loader=loader_seen,
                        fraction=0.1,
                        seed=args.get("seed", 42),
                        balanced=True,
                        batch_size=args.get("flat_eval_batch_size", 32)
                    )

                # ================= weight-space flatness =================
                if args.get("flat_eval", False) and (flat_loader is not None):
                    net.eval()  # switch to eval for metric extraction

                    flat_cfg = FlatnessConfig(
                        args=args,
                        fisher_rao=bool(args.get("fisher_rao", True)),
                        relative_flatness=bool(args.get("relative_flatness", True)),
                        save_metrics_path=os.path.join(log_dir, "flatness"),
                        save_prefix=f"{os.path.basename(logfilename)}_t{task:02d}",
                    )

                    # # 保存并临时将所有参数 requires_grad=True，用于全局二阶评估
                    _saved_requires = [(p, bool(p.requires_grad)) for _, p in net.named_parameters()]
                    # for _, _p in net.named_parameters():
                    #     if not _p.requires_grad:
                    #         _p.requires_grad_(True)
                    try:
                        flat_metrics = evaluate_flatness_metrics(
                            net,
                            flat_loader,
                            device=model._device,
                            config=flat_cfg,
                        )
                    finally:
                        # 恢复原 requires_grad 设置
                        for _p, _old in _saved_requires:
                            _p.requires_grad_(_old)

                    logging.info("Flatness metrics (task %d): %s", task, flat_metrics)

                # ================= feature-space flatness =================
                if args.get("feature_flat_eval", False) and (flat_loader is not None):
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
                        flat_loader,                 # 复用同一 flat_loader
                        config=feature_cfg,
                    )
                    logging.info("Feature flatness metrics (task %d): %s", task, feature_metrics)

                    if args.get("attention_probe_eval", False):
                        # Enhanced attention probe export (DINO-style + optional Grad-CAM)
                        import time
                        from torch.utils.data import Subset
                        import torch.nn.functional as F

                        probe_dir = os.path.join(log_dir, "attention_probe")
                        os.makedirs(probe_dir, exist_ok=True)

                        K = max(1, int(args.get("attention_probe_samples", 64)))
                        seed = int(args.get("attention_probe_seed", 42))
                        stamp = time.strftime("%Y%m%d-%H%M%S")
                        probe_path = os.path.join(probe_dir, f"{save_prefix}_attention_probe_{stamp}.pt")

                        probe_dataset = None
                        probe_loader = None
                        attn_handle = None
                        attn_bwd_handle = None
                        last_attn_cache = {}
                        last_attn_grad = {}

                        try:
                            # Build reproducible subset from test set
                            total_classes = getattr(model, "_total_classes", data_manager.nb_classes)
                            probe_dataset = data_manager.get_dataset(
                                np.arange(0, total_classes),
                                source="test",
                                mode="test",
                            )
                            N = len(probe_dataset)
                            if N == 0:
                                raise StopIteration

                            rng = np.random.RandomState(seed)
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
                            target_device = device_override or getattr(model, "_device", None)
                            if isinstance(target_device, str):
                                target_device = torch.device(target_device)
                            if target_device is None:
                                target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                            probe_inputs = probe_inputs.to(target_device, non_blocking=True)
                            net.eval()

                            backbone = getattr(net, "backbone", net)
                            blocks = getattr(backbone, "blocks", None)

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
                                if blocks is not None and hasattr(blocks[-1], "attn") and hasattr(blocks[-1].attn, "attn_drop"):
                                    attn_mod = blocks[-1].attn.attn_drop
                                    attn_handle = attn_mod.register_forward_hook(_hook_last_attn)
                                    # Optional: enable Grad-CAM on attention weights
                                    if bool(args.get("attention_probe_gradcam", False)):
                                        attn_bwd_handle = attn_mod.register_full_backward_hook(_hook_last_attn_bwd)

                                # Forward pass
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

                            # Build attention maps (DINO-style)
                            attn_heads_up = None
                            attn_avg_up = None
                            ps = None
                            try:
                                attn = last_attn_cache.get("attn", None)  # [B, heads, N, N]
                                if attn is not None and attn.ndim == 4:
                                    def _resolve_patch_size(m):
                                        pe = getattr(m, "patch_embed", None)
                                        ps_ = getattr(pe, "patch_size", None)
                                        return ps_[0] if isinstance(ps_, tuple) else (int(ps_) if ps_ is not None else None)

                                    ps = _resolve_patch_size(backbone)
                                    B, C, H, W = probe_inputs.shape
                                    if ps is None or ps <= 0:
                                        # Fallback: infer from token count (assumes cls token + h*w)
                                        Ntok = attn.shape[-1]
                                        p = int(round((H * W / (Ntok - 1)) ** 0.5)) if Ntok > 1 else 16
                                        ps = max(1, p)

                                    H2, W2 = H - (H % ps), W - (W % ps)
                                    h, w = max(1, H2 // ps), max(1, W2 // ps)

                                    A = attn[:, :, 0, 1:]
                                    A = A.reshape(A.shape[0], A.shape[1], h, w)

                                    A_up = F.interpolate(A, size=(H2, W2), mode="bilinear", align_corners=False)
                                    A_avg = A.mean(dim=1, keepdim=True)
                                    A_avg_up = F.interpolate(A_avg, size=(H2, W2), mode="bilinear", align_corners=False)

                                    def _norm(x):
                                        x_min = x.amin(dim=(-2, -1), keepdim=True)
                                        x_max = x.amax(dim=(-2, -1), keepdim=True)
                                        return (x - x_min) / (x_max - x_min + 1e-8)

                                    attn_heads_up = _norm(A_up).cpu()
                                    attn_avg_up = _norm(A_avg_up).cpu()
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
                            if gradcam_up is not None:
                                payload["gradcam_cls"] = gradcam_up     # [B, 1,    H', W']

                            try:
                                torch.save(payload, probe_path)
                                logging.info(
                                    "[AttentionProbe] Saved %d samples (features%s%s%s) to %s",
                                    int(probe_labels.shape[0]),
                                    ", attn_heads" if "attn_heads" in payload else "",
                                    ", attn_avg" if "attn_avg" in payload else "",
                                    ", gradcam_cls" if "gradcam_cls" in payload else "",
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
                            probe_loader = None
                            probe_dataset = None

                    # ---- atomic write of feature metrics ----
                    if args.get("feature_flat_eval", False) and is_main:
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

       

        # ===================== Last-step FIRST-vs-LAST comparison =====================
        if task == nb_tasks - 1 and (args.get("feature_cka_eval", False) or args.get("feature_proto_eval", False)):
            # -------- paths & tags --------
            feature_dir = os.path.join(log_dir, "feature_flatness")
            os.makedirs(feature_dir, exist_ok=True)
            base_stub = os.path.basename(logfilename)
            step_tag_last = f"t{task:02d}"
            step_tag_first = "t00"

            anchor_first_path = os.path.join(feature_dir, f"{base_stub}_{step_tag_first}_anchors_seen.pt")
            anchor_last_path  = os.path.join(feature_dir, f"{base_stub}_{step_tag_last}_anchors_seen.pt")
            proto_first_path  = os.path.join(feature_dir, f"{base_stub}_{step_tag_first}_prototypes.pt")
            proto_last_path   = os.path.join(feature_dir, f"{base_stub}_{step_tag_last}_prototypes.pt")

            # -------- devices & loaders --------
            device_override = getattr(model, "_device", None)
            if isinstance(device_override, str):
                device_override = torch.device(device_override)

            # 首步类集合（只比较这些类）
            start0, end0 = class_ranges[0][0], class_ranges[0][1]

            # 临时句柄，统一在 finally 清理
            seen_test_loader_first = None   # 末步上、用于“首步类集合”的测试 loader
            dataset_first = None
            X0 = None; y0 = None            # 首步锚点
            XT = None; yT = None            # 末步（按首步类集合）锚点
            prot0 = None; protT = None      # 首步/末步原型字典
            counts0 = None; countsT = None

            try:
                # ---------------------------------------------------------
                # 1) CKA: t00 vs tT  （仅当首步锚点存在；末步锚点不存在时现场抽取）
                # ---------------------------------------------------------
                if args.get("feature_cka_eval", False):
                    if os.path.exists(anchor_first_path):
                        first = torch.load(anchor_first_path, map_location="cpu")
                        X0 = first.get("features", None)
                        y0 = first.get("labels", None)
                        classes0 = first.get("classes", (start0, end0))
                        # 以文件中记录为准，避免索引偏差
                        start0, end0 = int(classes0[0]), int(classes0[1])
                    else:
                        logging.info("[FIRST-vs-LAST][CKA] first-step anchors not found: %s. Skip CKA(first).",
                                    anchor_first_path)

                    if X0 is not None and X0.numel() > 0:
                        # 末步特征：优先用已保存的 tT anchors；若不存在则现场抽取（仅首步类集合）
                        if os.path.exists(anchor_last_path):
                            last = torch.load(anchor_last_path, map_location="cpu")
                            XT_all = last.get("features", None)
                            yT_all = last.get("labels", None)
                            if (XT_all is not None) and (yT_all is not None):
                                # 过滤到首步类集合
                                mask = (yT_all >= start0) & (yT_all < end0)
                                if mask.any():
                                    XT = XT_all[mask]
                                    yT = yT_all[mask]
                        if XT is None or XT.numel() == 0:
                            # 现场抽末步、首步类集合的 test 特征
                            dataset_first = data_manager.get_dataset(
                                np.arange(start0, end0), source="test", mode="test"
                            )
                            seen_test_loader_first = DataLoader(
                                dataset_first, batch_size=64, shuffle=False, num_workers=0, pin_memory=False
                            )
                            anchor_max_batches = int(args.get("feature_cka_max_batches", 8))
                            anchor_max_samples = int(args.get("feature_cka_max_samples", 2048))
                            XT, yT = extract_features_and_labels(
                                model._network,
                                seen_test_loader_first,
                                device_override or model._device,
                                max_batches=anchor_max_batches,
                                max_samples=anchor_max_samples,
                            )

                        # 进行配对并计算 CKA
                        if (XT is not None) and XT.numel() > 0:
                            n_pairs = min(int(X0.shape[0]), int(XT.shape[0]))
                            if n_pairs > 0:
                                cka_first = linear_cka(
                                    X0[:n_pairs].to(model._device),
                                    XT[:n_pairs].to(model._device),
                                )
                                # 写结果
                                cka_json = os.path.join(feature_dir, f"{base_stub}_{step_tag_last}_cka_seen_first.json")
                                with open(cka_json, "w", encoding="utf-8") as fh:
                                    json.dump(
                                        {
                                            "ref_step": 0,
                                            "cur_step": int(task),
                                            "classes": [start0, end0],
                                            "cka_stab_seen_first": float(cka_first),
                                            "n_pairs": int(n_pairs),
                                        },
                                        fh,
                                        indent=2,
                                    )
                                logging.info("[FIRST-vs-LAST][CKA] t00 vs %s: %.4f (pairs=%d)",
                                            step_tag_last, float(cka_first), int(n_pairs))
                            else:
                                logging.info("[FIRST-vs-LAST][CKA] insufficient pairs for CKA: n_pairs=0")

                # ---------------------------------------------------------
                # 2) Prototype drift: t00 vs tT  （仅当首步原型存在；末步原型不存在时现场抽取）
                # ---------------------------------------------------------
                if args.get("feature_proto_eval", False):
                    if os.path.exists(proto_first_path):
                        first = torch.load(proto_first_path, map_location="cpu")
                        prot0 = first.get("prototypes", None) or {}
                        counts0 = first.get("counts", None) or {}
                        classes0 = first.get("classes", (start0, end0))
                        start0, end0 = int(classes0[0]), int(classes0[1])
                    else:
                        logging.info("[FIRST-vs-LAST][PROTO] first-step prototypes not found: %s. Skip PROTO(first).",
                                    proto_first_path)

                    if prot0 is not None and len(prot0) > 0:
                        if os.path.exists(proto_last_path):
                            last = torch.load(proto_last_path, map_location="cpu")
                            protT_all = last.get("prototypes", None) or {}
                            # 仅取首步类集合的原型
                            protT = {int(k): v for k, v in protT_all.items() if (int(k) >= start0 and int(k) < end0)}
                        else:
                            # 现场抽特征并在末步上计算“首步类集合”的原型
                            if seen_test_loader_first is None:
                                dataset_first = data_manager.get_dataset(
                                    np.arange(start0, end0), source="test", mode="test"
                                )
                                seen_test_loader_first = DataLoader(
                                    dataset_first, batch_size=64, shuffle=False, num_workers=0, pin_memory=False
                                )
                            proto_max_batches = int(args.get("feature_proto_max_batches",
                                                            int(args.get("feature_cka_max_batches", 8))))
                            proto_max_samples = int(args.get("feature_proto_max_samples",
                                                            int(args.get("feature_cka_max_samples", 2048))))
                            XTp, yTp = extract_features_and_labels(
                                model._network,
                                seen_test_loader_first,
                                device_override or model._device,
                                max_batches=proto_max_batches,
                                max_samples=proto_max_samples,
                            )
                            protT = {}
                            countsT = {}
                            if XTp is not None and XTp.numel() > 0:
                                for cls in torch.unique(yTp).tolist():
                                    if int(cls) >= start0 and int(cls) < end0:
                                        mask = yTp == cls
                                        n = int(mask.sum().item())
                                        if n > 0:
                                            protT[int(cls)] = XTp[mask].mean(dim=0).cpu()
                                            countsT[int(cls)] = n

                            # 保存末步的 prototypes（仅首步类集合），便于后续直接加载使用
                            try:
                                overwrite_last = bool(args.get("feature_overwrite_last", False))
                            except Exception:
                                overwrite_last = False
                            if len(protT) > 0 and (overwrite_last or not os.path.exists(proto_last_path)):
                                torch.save(
                                    {"prototypes": protT, "counts": countsT, "classes": (start0, end0)},
                                    proto_last_path,
                                )
                                logging.info(
                                    "[FIRST-vs-LAST][PROTO] last-step prototypes saved: %s (classes=%d)",
                                    proto_last_path, len(protT)
                                )

                        # 对齐共同类别并计算 L2 / 1-cos
                        if protT is not None and len(protT) > 0:
                            l2_list, cos_list, cls_list = [], [], []
                            for cls, mu0 in prot0.items():
                                cls = int(cls)
                                if cls in protT:
                                    v0 = mu0.to(torch.float32)
                                    vT = protT[cls].to(torch.float32)
                                    l2_list.append(torch.norm(v0 - vT, p=2).item())
                                    cos_list.append(1.0 - float(torch.nn.functional.cosine_similarity(
                                        v0.unsqueeze(0), vT.unsqueeze(0)).item()))
                                    cls_list.append(cls)

                            drift_json = os.path.join(feature_dir, f"{base_stub}_{step_tag_last}_prototype_drift_first.json")
                            mean_l2 = float(np.mean(l2_list)) if len(l2_list) > 0 else None
                            mean_cos = float(np.mean(cos_list)) if len(cos_list) > 0 else None
                            with open(drift_json, "w", encoding="utf-8") as fh:
                                json.dump(
                                    {
                                        "ref_step": 0,
                                        "cur_step": int(task),
                                        "classes": [start0, end0],
                                        "drift_l2": l2_list,
                                        "drift_cos": cos_list,
                                        "drift_l2_mean": mean_l2,
                                        "drift_cos_mean": mean_cos,
                                        "n_classes": len(cls_list),
                                    },
                                    fh,
                                    indent=2,
                                )
                            logging.info("[FIRST-vs-LAST][PROTO] t00 vs %s: L2_mean=%s, cos_mean=%s (classes=%d)",
                                        step_tag_last, str(mean_l2), str(mean_cos), len(cls_list))
                        else:
                            logging.info("[FIRST-vs-LAST][PROTO] no valid prototypes at last step for first-class set.")

                    # 若上面未生成/保存过，并且文件仍不存在，则在末步上抽取并保存“首步类集合”的 prototypes
                    if not os.path.exists(proto_last_path):
                        try:
                            if seen_test_loader_first is None:
                                dataset_first = data_manager.get_dataset(
                                    np.arange(start0, end0), source="test", mode="test"
                                )
                                seen_test_loader_first = DataLoader(
                                    dataset_first, batch_size=64, shuffle=False, num_workers=0, pin_memory=False
                                )
                            proto_max_batches = int(args.get("feature_proto_max_batches",
                                                            int(args.get("feature_cka_max_batches", 8))))
                            proto_max_samples = int(args.get("feature_proto_max_samples",
                                                            int(args.get("feature_cka_max_samples", 2048))))
                            XTp, yTp = extract_features_and_labels(
                                model._network,
                                seen_test_loader_first,
                                device_override or model._device,
                                max_batches=proto_max_batches,
                                max_samples=proto_max_samples,
                            )
                            protT_fallback, countsT_fallback = {}, {}
                            if XTp is not None and XTp.numel() > 0:
                                for cls in torch.unique(yTp).tolist():
                                    c = int(cls)
                                    if c >= start0 and c < end0:
                                        mask = yTp == c
                                        n = int(mask.sum().item())
                                        if n > 0:
                                            protT_fallback[c] = XTp[mask].mean(dim=0).cpu()
                                            countsT_fallback[c] = n
                            if len(protT_fallback) > 0:
                                torch.save(
                                    {"prototypes": protT_fallback, "counts": countsT_fallback, "classes": (start0, end0)},
                                    proto_last_path,
                                )
                                logging.info(
                                    "[FIRST-vs-LAST][PROTO] last-step prototypes saved (fallback): %s (classes=%d)",
                                    proto_last_path, len(protT_fallback)
                                )
                        except Exception as e:
                            logging.warning("[FIRST-vs-LAST][PROTO] unable to compute/save last-step prototypes (fallback): %s", str(e))

            finally:
                # ---- single cleanup block ----
                seen_test_loader_first = None
                dataset_first = None
                X0 = None; y0 = None
                XT = None; yT = None
                prot0 = None; protT = None
                counts0 = None; countsT = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        
        
        
    # 5) Summaries
    # ---------------------------
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
                num_workers=num_workers
            )

        # Helper: linear probe on an OOD DataManager using all classes
        def _ood_linear_probe(dm, *, tag: str):
            try:
                # Resolve model network and ensure eval mode
                net = getattr(model, "_network", model)
                if hasattr(net, "module"):
                    net = net.module
                was_training = net.training
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

                # All OOD classes, offset=0 in their local indexing
                start_seen = 0
                end_seen = int(dm.nb_classes)
                num_classes = end_seen - start_seen

                # Build OOD train/test loaders for all classes
                train_ds = dm.get_dataset(np.arange(start_seen, end_seen), source="train", mode=args.get("probe_train_mode", "train"))
                test_ds  = dm.get_dataset(np.arange(start_seen, end_seen), source="test",  mode=args.get("probe_test_mode",  "test"))

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

                # Train softmax linear head on final model features
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

                # Train/eval softmax linear head on BASE features (original model)
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

        # Evaluation settings
        ood_bs = int(args.get("ood_eval_batch_size", 128))
        ood_workers = int(args.get("ood_eval_num_workers", 0))
        class_shuffle_ood = args.get("class_shuffle", False)

        # # Tiny-ImageNet-R
        # if bool(args.get("ood_imagener_r", True)):
        #     ood_args_c = copy.deepcopy(args)
        #     ood_args_c["dataset"] = "imagenetr"
        #     dm_c = DataManager("imagenetr", class_shuffle_ood, args["seed"], 200, 0, ood_args_c)
        #     loader_c = _build_ood_loader(dm_c, ood_bs, ood_workers)
        #     acc_c = model.evaluate_full_dataset(loader_c)
        #     # Linear probe on OOD (final vs base)
        #     lp_final, lp_base = _ood_linear_probe(dm_c, tag="Tiny-ImageNet-R")
        #     ood_results["imagenetr"] = {
        #         "top1": float(acc_c.get("top1", 0.0)),
        #         "top5": float(acc_c.get("top5", 0.0)),
        #         "lp_softmax_final": float(lp_final) if lp_final == lp_final else 0.0,
        #         "lp_softmax_base": float(lp_base) if lp_base == lp_base else 0.0,
        #     }
        #     logging.info(
        #         "[OOD][Tiny-ImageNet-R] top1=%.2f | top5=%.2f | LP(final)=%.2f | LP(base)=%.2f",
        #         ood_results["imagenetr"]["top1"], ood_results["imagenetr"]["top5"],
        #         ood_results["imagenetr"].get("lp_softmax_final", float("nan")),
        #         ood_results["imagenetr"].get("lp_softmax_base", float("nan"))
        #     )
        #     # Persist into unified metrics json
        #     _write_final_metrics(metrics_book_path, "imagenetr", final_metrics=ood_results["imagenetr"], final_matrix=None)


        # Tiny-ImageNet-C
        # if bool(args.get("ood_imagener_c", False)):
        #     ood_args_c = copy.deepcopy(args)
        #     ood_args_c["dataset"] = "tiny_imagenetc"
        #     dm_c = DataManager("tiny_imagenetc", class_shuffle_ood, args["seed"], 200, 0, ood_args_c)
        #     loader_c = _build_ood_loader(dm_c, ood_bs, ood_workers)
        #     acc_c = model.evaluate_full_dataset(loader_c)
        #     lp_final, lp_base = _ood_linear_probe(dm_c, tag="Tiny-ImageNet-C")
        #     ood_results["tiny_imagenetc"] = {
        #         "top1": float(acc_c.get("top1", 0.0)),
        #         "top5": float(acc_c.get("top5", 0.0)),
        #         "lp_softmax_final": float(lp_final) if lp_final == lp_final else 0.0,
        #         "lp_softmax_base": float(lp_base) if lp_base == lp_base else 0.0,
        #     }
        #     logging.info(
        #         "[OOD][Tiny-ImageNet-C] top1=%.2f | top5=%.2f | LP(final)=%.2f | LP(base)=%.2f",
        #         ood_results["tiny_imagenetc"]["top1"], ood_results["tiny_imagenetc"]["top5"],
        #         ood_results["tiny_imagenetc"].get("lp_softmax_final", float("nan")),
        #         ood_results["tiny_imagenetc"].get("lp_softmax_base", float("nan"))
        #     )
        #     # Persist into unified metrics json
        #     _write_final_metrics(metrics_book_path, "tiny_imagenetc", final_metrics=ood_results["tiny_imagenetc"], final_matrix=None)


        # Tiny-ImageNet-A
        if bool(args.get("ood_imagener_a", False)):
            ood_args_c = copy.deepcopy(args)
            ood_args_c["dataset"] = "imageneta"
            dm_c = DataManager("imageneta", class_shuffle_ood, args["seed"], 200, 0, ood_args_c)
            loader_c = _build_ood_loader(dm_c, ood_bs, ood_workers)
            acc_c = model.evaluate_full_dataset(loader_c)
            # Linear probe on OOD (final vs base)
            lp_final, lp_base = _ood_linear_probe(dm_c, tag="Tiny-ImageNet-A")
            ood_results["imageneta"] = {
                "top1": float(acc_c.get("top1", 0.0)),
                # "top5": float(acc_c.get("top5", 0.0)),
                "lp_softmax_final": float(lp_final) if lp_final == lp_final else 0.0,
                "lp_softmax_base": float(lp_base) if lp_base == lp_base else 0.0,
            }
            logging.info(
                "[OOD][Tiny-ImageNet-A] top1=%.2f  | LP(final)=%.2f | LP(base)=%.2f",
                ood_results["imageneta"]["top1"], 
                ood_results["imageneta"].get("lp_softmax_final", float("nan")),
                ood_results["imageneta"].get("lp_softmax_base", float("nan"))
            )
            # Persist into unified metrics json
            _write_final_metrics(metrics_book_path, "imageneta", final_metrics=ood_results["imageneta"], final_matrix=None)




        # Tiny-ImageNet-P
        # if bool(args.get("ood_imagenet_p", True)):
        #     ood_args_p = copy.deepcopy(args)
        #     ood_args_p["dataset"] = "tiny_imagenetp"
        #     dm_p = DataManager("tiny_imagenetp", class_shuffle_ood, args["seed"], 200, 0, ood_args_p)
        #     loader_p = _build_ood_loader(dm_p, ood_bs, ood_workers)
        #     acc_p = model.evaluate_full_dataset(loader_p)
        #     lp_final, lp_base = _ood_linear_probe(dm_p, tag="Tiny-ImageNet-P")
        #     ood_results["tiny_imagenetp"] = {
        #         "top1": float(acc_p.get("top1", 0.0)),
        #         "top5": float(acc_p.get("top5", 0.0)),
        #         "lp_softmax_final": float(lp_final) if lp_final == lp_final else 0.0,
        #         "lp_softmax_base": float(lp_base) if lp_base == lp_base else 0.0,
        #     }
        #     logging.info(
        #         "[OOD][Tiny-ImageNet-P] top1=%.2f | top5=%.2f | LP(final)=%.2f | LP(base)=%.2f",
        #         ood_results["tiny_imagenetp"]["top1"], ood_results["tiny_imagenetp"]["top5"],
        #         ood_results["tiny_imagenetp"].get("lp_softmax_final", float("nan")),
        #         ood_results["tiny_imagenetp"].get("lp_softmax_base", float("nan"))
        #     )
        #     # Persist into unified metrics json
        #     _write_final_metrics(metrics_book_path, "ood_tiny_tiny_imagenetp", final_metrics=ood_results["tiny_imagenetp"], final_matrix=None)
    
    
    
    except Exception as _ood_exc:
        logging.exception("[OOD] Evaluation failed: %s", _ood_exc)

    # Save the final trained model once (backbone + head)
    # try:
    #     if bool(args.get("save_final_model", False)):
    #         net_to_save = getattr(model, "_network", None)
    #         if net_to_save is not None:
    #             if hasattr(net_to_save, "module"):
    #                 net_to_save = net_to_save.module
    #             final_ckpt = {
    #                 "tasks": int(nb_tasks - 1),
    #                 "model_state_dict": net_to_save.state_dict(),
    #             }
    #             final_path = os.path.join(checkpoint_dir, "final_model.pt")
    #             torch.save(final_ckpt, final_path)
    #             logging.info("Saved final model to %s", final_path)
    # except Exception as _save_exc:
    #     logging.exception("[FinalSave] Failed to save final model: %s", _save_exc)

    #  2) CNN / NME 汇总矩阵与遗忘
    # 始终在最后阶段输出 CNN / NME 的最终矩阵与指标
    if True:
        T = task + 1
        run_dir = log_dir
        run_stub = os.path.basename(logfilename)

        if len(cnn_matrix) > 0:
            cnn_time_by_task = build_eval_matrix(cnn_matrix, T, orientation="time_by_task")
            # cnn_task_by_time = cnn_time_by_task.T
            log_matrix(cnn_time_by_task, "CNN Evaluation", orientation="time_by_task")
            # log_matrix(cnn_task_by_time, "CNN Evaluation", orientation="task_by_time")
            forgetting = np.nanmean((np.nanmax(cnn_time_by_task, axis=0) - cnn_time_by_task[T-1, :])[:T-1])
            logging.info('Forgetting (CNN): %s', forgetting)
            if args.get("save_legacy_artifacts", False):
                save_matrix(cnn_time_by_task, run_dir, run_stub, "cnn_R_time_by_task")
                # save_matrix(cnn_task_by_time, run_dir, run_stub, "cnn_R_task_by_time")

            # 最终总结写入统一 JSON（含最终矩阵）
            _write_final_metrics(
                metrics_book_path, "cnn",
                final_metrics=compute_sequence_metrics(cnn_time_by_task),
                final_matrix=cnn_time_by_task,
            )

        if len(nme_matrix) > 0:
            nme_time_by_task = build_eval_matrix(nme_matrix, T, orientation="time_by_task")
            # nme_task_by_time = nme_time_by_task.T
            log_matrix(nme_time_by_task, "NME Evaluation", orientation="time_by_task")
            # log_matrix(nme_task_by_time, "NME Evaluation", orientation="task_by_time")
            forgetting = np.nanmean((np.nanmax(nme_time_by_task, axis=0) - nme_time_by_task[T-1, :])[:T-1])
            logging.info('Forgetting (NME): %s', forgetting)
            if args.get("save_legacy_artifacts", False):
                save_matrix(nme_time_by_task, run_dir, run_stub, "nme_R_time_by_task")
                # save_matrix(nme_task_by_time, run_dir, run_stub, "nme_R_task_by_time")

            # 最终总结写入统一 JSON（含最终矩阵）
            _write_final_metrics(
                metrics_book_path, "nme",
                final_metrics=compute_sequence_metrics(nme_time_by_task),
                final_matrix=nme_time_by_task,
            )
