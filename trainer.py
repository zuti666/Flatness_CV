import sys
import logging
import copy
import torch
from torch.utils.data import DataLoader
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from evaluation.metrics import compute_sequence_metrics
from evaluation.probe import evaluate_linear_probe
import os
import numpy as np


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
    
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])

    # Ensure experiment log directory exists
    os.makedirs(logs_name, exist_ok=True)

    # Store checkpoints and auxiliary artifacts alongside logs for easier tracking
    checkpoint_root = os.path.join("logs", args["model_name"], args["dataset"], str(init_cls))
    os.makedirs(checkpoint_root, exist_ok=True)
    checkpoint_dir = os.path.join(checkpoint_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Downstream components expect a trailing separator when concatenating filenames
    args["filepath"] = checkpoint_dir if checkpoint_dir.endswith(os.sep) else checkpoint_dir + os.sep
    args.setdefault("feature_flat_save_path", checkpoint_dir)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
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

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    
    model = factory.get_model(args["model_name"], args)

    # Ensure the network/backbone is on-device for any pre-training evals
    try:
        # Prefer moving just the backbone to avoid unintended wrapping here
        model._network.backbone.to(model._device)
    except AttributeError:
        # Fallback: move the whole network
        try:
            model._network.to(model._device)
        except Exception:
            pass

    nb_tasks = data_manager.nb_tasks
    class_ranges = [data_manager.get_task_class_range(task_idx) for task_idx in range(nb_tasks)]
    eval_batch_size = args.get("eval_batch_size", args.get("batch_size", 128))
    eval_num_workers = args.get("eval_num_workers", 2)


    # ---------------------------
    # 2) Switches for new metrics
    # ---------------------------
    head_eval = args.get("head_eval", False)
    head_R = np.full((nb_tasks, nb_tasks), np.nan, dtype=float)
    # 线性探针评测矩阵（通过 linear_probe_eval 控制是否启用）
    probe_R = (
        np.full((nb_tasks, nb_tasks), np.nan, dtype=float)
        if args.get("linear_probe_eval", False)
        else None
    )

    # ---------------------------
    # 3) Helpers
    # ---------------------------
    def _build_loader(class_range, source="test", mode="test", shuffle=False):
        dataset = data_manager.get_dataset(
            np.arange(class_range[0], class_range[1]),
            source=source,
            mode=mode,
        )
        return DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=shuffle,
            num_workers=eval_num_workers,
        )

    def _evaluate_head_row(task_idx):
        row = np.full(nb_tasks, np.nan, dtype=float)
        for j in range(task_idx + 1):
            loader = _build_loader(class_ranges[j], source="test", mode="test")
            row[j] = model._compute_accuracy(model._network, loader)
        return row

    def _evaluate_probe_row():
        if probe_R is None:
            return None

        row = np.full(nb_tasks, np.nan, dtype=float)
        train_mode = args.get("probe_train_mode", "test")
        test_mode = args.get("probe_test_mode", "test")
        max_train_batches = args.get("probe_train_max_batches", None)
        max_test_batches = args.get("probe_test_max_batches", None)
        l2_reg = args.get("probe_ridge_lambda", 1e-3)

        for j, (start, end) in enumerate(class_ranges):
            train_loader = _build_loader((start, end), source="train", mode=train_mode)
            test_loader = _build_loader((start, end), source="test", mode=test_mode)

            acc = evaluate_linear_probe(
                model._network,
                train_loader,
                test_loader,
                class_offset=start,
                num_classes=end - start,
                device=model._device,
                l2_reg=l2_reg,
                max_train_batches=max_train_batches,
                max_test_batches=max_test_batches,
            )

            row[j] = acc

        return row

    # 线性探针基线（仅当启用 linear_probe_eval 时才会进入）
    if probe_R is not None and args.get("linear_probe_eval_base", True):
        base_probe_row = _evaluate_probe_row()
        if base_probe_row is not None:
            logging.info(
                "Linear probe baseline accuracies (pre-training): %s",
                np.array2string(
                    base_probe_row,
                    precision=2,
                    formatter={"float_kind": lambda x: f"{x:.2f}"},
                ),
            )


    # ---------------------------
    # 4) Main CL loop
    # ---------------------------
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    logging.info("Start traing CL")
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()

        # ---- Head metrics (gated by head_eval) ----
        if head_eval:
            head_row = _evaluate_head_row(task)
            head_R[task, :] = head_row
            head_metrics = compute_sequence_metrics(head_R[: task + 1, : task + 1])

            logging.info(
                "Head metrics task %d => CA=%.2f, FAA_partial=%.2f",
                task,
                head_metrics["CA"][task],
                head_metrics["FAA"],
            )
            if not np.isnan(head_metrics["BWT"][task]):
                logging.info("Head BWT[%d]=%.2f", task, head_metrics["BWT"][task])
            if not np.isnan(head_metrics["FG"][task]):
                logging.info("Head FG[%d]=%.2f", task, head_metrics["FG"][task])

        # ---- Linear probe metrics (only if enabled) ----
        if probe_R is not None:
            probe_row = _evaluate_probe_row()
            if probe_row is not None:
                probe_R[task, :] = probe_row
                probe_metrics = compute_sequence_metrics(probe_R[: task + 1, :])
                logging.info(
                    "Linear probe metrics task %d => CA=%.2f, FAA_partial=%.2f",
                    task,
                    probe_metrics["CA"][task],
                    probe_metrics["FAA"],
                )
                if not np.isnan(probe_metrics["BWT"][task]):
                    logging.info(
                        "Linear probe BWT[%d]=%.2f",
                        task,
                        probe_metrics["BWT"][task],
                    )
                if not np.isnan(probe_metrics["FG"][task]):
                    logging.info(
                        "Linear probe FG[%d]=%.2f",
                        task,
                        probe_metrics["FG"][task],
                    )

        # ---- Optional: flatness in weight space ----
        if args.get("flat_eval", False):
            from eval_flat.eval_flatness_weight_Loss import FlatnessConfig, evaluate_flatness_metrics

            train_loader = getattr(model, "train_loader", None)
            if train_loader is not None:
                try:
                    flat_cfg = FlatnessConfig(
                        rho=args.get("flat_eval_rho", 0.05),
                        num_random_samples=args.get("flat_eval_num_samples", 10),
                        gaussian_std=args.get("flat_eval_gaussian_std", None),
                        max_batches=args.get("flat_eval_max_batches", 1),
                        power_iters=args.get("flat_eval_power_iters", 5),
                        trace_samples=args.get("flat_eval_trace_samples", 5),
                        grad_batches=args.get("flat_eval_grad_batches", 1),
                        max_examples_per_batch=args.get("flat_eval_max_examples_per_batch", 128),
                        save_metrics_path=args.get("feature_flat_save_path", None),
                        save_prefix=f"task{task}",
                    )
                    # Optional: loss landscape visualization controls
                    setattr(flat_cfg, "loss_land_1d", args.get("loss_land_1d", False))
                    setattr(flat_cfg, "loss_land_2d", args.get("loss_land_2d", False))
                    setattr(flat_cfg, "loss_land_radius", args.get("loss_land_radius", 0.5))
                    setattr(flat_cfg, "loss_land_num_points", args.get("loss_land_num_points", 21))
                    setattr(flat_cfg, "loss_land_max_batches", args.get("loss_land_max_batches", 1))
                    setattr(flat_cfg, "loss_land_filter_norm", args.get("loss_land_filter_norm", True))
                    # attention: this is in train
                    flat_metrics = evaluate_flatness_metrics(
                        model._network,
                        train_loader,
                        device=model._device,
                        config=flat_cfg,
                        known_classes=model._known_classes,
                    )
                    logging.info("Flatness metrics (task %d): %s", task, flat_metrics)
                except Exception as err:
                    logging.exception("Flatness evaluation failed at task %d: %s", task, err)


        # ---- Optional: flatness in feature space ----
        if args.get("feature_flat_eval", False):
            from eval_flat.eval_flat_feature import FeatureFlatnessConfig, evaluate_feature_metrics

            feature_loader = getattr(model, "train_loader", None)
            if feature_loader is not None:
                try:
                    device_override = getattr(model, "_device", None)
                    if isinstance(device_override, str):
                        device_override = torch.device(device_override)
                    feature_cfg = FeatureFlatnessConfig(
                        max_batches=args.get("feature_flat_max_batches", None),
                        topk_eigen=args.get("feature_flat_topk", 5),
                        eps=args.get("feature_flat_eps", 1e-12),
                        rank_tol=args.get("feature_flat_rank_tol", 1e-6),
                        save_matrix_path=args.get("feature_flat_save_path", None),
                        save_prefix=f"task{task}",
                        device_override=device_override,
                    )
                    feature_metrics = evaluate_feature_metrics(
                        model._network,
                        feature_loader,
                        config=feature_cfg,
                    )
                    logging.info(
                        "Feature flatness metrics (task %d): %s", task, feature_metrics
                    )
                except Exception as err:
                    logging.exception(
                        "Feature flatness evaluation failed at task %d: %s", task, err
                    )
        model.after_task()


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
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

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
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    # ---------------------------
    # 5) Summaries
    # ---------------------------
    if head_eval:
        final_head_metrics = compute_sequence_metrics(head_R)
        logging.info(
            "Head summary => ACA=%.2f, ABWT=%s, AFG=%s, FAA=%.2f",
            final_head_metrics["ACA"],
            f"{final_head_metrics['ABWT']:.2f}" if not np.isnan(final_head_metrics["ABWT"]) else "nan",
            f"{final_head_metrics['AFG']:.2f}" if not np.isnan(final_head_metrics["AFG"]) else "nan",
            final_head_metrics["FAA"],
        )

        # Save head_R matrix to files for this run
        try:
            run_dir = logs_name
            run_stub = os.path.basename(logfilename)
            head_npy_path = os.path.join(run_dir, f"{run_stub}_head_R.npy")
            head_csv_path = os.path.join(run_dir, f"{run_stub}_head_R.csv")
            np.save(head_npy_path, head_R)
            np.savetxt(head_csv_path, head_R, delimiter=",", fmt="%.6f")
            logging.info("Saved head_R to %s and %s", head_npy_path, head_csv_path)
        except Exception as err:
            logging.exception("Failed to save head_R matrices: %s", err)

    if probe_R is not None:
        final_probe_metrics = compute_sequence_metrics(probe_R)
        logging.info(
            "Linear probe summary => ACA=%.2f, ABWT=%s, AFG=%s, FAA=%.2f",
            final_probe_metrics["ACA"],
            f"{final_probe_metrics['ABWT']:.2f}" if not np.isnan(final_probe_metrics["ABWT"]) else "nan",
            f"{final_probe_metrics['AFG']:.2f}" if not np.isnan(final_probe_metrics["AFG"]) else "nan",
            final_probe_metrics["FAA"],
        )

        # Save probe_R matrix to files for this run
        try:
            run_dir = logs_name
            run_stub = os.path.basename(logfilename)
            probe_npy_path = os.path.join(run_dir, f"{run_stub}_probe_R.npy")
            probe_csv_path = os.path.join(run_dir, f"{run_stub}_probe_R.csv")
            np.save(probe_npy_path, probe_R)
            np.savetxt(probe_csv_path, probe_R, delimiter=",", fmt="%.6f")
            logging.info("Saved probe_R to %s and %s", probe_npy_path, probe_csv_path)
        except Exception as err:
            logging.exception("Failed to save probe_R matrices: %s", err)

    if args.get("print_forget", False):
        if len(cnn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(cnn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (CNN):')
            print(np_acctable)
            logging.info('Forgetting (CNN): {}'.format(forgetting))
        if len(nme_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (NME):')
            print(np_acctable)
        logging.info('Forgetting (NME): {}'.format(forgetting))


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
