import argparse
import os
import json
import logging
import copy
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from utils import factory
from utils.data_manager import DataManager
from eval_flat.loss_landscape import compute_loss_landscape_v1


def _parse_override_pairs(pairs) -> Dict[str, object]:
    if not pairs:
        return {}

    def _convert(value: str):
        lower = value.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower in {"null", "none"}:
            return None
        try:
            if value.startswith("[") or value.startswith("{"):
                return json.loads(value)
        except json.JSONDecodeError:
            pass
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' is not in key=value format")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = _convert(value.strip())
    return overrides


def _device_from_args(args: Dict[str, Any]) -> torch.device:
    dev = args.get("device", "cuda:0")
    if isinstance(dev, (list, tuple)) and dev:
        dev = dev[0]
    return torch.device(dev)


def _build_full_test_loader(dm: DataManager, args: Dict[str, Any]) -> DataLoader:
    indices = np.arange(0, dm.nb_classes)
    ds = dm.get_dataset(indices, source="test", mode="test")
    return DataLoader(
        ds,
        batch_size=int(args.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(args.get("eval_num_workers", 0)),
        persistent_workers=(int(args.get("eval_num_workers", 0)) > 0),
    )


def _default_flatness_dir(method: str, args: Dict[str, Any]) -> str:
    # logs_all/<method>/<dataset>/<seed>/<optimizer>/<prefix>/flatness
    dataset = str(args.get("dataset", "dataset"))
    seed = args.get("seed", 0)
    seed = seed[0] if isinstance(seed, (list, tuple)) and seed else seed
    optimizer = str(args.get("optimizer_type", "sgd"))
    prefix = str(args.get("prefix", "test"))
    return os.path.join("logs_all", method, dataset, str(seed), optimizer, prefix, "flatness")


def _default_inc_lora_flatness_dir(method: str, args: Dict[str, Any]) -> str:
    # logs_inc_lora/<method>/<optimizer>/<dataset>/<seed>/<prefix>/<increment>/flatness
    dataset = str(args.get("dataset", "dataset"))
    seed = args.get("seed", 0)
    seed = seed[0] if isinstance(seed, (list, tuple)) and seed else seed
    optimizer = str(args.get("optimizer_type", "sgd"))
    prefix = str(args.get("prefix", "exp"))
    inc = args.get("increment", 0)
    return os.path.join(
        "logs_inc_lora", method, optimizer, dataset, str(seed), prefix, str(inc), "flatness"
    )


def _default_inc_lora_ckpt_dir(method: str, args: Dict[str, Any]) -> str:
    # logs_inc_lora/<method>/<optimizer>/<dataset>/<seed>/<prefix>/checkpoints
    dataset = str(args.get("dataset", "dataset"))
    seed = args.get("seed", 0)
    seed = seed[0] if isinstance(seed, (list, tuple)) and seed else seed
    optimizer = str(args.get("optimizer_type", "sgd"))
    prefix = str(args.get("prefix", "exp"))
    return os.path.join(
        "logs_inc_lora", method, optimizer, dataset, str(seed), prefix, "checkpoints"
    )


def _build_and_load_lora_model(method: str, args: Dict[str, Any], dm: DataManager) -> Optional[torch.nn.Module]:
    """Instantiate a LoRA learner and restore the final trained snapshot.

    - seqlora: only the last LoRA module is used at eval
    - inclora/sdlora/olora: all LoRA modules up to the last task are composed
    """
    from utils import factory as _factory
    try:
        cfg = copy.deepcopy(args)
        cfg["model_name"] = method

        ckpt_dir = cfg.get("lora_ckpt_dir", None)
        if not ckpt_dir:
            ckpt_dir = _default_inc_lora_ckpt_dir(method, cfg)
        if not ckpt_dir.endswith(os.sep):
            ckpt_dir = ckpt_dir + os.sep
        cfg["filepath"] = ckpt_dir

        learner = _factory.get_model(cfg["model_name"], cfg)

        # Restore the final task snapshot (weights + FC)
        last_task_idx = int(dm.nb_tasks) - 1
        if last_task_idx < 0:
            logging.warning("[eval_all] No tasks found; skip LoRA load.")
            return None
        if hasattr(learner, "restore_task_snapshot"):
            learner.restore_task_snapshot(dm, last_task_idx)

        net = getattr(learner, "_network", learner)
        if hasattr(net, "module"):
            net = net.module
        return net
    except Exception:
        logging.exception("[eval_all] Failed to build/load LoRA model for method=%s", method)
        return None


def _load_full_model_state(net: torch.nn.Module, ckpt_path: str) -> bool:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        logging.warning("[eval_all] Finetune checkpoint missing: %s", ckpt_path)
        return False
    payload = torch.load(ckpt_path, map_location="cpu")
    state = None
    if isinstance(payload, dict):
        state = payload.get("model_state_dict", None) or payload.get("state_dict", None) or payload
    else:
        state = payload
    try:
        missing, unexpected = net.load_state_dict(state, strict=False)
        logging.info(
            "[eval_all] Loaded finetune state: missing=%d unexpected=%d",
            len(missing), len(unexpected),
        )
        return True
    except Exception:
        logging.exception("[eval_all] load_state_dict failed for %s", ckpt_path)
        return False


def _load_linearprobe_fc(
    net: torch.nn.Module,
    fc_dir: Optional[str] = None,
    fc_task_id: Optional[int] = None,
    fc_state_path: Optional[str] = None,
) -> bool:
    # Prefer a full state file if provided
    if fc_state_path and os.path.isfile(fc_state_path):
        try:
            state = torch.load(fc_state_path, map_location="cpu")
            if hasattr(net, "fc") and hasattr(net.fc, "load_state_dict"):
                net.fc.load_state_dict(state)
                logging.info("[eval_all] Loaded FC from %s", fc_state_path)
                return True
        except Exception:
            logging.exception("[eval_all] Failed loading FC state: %s", fc_state_path)

    # Fallback to weight/bias or state in a directory
    if fc_dir and fc_task_id is not None:
        state_path = os.path.join(fc_dir, f"fc_state_{int(fc_task_id)}.pt")
        weight_path = os.path.join(fc_dir, f"CLs_weight{int(fc_task_id)}.pt")
        bias_path = os.path.join(fc_dir, f"CLs_bias{int(fc_task_id)}.pt")
        try:
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu")
                net.fc.load_state_dict(state)
                logging.info("[eval_all] Loaded FC(full) from %s", state_path)
                return True
            if os.path.exists(weight_path):
                weight = torch.load(weight_path, map_location="cpu")
                net.fc.weight.data.copy_(weight.to(net.fc.weight.device))
                if hasattr(net.fc, "bias") and net.fc.bias is not None and os.path.exists(bias_path):
                    bias = torch.load(bias_path, map_location="cpu")
                    net.fc.bias.data.copy_(bias.to(net.fc.bias.device))
                logging.info("[eval_all] Loaded FC(w/b) from %s", fc_dir)
                return True
        except Exception:
            logging.exception("[eval_all] Failed loading FC from dir: %s", fc_dir)
    return False


def eval_all(args: Dict[str, Any]):
    # 1) Logging setup
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [eval_all] %(message)s")

    # 2) DataManager and a test loader that covers all classes
    dm = DataManager(
        args["dataset"],
        args.get("class_shuffle", args.get("shuffle", False)),
        args["seed"][0] if isinstance(args.get("seed"), (list, tuple)) else args.get("seed", 0),
        args.get("init_cls", 200),
        args.get("increment", 0),
        args,
    )
    test_loader = _build_full_test_loader(dm, args)

    device = _device_from_args(args)
    ce = torch.nn.CrossEntropyLoss(reduction="mean")

    # 3) Finetune path (replace entire model)
    try:
        finetune_cfg = copy.deepcopy(args)
        finetune_cfg["model_name"] = finetune_cfg.get("finetune_model_name", "finetune")
        learner_ft = factory.get_model(finetune_cfg["model_name"], finetune_cfg)
        model_ft = getattr(learner_ft, "_network", learner_ft)
        if hasattr(model_ft, "module"):
            model_ft = model_ft.module
        model_ft.to(device)

        ckpt_path = args.get("finetune_ckpt", None)
        if ckpt_path is None:
            # logs_all/<model>/<dataset>/<seed>/<optimizer>/<prefix>/checkpoints/final_model.pt
            default_ckpt_dir = os.path.join(
                "logs_all",
                str(finetune_cfg["model_name"]),
                str(args.get("dataset")),
                str(args.get("seed", [0])[0] if isinstance(args.get("seed"), (list, tuple)) else args.get("seed", 0)),
                str(args.get("optimizer_type", "sgd")),
                str(args.get("prefix", "test")),
                "checkpoints",
            )
            ckpt_path = os.path.join(default_ckpt_dir, "final_model.pt")

        if _load_full_model_state(model_ft, ckpt_path):
            out_dir = args.get("finetune_flatness_dir", _default_flatness_dir("finetune", args))
            os.makedirs(out_dir, exist_ok=True)
            _ = compute_loss_landscape_v1(
                model=model_ft,
                test_loader=test_loader,
                device=device,
                criterion=ce,
                output_dir=out_dir,
                save_file_name="finetune_lossland_2d",
                eval_task_id=0,
                class_incremental=False,
                x_range=(-1.0, 1.0),
                y_range=(-1.0, 1.0),
                num_points=int(args.get("eval_lossland_num_points", 20)),
                max_batches=int(args.get("eval_lossland_max_batches", args.get("flat_eval_loss_max_batches", 5) or 5)),
                sample_batches=False,
                param_name_exclude_substr="shared",
                seed=int(args.get("loss_land_seed", args.get("seed", [42])[0] if isinstance(args.get("seed"), (list, tuple)) else 42)),
            )
        else:
            logging.warning("[eval_all] Skip finetune: checkpoint not loaded.")
    except Exception:
        logging.exception("[eval_all] Finetune evaluation failed")

    # 4) Linear-probe path (replace FC only)
    try:
        lp_cfg = copy.deepcopy(args)
        lp_cfg["model_name"] = lp_cfg.get("linearprobe_model_name", "linearprobe")
        learner_lp = factory.get_model(lp_cfg["model_name"], lp_cfg)
        model_lp = getattr(learner_lp, "_network", learner_lp)
        if hasattr(model_lp, "module"):
            model_lp = model_lp.module
        model_lp.to(device)

        # Ensure FC matches all classes
        try:
            if hasattr(model_lp, "update_fc"):
                model_lp.update_fc(dm.nb_classes)
        except Exception:
            logging.exception("[eval_all] update_fc failed for linear probe model")

        # Load a saved FC if provided
        fc_loaded = _load_linearprobe_fc(
            model_lp,
            fc_dir=args.get("linearprobe_fc_dir", None),
            fc_task_id=args.get("linearprobe_fc_task_id", None),
            fc_state_path=args.get("linearprobe_fc_state_path", None),
        )
        if not fc_loaded:
            logging.warning("[eval_all] No linear-probe FC provided; proceeding with current FC.")

        out_dir = args.get("linearprobe_flatness_dir", _default_flatness_dir("linearprobe", args))
        os.makedirs(out_dir, exist_ok=True)
        _ = compute_loss_landscape_v1(
            model=model_lp,
            test_loader=test_loader,
            device=device,
            criterion=ce,
            output_dir=out_dir,
            save_file_name="linearprobe_lossland_2d",
            eval_task_id=0,
            class_incremental=False,
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
            num_points=int(args.get("eval_lossland_num_points", 20)),
            max_batches=int(args.get("eval_lossland_max_batches", args.get("flat_eval_loss_max_batches", 5) or 5)),
            sample_batches=False,
            param_name_exclude_substr="shared",
            seed=int(args.get("loss_land_seed", args.get("seed", [42])[0] if isinstance(args.get("seed"), (list, tuple)) else 42)),
        )
    except Exception:
        logging.exception("[eval_all] Linear-probe evaluation failed")

    # 5) LoRA path (incremental methods)
    try:
        lora_methods = args.get("lora_eval_methods", None)
        base_model = str(args.get("model_name", "")).lower()
        if lora_methods is None and ("lora" in base_model):
            lora_methods = [base_model]
        if isinstance(lora_methods, str):
            lora_methods = [lora_methods]

        valid_methods = {"seqlora", "inclora", "sdlora", "olora"}
        if lora_methods:
            for m in lora_methods:
                m_low = str(m).lower()
                if m_low not in valid_methods:
                    logging.info("[eval_all] Skip unsupported LoRA method: %s", m)
                    continue

                model_lora = _build_and_load_lora_model(m_low, args, dm)
                if model_lora is None:
                    logging.warning("[eval_all] Skip %s: model not built/loaded.", m_low)
                    continue

                out_dir = args.get(f"{m_low}_flatness_dir", _default_inc_lora_flatness_dir(m_low, args))
                os.makedirs(out_dir, exist_ok=True)
                save_stub = f"{m_low}_lossland_2d"
                _ = compute_loss_landscape_v1(
                    model=model_lora,
                    test_loader=test_loader,
                    device=device,
                    criterion=ce,
                    output_dir=out_dir,
                    save_file_name=save_stub,
                    eval_task_id=dm.nb_tasks - 1,
                    class_incremental=False,
                    x_range=(-1.0, 1.0),
                    y_range=(-1.0, 1.0),
                    num_points=int(args.get("eval_lossland_num_points", 20)),
                    max_batches=int(args.get("eval_lossland_max_batches", args.get("flat_eval_loss_max_batches", 5) or 5)),
                    sample_batches=False,
                    param_name_exclude_substr="shared",
                    seed=int(args.get("loss_land_seed", args.get("seed", [42])[0] if isinstance(args.get("seed"), (list, tuple)) else 42)),
                )
    except Exception:
        logging.exception("[eval_all] LoRA evaluation failed")


def setup_parser():
    p = argparse.ArgumentParser("Evaluate full-dataset loss-landscape (finetune + linear probe + LoRA)")
    p.add_argument("--config", required=True, help="Path to training config JSON/YAML")
    p.add_argument("--override", nargs="+", help="Override config entries via key=value pairs")
    return p


def main():
    ns = setup_parser().parse_args()
    cfg = load_config(ns.config)

    # Merge CLI args into cfg
    cli = vars(ns)
    override_pairs = cli.pop("override", None)
    cfg.update({k: v for k, v in cli.items() if k != "config" and k != "override"})
    cfg["config"] = ns.config
    cfg.update(_parse_override_pairs(override_pairs))

    # Normalize required keys
    if "device" not in cfg:
        cfg["device"] = ["cuda:0"]
    if not isinstance(cfg.get("seed", 42), (list, tuple)):
        cfg["seed"] = [cfg.get("seed", 42)]

    eval_all(cfg)


if __name__ == "__main__":
    main()
