"""
run_mech_experiment.py
----------------------
Standalone entry point for mechanism evaluation experiments.

Usage
-----
    conda activate Pilot
    cd /disk0/users/liying/Flatness_CV
    python evaluation_CL_mechanism/run_mech_experiment.py \
        --config config_exps/exps14_mech/ours_mech.yaml

What this does
--------------
1. Monkey-patches utils.factory.get_model to add  model_name: ogd_fisher3_mech
   → so factory.py is NEVER modified.
2. Loads and runs the normal PyCIL-style training pipeline (eval_all.py logic).
3. After training, optionally generates all four figures.

Mechanism-specific YAML keys  (all optional, have defaults):
  model_name            : ogd_fisher3_mech   # required for hook
  cl_mechanism_eval     : true
  cl_eval_max_batches   : 20
  cl_eval_n_noise_samples: 100
  cl_eval_power_iters   : 15
  cl_eval_noise_topk    : 3
  cl_eval_backend       : emp_fisher
  cl_eval_batch_size    : 64

Figure generation (post-training):
  python evaluation_CL_mechanism/run_mech_experiment.py \
      --figures_only \
      --mechanism_roots ours:/path/to/cl_mechanism sam:/path/to/cl_mechanism \
      --cl_metrics_files ours:path1.json,path2.json  \
      --out_dir summaries/mech_figs/
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Make sure project root is on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(filename)s] => %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Step 1: Monkey-patch factory BEFORE any imports that use it ───────────────

def _register_mech_model() -> None:
    """
    Add 'ogd_fisher3_mech' to utils.factory.get_model without modifying
    factory.py.  Safe to call multiple times (idempotent).
    """
    import utils.factory as _factory
    from models_Project2.models_Full.OGD_Fisher3_mech import Learner as _MechLearner

    _orig = _factory.get_model

    def _patched_get_model(model_name: str, args: dict):
        if model_name.lower() in {"ogd_fisher3_mech", "ogd_fisher3_mechanism",
                                   "ogd_rwp_fisher_3_mech"}:
            return _MechLearner(args)
        return _orig(model_name, args)

    # Guard against double-patching
    if not getattr(_factory.get_model, "_mech_patched", False):
        _patched_get_model._mech_patched = True
        _factory.get_model = _patched_get_model
        logger.info("[run_mech] factory patched: 'ogd_fisher3_mech' registered")


# ── Step 2: Training pipeline (mirrors eval_all.py) ──────────────────────────

def run_training(config_path: str) -> None:
    """Load config and run the standard PyCIL training loop."""
    _register_mech_model()

    # Import after patch so get_model is already updated
    import torch
    import numpy as np
    from utils.config import load_config as load_yaml_config
    from utils.data_manager import DataManager
    from utils.random_reproduce import set_random as seed_everything
    from utils.factory import get_model

    args = load_yaml_config(config_path)

    # Device: convert ['0'] → [torch.device('cuda:0')]
    from utils.random_reproduce import set_device
    set_device(args)

    # Seed
    seed = args.get("seed", [1993])
    if isinstance(seed, list):
        seed = seed[0]
    seed = int(seed)
    seed_everything(seed)
    args["seed"] = seed

    logger.info("[run_mech] config=%s  model_name=%s  seed=%d",
                config_path, args.get("model_name", "?"), seed)

    # Data
    data_manager = DataManager(
        dataset_name = args["dataset"],
        shuffle      = args.get("class_shuffle", True),
        seed         = seed,
        init_cls     = args["init_cls"],
        increment    = args["increment"],
        args         = args,
    )

    # Model
    model = get_model(args["model_name"], args)

    # Train task by task
    for task in range(data_manager.nb_tasks):
        model.incremental_train(data_manager)
        model.eval_task()
        model.after_task()

    logger.info("[run_mech] training complete")


# ── Step 3: Figure generation (standalone, no training needed) ────────────────

def run_figures(
    mechanism_roots: dict,
    cl_metrics_files: dict,
    out_dir: str,
) -> None:
    from evaluation_CL_mechanism.visualize import generate_all_figures
    generate_all_figures(mechanism_roots, cl_metrics_files, out_dir)


# ── Figure A  specifically: from existing exps13 results ─────────────────────

def run_figure_A_from_exps13(
    exps13_log_root: str = "outputs_logs/logs_inc",
    out_dir: str = "summaries/mech_figs",
) -> None:
    """
    Generate Figure A (Plasticity–Stability) from the already-completed
    exps13_claude_01 experiments without any new training.

    exps13_log_root : base path containing
                      .../finetune/sgd/imagenetr/{seed}/exps13_*/...
    """
    import glob

    _register_mech_model()   # not strictly needed here but keeps state clean
    from evaluation_CL_mechanism.visualize import figure_A_plasticity_stability

    method_map = {
        "finetune": ["finetune"],
        "ogd":      ["ogd"],
        "gaussian": ["ogd_rwp_gaussian"],
        "sam":      ["sam_ogd"],
        "ours":     ["ogd_fisher3"],
    }
    seeds = ["1993", "42", "2048"]

    cl_metrics_files: dict = {}
    for method_key, dirs in method_map.items():
        paths = []
        for method_dir in dirs:
            for seed in seeds:
                pattern = os.path.join(
                    exps13_log_root, method_dir, "sgd", "imagenetr",
                    seed, "exps13_claude_01_*", "exp_run", "20",
                    "*_cl_metrics.json",
                )
                found = sorted(glob.glob(pattern))
                paths.extend(found)
        if paths:
            cl_metrics_files[method_key] = paths
            logger.info("[run_mech] %s: found %d cl_metrics files", method_key, len(paths))

    os.makedirs(out_dir, exist_ok=True)
    figure_A_plasticity_stability(cl_metrics_files, out_dir, section="cnn")
    figure_A_plasticity_stability(cl_metrics_files, out_dir, section="nme")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Mechanism evaluation runner")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config path for training run")
    p.add_argument("--figures_only", action="store_true",
                   help="Skip training; only generate figures from saved JSONs")
    p.add_argument("--figure_A_only", action="store_true",
                   help="Generate Figure A from existing exps13 results (no new training)")
    p.add_argument("--exps13_log_root", type=str,
                   default="outputs_logs/logs_inc",
                   help="Root for exps13 outputs (for --figure_A_only)")
    p.add_argument("--mechanism_roots", nargs="+", default=[],
                   help="method:path pairs, e.g. ours:/path/to/cl_mechanism")
    p.add_argument("--cl_metrics_files", nargs="+", default=[],
                   help="method:path1,path2 pairs for Figure A")
    p.add_argument("--out_dir", type=str,
                   default="summaries/mech_figs",
                   help="Output directory for figures")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.figure_A_only:
        logger.info("[run_mech] Generating Figure A from exps13 ...")
        run_figure_A_from_exps13(
            exps13_log_root=args.exps13_log_root,
            out_dir=args.out_dir,
        )
        return

    if args.figures_only:
        mechanism_roots = {}
        for item in args.mechanism_roots:
            k, v = item.split(":", 1)
            mechanism_roots[k] = v

        cl_metrics_files = {}
        for item in args.cl_metrics_files:
            k, v = item.split(":", 1)
            cl_metrics_files[k] = v.split(",")

        run_figures(mechanism_roots, cl_metrics_files, args.out_dir)
        return

    if args.config is None:
        logger.error("Must provide --config for training mode")
        sys.exit(1)

    run_training(args.config)


if __name__ == "__main__":
    main()
