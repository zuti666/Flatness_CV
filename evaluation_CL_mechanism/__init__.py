"""
evaluation_CL_mechanism
=======================
Online, per-task-boundary mechanistic evaluation for CL experiments.

Purpose
-------
Answer three CL-specific questions that cannot be answered by final accuracy alone:

  1. Does the method preserve *old-task geometry* more than alternatives?
     → S_old = λ_max(H_old),  L_old,  tr(H_old · Σ_noise)

  2. Does the method do so *without crushing new-task learning*?
     → S_new = λ_max(H_new),  L_new

  3. Does it act through *noise–curvature interaction* (not just flat minima)?
     → tr(H_old · Σ_fisher)  vs  tr(H_old · Σ_gaussian)
     → cos(ε, u₁^old)  for Fisher vs Gaussian noise

Design principles
-----------------
- Zero modification to existing code.  All new code lives here.
- Existing modules are imported and called unchanged.
- Evaluation runs *online* at each task boundary inside the training loop.
- Only lightweight JSON artefacts are written to disk; no full checkpoints.
- A subclass ``models_CL/OGD_Fisher3_mech.py`` injects the hook into training.
- A standalone runner ``run_mech_experiment.py`` monkey-patches the model
  factory so no change to ``utils/factory.py`` is needed.

Public API
----------
CLMechanismEvaluator   main orchestrator (called by OGD_Fisher3_mech)
load_task_results      load per-task JSON → dict
summarize_results      aggregate across tasks → summary dict
"""

from evaluation_CL_mechanism.cl_evaluator import CLMechanismEvaluator
from evaluation_CL_mechanism.io_utils import load_task_results, summarize_results

__all__ = [
    "CLMechanismEvaluator",
    "load_task_results",
    "summarize_results",
]
