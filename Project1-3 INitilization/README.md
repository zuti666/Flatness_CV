# PGSR-LoRA / Project 1-3 Initialization

The full algorithm, claim boundaries, GPU protocol, audited results, and next
experiments are documented in [`wiki.md`](wiki.md). The shared-\(W_2\)
fork experiment is summarized in
[`ORIGINAL_IDEA_RESULTS.md`](results/original_idea_validation/analysis/ORIGINAL_IDEA_RESULTS.md),
with machine-readable values in
[`original_idea_results.json`](results/original_idea_validation/analysis/original_idea_results.json).

The completed original-idea run used one prefix GPU and five parallel branch
GPUs:

```bash
PREFIX_GPU=6 BRANCH_GPUS=0,2,3,4,5 \
  bash "Project1-3 INitilization/scripts/run_original_idea_gpu.sh"
```

It trains T1→T2 once, restores the same checkpoint for history0, history1,
fresh, perpendicular, and raw-history branches, and then runs the analyzer.
The launcher refuses to overwrite the archived result. Its provenance snapshot
is under `results/original_idea_launcher/20260808_110739/`.

Checkpoint-level outcome: all five initial predictors were bitwise identical;
the effective \(BA\) trajectories separated after step 1; the history-only raw
selector matched the observed held-out-AUC oracle, but it selected the trivial
latest branch, the main prior-MAP selected fresh, and every candidate finished
with identical T3 Class-IL, FAA, and BWT. This is mechanism evidence, not a
general performance result.

Run the dependency-free component regressions with:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python \
  "Project1-3 INitilization/tests/test_pgsr_regressions.py"
```

The earlier one-path engineering pilot remains archived in
[`GPU_PILOT_SUMMARY.md`](results/GPU_PILOT_SUMMARY.md). It is not the evidence
source for the shared-checkpoint causal comparison.
