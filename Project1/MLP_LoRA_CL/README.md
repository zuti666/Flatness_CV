# MLP–LoRA controlled continual-learning study

This folder is a self-contained small-model testbed for the story developed
from `Project1` and `RebuttalReply`: SAM can remain useful in Factor LoRA because
time-varying factor-chart dynamics reshape the effective-weight path. Mature-state
gauge interventions causally confirm chart sensitivity, while also showing that a
static pullback scalar is not sufficient. Whether lower directional curvature
becomes lower forgetting depends on the accompanying first-order task interference.

The implementation uses one `784 -> 32 -> 32 -> 10` MLP and changes only the
parameterization of its `32 x 32` middle update: Dense, dimension-matched random
subspace, fixed initial/mature LoRA tangent, projected rank manifold, standard
Factor LoRA, or balanced Factor LoRA. It includes strict common Task-A endpoints,
function-preserving gauge interventions, controlled task directions,
SGD/SAM/GAM, exact directional HVP/GGN diagnostics, standard CL metrics, and
current-loss/effective-drift matched analysis.

Start with the detailed [WIKI.md](WIKI.md).

The completed experimental sequence and evidence are summarized in
[WIKI.md](WIKI.md). Finite-difference HVP accuracy is retained as a separate GAM
appendix hypothesis; the SAM results do not establish it.

## Quick start

Run the download-free smoke test:

```bash
cd Project1/MLP_LoRA_CL
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python run_experiment.py --config configs/smoke.yaml
```

Rotated MNIST configs use `data.download: true` by default. A full grid checks
and, when needed, downloads both MNIST splits before launching any training run.
To prepare and validate only the dataset:

```bash
python run_grid.py --config configs/focus_stage1.yaml --prepare-data-only
```

Then run:

```bash
python run_grid.py --config configs/pilot.yaml --dry-run
python run_grid.py --config configs/pilot.yaml
python plot_results.py --input outputs --output figures
python analyze_results.py --input outputs --output analysis
```

Stage-3 path-safety and mature-factor gauge experiments use:

```bash
python analyze_stage3_p0_path_safety.py \
  --input outputs/focus_stage2c_geometry_formal --output analysis/stage3_p0
python run_stage3_p1_mature_gauge.py \
  --config configs/stage3_p1_mature_gauge_formal.yaml --skip-completed
python analyze_stage3_p1_mature_gauge.py \
  --input outputs/stage3_p1_mature_gauge_formal \
  --output analysis/stage3_p1_mature_gauge_formal
python plot_stage3_p0_p1.py \
  --p0 analysis/stage3_p0 --p1 analysis/stage3_p1_mature_gauge_formal \
  --output figures
python run_stage3_p1_mature_gauge.py \
  --config configs/stage3_p2_dynamic_chart_pilot.yaml --skip-completed
python analyze_stage3_p2_dynamic_chart.py \
  --input outputs/stage3_p2_dynamic_chart_pilot \
  --output analysis/stage3_p2_dynamic_chart_pilot
python plot_stage3_p2_dynamic_chart.py \
  --analysis analysis/stage3_p2_dynamic_chart_pilot --output figures
python run_stage3_p1_mature_gauge.py \
  --config configs/stage3_p2b_step_taylor_pilot.yaml --skip-completed
python analyze_stage3_p2b_step_taylor.py \
  --input outputs/stage3_p2b_step_taylor_pilot \
  --output analysis/stage3_p2b_step_taylor_pilot
python run_stage3_p1_mature_gauge.py \
  --config configs/stage3_p2c_every_step_taylor.yaml --skip-completed
python run_stage3_p1_mature_gauge.py \
  --config configs/stage3_p2d_path_matched_gauge_pilot.yaml --skip-completed
python analyze_stage3_p2d_path_matched_gauge.py \
  --input outputs/stage3_p2d_path_matched_gauge_pilot \
  --output analysis/stage3_p2d_path_matched_gauge_pilot
```

One run writes its resolved configuration, accuracy/loss matrices, CL summary,
per-epoch history, exact transition diagnostics, tangent-space geometry, and
checkpoints below `outputs/`.

## Layout

```text
configs/              smoke, pilot, and maximal final grids
small_cl/data.py      Rotated MNIST and deterministic synthetic smoke data
small_cl/models.py    common-base MLP and five controlled parameterizations
small_cl/optimizers.py SGD, SAM, finite-difference GAM, exact-HVP GAM
small_cl/diagnostics.py exact HΔ, ΔᵀHΔ, Taylor terms, reachable-space coverage
small_cl/gauge.py      exact mature-factor gauge transforms and audits
small_cl/metrics.py   FAA, AAA, BWT, forgetting
run_experiment.py     one configuration
run_grid.py           factorial grid expansion
plot_results.py       aggregation and core figures
analyze_results.py    RQ1 predictor, RQ2 safe-route, RQ3 matched-gain tables
run_stage3_p1_mature_gauge.py shared-prefix causal gauge branching
analyze_stage3_p2_dynamic_chart.py multi-batch dynamic-chart tests
tests/                unit and end-to-end smoke tests
```

The synthetic dataset is only for code validation. Paper conclusions must use
Rotated MNIST and the seed protocol in the wiki.
