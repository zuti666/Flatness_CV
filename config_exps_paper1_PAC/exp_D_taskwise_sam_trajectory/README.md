# Exp D: Task-Wise SAM Trajectory Diagnostic

Purpose: test whether sharpness-aware optimization only improves the task where it is applied, or whether it changes the posterior trajectory and affects retention after the next task.

Default setting:

- Dataset: `cifar10_224`, two tasks, fixed class split `{0,1,2,3,4}` then `{5,6,7,8,9}`.
- Method: `SeqLoRA`, rank 16, ViT-B/16.
- Main perturbation support: raw LoRA factor space `(A,B)`.
- Default perturbation optimizers: `sam_factor` and `random_factor`.
- For each perturbation optimizer, the launcher runs the same four optimizer schedules:
  - `sgd_sgd`: task 1 SGD, task 2 SGD.
  - `<perturb>_sgd`: task 1 perturbation-aware optimizer, task 2 SGD.
  - `sgd_<perturb>`: task 1 SGD, task 2 perturbation-aware optimizer.
  - `<perturb>_<perturb>`: both tasks use the perturbation-aware optimizer.

Default variants:

- `sgd_sgd`
- `sam_factor_sgd`, `sgd_sam_factor`, `sam_factor_sam_factor`
- `random_factor_sgd`, `sgd_random_factor`, `random_factor_random_factor`

Run:

```bash
PYTHON_BIN=/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python \
TASKWISE_GPUS="0 1 2 3" \
bash config_exps_paper1_PAC/exp_D_taskwise_sam_trajectory/run_taskwise_sam_trajectory.sh
```

To use another support for the same trajectory diagnostic:

```bash
TASKWISE_PERTURB_OPTS="sam_delta random_delta" \
bash config_exps_paper1_PAC/exp_D_taskwise_sam_trajectory/run_taskwise_sam_trajectory.sh
```

Summarize:

```bash
python config_exps_paper1_PAC/exp_D_taskwise_sam_trajectory/summarize_taskwise_sam_trajectory.py
```

Core quantities:

- `A_1_1`: task 1 accuracy after task 1.
- `A_1_2`: task 1 accuracy after task 2.
- `A_2_2`: task 2 accuracy after task 2.
- `Current = (A_1_1 + A_2_2) / 2`.
- `Retention = A_1_2`.
- `F1_acc = A_1_1 - A_1_2`.
- `FinalAvg = (A_1_2 + A_2_2) / 2`.

Main contrasts:

- `Delta_T1_flat_A12 = A_1_2(perturb,sgd) - A_1_2(sgd,sgd)`.
- `Delta_T2_protect_A12 = A_1_2(sgd,perturb) - A_1_2(sgd,sgd)`.
- `Interaction_A12 = [A_1_2(perturb,perturb)-A_1_2(perturb,sgd)] - [A_1_2(sgd,perturb)-A_1_2(sgd,sgd)]`.
