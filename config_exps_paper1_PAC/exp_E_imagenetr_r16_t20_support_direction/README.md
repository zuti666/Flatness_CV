# Exp E: ImageNet-R r16 T20 Support x Direction Main Experiment

Purpose: main ImageNet-R experiment for comparing two perturbation directions across five perturbation locations.

Dataset and model:

- `imagenetr`, 20 tasks x 10 classes.
- `SeqLoRA`, rank 16.
- `ViT-B/16`.
- `class_shuffle: false`.
- seed `1993`.

Variants:

- Baseline: `sgd`.
- SAM direction: `sam_factor`, `sam_full`, `sam_delta`, `sam_all`, `sam_frozen`.
- Gaussian direction: `random_factor`, `random_full`, `random_delta`, `random_all`, `random_frozen`.

The old `sam_random` random-support control is intentionally excluded from this 11-config main set because it is a different support, not one of the five shared perturbation locations.

Run all variants:

```bash
PYTHON_BIN=/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python \
EXP_E_GPUS="0 1 2 3 4 5 6 7" \
bash config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction/run_exp_E_support_direction.sh
```

For faster performance-only training, disable post-hoc flatness evaluation:

```bash
EXP_E_FLAT_EVAL=false \
bash config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction/run_exp_E_support_direction.sh
```

Run a subset:

```bash
EXP_E_VARIANTS="sgd sam_factor random_factor" \
bash config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction/run_exp_E_support_direction.sh
```

Summarize:

```bash
python config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction/summarize_exp_E_support_direction.py
```

Launcher logs:

```text
outputs_logs/exp_E_imagenetr_r16_t20_support_direction_launcher_logs/
```

Training logs and metrics:

```text
outputs_logs/logs_inc_lora/seqlora/<variant>/imagenetr/1993/exp_E_imagenetr_r16_t20_support_direction_seqlora_<variant>/exp_run/10/
```
