#!/usr/bin/env bash
# FC-split experiment: LoRA lrate=0.04, FC fc_lrate=0.1 (2.5x ratio).
# Validates whether a higher FC-head LR improves per-task learning vs no-split baseline.
#
# Baselines (already done, no fc split):
#   cifar100   lr=0.04  T0=96.85  (logs_expand_lr_equal_grid_v2)
#   imagenetr  lr=0.04  T0=88.84  (logs_expand_lr_equal_suggested)
#   imageneta  lr=0.04  T0=82.17  (logs_expand_lr_equal_grid_v2)
#   domainnet  lr=0.04  T0=80.97  (logs_expand_lr_equal_grid_v2)
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_Expand_fc_split"
LOG_DIR="$ROOT/logs_expand_fc_split"
PYTHON_BIN="${PYTHON_BIN:-/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python}"

export FLATNESS_CV_DATA_ROOT="${FLATNESS_CV_DATA_ROOT:-/data/140-0/datasets}"

TMP_ROOT="/tmp/$(whoami)/flatness_cv_fc_split"
mkdir -p "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"

mkdir -p "$LOG_DIR"

CONFIGS=(
  as1_normfisher_lam2000_cifar100_t10c10_r10_sgd_initlr0p04_lr0p04_fclr0p1_e20_rho005_f100.yaml
  as1_normfisher_lam2000_imagenetr_t10c20_r10_sgd_initlr0p04_lr0p04_fclr0p1_e50_rho005_f100.yaml
  as1_normfisher_lam2000_imageneta_t10c20_r10_sgd_initlr0p04_lr0p04_fclr0p1_e20_rho005_f100.yaml
  as1_normfisher_lam2000_domainnet_t5c69_r30_sgd_initlr0p04_lr0p04_fclr0p1_e5_rho005_f100.yaml
)
GPUS=(0 1 2 3)

if [[ "${#CONFIGS[@]}" -ne "${#GPUS[@]}" ]]; then
  echo "[ERROR] CONFIGS/GPUS length mismatch" >&2
  exit 1
fi

printf '========================================\n'
printf '[%s] FC-split run: LoRA lr=0.04, FC lr=0.1\n' "$(date)"
printf 'Config dir : %s\n' "$CONFIG_DIR"
printf 'Log dir    : %s\n' "$LOG_DIR"
printf 'Data root  : %s\n' "$FLATNESS_CV_DATA_ROOT"
printf 'GPUs       : %s\n' "${GPUS[*]}"
printf '========================================\n'

pids=()
pid_cfgs=()

for i in "${!CONFIGS[@]}"; do
  cfg_name="${CONFIGS[$i]}"
  gpu="${GPUS[$i]}"
  cfg="$CONFIG_DIR/$cfg_name"
  stem="${cfg_name%.yaml}"
  log="$LOG_DIR/${stem}_gpu${gpu}.log"

  if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] Missing config: $cfg" >&2
    exit 1
  fi

  printf '[%s] Launch  %-80s  on GPU %s\n' "$(date)" "$cfg_name" "$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m src.main --config="$cfg" > "$log" 2>&1 &
  pids+=("$!")
  pid_cfgs+=("$cfg_name")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    printf '[%s] ✓ %s\n' "$(date)" "${pid_cfgs[$i]}"
  else
    printf '[%s] ✗ FAILED: %s  (log: %s)\n' "$(date)" "${pid_cfgs[$i]}" \
      "$LOG_DIR/${pid_cfgs[$i]%.yaml}_gpu${GPUS[$i]}.log" >&2
    status=1
  fi
done

printf '========================================\n'
printf 'Logs: %s\n' "$LOG_DIR"
exit "$status"
