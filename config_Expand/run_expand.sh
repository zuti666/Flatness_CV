#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_Expand"
LOG_DIR="$ROOT/logs_expand"
PYTHON_BIN="${PYTHON_BIN:-/home-local/liying/.conda/envs/Pilot_new_local/bin/python}"

# Redirect multiprocessing/DataLoader tmp files to local disk to avoid
# NFS ".nfs*" busy errors during worker cleanup.
TMP_ROOT="/tmp/$(whoami)/flatness_cv_expand"
mkdir -p "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"

mkdir -p "$LOG_DIR"

CONFIGS=(
  as1_normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr002_e50_rho005_f100.yaml
  as1_normfisher_lam2000_imageneta_t10c20_r10_sgd_lr002_e20_rho005_f100.yaml
  as1_normfisher_lam2000_cifar100_t10c10_r10_sgd_lr002_e20_rho005_f100.yaml
  # as1_normfisher_lam2000_domainnet_t5c69_r30_sgd_lr002_e5_rho005_f100.yaml

)
GPUS=(4 5 6)

printf '========================================\n'
printf '[%s] config_Expand run\n' "$(date)"
printf 'Config dir : %s\n' "$CONFIG_DIR"
printf 'Log dir    : %s\n' "$LOG_DIR"
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

  printf '[%s] Launch  %-60s  on GPU %s\n' "$(date)" "$cfg_name" "$gpu"
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
