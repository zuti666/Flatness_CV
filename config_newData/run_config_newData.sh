#!/usr/bin/env bash
# Run the tuned-base configs collected in config_newData.
#
# Usage:
#   cd /data/140-0/users/liying/Flatness_CV
#   GPUS="0 1 2" bash config_newData/run_config_newData.sh

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_newData"
LOG_DIR="$ROOT/logs_newData/base"
PYTHON_BIN="${PYTHON_BIN:-/home-local/liying/.conda/envs/Pilot_new_local/bin/python}"

TMP_ROOT="/tmp/$(whoami)/flatness_cv_config_newData"
mkdir -p "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"
mkdir -p "$LOG_DIR"

read -r -a GPUS <<< "${GPUS:-0 1 2}"
CFGS=(
    base_as1_normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr001_e50_rho005_f100.yaml
    base_as1_normfisher_lam2000_imageneta_t10c20_r10_sgd_lr001_e30_rho005_f100.yaml
    base_cifar100_t10c10_r10_sgd_lr001_e30.yaml
)

if [[ "${#GPUS[@]}" -lt "${#CFGS[@]}" ]]; then
    echo "[ERROR] Need at least ${#CFGS[@]} GPU ids in GPUS, got: ${GPUS[*]}" >&2
    exit 1
fi

printf '\n========================================\n'
printf '[%s] config_newData runs\n' "$(date)"
printf 'Configs: %s\n' "$CONFIG_DIR"
printf 'Logs:    %s\n' "$LOG_DIR"
printf 'GPUs:    %s\n' "${GPUS[*]}"
printf '========================================\n'

pids=()
pid_cfgs=()

for i in "${!CFGS[@]}"; do
    cfg_name="${CFGS[$i]}"
    gpu="${GPUS[$i]}"
    cfg="$CONFIG_DIR/$cfg_name"
    stem="${cfg_name%.yaml}"
    log="$LOG_DIR/${stem}_gpu${gpu}.log"

    if [[ ! -f "$cfg" ]]; then
        echo "[ERROR] Missing config: $cfg" >&2
        exit 1
    fi

    printf '[%s] Launch  %-72s  GPU %s\n' "$(date)" "$cfg_name" "$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m src.main --config="$cfg" \
        > "$log" 2>&1 &
    pids+=("$!")
    pid_cfgs+=("$cfg_name (GPU $gpu, log: $log)")
done

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        printf '[%s] OK  %s\n' "$(date)" "${pid_cfgs[$i]}"
    else
        printf '[%s] FAILED: %s\n' "$(date)" "${pid_cfgs[$i]}" >&2
        status=1
    fi
done

if [[ $status -ne 0 ]]; then
    echo "[ERROR] One or more runs failed." >&2
    exit 1
fi

printf '\n========================================\n'
printf '[%s] All config_newData runs complete.\n' "$(date)"
printf 'Logs: %s\n' "$LOG_DIR"
printf '========================================\n'
