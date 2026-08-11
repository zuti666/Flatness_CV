#!/usr/bin/env bash
# Run ImageNet-A gamma=1.0 KL mechanism diagnostics.
#
# Usage:
#   cd /data/140-0/users/liying/Flatness_CV
#   GPUS="3 4" bash config_kl_mechanism_imageneta_g1/run_g1_kl_mechanism_imageneta.sh

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_kl_mechanism_imageneta_g1"
LOG_DIR="$ROOT/logs_kl_mechanism_imageneta_g1"
PYTHON_BIN="${PYTHON_BIN:-/home-local/liying/.conda/envs/Pilot_new_local/bin/python}"

if [[ -n "${GPUS:-}" ]]; then
    read -r -a GPU_LIST <<< "$GPUS"
else
    GPU_LIST=(3 4)
fi

CFGS=(
    as2_normfisher_kl_lam2000_ewcg100_imageneta_t10c20_r10_sgd_lr004_e20_rho02_nr02_g01.yaml
    as2_normfisher_kl_lam1900_ewcg100_imageneta_t10c20_r10_sgd_lr004_e20_rho02_nr02_g01.yaml
)

if (( ${#GPU_LIST[@]} < ${#CFGS[@]} )); then
    echo "[ERROR] Need at least ${#CFGS[@]} GPUs, got: ${GPU_LIST[*]}" >&2
    exit 1
fi

TMP_ROOT="/tmp/$(whoami)/flatness_cv_kl_mechanism_imageneta_g1"
mkdir -p "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"
mkdir -p "$LOG_DIR"

printf '\n========================================\n'
printf '[%s] ImageNet-A gamma=1.0 KL mechanism diagnostics\n' "$(date)"
printf 'Configs: %s\n' "$CONFIG_DIR"
printf 'Logs:    %s\n' "$LOG_DIR"
printf 'GPUs:    %s\n' "${GPU_LIST[*]}"
printf 'Outputs root: outputs_logs/config_kl_mechanism_imageneta_g1\n'
printf '========================================\n'

pids=()
pid_cfgs=()

for i in "${!CFGS[@]}"; do
    cfg_name="${CFGS[$i]}"
    gpu="${GPU_LIST[$i]}"
    cfg="$CONFIG_DIR/$cfg_name"
    stem="${cfg_name%.yaml}"
    log="$LOG_DIR/${stem}_gpu${gpu}.log"

    if [[ ! -f "$cfg" ]]; then
        echo "[ERROR] Missing config: $cfg" >&2
        exit 1
    fi

    printf '[%s] Launch  %-88s  GPU %s\n' "$(date)" "$cfg_name" "$gpu"
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
printf '[%s] All ImageNet-A gamma=1.0 KL mechanism runs complete.\n' "$(date)"
printf 'Logs: %s\n' "$LOG_DIR"
printf 'Outputs root: outputs_logs/config_kl_mechanism_imageneta_g1\n'
printf '========================================\n'
