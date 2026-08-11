#!/usr/bin/env bash
# Run EWC gamma=0.99 sweep on ImageNet-A and ImageNet-R only.
# GPUs: 3, 4, 5, 7
#
# Usage:
#   cd /data/140-0/users/liying/Flatness_CV
#   bash config_Expand_ewc_gamma099/run_ewc_gamma099_imagenet_ar.sh

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_Expand_ewc_gamma099"
LOG_DIR="$ROOT/logs_expand_ewc_gamma099"
PYTHON_BIN="${PYTHON_BIN:-/home-local/liying/.conda/envs/Pilot_new_local/bin/python}"

TMP_ROOT="/tmp/$(whoami)/flatness_cv_ewc_gamma099"
mkdir -p "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"
mkdir -p "$LOG_DIR"

GPUS=(3 4 5 7)
CFGS=(
    as2_normfisher_lam2000_ewcg099_imageneta_t10c20_r10_sgd_lr004_e20_rho02_nr02_g01_paper1agg.yaml
    as2_normfisher_lam1500_ewcg099_imageneta_t10c20_r10_sgd_lr004_e20_rho02_nr02_g01_paper1agg.yaml
    as2_normfisher_lam2000_ewcg099_imagenetr_t10c20_r10_sgd_lr004_e50_rho02_nr02_g01_paper1agg.yaml
    as2_normfisher_lam1500_ewcg099_imagenetr_t10c20_r10_sgd_lr004_e50_rho02_nr02_g01_paper1agg.yaml
)

printf '\n========================================\n'
printf '[%s] EWC gamma=0.99 sweep: ImageNet-A/R\n' "$(date)"
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

    printf '[%s] Launch  %-90s  GPU %s\n' "$(date)" "$cfg_name" "$gpu"
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
printf '[%s] All EWC gamma=0.99 ImageNet-A/R runs complete.\n' "$(date)"
printf 'Logs: %s\n' "$LOG_DIR"
printf 'Outputs root: outputs_logs/config_Expand_ewc_gamma099\n'
printf '========================================\n'
