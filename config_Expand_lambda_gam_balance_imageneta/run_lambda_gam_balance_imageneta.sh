#!/usr/bin/env bash
# Run the ImageNet-A EWC/GAM balance sweep.
#
# Usage:
#   cd /data/140-0/users/liying/Flatness_CV
#   RUN_GPU=6 bash config_Expand_lambda_gam_balance_imageneta/run_lambda_gam_balance_imageneta.sh

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_Expand_lambda_gam_balance_imageneta"
LOG_DIR="$ROOT/logs_expand_lambda_gam_balance_imageneta"
PYTHON_BIN="${PYTHON_BIN:-/home-local/liying/.conda/envs/Pilot_new_local/bin/python}"
RUN_GPU="${RUN_GPU:-6}"

TMP_ROOT="/tmp/$(whoami)/flatness_cv_lambda_gam_balance_imageneta"
mkdir -p "$TMP_ROOT" "$LOG_DIR"
export TMPDIR="$TMP_ROOT"

CFGS=(
    as2_normfisher_lam5e5_ewcg099_imageneta_t10c20_r10_sgd_lr004_e20_rho002_nr02_g003.yaml
    as2_normfisher_lam2e5_ewcg099_imageneta_t10c20_r10_sgd_lr004_e20_rho002_nr02_g003.yaml
    as2_normfisher_lam1e6_ewcg099_imageneta_t10c20_r10_sgd_lr004_e20_rho002_nr02_g003.yaml
)

printf '[%s] ImageNet-A lambda/GAM balance sweep\n' "$(date)"
printf 'GPU:     %s\n' "$RUN_GPU"
printf 'Configs: %s\n' "$CONFIG_DIR"
printf 'Logs:    %s\n' "$LOG_DIR"

for cfg_name in "${CFGS[@]}"; do
    cfg="$CONFIG_DIR/$cfg_name"
    stem="${cfg_name%.yaml}"
    log="$LOG_DIR/${stem}_gpu${RUN_GPU}.log"

    if [[ ! -f "$cfg" ]]; then
        echo "[ERROR] Missing config: $cfg" >&2
        exit 1
    fi

    printf '[%s] Launch %s on GPU %s\n' "$(date)" "$cfg_name" "$RUN_GPU"
    CUDA_VISIBLE_DEVICES="$RUN_GPU" "$PYTHON_BIN" -m src.main --config="$cfg" > "$log" 2>&1
    printf '[%s] Done   %s\n' "$(date)" "$cfg_name"
done

printf '[%s] Sweep complete. Logs: %s\n' "$(date)" "$LOG_DIR"
