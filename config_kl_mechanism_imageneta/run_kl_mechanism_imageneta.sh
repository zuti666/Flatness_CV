#!/usr/bin/env bash
# Run the ImageNet-A KL mechanism diagnostic.
#
# Usage:
#   cd /data/140-0/users/liying/Flatness_CV
#   GPU=3 bash config_kl_mechanism_imageneta/run_kl_mechanism_imageneta.sh

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_kl_mechanism_imageneta"
LOG_DIR="$ROOT/logs_kl_mechanism_imageneta"
PYTHON_BIN="${PYTHON_BIN:-/home-local/liying/.conda/envs/Pilot_new_local/bin/python}"
GPU="${GPU:-3}"

TMP_ROOT="/tmp/$(whoami)/flatness_cv_kl_mechanism_imageneta"
mkdir -p "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"
mkdir -p "$LOG_DIR"

CFG="as2_normfisher_kl_lam2000_ewcg099_imageneta_t10c20_r10_sgd_lr004_e20_rho02_nr02_g01.yaml"
LOG="$LOG_DIR/${CFG%.yaml}_gpu${GPU}.log"

printf '\n========================================\n'
printf '[%s] ImageNet-A KL mechanism diagnostic\n' "$(date)"
printf 'Config: %s/%s\n' "$CONFIG_DIR" "$CFG"
printf 'Log:    %s\n' "$LOG"
printf 'GPU:    %s\n' "$GPU"
printf 'Outputs root: outputs_logs/config_kl_mechanism_imageneta\n'
printf '========================================\n'

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m src.main --config="$CONFIG_DIR/$CFG" \
    > "$LOG" 2>&1

printf '\n[%s] Done. Log: %s\n' "$(date)" "$LOG"
