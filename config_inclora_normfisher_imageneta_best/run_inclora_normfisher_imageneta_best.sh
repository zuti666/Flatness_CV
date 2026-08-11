#!/usr/bin/env bash
# Run IncLoRA-specific FS-LoRA on ImageNet-A with the current best params.
#
# Usage:
#   cd /data/140-0/users/liying/Flatness_CV
#   GPU=3 bash config_inclora_normfisher_imageneta_best/run_inclora_normfisher_imageneta_best.sh

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_inclora_normfisher_imageneta_best"
LOG_DIR="$ROOT/logs_inclora_normfisher_imageneta_best"
PYTHON_BIN="${PYTHON_BIN:-/home-local/liying/.conda/envs/Pilot_new_local/bin/python}"
GPU="${GPU:-3}"

TMP_ROOT="/tmp/$(whoami)/flatness_cv_inclora_normfisher_imageneta_best"
mkdir -p "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"
mkdir -p "$LOG_DIR"

CFG="inclora_normfisher_gam_imageneta_best_g099_lam2000_lr004_rho02.yaml"
LOG="$LOG_DIR/${CFG%.yaml}_gpu${GPU}2.log"

printf '\n========================================\n'
printf '[%s] IncLoRA-FS ImageNet-A best-params run\n' "$(date)"
printf 'Config: %s/%s\n' "$CONFIG_DIR" "$CFG"
printf 'Log:    %s\n' "$LOG"
printf 'GPU:    %s\n' "$GPU"
printf 'Outputs root: outputs_logs/config_inclora_normfisher_imageneta_best\n'
printf '========================================\n'

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m src.main --config="$CONFIG_DIR/$CFG" \
    > "$LOG" 2>&1

printf '\n[%s] Done. Log: %s\n' "$(date)" "$LOG"
