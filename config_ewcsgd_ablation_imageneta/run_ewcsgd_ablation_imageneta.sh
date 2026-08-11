#!/usr/bin/env bash
# EWC+SGD ablation on ImageNet-A (no GAM).
#
# Sequential mode (all 5 configs on one GPU):
#   cd /data/140-0/users/liying/Flatness_CV
#   RUN_GPU=0 bash config_ewcsgd_ablation_imageneta/run_ewcsgd_ablation_imageneta.sh
#
# Single config mode:
#   RUN_GPU=1 ONLY=c2 bash config_ewcsgd_ablation_imageneta/run_ewcsgd_ablation_imageneta.sh
#
# Parallel mode (run each in background on different GPUs):
#   RUN_GPU=0 ONLY=c1 bash ... &
#   RUN_GPU=1 ONLY=c2 bash ... &
#   RUN_GPU=2 ONLY=c3 bash ... &
#   RUN_GPU=3 ONLY=c4 bash ... &
#   RUN_GPU=4 ONLY=c5 bash ... &
#   wait

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_ewcsgd_ablation_imageneta"
LOG_DIR="$ROOT/logs_ewcsgd_ablation_imageneta"
PYTHON_BIN="${PYTHON_BIN:-/home-local/liying/.conda/envs/Pilot_new_local/bin/python}"
RUN_GPU="${RUN_GPU:-0}"
ONLY="${ONLY:-}"   # set to e.g. "c2" to run only that config

TMP_ROOT="/tmp/$(whoami)/flatness_cv_ewcsgd_ablation"
mkdir -p "$TMP_ROOT" "$LOG_DIR"
export TMPDIR="$TMP_ROOT"

# Config list: (stem  description)
declare -A CFG_DESC
CFG_DESC["c1"]="ewclora SGD lam=2000  gamma=0.99 lr=0.04  [no-GAM baseline]"
CFG_DESC["c2"]="ewclora SGD lam=10000 gamma=0.99 lr=0.04  [stronger EWC]"
CFG_DESC["c3"]="ewclora SGD lam=50000 gamma=0.99 lr=0.04  [upper-bound probe]"
CFG_DESC["c4"]="ewclora SGD lam=10000 gamma=0.90 lr=0.03  [stable low-lr]"
CFG_DESC["c5"]="inclora  SGD lam=5000  gamma=0.99 lr=0.05  [arch ablation]"

CFGS=(c1 c2 c3 c4 c5)

printf '[%s] EWC+SGD ablation — ImageNet-A\n' "$(date)"
printf 'GPU:        %s\n' "$RUN_GPU"
printf 'Config dir: %s\n' "$CONFIG_DIR"
printf 'Log dir:    %s\n' "$LOG_DIR"
[[ -n "$ONLY" ]] && printf 'ONLY:       %s\n' "$ONLY"
echo

for stem in "${CFGS[@]}"; do
    [[ -n "$ONLY" && "$stem" != "$ONLY" ]] && continue

    # find the yaml matching the stem prefix
    cfg=$(ls "$CONFIG_DIR"/${stem}_*.yaml 2>/dev/null | head -1)
    if [[ -z "$cfg" ]]; then
        echo "[ERROR] No yaml found for stem '$stem' in $CONFIG_DIR" >&2
        exit 1
    fi

    cfg_name=$(basename "$cfg")
    log="$LOG_DIR/${stem}_gpu${RUN_GPU}.log"

    printf '[%s] %-6s %s\n' "$(date)" "$stem" "${CFG_DESC[$stem]}"
    printf '            cfg: %s\n' "$cfg_name"
    printf '            log: %s\n' "$log"

    CUDA_VISIBLE_DEVICES="$RUN_GPU" "$PYTHON_BIN" -m src.main --config="$cfg" > "$log" 2>&1
    status=$?

    if [[ $status -eq 0 ]]; then
        # quick FAA from final CNN top1 curve line
        last_curve=$(grep "CNN top1 curve" "$log" | tail -1)
        printf '[%s] Done   %s  |  %s\n' "$(date)" "$stem" "$last_curve"
    else
        printf '[%s] FAILED %s (exit %d) — see %s\n' "$(date)" "$stem" "$status" "$log" >&2
    fi
    echo
done

printf '[%s] All configs complete. Logs: %s\n' "$(date)" "$LOG_DIR"
