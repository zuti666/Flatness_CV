#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_exps_paper1_PAC/exp_B_imager_r16_t20_seed1993"
LOG_DIR="$ROOT/outputs_logs/exp_B_imager_r16_t20_seed1993_launcher_logs"
mkdir -p "$LOG_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "Pilot"; then
  PYTHON_CMD=(conda run -n Pilot python)
else
  PYTHON_CMD=(python3)
fi

VARIANTS="${EXP_B_RANDOM_VARIANTS:-random_factor random_full random_delta random_all random_frozen}"
GPUS=(${EXP_B_GPUS:-0 1 2 3 4 5 6})

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

declare -a PIDS=()
declare -a NAMES=()
status=0

wait_group() {
  local j
  for j in "${!PIDS[@]}"; do
    if wait "${PIDS[$j]}"; then
      echo "[$(date)] completed ${NAMES[$j]}"
    else
      echo "[$(date)] failed ${NAMES[$j]} (see $LOG_DIR)" >&2
      status=1
    fi
  done
  PIDS=()
  NAMES=()
}

i=0
for variant in $VARIANTS; do
  cfg="$CONFIG_DIR/seqlora_imagenetr_r16_t20_seed1993_${variant}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] missing config: $cfg" >&2
    exit 1
  fi
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  log="$LOG_DIR/${variant}.log"
  echo "[$(date)] launch exp_B random ${variant} on CUDA_VISIBLE_DEVICES=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" -m src.main --config "$cfg" > "$log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$variant")
  i=$((i + 1))
  if (( ${#PIDS[@]} >= ${#GPUS[@]} )); then
    wait_group
  fi
done

if (( ${#PIDS[@]} > 0 )); then
  wait_group
fi

echo "Launcher logs: $LOG_DIR"
exit "$status"
