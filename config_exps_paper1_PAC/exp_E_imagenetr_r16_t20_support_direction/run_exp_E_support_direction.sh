#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction"
LOG_DIR="$ROOT/outputs_logs/exp_E_imagenetr_r16_t20_support_direction_launcher_logs"
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

VARIANTS="${EXP_E_VARIANTS:-sgd sam_factor sam_full sam_delta sam_all sam_frozen random_factor random_full random_delta random_all random_frozen}"
GPUS=(${EXP_E_GPUS:-1 2 3 5})

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

build_override_args() {
  local args=()
  if [[ -n "${EXP_E_FLAT_EVAL:-}" ]]; then
    args+=(flat_eval="$EXP_E_FLAT_EVAL")
  fi
  if [[ -n "${EXP_E_SEED:-}" ]]; then
    args+=(seed="[$EXP_E_SEED]")
  fi
  if (( ${#args[@]} > 0 )); then
    printf '%s\n' "--override" "${args[@]}"
  fi
}

i=0
for variant in $VARIANTS; do
  cfg="$CONFIG_DIR/seqlora_imagenetr_r16_t20_${variant}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] missing config: $cfg" >&2
    exit 1
  fi
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  log="$LOG_DIR/${variant}.log"
  echo "[$(date)] launch exp_E ${variant} on CUDA_VISIBLE_DEVICES=${gpu}"
  mapfile -t override_args < <(build_override_args)
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" -m src.main --config "$cfg" "${override_args[@]}" > "$log" 2>&1 &
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
