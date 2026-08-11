#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

CONFIG="$ROOT/config_exps_paper1_PAC/exp_D_taskwise_sam_trajectory/seqlora_cifar10_224_t2_r16_base.yaml"
LOG_DIR="$ROOT/outputs_logs/exp_D_taskwise_sam_trajectory_launcher_logs"
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

# Default diagnostic is raw LoRA factor space.  This runs the same 2x2
# task-position intervention for adversarial SAM and normalized Gaussian noise.
PERTURB_OPTS="${TASKWISE_PERTURB_OPTS:-sam_factor random_factor}"
VARIANT_FILTER="${TASKWISE_VARIANTS:-}"
GPUS=(${TASKWISE_GPUS:-0 1 2 3})

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

should_run_variant() {
  local name="$1"
  if [[ -z "$VARIANT_FILTER" ]]; then
    return 0
  fi
  local item
  for item in $VARIANT_FILTER; do
    if [[ "$item" == "$name" ]]; then
      return 0
    fi
  done
  return 1
}

add_run() {
  local variant="$1"
  local schedule="$2"
  if should_run_variant "$variant"; then
    RUN_VARIANTS+=("$variant")
    RUN_SCHEDULES+=("$schedule")
  fi
}

declare -a RUN_VARIANTS=()
declare -a RUN_SCHEDULES=()
add_run "sgd_sgd" '["sgd","sgd"]'

for perturb_opt in $PERTURB_OPTS; do
  tag="$perturb_opt"
  add_run "${tag}_sgd" "[\"${perturb_opt}\",\"sgd\"]"
  add_run "sgd_${tag}" "[\"sgd\",\"${perturb_opt}\"]"
  add_run "${tag}_${tag}" "[\"${perturb_opt}\",\"${perturb_opt}\"]"
done

if (( ${#RUN_VARIANTS[@]} == 0 )); then
  echo "[ERROR] no variants selected. TASKWISE_VARIANTS=${VARIANT_FILTER}" >&2
  exit 1
fi

i=0
for idx in "${!RUN_VARIANTS[@]}"; do
  variant="${RUN_VARIANTS[$idx]}"
  schedule="${RUN_SCHEDULES[$idx]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  prefix="exp_D_taskwise_sam_trajectory_${variant}"
  log="$LOG_DIR/${variant}.log"
  echo "[$(date)] launch exp_D ${variant} schedule=${schedule} on CUDA_VISIBLE_DEVICES=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" -m src.main \
    --config "$CONFIG" \
    --override optimizer_type="sgd" optimizer_type_by_task="$schedule" prefix="$prefix" \
    > "$log" 2>&1 &
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
