#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_exps_paper1_PAC/exp_7_RQ1RQ2"
LOG_DIR="$ROOT/outputs_logs/exp_7_RQ1RQ2_launcher_logs"

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

SGD_GPU="${RQ1_SGD_GPU:-6}"
SAM_GPU="${RQ1_SAM_GPU:-7}"

declare -a PIDS=()
declare -a NAMES=()

launch() {
  local name="$1"
  local cfg="$2"
  local gpu="$3"
  local log="$LOG_DIR/${name}.log"

  if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] missing config: $cfg" >&2
    exit 1
  fi

  echo "[$(date)] launch ${name} on CUDA_VISIBLE_DEVICES=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" -m src.main --config "$cfg" > "$log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

launch "rq1_seqlora_imagenetr_r16_sgd" "$CONFIG_DIR/rq1_seqlora_imagenetr_r16_sgd.yaml" "$SGD_GPU"
launch "rq1_seqlora_imagenetr_r16_sam" "$CONFIG_DIR/rq1_seqlora_imagenetr_r16_sam.yaml" "$SAM_GPU"

status=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[$(date)] completed ${NAMES[$i]}"
  else
    echo "[$(date)] failed ${NAMES[$i]} (see $LOG_DIR)" >&2
    status=1
  fi
done

echo "Launcher logs: $LOG_DIR"
exit "$status"
