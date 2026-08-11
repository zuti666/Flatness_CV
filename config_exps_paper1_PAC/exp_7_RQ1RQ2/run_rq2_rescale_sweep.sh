#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

CONFIG="$ROOT/config_exps_paper1_PAC/exp_7_RQ1RQ2/rq2_seqlora_imagenetr_r16_sgd_task10.yaml"
LOG_DIR="$ROOT/outputs_logs/exp_7_RQ1RQ2_launcher_logs"
GPU="${RQ2_GPU:-1}"
C_VALUES="${RQ2_C_VALUES:-0.1 0.3 1.0 3.0 10.0}"

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

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] missing config: $CONFIG" >&2
  exit 1
fi

for c in $C_VALUES; do
  tag=$(printf '%s' "$c" | sed 's/\./p/g')
  prefix="exp_7_RQ2_seqlora_imagenetr_r16_sgd_task10_c${tag}"
  log="$LOG_DIR/rq2_rescale_c${tag}.log"

  echo "[$(date)] run RQ2 c=${c} on CUDA_VISIBLE_DEVICES=${GPU}"
  CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON_CMD[@]}" -m src.main \
    --config "$CONFIG" \
    --override \
      "prefix=${prefix}" \
      "flat_eval_lora_rescale_factor=${c}" \
      "device=[\"0\"]" \
    > "$log" 2>&1
done

echo "Launcher logs: $LOG_DIR"
