#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

CONFIG="$ROOT/config_exps_paper1_PAC/exp_F_imagenetr_r16_t20_forked_taskwise/seqlora_imagenetr_r16_t20_fork_base.yaml"
LOG_DIR="$ROOT/outputs_logs/exp_F_imagenetr_r16_t20_forked_taskwise_launcher_logs"
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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

SEED="${EXP_F_SEED:-1993}"
PREFIX_GPU="${EXP_F_PREFIX_GPU:-0}"
GPUS=(${EXP_F_GPUS:-0 1 2 3})
FORCE_RERUN="${EXP_F_FORCE_RERUN:-0}"
VARIANT_FILTER="${EXP_F_VARIANTS:-sgd_sgd sgd_sam_factor sam_factor_sgd sam_factor_sam_factor sgd_random_factor}"

PREFIX_SGD="exp_F_imagenetr_r16_t20_prefix_sgd_t0_9"
PREFIX_SAM="exp_F_imagenetr_r16_t20_prefix_sam_factor_t0_9"

metrics_path_for() {
  local opt_tag="$1"
  local prefix="$2"
  echo "$ROOT/outputs_logs/logs_inc_lora/seqlora/${opt_tag}/imagenetr/${SEED}/${prefix}/exp_run/10/${prefix}_vit_base_patch16_224_cl_metrics.json"
}

ckpt_path_for() {
  local opt_tag="$1"
  local prefix="$2"
  echo "$ROOT/outputs_logs/logs_inc_lora/seqlora/${opt_tag}/imagenetr/${SEED}/${prefix}/exp_run/checkpoints"
}

run_prefix() {
  local opt="$1"
  local prefix="$2"
  local ckpt_dir
  local metrics_json
  ckpt_dir=$(ckpt_path_for "$opt" "$prefix")
  metrics_json=$(metrics_path_for "$opt" "$prefix")
  if [[ "$FORCE_RERUN" != "1" && -f "$ckpt_dir/lora_w_a_9.pt" && -f "$ckpt_dir/fc_state_9.pt" && -f "$metrics_json" ]]; then
    echo "[$(date)] reuse prefix ${prefix}: $ckpt_dir"
    return 0
  fi
  echo "[$(date)] train prefix ${prefix} optimizer=${opt} tasks=0..9 on CUDA_VISIBLE_DEVICES=${PREFIX_GPU}"
  CUDA_VISIBLE_DEVICES="$PREFIX_GPU" "${PYTHON_CMD[@]}" -m src.main \
    --config "$CONFIG" \
    --override prefix="$prefix" optimizer_type="$opt" max_train_tasks=10 \
    > "$LOG_DIR/${prefix}.log" 2>&1
  [[ -f "$ckpt_dir/lora_w_a_9.pt" && -f "$ckpt_dir/fc_state_9.pt" && -f "$metrics_json" ]]
}

should_run_variant() {
  local name="$1"
  local item
  for item in $VARIANT_FILTER; do
    if [[ "$item" == "$name" ]]; then
      return 0
    fi
  done
  return 1
}

copy_prefix_checkpoint() {
  local src_ckpt="$1"
  local branch_prefix="$2"
  local branch_ckpt="$ROOT/outputs_logs/logs_inc_lora/seqlora/exp_F_forked_taskwise/imagenetr/${SEED}/${branch_prefix}/exp_run/checkpoints"
  mkdir -p "$branch_ckpt"
  cp -a "$src_ckpt/." "$branch_ckpt/"
  echo "$branch_ckpt"
}

SGD20='["sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd"]'
SGD10_SAM10='["sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor"]'
SAM10_SGD10='["sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd"]'
SAM20='["sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor","sam_factor"]'
SGD10_RANDOM10='["sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","sgd","random_factor","random_factor","random_factor","random_factor","random_factor","random_factor","random_factor","random_factor","random_factor","random_factor"]'

run_prefix "sgd" "$PREFIX_SGD"
run_prefix "sam_factor" "$PREFIX_SAM"

SGD_CKPT=$(ckpt_path_for "sgd" "$PREFIX_SGD")
SGD_METRICS=$(metrics_path_for "sgd" "$PREFIX_SGD")
SAM_CKPT=$(ckpt_path_for "sam_factor" "$PREFIX_SAM")
SAM_METRICS=$(metrics_path_for "sam_factor" "$PREFIX_SAM")

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

launch_branch() {
  local variant="$1"
  local src_ckpt="$2"
  local src_metrics="$3"
  local schedule="$4"
  local gpu="$5"
  if ! should_run_variant "$variant"; then
    return 0
  fi
  local branch_prefix="exp_F_imagenetr_r16_t20_fork_${variant}"
  local branch_ckpt
  branch_ckpt=$(copy_prefix_checkpoint "$src_ckpt" "$branch_prefix")
  local log="$LOG_DIR/${variant}.log"
  echo "[$(date)] launch fork ${variant} from ${branch_ckpt} on CUDA_VISIBLE_DEVICES=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" -m src.main \
    --config "$CONFIG" \
    --override \
      prefix="$branch_prefix" \
      optimizer_type="sgd" \
      optimizer_type_by_task="$schedule" \
      optimizer_tag_override="exp_F_forked_taskwise" \
      max_train_tasks=20 \
      resume_from_task=9 \
      resume_from_checkpoint_dir="$branch_ckpt" \
      resume_from_metrics_json="$src_metrics" \
    > "$log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$variant")
  if (( ${#PIDS[@]} >= ${#GPUS[@]} )); then
    wait_group
  fi
}

i=0
launch_branch "sgd_sgd" "$SGD_CKPT" "$SGD_METRICS" "$SGD20" "${GPUS[$((i++ % ${#GPUS[@]}))]}"
launch_branch "sgd_sam_factor" "$SGD_CKPT" "$SGD_METRICS" "$SGD10_SAM10" "${GPUS[$((i++ % ${#GPUS[@]}))]}"
launch_branch "sam_factor_sgd" "$SAM_CKPT" "$SAM_METRICS" "$SAM10_SGD10" "${GPUS[$((i++ % ${#GPUS[@]}))]}"
launch_branch "sam_factor_sam_factor" "$SAM_CKPT" "$SAM_METRICS" "$SAM20" "${GPUS[$((i++ % ${#GPUS[@]}))]}"
launch_branch "sgd_random_factor" "$SGD_CKPT" "$SGD_METRICS" "$SGD10_RANDOM10" "${GPUS[$((i++ % ${#GPUS[@]}))]}"

if (( ${#PIDS[@]} > 0 )); then
  wait_group
fi

echo "Launcher logs: $LOG_DIR"
echo "Summarize with:"
echo "python config_exps_paper1_PAC/exp_F_imagenetr_r16_t20_forked_taskwise/summarize_forked_taskwise.py"
exit "$status"
