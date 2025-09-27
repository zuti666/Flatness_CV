#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CONFIG="${REPO_ROOT}/exps/sdlora_c100.json"
CHECKPOINT_DIR="${REPO_ROOT}/logs/sdlora/cifar224/0/checkpoints"
FEATURE_FLAT_DIR="${CHECKPOINT_DIR}/feature_flat"
LOG_DIR="${REPO_ROOT}/logs/sdlora/cifar224/eval"

mkdir -p "${LOG_DIR}" "${FEATURE_FLAT_DIR}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/eval_${TIMESTAMP}.log"

echo "Evaluating checkpoints in ${CHECKPOINT_DIR}" | tee "${LOG_FILE}"

python "${REPO_ROOT}/evaluation/eval_cl.py" \
  --config "${CONFIG}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --tasks all \
  --flat-eval \
  --flat-rho 0.05 \
  --flat-samples 16 \
  --flat-max-batches 2 \
  --flat-power-iters 6 \
  --flat-trace-samples 8 \
  --flat-grad-batches 2 \
  --feature-flat-eval \
  --feature-flat-max-batches 2 \
  --feature-flat-topk 10 \
  --feature-flat-save-path "${FEATURE_FLAT_DIR}" \
  --log-level INFO | tee -a "${LOG_FILE}"

echo "Evaluation complete. Results stored in ${LOG_FILE}" | tee -a "${LOG_FILE}"
