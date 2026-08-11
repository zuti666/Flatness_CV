#!/usr/bin/env bash
# run_gam_sweep.sh
# Sequential by dataset, 4-way parallel within each dataset.
# Order: ImageNet-A → ImageNet-R → CIFAR-100
# GPUs:  0, 1, 3, 4
#
# Usage:
#   cd /data/140-0/users/liying/Flatness_CV
#   bash config_Expand_gam_sweep/run_gam_sweep.sh

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_Expand_gam_sweep"
LOG_DIR="$ROOT/outputs_logs/logs_gam_sweep"
PYTHON_BIN="${PYTHON_BIN:-/home-local/liying/.conda/envs/Pilot_new_local/bin/python}"

TMP_ROOT="/tmp/$(whoami)/flatness_cv_gam_sweep"
mkdir -p "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"
mkdir -p "$LOG_DIR"

GPUS=(0 1 3 4)

# ── helper: run one stage (array of 4 config filenames) in parallel ──────────
run_stage() {
    local stage_name="$1"
    shift
    local cfgs=("$@")

    printf '\n========================================\n'
    printf '[%s] Stage: %s\n' "$(date)" "$stage_name"
    printf '========================================\n'

    local pids=()
    local pid_cfgs=()

    for i in "${!cfgs[@]}"; do
        local cfg_name="${cfgs[$i]}"
        local gpu="${GPUS[$i]}"
        local cfg="$CONFIG_DIR/$cfg_name"
        local stem="${cfg_name%.yaml}"
        local log="$LOG_DIR/${stem}_gpu${gpu}.log"

        if [[ ! -f "$cfg" ]]; then
            echo "[ERROR] Missing config: $cfg" >&2
            exit 1
        fi

        printf '[%s] Launch  %-75s  GPU %s\n' "$(date)" "$cfg_name" "$gpu"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m src.main --config="$cfg" \
            > "$log" 2>&1 &
        pids+=("$!")
        pid_cfgs+=("$cfg_name (GPU $gpu, log: $log)")
    done

    local status=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            printf '[%s] ✓  %s\n' "$(date)" "${pid_cfgs[$i]}"
        else
            printf '[%s] ✗  FAILED: %s\n' "$(date)" "${pid_cfgs[$i]}" >&2
            status=1
        fi
    done

    if [[ $status -ne 0 ]]; then
        echo "[ERROR] Stage '$stage_name' had failures — aborting." >&2
        exit 1
    fi
    printf '[%s] Stage "%s" complete.\n' "$(date)" "$stage_name"
}

# ── Stage 1: ImageNet-A  (4 configs × GPU 0,1,3,4) ──────────────────────────
IMAGENETA_CFGS=(
    as2_normfisher_lam2000_imageneta_t10c20_r10_sgd_lr004_e20_rho002_nr02_g003.yaml
    as2_normfisher_lam2000_imageneta_t10c20_r10_sgd_lr004_e20_rho005_nr01_g005.yaml
    as2_normfisher_lam2000_imageneta_t10c20_r10_sgd_lr004_e20_rho002_nr01_g01.yaml
    as2_normfisher_lam2000_imageneta_t10c20_r10_sgd_lr004_e20_rho02_nr02_g01_paper1agg.yaml
)
run_stage "ImageNet-A" "${IMAGENETA_CFGS[@]}"

# ── Stage 2: ImageNet-R  (4 configs × GPU 0,1,3,4) ──────────────────────────
IMAGENETR_CFGS=(
    as2_normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr004_e50_rho002_nr02_g003.yaml
    as2_normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr004_e50_rho005_nr01_g005.yaml
    as2_normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr004_e50_rho002_nr01_g01.yaml
    as2_normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr004_e50_rho02_nr02_g01_paper1agg.yaml
)
run_stage "ImageNet-R" "${IMAGENETR_CFGS[@]}"

# ── Stage 3: CIFAR-100  (4 configs × GPU 0,1,3,4) ───────────────────────────
CIFAR100_CFGS=(
    as2_normfisher_lam2000_cifar100_t10c10_r10_sgd_lr004_e20_rho002_nr02_g003.yaml
    as2_normfisher_lam2000_cifar100_t10c10_r10_sgd_lr004_e20_rho005_nr01_g005.yaml
    as2_normfisher_lam2000_cifar100_t10c10_r10_sgd_lr004_e20_rho002_nr01_g01.yaml
    as2_normfisher_lam2000_cifar100_t10c10_r10_sgd_lr004_e20_rho02_nr02_g01_paper1agg.yaml
)
run_stage "CIFAR-100" "${CIFAR100_CFGS[@]}"

printf '\n========================================\n'
printf '[%s] All stages complete. Logs: %s\n' "$(date)" "$LOG_DIR"
printf '========================================\n'
