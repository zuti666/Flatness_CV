#!/usr/bin/env bash
# Run C1/C2/D1/D2 on CIFAR-100 and DomainNet.
# GPUs: 0, 1, 2, 6
#
# Usage:
#   cd /data/140-0/users/liying/Flatness_CV
#   bash config_Expand_cifar_domainnet_c1d2/run_cifar_domainnet_c1d2.sh

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONFIG_DIR="$ROOT/config_Expand_cifar_domainnet_c1d2"
LOG_DIR="$ROOT/logs_expand_cifar_domainnet_c1d2"
PYTHON_BIN="${PYTHON_BIN:-/home-local/liying/.conda/envs/Pilot_new_local/bin/python}"

TMP_ROOT="/tmp/$(whoami)/flatness_cv_cifar_domainnet_c1d2"
mkdir -p "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"
mkdir -p "$LOG_DIR" "$ROOT/outputs_logs/config_Expand_cifar_domainnet_c1d2"

GPUS=(0 1 2 6)
CFGS=(
    as1_normfisher_lam1600_ewcg095_cifar100_t10c10_r10_sgd_initlr0p04_lr0p03_e20_rho005_nr01_g01_c1.yaml
    as1_normfisher_lam1500_ewcg099_cifar100_t10c10_r10_sgd_initlr0p04_lr0p025_e20_rho005_nr01_g01_c2.yaml
    as1_normfisher_lam1600_ewcg095_domainnet_t5c69_r30_sgd_initlr0p05_lr0p05_e5_rho005_nr01_g01_d1.yaml
    as1_normfisher_lam1600_ewcg095_domainnet_t5c69_r30_sgd_initlr0p06_lr0p04_e5_rho005_nr01_g01_d2.yaml
)

printf '\n========================================\n'
printf '[%s] CIFAR/DomainNet C1-C2-D1-D2 sweep\n' "$(date)"
printf 'Configs: %s\n' "$CONFIG_DIR"
printf 'Logs:    %s\n' "$LOG_DIR"
printf 'Outputs: outputs_logs/config_Expand_cifar_domainnet_c1d2\n'
printf 'GPUs:    %s\n' "${GPUS[*]}"
printf '========================================\n'

pids=()
pid_cfgs=()

for i in "${!CFGS[@]}"; do
    cfg_name="${CFGS[$i]}"
    gpu="${GPUS[$i]}"
    cfg="$CONFIG_DIR/$cfg_name"
    stem="${cfg_name%.yaml}"
    log="$LOG_DIR/${stem}_gpu${gpu}.log"

    if [[ ! -f "$cfg" ]]; then
        echo "[ERROR] Missing config: $cfg" >&2
        exit 1
    fi

    printf '[%s] Launch  %-100s  GPU %s\n' "$(date)" "$cfg_name" "$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m src.main --config="$cfg" \
        > "$log" 2>&1 &
    pids+=("$!")
    pid_cfgs+=("$cfg_name (GPU $gpu, log: $log)")
done

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        printf '[%s] OK  %s\n' "$(date)" "${pid_cfgs[$i]}"
    else
        printf '[%s] FAILED: %s\n' "$(date)" "${pid_cfgs[$i]}" >&2
        status=1
    fi
done

if [[ $status -ne 0 ]]; then
    echo "[ERROR] One or more runs failed." >&2
    exit 1
fi

printf '\n========================================\n'
printf '[%s] All C1/C2/D1/D2 runs complete.\n' "$(date)"
printf 'Logs: %s\n' "$LOG_DIR"
printf 'Outputs root: outputs_logs/config_Expand_cifar_domainnet_c1d2\n'
printf '========================================\n'
