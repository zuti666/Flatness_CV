#!/usr/bin/env bash
set -euo pipefail

gpu_index="$1"
job_offset="$2"
job_limit="$3"
python_bin="/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python"
config_path="configs/stage3_p3_step_normalized_formal.yaml"
poll_count=0

while true; do
    used_mib="$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= 1000 )); then
        break
    fi
    if (( poll_count % 10 == 0 )); then
        echo "waiting gpu=${gpu_index} used_mib=${used_mib} offset=${job_offset} limit=${job_limit}"
    fi
    poll_count=$((poll_count + 1))
    sleep 30
done

echo "starting gpu=${gpu_index} offset=${job_offset} limit=${job_limit}"
export CUDA_VISIBLE_DEVICES="${gpu_index}"
exec "${python_bin}" -u run_stage3_p3_step_normalized.py \
    --config "${config_path}" \
    --offset "${job_offset}" \
    --limit "${job_limit}" \
    --skip-completed
