#!/usr/bin/env bash
set -euo pipefail

# Run all configs in exp_1_cflat-rwp_imageR_r16_t20 with updated paths only.

# # ===== Wave 1: inclora + seqlora (gam)
# mkdir -p ./logs_inc_lora/inclora/gam/imagenetr
# python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/inclora_inr_gam_t20c10_r16.yaml \
#   >> ./logs_inc_lora/inclora/gam/imagenetr/inclora_inr_gam_t20c10_r16_train_eval_redo.log 2>&1 & p1=$!

# mkdir -p ./logs_inc_lora/seqlora/gam/imagenetr
# python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/seqlora_inr_gam_t20c10_r16.yaml \
#   >> ./logs_inc_lora/seqlora/gam/imagenetr/seqlora_inr_gam_t20c10_r16_train_eval_redo.log 2>&1 & p3=$!

# mkdir -p ./logs_inc_lora/olora/gam/imagenetr
# python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/olora_inr_gam_t20c10_r16.yaml \
#   >> ./logs_inc_lora/olora/gam/imagenetr/olora_inr_gam_t20c10_r16_train_eval_redo.log 2>&1 & p5=$!


# # ===== Wave 3: PerTask LP (cflat/rwp) + PerTask FT (cflat/rwp)
# mkdir -p ./logs_inc/linearprobe/gam/imagenetr
# python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/LP_gam_imageR_t20.yaml \
#   >> ./logs_inc/linearprobe/gam/imagenetr/LP_gam_imageR_t20_train_eval_redo.log 2>&1 & p9=$!

# mkdir -p ./logs_inc/finetune/gam/imagenetr
# python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/FT_gam_imageR_t20.yaml \
#   >> ./logs_inc/finetune/gam/imagenetr/FT_gam_imageR_t20_train_eval_redo.log 2>&1 & p11=$!

# wait $p9 $p11 $p1 $p3  $p5  
# echo "GASM ."



