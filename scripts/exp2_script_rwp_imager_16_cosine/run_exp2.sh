#!/usr/bin/env bash
set -euo pipefail

# Run all configs in exp_1_cflat-rwp_imageR_r16_t20 with updated paths only.

# # ===== Wave 1: inclora + seqlora (cflat/rwp)
# # mkdir -p ./logs_inc_lora/inclora/cflat/imagenetr
# # python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/inclora_inr_cflat_t20c10_r16.yaml \
# #   >> ./logs_inc_lora/inclora/cflat/imagenetr/inclora_inr_cflat_t20c10_r16_train1.log 2>&1 & p1=$!
# # mkdir -p ./logs_inc_lora/inclora/rwp/imagenetr
# python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/inclora_inr_rwp_t20c10_r16.yaml \
#   >> ./logs_inc_lora/inclora/rwp/imagenetr/inclora_inr_rwp_t20c10_r16_train1.log 2>&1 & p2=$!

# # mkdir -p ./logs_inc_lora/seqlora/cflat/imagenetr
# # python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/seqlora_inr_cflat_t20c10_r16.yaml \
# #   >> ./logs_inc_lora/seqlora/cflat/imagenetr/seqlora_inr_cflat_t20c10_r16_train1.log 2>&1 & p3=$!
# # mkdir -p ./logs_inc_lora/seqlora/rwp/imagenetr
# python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/seqlora_inr_rwp_t20c10_r16.yaml \
#   >> ./logs_inc_lora/seqlora/rwp/imagenetr/seqlora_inr_rwp_t20c10_r16_train1.log 2>&1 & p4=$!

# # echo "inclora + seqlora (cflat/rwp) ."
# # wait $p1 $p2 $p3 $p4
# # echo "inclora + seqlora (cflat/rwp) ."


# # ===== Wave 2: olora + inflora (cflat/rwp)
# # mkdir -p ./logs_inc_lora/olora/cflat/imagenetr
# # python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/olora_inr_cflat_t20c10_r16.yaml \
# #   >> ./logs_inc_lora/olora/cflat/imagenetr/olora_inr_cflat_t20c10_r16_train1.log 2>&1 & p5=$!
# # mkdir -p ./logs_inc_lora/olora/rwp/imagenetr
# python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/olora_inr_rwp_t20c10_r16.yaml \
#   >> ./logs_inc_lora/olora/rwp/imagenetr/olora_inr_rwp_t20c10_r16_train1.log 2>&1 & p6=$!

# # mkdir -p ./logs_inc_lora/inflora/cflat/imagenetr
# # python3 main.py --config=./exp_1_cflat-rwp_imageR_r16_t20/inflora_inr_cflat_t20c10_r16.yaml \
# #   >> ./logs_inc_lora/inflora/cflat/imagenetr/inflora_inr_cflat_t20c10_r16_train1.log 2>&1 & p7=$!
# # mkdir -p ./logs_inc_lora/inflora/rwp/imagenetr
# # python3 main.py --config=./exp_1_cflat-rwp_imageR_r16_t20/inflora_inr_rwp_t20c10_r16.yaml \
# #   >> ./logs_inc_lora/inflora/rwp/imagenetr/inflora_inr_rwp_t20c10_r16_train1.log 2>&1 & p8=$!
# # $p7 $p8




# # ===== Wave 3: PerTask LP (cflat/rwp) + PerTask FT (cflat/rwp)
# # mkdir -p ./logs_inc/linearprobe/cflat/imagenetr
# # python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/LP_cflat_imageR_t20.yaml \
# #   >> ./logs_inc/linearprobe/cflat/imagenetr/LP_cflat_imageR_t20_train1.log 2>&1 & p9=$!
# # mkdir -p ./logs_inc/linearprobe/rwp/imagenetr
# # python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/LP_rwp_imageR_t20.yaml \
# #   >> ./logs_inc/linearprobe/rwp/imagenetr/LP_rwp_imageR_t20_train1.log 2>&1 & p10=$!

# # mkdir -p ./logs_inc/finetune/cflat/imagenetr
# # python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/FT_cflat_imageR_t20.yaml \
# #   >> ./logs_inc/finetune/cflat/imagenetr/FT_cflat_imageR_t20_train1.log 2>&1 & p11=$!
# # mkdir -p ./logs_inc/finetune/rwp/imagenetr
# # python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/FT_rwp_imageR_t20.yaml \
# #   >> ./logs_inc/finetune/rwp/imagenetr/FT_rwp_imageR_t20_train1.log 2>&1 & p12=$!

# # wait $p9 $p10 $p11 $p12  $p5 $p6 
# # echo "PerTask LP (cflat/rwp) + PerTask FT (cflat/rwp) ."



# echo "olora + inflora (cflat/rwp) ."
# echo "All runs submitted (stdout logs in ./logs_inc* and ./logs_inc_lora*)."


# ---
python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/inclora_inr_rwp_t20c10_r16.yaml \
  >> ./logs_inc_lora/inclora/rwp/imagenetr/inclora_inr_rwp_t20c10_r16_SH.log 2>&1 & p2=$!

python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/seqlora_inr_rwp_t20c10_r16.yaml \
  >> ./logs_inc_lora/seqlora/rwp/imagenetr/seqlora_inr_rwp_t20c10_r16_SH.log 2>&1 & p4=$!

python3 main.py --config=./exp2_sam-rwp-flat_imageR_r16_t20/olora_inr_rwp_t20c10_r16.yaml \
  >> ./logs_inc_lora/olora/rwp/imagenetr/olora_inr_rwp_t20c10_r16_SH.log 2>&1 & p6=$!
