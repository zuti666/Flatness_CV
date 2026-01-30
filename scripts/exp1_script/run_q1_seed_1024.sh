#!/usr/bin/env bash
set -euo pipefail

# Run all configs in exp_1_sam-sgd_imageR_r16_t20_seed_1024 with updated paths only.


# # ===== Wave 3: PerTask LP (SAM/SGD) + PerTask FT (SAM/SGD)
# mkdir -p ./logs_inc/linearprobe/sam/imagenetr
# python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_1024/LP_sam_imageR_t20.yaml \
#   >> ./logs_inc/linearprobe/sam/imagenetr/LP_sam_imageR_t20_train_eval1024.log 2>&1 & p9=$!
# mkdir -p ./logs_inc/linearprobe/sgd/imagenetr
# python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_1024/LP_sgd_imageR_t20.yaml \
#   >> ./logs_inc/linearprobe/sgd/imagenetr/LP_sgd_imageR_t20_train_eval1024.log 2>&1 & p10=$!

# mkdir -p ./logs_inc/finetune/sam/imagenetr
# python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_1024/FT_sam_imageR_t20.yaml \
#   >> ./logs_inc/finetune/sam/imagenetr/FT_sam_imageR_t20_train_eval1024.log 2>&1 & p11=$!
# mkdir -p ./logs_inc/finetune/sgd/imagenetr
# python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_1024/FT_sgd_imageR_t20.yaml \
#   >> ./logs_inc/finetune/sgd/imagenetr/FT_sgd_imageR_t20_train_eval1024.log 2>&1 & p12=$!

# wait $p9 $p10 $p11 $p12
# echo "PerTask LP (SAM/SGD) + PerTask FT (SAM/SGD) ."



# # ===== Wave 1: inclora + seqlora (SAM/SGD)
# mkdir -p ./logs_inc_lora/inclora/sam/imagenetr
# python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_1024/inclora_inr_sam_t20c10_r16.yaml \
#   >> ./logs_inc_lora/inclora/sam/imagenetr/inclora_inr_sam_t20c10_r16_train_eval1024.log 2>&1 & p1=$!
# mkdir -p ./logs_inc_lora/inclora/sgd/imagenetr
# python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_1024/inclora_inr_sgd_t20c10_r16.yaml \
#   >> ./logs_inc_lora/inclora/sgd/imagenetr/inclora_inr_sgd_t20c10_r16_train_eval1024.log 2>&1 & p2=$!

# mkdir -p ./logs_inc_lora/seqlora/sam/imagenetr
# python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_1024/seqlora_inr_sam_t20c10_r16.yaml \
#   >> ./logs_inc_lora/seqlora/sam/imagenetr/seqlora_inr_sam_t20c10_r16_train_eval1024.log 2>&1 & p3=$!
# mkdir -p ./logs_inc_lora/seqlora/sgd/imagenetr
# python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_1024/seqlora_inr_sgd_t20c10_r16.yaml \
#   >> ./logs_inc_lora/seqlora/sgd/imagenetr/seqlora_inr_sgd_t20c10_r16_train_eval1024.log 2>&1 & p4=$!


# echo "inclora + seqlora (SAM/SGD) ."
# wait $p1 $p2 $p3 $p4

# ===== Wave 2: olora + inflora (SAM/SGD)
mkdir -p ./logs_inc_lora/olora/sam/imagenetr
python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_1024/olora_inr_sam_t20c10_r16.yaml \
  >> ./logs_inc_lora/olora/sam/imagenetr/olora_inr_sam_t20c10_r16_train_eval1024.log 2>&1 & p5=$!
mkdir -p ./logs_inc_lora/olora/sgd/imagenetr
python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_1024/olora_inr_sgd_t20c10_r16.yaml \
  >> ./logs_inc_lora/olora/sgd/imagenetr/olora_inr_sgd_t20c10_r16_train_eval1024.log 2>&1 & p6=$!

mkdir -p ./logs_inc_lora/olora/sam/imagenetr
python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_42/olora_inr_sam_t20c10_r16.yaml \
  >> ./logs_inc_lora/olora/sam/imagenetr/olora_inr_sam_t20c10_r16_train_eval_42_new.log 2>&1 & p7=$!
mkdir -p ./logs_inc_lora/olora/sgd/imagenetr
python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_42/olora_inr_sgd_t20c10_r16.yaml \
  >> ./logs_inc_lora/olora/sgd/imagenetr/olora_inr_sgd_t20c10_r16_train_eval_42_new.log 2>&1 & p8=$!





# mkdir -p ./logs_inc_lora/inflora/sam/imagenetr
# python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_1024/inflora_inr_sam_t20c10_r16.yaml \
#   >> ./logs_inc_lora/inflora/sam/imagenetr/inflora_inr_sam_t20c10_r16_train_eval1024.log 2>&1 & p7=$!
# mkdir -p ./logs_inc_lora/inflora/sgd/imagenetr
# python3 main.py --config=./exp_1_sam-sgd_imageR_r16_t20_seed_1024/inflora_inr_sgd_t20c10_r16.yaml \
#   >> ./logs_inc_lora/inflora/sgd/imagenetr/inflora_inr_sgd_t20c10_r16_train_eval1024.log 2>&1 & p8=$!

echo "olora + inflora (SAM/SGD) ."
wait  $p5 $p6 $p7 $p8 
echo "olora + inflora (SAM/SGD) ."






echo "olora + inflora (SAM/SGD) ."
echo "All runs submitted (stdout logs in ./logs_inc* and ./logs_inc_lora*)."
