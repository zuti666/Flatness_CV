#!/usr/bin/env bash
set -euo pipefail

# Run all configs in exp_1_sam-sgd_imageR_r16_t20 with updated paths only.
# ===== Wave 1: inclora 
mkdir -p ./logs_inc_lora/inclora/sam/imagenetr
python3 main.py --config=./exp4_ablation_length/inclora_inr_sam_t40c5_r16.yaml \
  >> ./logs_inc_lora/inclora/sam/imagenetr/inclora_inc_sam_t10c20_r16_train_2048_cosine_redonew2.log 2>&1 & p1=$!

mkdir -p ./logs_inc_lora/inclora/sgd/imagenetr
python3 main.py --config=./exp4_ablation_length/inclora_inr_sgd_t40c5_r16.yaml \
  >> ./logs_inc_lora/inclora/sgd/imagenetr/inclora_inc_sgd_t10c20_r16_train_2048_cosine_redonew3.log 2>&1 & p2=$!

mkdir -p ./logs_inc_lora/inclora/gam/imagenetr
python3 main.py --config=./exp4_ablation_length/inclora_inr_gam_t40c5_r16.yaml \
  >> ./logs_inc_lora/inclora/gam/imagenetr/inclora_inc_gam_t10c20_r16_train_2048_cosine.log 2>&1 & p7=$!


#  seqlora 
mkdir -p ./logs_inc_lora/seqlora/sam/imagenetr
python3 main.py --config=./exp4_ablation_length/seqlora_inr_sam_t40c5_r16.yaml \
  >> ./logs_inc_lora/seqlora/sam/imagenetr/seqlora_inc_sam_t10c20_r16_train_2048_cosine_redonew2.log 2>&1 & p3=$!

mkdir -p ./logs_inc_lora/seqlora/sgd/imagenetr
python3 main.py --config=./exp4_ablation_length/seqlora_inr_sgd_t40c5_r16.yaml \
  >> ./logs_inc_lora/seqlora/sgd/imagenetr/seqlora_inc_sgd_t10c20_r16_train_2048_cosine_redonew2.log 2>&1 & p4=$!


mkdir -p ./logs_inc_lora/seqlora/gam/imagenetr
python3 main.py --config=./exp4_ablation_length/seqlora_inr_gam_t40c5_r16.yaml \
  >> ./logs_inc_lora/seqlora/gam/imagenetr/seqlora_inc_gam_t10c20_r16_train_2048_cosine.log 2>&1 & p8=$!


#  olora 
mkdir -p ./logs_inc_lora/olora/sam/imagenetr
python3 main.py --config=./exp4_ablation_length/olora_inr_sam_t40c5_r16.yaml \
  >> ./logs_inc_lora/olora/sam/imagenetr/olora_inc_sam_t10c20_r16_train_2048_cosine_redonew2.log 2>&1 & p5=$!

mkdir -p ./logs_inc_lora/olora/sgd/imagenetr
python3 main.py --config=./exp4_ablation_length/olora_inr_sgd_t40c5_r16.yaml \
 >> ./logs_inc_lora/olora/sgd/imagenetr/olora_inc_sgd_t10c20_r16_train_2048_cosine_redonew2.log 2>&1 & p6=$!


mkdir -p ./logs_inc_lora/olora/gam/imagenetr
python3 main.py --config=./exp4_ablation_length/olora_inr_gam_t40c5_r16.yaml \
  >> ./logs_inc_lora/olora/gam/imagenetr/olora_inc_gam_t10c20_r16_train_2048_cosine.log 2>&1 & p9=$!





# mkdir -p ./logs_inc_lora/inclora/gam/imagenetr
# python3 main.py --config=./exp4_ablation_length/inclora_inr_gam_t40c5_r16.yaml \
#   >> ./logs_inc_lora/inclora/gam/imagenetr/inclora_inc_gam_t40c5_r16_train_2048_cosine.log 2>&1 & p7=$!


# mkdir -p ./logs_inc_lora/olora/gam/imagenetr
# python3 main.py --config=./exp4_ablation_length/olora_inr_gam_t40c5_r16.yaml \
#   >> ./logs_inc_lora/olora/gam/imagenetr/olora_inc_gam_t40c5_r16_train_2048_cosine.log 2>&1 & p9=$!
