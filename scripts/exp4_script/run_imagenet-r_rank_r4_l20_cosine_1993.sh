#!/usr/bin/env bash
set -euo pipefail

# Run all configs in exp_1_sam-sgd_imageR_r8_t20 with updated paths only.
# ===== Wave 1: inclora 
# mkdir -p ./logs_inc_lora/inclora/sam/imagenetr
# python3 main.py --config=./exp4_ablation_rank/inclora_inr_sam_t20c10_r4.yaml \
#   >> ./logs_inc_lora/inclora/sam/imagenetr/inclora_inc_sam_t20c10_r2_train_1993_cosine_redonew2.log 2>&1 & p1=$!

# mkdir -p ./logs_inc_lora/inclora/sgd/imagenetr
# python3 main.py --config=./exp4_ablation_rank/inclora_inr_sgd_t20c10_r4.yaml \
#   >> ./logs_inc_lora/inclora/sgd/imagenetr/inclora_inc_sgd_t20c10_r2_train_1993_cosine_redonew2.log 2>&1 & p2=$!

# python3 main.py --config=./exp4_ablation_rank/rank16/olora_inr_sam_t20c10_r16.yaml \
#   >> ./logs_inc_lora/olora/sam/imagenetr/olora_inc_sam_t20c10_r16_train_1993_cosine.log 2>&1 & 

# python3 main.py --config=./exp4_ablation_rank/rank16/olora_inr_sgd_t20c10_r16.yaml \
#   >> ./logs_inc_lora/olora/sgd/imagenetr/olora_inc_sgd_t20c10_r16_train_1993_cosine.log 2>&1 & 



# #  seqlora 
# mkdir -p ./logs_inc_lora/seqlora/sam/imagenetr
# python3 main.py --config=./exp4_ablation_rank/seqlora_inr_sam_t20c10_r4.yaml \
#   >> ./logs_inc_lora/seqlora/sam/imagenetr/seqlora_inc_sam_t20c10_r2_train_1993_cosine_redonew3.log 2>&1 & p3=$!

# mkdir -p ./logs_inc_lora/seqlora/sgd/imagenetr
# python3 main.py --config=./exp4_ablation_rank/seqlora_inr_sgd_t20c10_r4.yaml \
#   >> ./logs_inc_lora/seqlora/sgd/imagenetr/seqlora_inc_sgd_t20c10_r2_train_1993_cosine_redonew3.log 2>&1 & p4=$!



# #  olora 
# mkdir -p ./logs_inc_lora/olora/sam/imagenetr
# python3 main.py --config=./exp4_ablation_rank/olora_inr_sam_t20c10_r4.yaml \
#   >> ./logs_inc_lora/olora/sam/imagenetr/olora_inc_sam_t20c10_r2_train_1993_cosine_redonew2.log 2>&1 & p5=$!

# mkdir -p ./logs_inc_lora/olora/sgd/imagenetr
# python3 main.py --config=./exp4_ablation_rank/olora_inr_sgd_t20c10_r4.yaml \
#  >> ./logs_inc_lora/olora/sgd/imagenetr/olora_inc_sgd_t20c10_r2_train_1993_cosine_redonew2.log 2>&1 & p6=$!




# mkdir -p ./logs_inc_lora/inclora/gam/imagenetr
# python3 main.py --config=./exp4_ablation_rank/inclora_inr_gam_t20c10_r4.yaml \
#   >> ./logs_inc_lora/inclora/gam/imagenetr/inclora_inc_gam_t20c10_r2_train_1993_cosine.log 2>&1 & p7=$!



# mkdir -p ./logs_inc_lora/seqlora/gam/imagenetr
# python3 main.py --config=./exp4_ablation_rank/seqlora_inr_gam_t20c10_r4.yaml \
#   >> ./logs_inc_lora/seqlora/gam/imagenetr/seqlora_inc_gam_t20c10_r2_train_1993_cosine.log 2>&1 & p8=$!

# mkdir -p ./logs_inc_lora/olora/gam/imagenetr
# python3 main.py --config=./exp4_ablation_rank/olora_inr_gam_t20c10_r4.yaml \
#   >> ./logs_inc_lora/olora/gam/imagenetr/olora_inc_gam_t20c10_r2_train_1993_cosine.log 2>&1 & p9=$!



# mkdir -p ./logs_inc_lora/inclora/gam/imagenetr
# python3 main.py --config=./exp4_ablation_rank/rank8/inclora_inr_gam_t20c10_r8.yaml \
#   >> ./logs_inc_lora/inclora/gam/imagenetr/inclora_inc_gam_t20c10_r8_train_2048_cosine.log 2>&1 & p1=$!
# mkdir -p ./logs_inc_lora/seqlora/gam/imagenetr
# python3 main.py --config=./exp4_ablation_rank/rank8/seqlora_inr_gam_t20c10_r8.yaml \
#   >> ./logs_inc_lora/seqlora/gam/imagenetr/seqlora_inc_gam_t20c10_r8_train_2048_cosine.log 2>&1 & p2=$!


mkdir -p ./logs_inc_lora/olora/gam/imagenetr
python3 main.py --config=./exp4_ablation_rank/rank8/olora_inr_gam_t20c10_r8.yaml \
  >> ./logs_inc_lora/olora/gam/imagenetr/olora_inc_gam_t20c10_r8_train_2048_cosine2.log 2>&1 & p3=$!


# mkdir -p ./logs_inc_lora/inclora/sam/imagenetr ./logs_inc_lora/seqlora/sam/imagenetr ./logs_inc_lora/olora/sam/imagenetr
# mkdir -p ./logs_inc_lora/inclora/sgd/imagenetr ./logs_inc_lora/seqlora/sgd/imagenetr ./logs_inc_lora/olora/sgd/imagenetr
# python3 main.py --config=./exp4_ablation_rank/rank8/inclora_inr_sam_t20c10_r8.yaml \
#   >> ./logs_inc_lora/inclora/sam/imagenetr/inclora_inc_sam_t20c10_r8_train_2048_cosine.log 2>&1 & p4=$!
# python3 main.py --config=./exp4_ablation_rank/rank8/seqlora_inr_sam_t20c10_r8.yaml \
#   >> ./logs_inc_lora/seqlora/sam/imagenetr/seqlora_inc_sam_t20c10_r8_train_2048_cosine.log 2>&1 & p5=$!
# python3 main.py --config=./exp4_ablation_rank/rank8/olora_inr_sam_t20c10_r8.yaml \
#   >> ./logs_inc_lora/olora/sam/imagenetr/olora_inc_sam_t20c10_r8_train_2048_cosine.log 2>&1 & p6=$!


# python3 main.py --config=./exp4_ablation_rank/rank8/inclora_inr_sgd_t20c10_r8.yaml \
#   >> ./logs_inc_lora/inclora/sgd/imagenetr/inclora_inc_sgd_t20c10_r8_train_2048_cosine.log 2>&1 & p7=$!
# python3 main.py --config=./exp4_ablation_rank/rank8/seqlora_inr_sgd_t20c10_r8.yaml \
#   >> ./logs_inc_lora/seqlora/sgd/imagenetr/seqlora_inc_sgd_t20c10_r8_train_2048_cosine.log 2>&1 & p8=$!
# python3 main.py --config=./exp4_ablation_rank/rank8/olora_inr_sgd_t20c10_r8.yaml \
#   >> ./logs_inc_lora/olora/sgd/imagenetr/olora_inc_sgd_t20c10_r8_train_2048_cosine.log 2>&1 & p9=$!