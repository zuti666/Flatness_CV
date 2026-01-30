#!/usr/bin/env bash

# # 自动创建日志目录
# mkdir -p ./logs_inc_lora/{inclora,olora,sdlora,seqlora}/{sgd,sam}/tiny_imagenetc

# # ===== 批次 1：tiny_imagenetc（SGD+SAM 并行） 方法 inclora=====
# python3 main.py --config=./exps_inc_lora/inclora_inr_sgd_c10_r8.yaml \
#   >> ./logs_inc_lora/inclora/sgd/tiny_imagenetc/sgd_c10_r8_train_eval.log 2>&1 & p1=$!

# python3 main.py --config=./exps_inc_lora/inclora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/inclora/sam/tiny_imagenetc/sam_c10_r8_train_eval.log 2>&1 & p2=$!

# # wait $p1 $p2
# # ===== 批次 2：tiny_imagenetc（SGD+SAM 并行） 方法 olora=====
# python3 main.py --config=./exps_inc_lora/olora_inr_sgd_c10_r8.yaml \
#   >> ./logs_inc_lora/olora/sgd/tiny_imagenetc/sgd_c10_r8_train_eval.log 2>&1 & p3=$!

# python3 main.py --config=./exps_inc_lora/olora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/olora/sam/tiny_imagenetc/sam_c10_r8_train_eval.log 2>&1 & p4=$!


# wait $p1 $p2 $p3 $p4

# # ===== 批次 3：tiny_imagenetc（SGD+SAM 并行） 方法 sdlora=====
# python3 main.py --config=./exps_inc_lora/sdlora_inr_sgd_c10_r8.yaml \
#   >> ./logs_inc_lora/sdlora/sgd/tiny_imagenetc/sgd_c10_r8_train_eval.log 2>&1 & p1=$!

# python3 main.py --config=./exps_inc_lora/sdlora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/sdlora/sam/tiny_imagenetc/sam_c10_r8_train_eval.log 2>&1 & p2=$!

# # wait $p1 $p2
# # ===== 批次 4：tiny_imagenetc（SGD+SAM 并行） 方法 seqlora=====
# python3 main.py --config=./exps_inc_lora/seqlora_inr_sgd_c10_r8.yaml \
#   >> ./logs_inc_lora/seqlora/sgd/tiny_imagenetc/sgd_c10_r8_train_eval.log 2>&1 & p3=$!

# python3 main.py --config=./exps_inc_lora/seqlora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/seqlora/sam/tiny_imagenetc/sam_c10_r8_train_eval.log 2>&1 & p4=$!


# wait $p1 $p2 $p3 $p4
 

#  # ===== seqlora sam 评估未完成，重新跑
# python3 main.py --config=./exps_inc_lora/seqlora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/seqlora/sam/tiny_imagenetp/seqlora_inr_sam_c10_r8_train_eval.log 2>&1 & p1=$!


#  # ===== inclora sam  sgd评估未完成，重新跑
# python3 main.py --config=./exps_inc_lora/inclora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/inclora/sam/tiny_imagenetp/inclora_inr_sam_c10_r8_train_eval.log 2>&1 & p2=$!
# python3 main.py --config=./exps_inc_lora/inclora_inr_sgd_c10_r8.yaml \
#   >> ./logs_inc_lora/inclora/sgd/tiny_imagenetp/inclora_inr_sgd_c10_r8_train_eval.log 2>&1 & p3=$!

# wait $p1 $p2 $p3 




# ===== olora sgd-sam
# python3 main.py --config=./exps_inc_lora/olora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/olora/sam/imagenetr/olora_inr_sam_c10_r8_train_eval2.log 2>&1 & p2=$!
# python3 main.py --config=./exps_inc_lora/olora_inr_sgd_c10_r8.yaml \
#   >> ./logs_inc_lora/olora/sgd/imagenetr/olora_inr_sgd_c10_r8_train_eval2.log 2>&1 & p3=$!


# # ===== sdlora sgd-sam
# python3 main.py --config=./exps_inc_lora/sdlora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/sdlora/sam/imagenetr/sdlora_inr_sam_c10_r8.log 2>&1 & p1=$!
# python3 main.py --config=./exps_inc_lora/sdlora_inr_sgd_c10_r8.yaml \
#   >> ./logs_inc_lora/sdlora/sgd/imagenetr/sdlora_inr_sgd_c10_r8_train_eval2.log 2>&1 & p4=$!



# wait $p1 $p2 $p3 $p4 