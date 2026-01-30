#!/usr/bin/env bash
set -euo pipefail


# python3 main.py --config=./exps_lora/quick_verify_sdlora_inc_sam.yaml >> ./logs_exps_lora/inclora/imagenetr/sam/train_test1.log 2>&1 &

# Create stdout log directories (training itself writes to logs/<model>/<dataset>/<opt>/...)

# # SDLoRA
# python3 main.py --config=./exps_lora/sdlora_inr_sgd.yaml >> ./logs_exps_lora/sdlora/imagenetr/sgd_train.log 2>&1 & p1=$!
# python3 main.py --config=./exps_lora/sdlora_inr_sam.yaml >> ./logs_exps_lora/sdlora/imagenetr/sam_train.log 2>&1 & p2=$!
# wait $p1 $p2  # 两个都结束再继续

# # # IncLoRA
# python3 main.py --config=./exps_lora/inclora_inr_sgd.yaml >> ./logs_exps_lora/inclora/imagenetr/sgd_train.log 2>&1 & p3=$!
# python3 main.py --config=./exps_lora/inclora_inr_sam.yaml >> ./logs_exps_lora/inclora/imagenetr/sam_train.log 2>&1 & p4=$!
# wait $p3 $p4

# # # SeqLoRA
# python3 main.py --config=./exps_lora/seqlora_inr_sgd.yaml >> ./logs_exps_lora/seqlora/imagenetr/sgd_train.log 2>&1 & p5=$!
# python3 main.py --config=./exps_lora/seqlora_inr_sam.yaml >> ./logs_exps_lora/seqlora/imagenetr/sam_train.log 2>&1 & p6=$!
# wait $p5 $p6

# # # oLoRA
# python3 main.py --config=./exps_lora/olora_inr_sgd.yaml >> ./logs_exps_lora/olora/imagenetr/sgd_train.log 2>&1 & p7=$!
# python3 main.py --config=./exps_lora/olora_inr_sam.yaml >> ./logs_exps_lora/olora/imagenetr/sam_train.log 2>&1 & p8=$!
# wait $p7 $p8

# ## tuna 
# # python3 main.py --config=./exps_inc/tuna_sgd.yaml >> ./logs_inc/tuna/sgd/imagenetr 2>&1 & p7=$!
# python3 main.py --config=./exps_inc/tuna_sam.yaml >> ./logs_inc/tuna/sam/imagenetr 2>&1 & p8=$!
# # wait $p7 $p8


# ## mos
# python3 main.py --config=./exps_inc/mos_sgd.yaml >> ./logs_inc/mos/sgd/imagenetr/try_sgd.log  2>&1 & p8=$!
# python3 main.py --config=./exps_inc/mos_sam.yaml >> ./logs_inc/mos/sam/imagenetr/try_sam.log  2>&1 & p8=$!

# python3 main.py --config=./exps_inc_lora/inclora_inr_sgd_p_5.yaml \
#   >> ./logs_inc_lora/inclora/sgd/tiny_imagenetp/sgd_test_rankclass4.log 2>&1 & p3=$!

# python3 main.py --config=./exps_inc_lora/inclora_inr_sam_p_5.yaml \
#   >> ./logs_inc_lora/inclora/sam/tiny_imagenetp/sam_test_rankclass4.log 2>&1 & p4=$!

# wait $p3 $p4



# # ===== 批次 2：imagenetr（SGD+SAM 并行） 方法 olora=====
# python3 main.py --config=./exps_inc_lora/inclora_inr_sgd_c10_r8.yaml \
#   >> ./logs_inc_lora/inclora/sgd/tiny_imagenetc/sgd_c10_r16_train_eval.log 2>&1 & p1=$!

# python3 main.py --config=./exps_inc_lora/inclora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/inclora/sam/tiny_imagenetc/sam_c10_r16_train_eval.log 2>&1 & p2=$!


# # ===== 批次 2：imagenetr（SGD+SAM 并行） 方法 olora=====
# python3 main.py --config=./exps_inc_lora/seqlora_inr_sgd_c10_r8.yaml \
#   >> ./logs_inc_lora/seqlora/sgd/tiny_imagenetc/sgd_c10_r16_train_eval.log 2>&1 & p3=$!
# python3 main.py --config=./exps_inc_lora/seqlora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/seqlora/sam/tiny_imagenetc/sam_c10_r16_train_eval.log 2>&1 & p4=$!

# wait $p1 $p2 $p3 $p4

# ===== inclora sgd-sam
python3 main.py --config=./exps_inc_lora/inclora_inr_sam_c10_r8.yaml \
  >> ./logs_inc_lora/inclora/sam/imagenetr/inclora_inr_sam_c10_r16_train_eval.log 2>&1 & p2=$!
python3 main.py --config=./exps_inc_lora/inclora_inr_sgd_c10_r8.yaml \
  >> ./logs_inc_lora/inclora/sgd/imagenetr/inclora_inr_sgd_c10_r16_train_eval.log 2>&1 & p3=$!


# ===== seqlora sgd-sam
python3 main.py --config=./exps_inc_lora/seqlora_inr_sam_c10_r8.yaml \
  >> ./logs_inc_lora/seqlora/sam/imagenetr/seqlora_inr_sam_c10_r16_train_eval.log 2>&1 & p1=$!
python3 main.py --config=./exps_inc_lora/seqlora_inr_sgd_c10_r8.yaml \
  >> ./logs_inc_lora/seqlora/sgd/imagenetr/seqlora_inr_sgd_c10_r16_train_eval.log 2>&1 & p4=$!



wait $p1 $p2 $p3 $p4 