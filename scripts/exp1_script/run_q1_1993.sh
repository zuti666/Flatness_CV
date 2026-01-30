#!/usr/bin/env bash
set -euo pipefail

# Run all configs in exp_1_sam-sgd_imageR_r16_t20 with updated paths only.

# ===== Wave 1: inclora + seqlora (SAM/SGD)
mkdir -p ./logs_inc_lora/inclora/sam/imagenetr
python3 main.py --config=./exp1_lp_nme_1993/inclora_inr_sam_t20c10_r16.yaml \
  >> ./logs_inc_lora/inclora/sam/imagenetr/inclora_inr_sam_t20c10_r16_train_eval_1993_redo.log 2>&1 & p1=$!
mkdir -p ./logs_inc_lora/inclora/sgd/imagenetr
python3 main.py --config=./exp1_lp_nme_1993/inclora_inr_sgd_t20c10_r16.yaml \
  >> ./logs_inc_lora/inclora/sgd/imagenetr/inclora_inr_sgd_t20c10_r16_train_eval_1993_redo.log 2>&1 & p2=$!

mkdir -p ./logs_inc_lora/seqlora/sam/imagenetr
python3 main.py --config=./exp1_lp_nme_1993/seqlora_inr_sam_t20c10_r16.yaml \
  >> ./logs_inc_lora/seqlora/sam/imagenetr/seqlora_inr_sam_t20c10_r16_train_eval_1993_redo.log 2>&1 & p3=$!
mkdir -p ./logs_inc_lora/seqlora/sgd/imagenetr
python3 main.py --config=./exp1_lp_nme_1993/seqlora_inr_sgd_t20c10_r16.yaml \
  >> ./logs_inc_lora/seqlora/sgd/imagenetr/seqlora_inr_sgd_t20c10_r16_train_eval_1993_redo.log 2>&1 & p4=$!


echo "inclora + seqlora (SAM/SGD) ."
wait $p1 $p2 $p3 $p4
echo "inclora + seqlora (SAM/SGD) ."
