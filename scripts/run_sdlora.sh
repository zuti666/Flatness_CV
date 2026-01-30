# python3 main.py --config=./exps/sdlora_c100.json  >> ./logs/sdlora/cifar224/train_sdlora.log 2>&1 &
# python3 main.py --config=./exps/seqlora_c100.json  >> ./logs/seqlora/cifar224/train_seqlora.log 2>&1 &
# # python3 main.py --config=./exps/inclora_c100.json  >> ./logs/inclora/cifar224/train_inclora.log 2>&1 &
# python3 main.py --config=./exps/exps/olora_c100.json  >> ./logs/olora/cifar224/train_olora.log 2>&1 &


# python3 main.py --config=./exps/seqlora_c100.json  >> ./logs/seqlora/cifar224/train_seqlora.log 2>&1 &


# python3 main.py --config=./exps/sdlora_inr_gam_test.json >> ./logs/sdlora/imagenetr/test3/gam/test.log 2>&1 &

# python3 main.py --config=./exps/inclora_inr_sgd.json >> ./logs/inclora/imagenetr/sgd/train.log 2>&1 &
# python3 main.py --config=./exps/inclora_inr_sam.json >> ./logs/inclora/imagenetr/sam/train.log 2>&1 &


# python3 main.py --config=./exps/seqlora_inr_sgd.json >> ./logs/seqlora/imagenetr/sgd/train.log 2>&1 &
# python3 main.py --config=./exps/seqlora_inr_sam.json >> ./logs/seqlora/imagenetr/sam/train.log 2>&1 &

# python3 main.py --config=./exps_inc_lora/sdlora_inr_sgd.yaml >> ./logs_inc_lora/sdlora/imagenetr/logs_train.log 2>&1 &


# >> ./logs_exps_lora/sdlora/imagenetr/sgd/train2.log 2>&1 &

# finetune 
# python3 main.py --config=./exps_inc/finetune_inr_sgd.yaml >> ./logs_inc/finetune/imagenetr/logs_train2.log 2>&1 &


# python3 main.py --config=./exps_all/quick_verify_finetune_all_sam.yaml >> ./logs_all/finetune/imagenetr/2048/test_eval.log 2>&1


# python3 main.py --config=./exps_all/quick_verify_seqlora_alldata_sam.yaml >> ./logs_all/seqlora/imagenetr/2048/test_seqlora_eval.log 2>&1


# # # IncLoRA
# python3 main.py --config=./exps_inc_lora/inclora_inr_sgd.yaml >> ./logs_inc_lora/inclora/sam/tiny_imagenetp/sgd_train_0.log 2>&1 & p3=$!
# python3 main.py --config=./exps_inc_lora/inclora_inr_sam.yaml >> ./logs_inc_lora/inclora/sam/tiny_imagenetp/sam_train_0.log 2>&1 & p4=$!


# wait $p3 $p4 

# python3 main.py --config=./exps_inc_lora/inclora_inr_sgd.yaml >> ./logs_inc_lora/inclora/sgd/tiny_imagenetp/sgd_test_rankclass.log 2>&1 & p3=$!


# python3 main.py --config=./exps_inc_lora/inclora_inr_sam.yaml >> ./logs_inc_lora/inclora/sam/tiny_imagenetp/sam_test_rankclass.log 2>&1 & p4=$!

# wait $p3 $p4 


# oLoRA
# python3 main.py --config=./exps_inc_lora/olora_inr_sgd.yaml >> ./logs_inc_lora/olora/sgd/imagenetr/train_eva_1.log 2>&1 & p3=$!
# python3 main.py --config=./exps_inc_lora/olora_inr_sam.yaml >> ./logs_inc_lora/olora/sam/imagenetr/train_eva_1.log 2>&1 & p4=$!
# wait $p3 $p4

# sdLoRA
# python3 main.py --config=./exps_inc_lora/sdlora_inr_sgd.yaml >> ./logs_inc_lora/sdlora/sgd/imagenetr/train_eva_1.log 2>&1 & p5=$!
# python3 main.py --config=./exps_inc_lora/sdlora_inr_sam.yaml >> ./logs_inc_lora/sdlora/sam/imagenetr/train_eva_1.log 2>&1 & p6=$!
# wait $p5 $p6

# # # SeqLoRA
# python3 main.py --config=./exps_inc_lora/seqlora_inr_sgd.yaml >> ./logs_inc_lora/seqlora/imagenetr/sgd_train_0.log 2>&1 & p5=$!
# python3 main.py --config=./exps_inc_lora/seqlora_inr_sam.yaml >> ./logs_inc_lora/seqlora/imagenetr/sam_train_0.log 2>&1 & p6=$!
# wait $p3 $p4  wait $p5 $p6
# echo "[进度] IncLoRA 两个任务已完成 (SGD + SAM)"
# echo "[进度] SeqLoRA 两个任务已完成 (SGD + SAM)"

# Finetune
# python3 main.py --config=./exps_inc/finetune_inr_sam.yaml >> ./logs_inc_lora/finetune/imagenetr/sam_train_0.log 2>&1 & p5=$!
# python3 main.py --config=./exps_inc/finetune_inr_sgd_iamgec.yaml >> ./logs_inc_lora/finetune/sgd/imagenetr/sgd_train_0.log 2>&1 & 
# wait $p5 $p6
# echo "[进度] Finetune 两个任务已完成 (SAM + SGD)"


# python3 main.py --config=./exps_all/quick_verify_seqlora_alldata_sam.yaml >> ./logs_all/seqlora/imagenetr/2048/test_falt_eval2.log 2>&1 & p6=$!


# python3 main.py --config=./exps_inc/finetune_inr_sgd.yaml >> ./logs_inc_lora/finetune/sgd/imagenetr/sgd_train_0.log 2>&1 & 

# train


# python3 main.py --config=./exps_inc/finetune_inr_sgd.yaml >> ./logs_inc_lora/finetune/sgd/imagenetr/test_lr1.log 2>&1 &

# python3 main.py --config=./exps_inc/LinearProbe-FineTuning_inr_sgd.yaml >> ./logs_inc_lora/linearprobe_finetune/sgd/imagenetr/test_lr1.log 2>&1 &


# python3 main.py --config=./exps_all/LPFT_EFM_all_sgd2.yaml >> ./logs_inc_lora/lpft_efm/sgd/imagenetr/test_lr1.log 2>&1 &


# python3 main.py --config=./exps_inc/LPFT_inr_sgd_imagec.yaml >> ./logs_inc_lora/linearprobe_finetune/sgd/imagenetr/test_lr1.log 2>&1 &


# python3 main.py --config=./exps_inc/LPFT_EFM_inr_sgd_imagec.yaml >> ./logs_inc_lora/lpft_efm/sgd/imagenetr/test_lr2.log 2>&1 &


# python3 main.py --config=./exps_inc/finetune_inr_sgd_imagec.yaml >> ./logs_inc_lora/finetune/sgd/imagenetr/test_lr2.log 2>&1 &


# python3 main.py --config=./exps_inc/tuna.yaml >> ./logs_inc/tuna/sgd/imagenetr/test_lr1.log 2>&1 &


# python3 main.py --config=./exps_inc/tuna_efm.yaml >> ./logs_inc/tuna_efm/sgd/imagenetr/test_lr1.log 2>&1 &


# python3 main.py --config=./exps_inc/finetune_inr_sgd_imager_origin.yaml >> ./logs_inc/finetune/sgd/imagenetr/check_origin.log 2>&1 &


# python3 main.py --config=./exps_inc/LinearProbe-FineTuning_inr_sgd.yaml >> ./logs_inc_lora/linearprobe_finetune/sgd/imagenetr/sgd_train_0.log 2>&1 &



# python3 main.py --config=./exps_inc/LPFT_EFM_inr_sgd.yaml >> ./logs_inc_lora/lpft_efm/sgd/imagenetr/sgd_train_0.log 2>&1 &

#!/usr/bin/env bash
# set -euo pipefail




# 自动创建日志目录
# mkdir -p ./logs_inc_lora/inclora/sgd/imagenetr \
#          ./logs_inc_lora/inclora/sam/imagenetr \
#          ./logs_inc_lora/inclora/sgd/tiny_imagenetp \
#          ./logs_inc_lora/inclora/sam/tiny_imagenetp

# 如需保留你原来的固定日志文件名（可能被并发追加），保持如下两行；
# 如果想避免并发“互踩”，把下方的两处 *.log 换成带时间戳版本：
#   sgd_test_rankclass_$(date +%F_%H-%M-%S).log
#   sam_test_rankclass_$(date +%F_%H-%M-%S).log

# ===== 批次 1：imagenetr（SGD+SAM 并行）=====
# python3 main.py --config=./exps_inc_lora/inclora_inr_sgd_c_50.yaml \
#   >> ./logs_inc_lora/inclora/sgd/imagenetr/sgd_test_rankclass.log 2>&1 & p1=$!

# python3 main.py --config=./exps_inc_lora/inclora_inr_sam_c_50.yaml \
#   >> ./logs_inc_lora/inclora/sam/imagenetr/sam_test_rankclass.log 2>&1 & p2=$!

# wait $p1 $p2

# ===== 批次 2：tiny_imagenetp（SGD+SAM 并行）=====
# python3 main.py --config=./exps_inc_lora/inclora_inr_sgd_p_5.yaml \
#   >> ./logs_inc_lora/inclora/sgd/tiny_imagenetp/sgd_test_rankclass2.log 2>&1 & p3=$!

# python3 main.py --config=./exps_inc_lora/inclora_inr_sam_p_5.yaml \
#   >> ./logs_inc_lora/inclora/sam/tiny_imagenetp/sam_test_rankclass2.log 2>&1 & p4=$!

# wait $p3 $p4

# mkdir -p ./logs_inc_lora/{inclora,olora,sdlora,seqlora}/{sgd,sam}/imagenetr




# ===== inclora sgd-sam
python3 main.py --config=./exps_inc_lora/inclora_inr_sam_c10_r8.yaml \
  >> ./logs_inc_lora/inclora/sam/imagenetr/inclora_inr_sam_c5_r32_train_eval2.log 2>&1 & p2=$!
python3 main.py --config=./exps_inc_lora/inclora_inr_sgd_c10_r8.yaml \
  >> ./logs_inc_lora/inclora/sgd/imagenetr/inclora_inr_sgd_c5_r32_train_eval2.log 2>&1 & p3=$!


# ===== seqlora sgd-sam
python3 main.py --config=./exps_inc_lora/seqlora_inr_sam_c10_r8.yaml \
  >> ./logs_inc_lora/seqlora/sam/imagenetr/seqlora_inr_sam_c5_r32_train_eval2.log 2>&1 & p1=$!
python3 main.py --config=./exps_inc_lora/seqlora_inr_sgd_c10_r8.yaml \
  >> ./logs_inc_lora/seqlora/sgd/imagenetr/seqlora_inr_sgd_c5_r32_train_eval2.log 2>&1 & p4=$!



wait $p1 $p2 $p3 $p4 


# ===== olora sgd-sam
python3 main.py --config=./exps_inc_lora/olora_inr_sam_c10_r8.yaml \
  >> ./logs_inc_lora/olora/sam/imagenetr/olora_inr_sam_c5_r32_train_eval2_true.log 2>&1 & p2=$!
python3 main.py --config=./exps_inc_lora/olora_inr_sgd_c10_r8.yaml \
  >> ./logs_inc_lora/olora/sgd/imagenetr/olora_inr_sgd_c5_r32_train_eval2_true.log 2>&1 & p3=$!


# ===== sdlora sgd-sam
# python3 main.py --config=./exps_inc_lora/sdlora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/sdlora/sam/imagenetr/sdlora_inr_sam_c5$$


# # ===== ft sgd-sam
# python3 main.py --config=./exps_inc/FT_origin_sam_imageP.yaml \
#   >> ./logs_inc/finetune/sam/imagenetr/FT_inr_sam_c10_train_eval2_true.log 2>&1 & p1=$!
# python3 main.py --config=./exps_inc/FT_origin_sgd_imageP.yaml \
#   >> ./logs_inc/finetune/sgd/imagenetr/FT_inr_sgd_c10_train_eval2_true.log 2>&1 & p4=$!


# wait $p1 $p4




# python3 main.py --config=./exps_inc_lora/inclora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/inclora/sam/imagenetr/sam_c10_r4_train_eval2.log 2>&1 & p2=$!

# # wait $p1 $p2
# # ===== 批次 2：imagenetr（SGD+SAM 并行） 方法 olora=====
# python3 main.py --config=./exps_inc_lora/seqlora_inr_sgd_c10_r8.yaml \
#   >> ./logs_inc_lora/seqlora/sgd/imagenetr/sgd_c10_r4_train_eval2.log 2>&1 & p3=$!

# python3 main.py --config=./exps_inc_lora/seqlora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/seqlora/sam/imagenetr/sam_c10_r4_train_eval2.log 2>&1 & p4=$!

# # ===== 批次 3：imagenetr（SGD+SAM 并行） 方法 sdlora=====
# python3 main.py --config=./exps_inc_lora/sdlora_inr_sgd_c10_r8.yaml \
#   >> ./logs_inc_lora/sdlora/sgd/tiny_imagenetc/sgd_c10_r8_train_eval2.log 2>&1 & p1=$!

# python3 main.py --config=./exps_inc_lora/sdlora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/sdlora/sam/tiny_imagenetc/sam_c10_r8_train_eval2.log 2>&1 & p2=$!

# # wait $p1 $p2
# # ===== 批次 4：imagenetr（SGD+SAM 并行） 方法 seqlora=====
# python3 main.py --config=./exps_inc_lora/olora_inr_sgd_c10_r8.yaml \
#   >> ./logs_inc_lora/olora/sgd/tiny_imagenetc/sgd_c10_r8_train_eval2.log 2>&1 & p3=$!

# python3 main.py --config=./exps_inc_lora/olora_inr_sam_c10_r8.yaml \
#   >> ./logs_inc_lora/olora/sam/tiny_imagenetc/sam_c10_r8_train_eval2.log 2>&1 & p4=$!


# wait $p1 $p2 $p3 $p4
 