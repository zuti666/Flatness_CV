# eval_origin_model on imagenet-p 
# python3 main.py --config=./exps_all/finetune_all_sgd_eval_c.yaml >> ./logs_all/finetune/sgd/tiny_imagenetp/eval_originmodel_imagep.log 2>&1 &


# & p1=$!
# train_origin_model
# python3 main.py --config=./exps_all/finetune_all_sgd_c_10cos.yaml >> ./logs_all/finetune/sgd/tiny_imagenetc/finetune_all_sgd_c_10cos.log 2>&1 & p1=$!

# # finetune 整个模型 
# python3 main.py --config=./exps_all/FT_all_sgd_eval_c.yaml \
#     >> ./logs_all/finetune/sgd/tiny_imagenetp/finetune_imagep.log 2>&1 & p1=$!



# # # LinearPrbe 整个模型
# python3 main.py --config=./exps_all/LP_all_sgd_imageP.yaml \
#   >> ./logs_all/linearprobe/sgd/tiny_imagenetp/LP_all_sgd_imageP.log 2>&1 & p2=$!


# # 单个LoRA 训练整个模型
# # exps_all/seqlora_all_sgd.yaml
# python3 main.py --config=./exps_all/seqlora_all_sgd.yaml \
#   >> ./logs_all/seqlora/sgd/tiny_imagenetp/LORA_ALL_RANK8.log 2>&1 & p3=$!


# wait $p1 $p2 $p3 


# # PerTaskFT

# python3 main.py --config=./exps_inc/FT_origin_sam_imageP.yaml \
#   >> ./logs_inc/finetune/sam/tiny_imagenetp/sdlora_inr_sam_c10_r8.log 2>&1 & p3=$!

# python3 main.py --config=./exps_inc/FT_origin_sgd_imageP.yaml \
#   >> ./logs_inc/finetune/sgd/tiny_imagenetp/sdlora_inr_sgd_c10_r8.log 2>&1 & p4=$!


# ===== olora sgd-sam
python3 main.py --config=./exps_inc_lora/olora_inr_sam_c10_r8.yaml \
  >> ./logs_inc_lora/olora/sam/imagenetr/olora_inr_sam_c10_r16_train_eval2_true.log 2>&1 & p2=$!
python3 main.py --config=./exps_inc_lora/olora_inr_sgd_c10_r8.yaml \
  >> ./logs_inc_lora/olora/sgd/imagenetr/olora_inr_sgd_c10_r16_train_eval2_true.log 2>&1 & p3=$!


# ===== sdlora sgd-sam
python3 main.py --config=./exps_inc_lora/sdlora_inr_sam_c10_r8.yaml \
  >> ./logs_inc_lora/sdlora/sam/imagenetr/sdlora_inr_sam_c10_r16__train_eval2_true.log 2>&1 & p1=$!
python3 main.py --config=./exps_inc_lora/sdlora_inr_sgd_c10_r8.yaml \
  >> ./logs_inc_lora/sdlora/sgd/imagenetr/sdlora_inr_sgd_c10_r16_train_eval2_true.log 2>&1 & p4=$!



wait $p1 $p2 $p3 $p4 