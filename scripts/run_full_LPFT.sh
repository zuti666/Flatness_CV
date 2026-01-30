# export HF_HUB_OFFLINE=1
# # Finetune SGD
# python3 main.py --config=./exps_all/finetune_all_sgd2.yaml  >> ./logs_all/finetune/sgd/imagenetr/sgd_lr_2.log 2>&1 & 

# # python3 main.py --config=./exps_all/finetune_all_sgd2.yaml  >> ./logs_all/finetune/sgd/imagenetr/sgd_lr_1.log 2>&1 & 
# # python3 main.py --config=./exps_all/LinearProbe-FineTuning_all_sgd2.yaml >> ./logs_all/linearprobe_finetune/sgd/imagenetr/sgd_test7.log 2>&1 & 
# # python3 main.py --config=./exps_all/LPFT_EFM_all_sgd2.yaml  >> ./logs_all/lpft_efm/sgd/imagenetr/sgd_test7.log 2>&1 & 




# python3 main.py --config=./exps_all/LinearProbe-FineTuning_all_sgd2.yaml >> ./logs_all/linearprobe_finetune/sgd/imagenetr/sgd_lr1.log 2>&1 & 


# python3 main.py --config=./exps_all/LPFT_EFM_all_sgd2.yaml >> ./logs_all/lpft_efm/sgd/imagenetr/sgd_lr13.log 2>&1 & 


# mkdir -p ./logs_all/linearprobe/sgd/tiny_imagenetc 

# python3 main.py --config=./exps_all/linearprobe_all_sgd_eval_c.yaml >> ./logs_all/linearprobe/sgd/tiny_imagenetc/test_lossland.log 2>&1 &


# PerTaskFT

python3 main.py --config=./exps_inc/FT_origin_sam_imageP.yaml \
  >> ./logs_inc/finetune/sam/tiny_imagenetp/sdlora_inr_sam_c10_r8.log 2>&1 & p3=$!

python3 main.py --config=./exps_inc/FT_origin_sgd_imageP.yaml \
  >> ./logs_inc/finetune/sgd/tiny_imagenetp/sdlora_inr_sgd_c10_r8.log 2>&1 & p4=$!