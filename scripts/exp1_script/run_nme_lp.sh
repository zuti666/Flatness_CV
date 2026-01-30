mkdir -p ./logs_inc/linearprobe/sam/imagenetr
python3 main.py --config=./exp1_lp_nme_1993/LP_sam_imageR_t20.yaml \
  >> ./logs_inc/linearprobe/sam/imagenetr/LP_sam_imageR_t20_train_eval1993_NME.log 2>&1 & p9=$!
mkdir -p ./logs_inc/linearprobe/sgd/imagenetr
python3 main.py --config=./exp1_lp_nme_1993/LP_sgd_imageR_t20.yaml \
  >> ./logs_inc/linearprobe/sgd/imagenetr/LP_sgd_imageR_t20_train_eval1993_NME.log 2>&1 & p10=$!
