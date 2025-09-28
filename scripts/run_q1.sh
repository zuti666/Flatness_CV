# # Baseline LoRA-only SGD
# python3 main.py --config exps/seqlora_c100.json --override optimizer_type=sgd flat_eval=true flat_eval_num_samples=16 linear_probe_eval=false >> ./logs/seqlora/cifar224/InR.log 2>&1 &
# # python3 main.py --config exps/inclora_c100.json --override optimizer_type=sgd flat_eval=true linear_probe_eval=false
# # python3 main.py --config exps/olora_c100.json  --override optimizer_type=sgd flat_eval=true linear_probe_eval=false
# # python3 main.py --config exps/sdlora_c100.json --override optimizer_type=sgd flat_eval=true linear_probe_eval=false

# # # LoRA-only SAM (ρ = 0.05; repeat for 0.10, 0.20)
# # python3 main.py --config exps/seqlora_c100.json --override optimizer_type=sam sam_rho=0.05 flat_eval=true flat_eval_rho=0.05
# # python3 main.py --config exps/inclora_c100.json --override optimizer_type=sam sam_rho=0.05 flat_eval=true flat_eval_rho=0.05
# # python3 main.py --config exps/olora_c100.json  --override optimizer_type=sam sam_rho=0.05 flat_eval=true flat_eval_rho=0.05
# # python3 main.py --config exps/sdlora_c100.json --override optimizer_type=sam sam_rho=0.05 flat_eval=true flat_eval_rho=0.05


# python3 main.py --config=./exps/sdlora_c100.json  >> ./logs/sdlora/cifar224/train_sdlora.log 2>&1 &

python3 main.py --config=./exps/sdlora_inr_sgd.json >> ./logs/sdlora/imagenet_r_sgd/train_sdlora_sgd.log 2>&1 &
# python3 main.py --config=./exps/sdlora_inr_sam.json >> ./logs/sdlora/imagenet_r_sam/train_sdlora_sam.log 2>&1 &