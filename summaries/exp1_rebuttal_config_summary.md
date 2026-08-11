# Exp1 Rebuttal Config Summary

## 1. Dataset Overview

| Dataset | yaml_dataset | Split | nb_tasks | sessions | LoRA families | LoRA rank | LoRA lr/epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aircraft | aircraft | 10 + inc 10 | 10 | 10 | IncLoRA / InfLoRA / OLoRA / SeqLoRA | 16 | lr=0.05 / 0.02, epoch=30 |
| cars196 | cars196 | 16 + inc 20 | 10 | 10 | IncLoRA / InfLoRA / OLoRA / SeqLoRA | 16 | lr=0.05, epoch=20 |
| cub200 | cub200 | 20 + inc 20 | 10 | 10 | IncLoRA / InfLoRA / OLoRA / SeqLoRA | 16 | lr=0.01, epoch=20 |
| flower | flowers | 12 + inc 10 | 10 | 10 | IncLoRA / InfLoRA / OLoRA / SeqLoRA | 16 | lr=0.005, epoch=10 |
| oxfordPet | pets | 5 + inc 4 | 9 | 9 | IncLoRA / InfLoRA / OLoRA / SeqLoRA | 16 | lr=0.005, epoch=10 |

## 2. GAM Consistency

| Dataset | Verdict | LoRA-GAM |
| --- | --- | --- |
| aircraft | LoRA-GAM不完全一致 | IncLoRA: gam_grad_rho=0.02, gam_grad_norm_rho=0.1, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false / InfLoRA: gam_grad_rho=0.01, gam_grad_norm_rho=0.1, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false / OLoRA: gam_grad_rho=0.01, gam_grad_norm_rho=0.1, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false / SeqLoRA: gam_grad_rho=0.01, gam_grad_norm_rho=0.1, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.02, gam_adaptive=false |
| cars196 | LoRA-GAM不完全一致 | IncLoRA: gam_grad_rho=0.05, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false / InfLoRA: gam_grad_rho=0.02, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false / OLoRA: gam_grad_rho=0.05, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false / SeqLoRA: gam_grad_rho=0.05, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false |
| cub200 | LoRA-GAM基本一致 | IncLoRA: yaml-no-explicit-gam-params / InfLoRA: yaml-no-explicit-gam-params / OLoRA: yaml-no-explicit-gam-params / SeqLoRA: yaml-no-explicit-gam-params |
| flower | LoRA-GAM不一致，且部分yaml未显式写GAM参数 | IncLoRA: gam_grad_rho=0.05, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false / InfLoRA: gam_grad_rho=0.05, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false / OLoRA: gam_grad_rho=0.05, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false / SeqLoRA: yaml-no-explicit-gam-params |
| oxfordPet | LoRA-GAM基本一致 | IncLoRA: yaml-no-explicit-gam-params / InfLoRA: yaml-no-explicit-gam-params / OLoRA: yaml-no-explicit-gam-params / SeqLoRA: yaml-no-explicit-gam-params |

## 3. Detailed Grid

| Dataset | yaml_dataset | Family | Optimizer | init_epoch | init_lr | epochs | lrate | files | extra_params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aircraft | aircraft | SeqLoRA | sgd | 30 | 0.05 | 30 | 0.05 | 1 | - |
| aircraft | aircraft | SeqLoRA | sam | 30 | 0.05 | 30 | 0.05 | 1 | sam_rho=0.05, sam_adaptive=false |
| aircraft | aircraft | SeqLoRA | gam | 30 | 0.05 | 30 | 0.05 | 1 | gam_grad_rho=0.01, gam_grad_norm_rho=0.1, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.02, gam_adaptive=false |
| aircraft | aircraft | SeqLoRA | rwp | 30 | 0.05 | 30 | 0.05 | 1 | rwp_std=0.05, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| aircraft | aircraft | SeqLoRA | cflat | 30 | 0.05 | 30 | 0.05 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| aircraft | aircraft | IncLoRA | sgd | 30 | 0.05 | 30 | 0.05 | 1 | - |
| aircraft | aircraft | IncLoRA | sam | 30 | 0.05 | 30 | 0.05 | 1 | sam_rho=0.05, sam_adaptive=false |
| aircraft | aircraft | IncLoRA | gam | 30 | 0.05 | 30 | 0.05 | 1 | gam_grad_rho=0.02, gam_grad_norm_rho=0.1, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false |
| aircraft | aircraft | IncLoRA | rwp | 30 | 0.05 | 30 | 0.05 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| aircraft | aircraft | IncLoRA | cflat | 30 | 0.05 | 30 | 0.05 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| aircraft | aircraft | OLoRA | sgd | 30 | 0.05 | 30 | 0.05 | 1 | - |
| aircraft | aircraft | OLoRA | sam | 30 | 0.05 | 30 | 0.05 | 1 | sam_rho=0.05, sam_adaptive=false |
| aircraft | aircraft | OLoRA | gam | 30 | 0.02 | 30 | 0.02 | 1 | gam_grad_rho=0.01, gam_grad_norm_rho=0.1, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false |
| aircraft | aircraft | OLoRA | rwp | 30 | 0.05 | 30 | 0.05 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| aircraft | aircraft | OLoRA | cflat | 30 | 0.05 | 30 | 0.05 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| aircraft | aircraft | InfLoRA | sgd | 30 | 0.05 | 30 | 0.05 | 1 | - |
| aircraft | aircraft | InfLoRA | sam | 30 | 0.05 | 30 | 0.05 | 1 | sam_rho=0.05, sam_adaptive=false |
| aircraft | aircraft | InfLoRA | gam | 30 | 0.05 | 30 | 0.05 | 1 | gam_grad_rho=0.01, gam_grad_norm_rho=0.1, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false |
| aircraft | aircraft | InfLoRA | rwp | 30 | 0.05 | 30 | 0.05 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| aircraft | aircraft | InfLoRA | cflat | 30 | 0.05 | 30 | 0.05 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| cars196 | cars196 | SeqLoRA | sgd | 20 | 0.05 | 20 | 0.05 | 1 | - |
| cars196 | cars196 | SeqLoRA | sam | 20 | 0.05 | 20 | 0.05 | 1 | sam_rho=0.05, sam_adaptive=false |
| cars196 | cars196 | SeqLoRA | gam | 20 | 0.05 | 20 | 0.05 | 1 | gam_grad_rho=0.05, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false |
| cars196 | cars196 | SeqLoRA | rwp | 20 | 0.05 | 20 | 0.05 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| cars196 | cars196 | SeqLoRA | cflat | 20 | 0.05 | 20 | 0.05 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| cars196 | cars196 | IncLoRA | sgd | 20 | 0.05 | 20 | 0.05 | 1 | - |
| cars196 | cars196 | IncLoRA | sam | 20 | 0.05 | 20 | 0.05 | 1 | sam_rho=0.05, sam_adaptive=false |
| cars196 | cars196 | IncLoRA | gam | 20 | 0.05 | 20 | 0.05 | 1 | gam_grad_rho=0.05, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false |
| cars196 | cars196 | IncLoRA | rwp | 20 | 0.05 | 20 | 0.05 | 1 | rwp_std=0.05, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| cars196 | cars196 | IncLoRA | cflat | 20 | 0.05 | 20 | 0.05 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| cars196 | cars196 | OLoRA | sgd | 20 | 0.05 | 20 | 0.05 | 1 | - |
| cars196 | cars196 | OLoRA | sam | 20 | 0.05 | 20 | 0.05 | 1 | sam_rho=0.05, sam_adaptive=false |
| cars196 | cars196 | OLoRA | gam | 20 | 0.05 | 20 | 0.05 | 1 | gam_grad_rho=0.05, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false |
| cars196 | cars196 | OLoRA | rwp | 20 | 0.05 | 20 | 0.05 | 1 | rwp_std=0.05, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| cars196 | cars196 | OLoRA | cflat | 20 | 0.05 | 20 | 0.05 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| cars196 | cars196 | InfLoRA | sgd | 20 | 0.05 | 20 | 0.05 | 1 | - |
| cars196 | cars196 | InfLoRA | sam | 20 | 0.05 | 20 | 0.05 | 1 | sam_rho=0.05, sam_adaptive=false |
| cars196 | cars196 | InfLoRA | gam | 20 | 0.05 | 20 | 0.05 | 1 | gam_grad_rho=0.02, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false |
| cars196 | cars196 | InfLoRA | rwp | 20 | 0.05 | 20 | 0.05 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| cars196 | cars196 | InfLoRA | cflat | 20 | 0.05 | 20 | 0.05 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| cub200 | cub200 | SeqLoRA | sgd | 20 | 0.01 | 20 | 0.01 | 1 | - |
| cub200 | cub200 | SeqLoRA | sam | 20 | 0.01 | 20 | 0.01 | 1 | sam_rho=0.05, sam_adaptive=false |
| cub200 | cub200 | SeqLoRA | gam | 20 | 0.01 | 20 | 0.01 | 1 | yaml-no-explicit-gam-params |
| cub200 | cub200 | SeqLoRA | rwp | 20 | 0.01 | 20 | 0.01 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| cub200 | cub200 | SeqLoRA | cflat | 20 | 0.01 | 20 | 0.01 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| cub200 | cub200 | IncLoRA | sgd | 20 | 0.01 | 20 | 0.01 | 1 | - |
| cub200 | cub200 | IncLoRA | sam | 20 | 0.01 | 20 | 0.01 | 1 | sam_rho=0.05, sam_adaptive=false |
| cub200 | cub200 | IncLoRA | gam | 20 | 0.01 | 20 | 0.01 | 1 | yaml-no-explicit-gam-params |
| cub200 | cub200 | IncLoRA | rwp | 20 | 0.01 | 20 | 0.01 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| cub200 | cub200 | IncLoRA | cflat | 20 | 0.01 | 20 | 0.01 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| cub200 | cub200 | OLoRA | sgd | 20 | 0.01 | 20 | 0.01 | 1 | - |
| cub200 | cub200 | OLoRA | sam | 20 | 0.01 | 20 | 0.01 | 1 | sam_rho=0.05, sam_adaptive=false |
| cub200 | cub200 | OLoRA | gam | 20 | 0.01 | 20 | 0.01 | 1 | yaml-no-explicit-gam-params |
| cub200 | cub200 | OLoRA | rwp | 20 | 0.01 | 20 | 0.01 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| cub200 | cub200 | OLoRA | cflat | 20 | 0.01 | 20 | 0.01 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| cub200 | cub200 | InfLoRA | sgd | 20 | 0.01 | 20 | 0.01 | 1 | - |
| cub200 | cub200 | InfLoRA | sam | 20 | 0.01 | 20 | 0.01 | 1 | sam_rho=0.05, sam_adaptive=false |
| cub200 | cub200 | InfLoRA | gam | 20 | 0.01 | 20 | 0.01 | 1 | yaml-no-explicit-gam-params |
| cub200 | cub200 | InfLoRA | rwp | 20 | 0.01 | 20 | 0.01 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| cub200 | cub200 | InfLoRA | cflat | 20 | 0.01 | 20 | 0.01 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| flower | flowers | SeqLoRA | sgd | 10 | 0.005 | 10 | 0.005 | 1 | - |
| flower | flowers | SeqLoRA | sam | 10 | 0.005 | 10 | 0.005 | 1 | sam_rho=0.05, sam_adaptive=false |
| flower | flowers | SeqLoRA | gam | 10 | 0.005 | 10 | 0.005 | 1 | yaml-no-explicit-gam-params |
| flower | flowers | SeqLoRA | rwp | 10 | 0.005 | 10 | 0.005 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| flower | flowers | SeqLoRA | cflat | 10 | 0.005 | 10 | 0.005 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| flower | flowers | IncLoRA | sgd | 10 | 0.005 | 10 | 0.005 | 1 | - |
| flower | flowers | IncLoRA | sam | 10 | 0.005 | 10 | 0.005 | 1 | sam_rho=0.05, sam_adaptive=false |
| flower | flowers | IncLoRA | gam | 10 | 0.005 | 10 | 0.005 | 1 | gam_grad_rho=0.05, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false |
| flower | flowers | IncLoRA | rwp | 10 | 0.005 | 10 | 0.005 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| flower | flowers | IncLoRA | cflat | 10 | 0.005 | 10 | 0.005 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| flower | flowers | OLoRA | sgd | 10 | 0.005 | 10 | 0.005 | 1 | - |
| flower | flowers | OLoRA | sam | 10 | 0.005 | 10 | 0.005 | 1 | sam_rho=0.05, sam_adaptive=false |
| flower | flowers | OLoRA | gam | 10 | 0.005 | 10 | 0.005 | 1 | gam_grad_rho=0.05, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false |
| flower | flowers | OLoRA | rwp | 10 | 0.005 | 10 | 0.005 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| flower | flowers | OLoRA | cflat | 10 | 0.005 | 10 | 0.005 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| flower | flowers | InfLoRA | sgd | 10 | 0.005 | 10 | 0.005 | 1 | - |
| flower | flowers | InfLoRA | sam | 10 | 0.005 | 10 | 0.005 | 1 | sam_rho=0.05, sam_adaptive=false |
| flower | flowers | InfLoRA | gam | 10 | 0.005 | 10 | 0.005 | 1 | gam_grad_rho=0.05, gam_grad_norm_rho=0.2, gam_grad_beta_1=1, gam_grad_beta_2=1.0, gam_grad_beta_3=1.0, gam_grad_gamma=0.03, gam_adaptive=false |
| flower | flowers | InfLoRA | rwp | 10 | 0.005 | 10 | 0.005 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| flower | flowers | InfLoRA | cflat | 10 | 0.005 | 10 | 0.005 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| oxfordPet | pets | SeqLoRA | sgd | 10 | 0.005 | 10 | 0.005 | 1 | - |
| oxfordPet | pets | SeqLoRA | sam | 10 | 0.005 | 10 | 0.005 | 1 | sam_rho=0.05, sam_adaptive=false |
| oxfordPet | pets | SeqLoRA | gam | 10 | 0.005 | 10 | 0.005 | 1 | yaml-no-explicit-gam-params |
| oxfordPet | pets | SeqLoRA | rwp | 10 | 0.005 | 10 | 0.005 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| oxfordPet | pets | SeqLoRA | cflat | 10 | 0.005 | 10 | 0.005 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| oxfordPet | pets | IncLoRA | sgd | 10 | 0.005 | 10 | 0.005 | 1 | - |
| oxfordPet | pets | IncLoRA | sam | 10 | 0.005 | 10 | 0.005 | 1 | sam_rho=0.05, sam_adaptive=false |
| oxfordPet | pets | IncLoRA | gam | 10 | 0.005 | 10 | 0.005 | 2 | yaml-no-explicit-gam-params |
| oxfordPet | pets | IncLoRA | rwp | 10 | 0.005 | 10 | 0.005 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| oxfordPet | pets | IncLoRA | cflat | 10 | 0.005 | 10 | 0.005 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| oxfordPet | pets | OLoRA | sgd | 10 | 0.005 | 10 | 0.005 | 1 | - |
| oxfordPet | pets | OLoRA | sam | 10 | 0.005 | 10 | 0.005 | 1 | sam_rho=0.05, sam_adaptive=false |
| oxfordPet | pets | OLoRA | gam | 10 | 0.005 | 10 | 0.005 | 1 | yaml-no-explicit-gam-params |
| oxfordPet | pets | OLoRA | rwp | 10 | 0.005 | 10 | 0.005 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| oxfordPet | pets | OLoRA | cflat | 10 | 0.005 | 10 | 0.005 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |
| oxfordPet | pets | InfLoRA | sgd | 10 | 0.005 | 10 | 0.005 | 1 | - |
| oxfordPet | pets | InfLoRA | sam | 10 | 0.005 | 10 | 0.005 | 1 | sam_rho=0.05, sam_adaptive=false |
| oxfordPet | pets | InfLoRA | gam | 10 | 0.005 | 10 | 0.005 | 1 | yaml-no-explicit-gam-params |
| oxfordPet | pets | InfLoRA | rwp | 10 | 0.005 | 10 | 0.005 | 1 | rwp_std=0.01, rwp_eta=0.1, rwp_beta=0.99, rwp_std_follow_lr=true, rwp_noise_type=Gauss_standard |
| oxfordPet | pets | InfLoRA | cflat | 10 | 0.005 | 10 | 0.005 | 1 | cflat_rho=0.05, cflat_lambda=0.01 |

## 4. Notes

- `flower` 目录内部的 `dataset` 字段写的是 `flowers`。
- `oxfordPet` 目录内部的 `dataset` 字段写的是 `pets`。
- `files` 表示同一 `dataset × family × optimizer` 下匹配到的 yaml 数量；大于 1 通常来自 seed42 变体或重复别名文件。
- `yaml-no-explicit-gam-params` 表示该配置虽使用 `optimizer_type: gam`，但 yaml 中没有显式写出 `gam_grad_*` / `grad_*` 参数。