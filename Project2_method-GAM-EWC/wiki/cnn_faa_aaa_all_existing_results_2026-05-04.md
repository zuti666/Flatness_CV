# CNN FAA / AAA Summary for Existing Results

Generated: 2026-05-04 from `/data/140-0/users/liying/Flatness_CV`.

This report uses `*_cl_metrics.json` as the source of truth. Rows with `cnn.final.FAA` and `cnn.final.AAA` are marked `final`; rows without `cnn.final` but with matrices are marked `partial_derived`, where FAA is the latest matrix last-row mean and AAA is the latest matrix lower-triangle mean. Actual model/optimizer come from the result path; YAML-derived parameters are kept separately with `cfg_*` columns.

## Output Files

- Detail CSV: `Project2_method-GAM-EWC/tables/cnn_faa_aaa_all_existing_results_2026-05-04.csv`

- Group stats CSV: `Project2_method-GAM-EWC/tables/cnn_faa_aaa_group_stats_2026-05-04.csv`

- Markdown: `Project2_method-GAM-EWC/wiki/cnn_faa_aaa_all_existing_results_2026-05-04.md`


## Coverage

| Item | Count |
| --- | --- |
| metric JSON files parsed | 1297 |
| rows with CNN FAA/AAA | 1290 |
| final rows from cnn.final | 1130 |
| partial rows derived from latest matrix | 160 |
| rows matched to YAML config | 503 |
| rows without usable CNN FAA/AAA | 7 |


## Best Final Result per Dataset

| Dataset | CNN FAA | CNN AAA | Actual model | Actual optimizer | Seed | Rank | LR | Epochs | lambda_flat | ewc_lambda | Prefix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 | 97.880 | 97.880 | seqlora | taskwise_sam_delta_sgd | 1993 |  |  |  |  |  | exp_D_smoke_taskwise_samdelta_sgd |
| ImageNet-R | 89.520 | 89.520 | seqlora | sgd | 521 |  |  |  |  |  | seqlora_curvloc_imagenetr_vitb16_r16_task1 |
| Flowers | 87.652 | 90.149 | inflora | gam | 0 | 16 | 0.0025 | 40 |  |  | inflora_inr_gam_flowers_ep40_lr00025_t20_rank16 |
| tiny_imagenetp | 86.083 | 88.137 | seqlora | gam | 1993 | 8 | 0.01 | 20 |  |  | seqlora_gam_imagenetp_t20_r8_1993 |
| CIFAR-100 | 85.900 | 88.244 | seqlora | gam | 0 |  |  |  |  |  | seqlora_inr_gam_cifar100_t10_rank16_eval |
| Het5 | 85.692 | 83.259 | inflora | gam | 0 | 16 | 0.02 | 40 |  |  | inflora_inr_gam_het5_t5_rank16_lr002_epoch40 |
| OxfordPet | 81.433 | 83.095 | inflora | gam | 0 | 16 | 0.0025 | 40 |  |  | inflora_inr_gam_pets_ep40_lr00025_t20_rank16 |
| tiny_imagenetc | 80.034 | 82.397 | seqlora | gam | 42 | 8 | 0.01 | 20 |  |  | seqlora_gam_imagenetc_t20_r8_42 |
| CUB200 | 75.169 | 76.191 | inflora | gam | 0 | 16 | 0.01 | 40 |  |  | inflora_inr_gam_cub200_t20-E40_rank16 |
| tiny_imagenetp_before | 67.375 | 76.590 | inclora | sgd | 42 |  |  |  |  |  | inclora_inr_sgd_imagenetp_cls10_rank4 copy |
| DomainNet | 57.246 | 65.361 | ewclora_normfisher_gam | gam_adam | 0 | 30 | 0.0005 | 5 | 1.0 | 2000 | paper4_as1normfisher_lam2000_domainnet_t5c69_r30_s0 |
| Cars196 | 53.353 | 52.300 | inclora | gam | 0 |  |  |  |  |  | inclora_inr_gam_cars196_t20_rank16_lr005_train-eval-paer |
| Aircraft | 47.081 | 43.873 | seqlora | gam | 0 |  |  |  |  |  | seqlora_inr_gam_hyper1_aircraft_t20_rank16_lr005_epoch30 |
| ImageNet-A | 45.999 | 56.367 | ewclora_normfisher_gam | gam_sgd_lr001_e30_rho005_f100 | 0 | 10 | 0.01 | 30 | 1.0 | 2000 | paper4_as1normfisher_lam2000_imageneta_t10c20_r10_sgd_lr001_e30_rho005_f100_s0 |


## PaperA / Paper4 / EWC-LoRA Focus Rows

| Status | Dataset | CNN FAA | CNN AAA | Forget | Actual model | Actual optimizer | Seed | Rank | LR | Epochs | lambda_flat | ewc_lambda | rho | Config match | Config |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final | Flowers | 86.747 | 89.743 | 4.584 | ewclora_normfisher_gam | gam | 0 | 16 | 0.0025 | 40 | 1.0 | 2000 |  | matched | config_exps_paper1_PAC/exp_paperA_cross_dataset/as1_normfisher_lam2000_flowers_t10c10_r16.yaml |
| final | OxfordPet | 78.824 | 81.197 | 16.546 | ewclora_normfisher_gam | gam | 0 | 16 | 0.0025 | 40 | 1.0 | 2000 |  | matched | config_exps_paper1_PAC/exp_paperA_cross_dataset/as1_normfisher_lam2000_oxfordpet_t9c4_r16.yaml |
| final | CUB200 | 74.224 | 74.332 | 11.709 | ewclora_normfisher_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 2000 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_newmethod/as1_normfisher_lam2000_cub200_t20.yaml |
| final | CUB200 | 74.220 | 74.284 | 11.563 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 1000 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_layer2_sweep/layer2_rawfisher_flat1p0_ewc1000_cub200_t20.yaml |
| final | CUB200 | 74.182 | 74.309 | 11.806 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 100 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_layer2_sweep/layer2_rawfisher_flat1p0_ewc100_cub200_t20.yaml |
| final | CUB200 | 74.181 | 74.255 | 11.667 | ewclora_normfisher_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 5000 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_newmethod/as1_normfisher_lam5000_cub200_t20.yaml |
| final | CUB200 | 74.097 | 74.316 | 11.753 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 500 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_layer2_sweep/layer2_rawfisher_flat1p0_ewc500_cub200_t20.yaml |
| final | CUB200 | 73.967 | 74.253 | 12.091 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 20 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_layer2_sweep/layer2_rawfisher_flat1p0_ewc20_cub200_t20.yaml |
| final | CUB200 | 73.931 | 74.241 | 11.986 | ewclora_normfisher_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 500 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_newmethod/as1_normfisher_lam500_cub200_t20.yaml |
| final | CUB200 | 73.881 | 74.207 | 12.039 | ewclora_normfisher_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 500.0 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_newmethod/as1_normfisher_dual_cub200_t20.yaml |
| final | CUB200 | 73.845 | 74.245 | 12.079 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 20.0 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_three_methods/layer2_explicit_gam_rawfisher_cub200_t20.yaml |
| final | CUB200 | 73.227 | 72.952 | 10.796 | ewclora_youyue_fitarchitecture | gam | 0 | 16 | 0.01 | 40 |  | 20.0 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_three_methods/layer1_mixed_gam_rawfisher_cub200_t20.yaml |
| final | ImageNet-R | 73.143 | 76.266 | 9.605 | ewclora_normfisher_gam | gam | 1993 | 16 | 0.01 | 20 | 1.0 | 2000 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_cross_dataset/as1_normfisher_lam2000_imagenetr_t20c10_r16.yaml |
| final | CIFAR-100 | 66.450 | 76.338 | 35.444 | ewclora_normfisher_gam | gam_adam | 0 | 10 | 0.0005 | 20 | 1.0 | 2000 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3/as1_normfisher_lam2000_cifar100_t10c10_r10.yaml |
| final | DomainNet | 57.246 | 65.361 | 33.655 | ewclora_normfisher_gam | gam_adam | 0 | 30 | 0.0005 | 5 | 1.0 | 2000 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3/as1_normfisher_lam2000_domainnet_t5c69_r30.yaml |
| final | Cars196 | 51.741 | 52.494 | 12.321 | ewclora_normfisher_gam | gam | 0 | 16 | 0.05 | 40 | 1.0 | 2000 | 0.05 | matched | config_exps_paper1_PAC/exp_paperA_cross_dataset/as1_normfisher_lam2000_cars196_t10c20_r16.yaml |
| final | Aircraft | 46.660 | 43.862 | 15.567 | ewclora_normfisher_gam | gam | 0 | 16 | 0.05 | 40 | 1.0 | 2000 | 0.01 | matched | config_exps_paper1_PAC/exp_paperA_cross_dataset/as1_normfisher_lam2000_aircraft_t10c10_r16.yaml |
| final | ImageNet-A | 45.999 | 56.367 | 8.300 | ewclora_normfisher_gam | gam_sgd_lr001_e30_rho005_f100 | 0 | 10 | 0.01 | 30 | 1.0 | 2000 | 0.05 | matched | config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3_sgd_tuned/as1_normfisher_lam2000_imageneta_t10c20_r10_sgd_lr001_e30_rho005_f100.yaml |
| final | ImageNet-R | 22.949 | 26.096 | 7.508 | ewclora_youyue_fitarchitecture | sgd | 0 | 10 | 0.0005 | 50 |  | 1.0e+07 |  | matched | config_exps_paper1_PAC/exp_paperA_reproduce_ewclora/ewclora_reproduce_imagenetr_t10_r10.yaml |
| final | ImageNet-A | 13.691 | 34.599 | 28.718 | ewclora_normfisher_gam | gam_adam | 0 | 10 | 0.0005 | 10 | 1.0 | 2000 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3/as1_normfisher_lam2000_imageneta_t10c20_r10.yaml |
| final | ImageNet-R | 9.143 | 16.684 | 91.797 | ewclora_normfisher_gam | gam_adam | 0 | 10 | 0.0005 | 50 | 1.0 | 2000 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3/as1_normfisher_lam2000_imagenetr_t10c20_r10.yaml |
| final | CUB200 | 2.596 | 3.478 | 8.109 | as1_normfisher_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 500.0 | 0.2 | matched:model_mismatch | config_exps_paper1_PAC/exp_paperA_newmethod/as1_normfisher_dual_cub200_t20.yaml |
| final | CUB200 | 2.110 | 2.932 | 6.198 | as1_normfisher_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 500 | 0.2 | matched:model_mismatch | config_exps_paper1_PAC/exp_paperA_newmethod/as1_normfisher_lam500_cub200_t20.yaml |
| final | ImageNet-A | 1.827 | 2.136 | 11.021 | as1_normfisher_gam | gam_adam | 1993 | 10 | 0.0005 | 10 | 1.0 | 2000 | 0.2 | matched:model_mismatch | config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3/as1_normfisher_lam2000_imageneta_t10c20_r10.yaml |
| final | CUB200 | 1.393 | 2.306 | 4.683 | as1_normfisher_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 2000 | 0.2 | matched:model_mismatch | config_exps_paper1_PAC/exp_paperA_newmethod/as1_normfisher_lam2000_cub200_t20.yaml |
| final | ImageNet-A | 1.341 | 2.708 | 3.258 | ewclora_normfisher_gam | gam_sgd | 0 | 10 | 0.0005 | 10 | 1.0 | 2000 | 0.2 | matched | config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3/as1_normfisher_lam2000_imageneta_t10c20_r10_sgd.yaml |
| final | CUB200 | 1.073 | 1.959 | 3.193 | as1_normfisher_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 500 | 0.2 | matched:model_mismatch | config_exps_paper1_PAC/exp_paperA_three_methods/layer3_as1_normfisher_lam500_cub200_t20.yaml |
| final | ImageNet-A | 1.011 | 2.004 | 2.528 | ewclora_youyue_fitarchitecture | sgd | 0 | 10 | 0.0005 | 20 |  | 1.0e+07 |  | matched | config_exps_paper1_PAC/exp_paperA_reproduce_ewclora/ewclora_reproduce_imageneta_t10_r10.yaml |
| final | CIFAR-100 | 1.000 | 9.085 | 24.278 | ewclora_youyue_fitarchitecture | sgd | 42 | 10 | 0.0005 | 20 |  | 1.0e+07 |  | matched | config_exps_paper1_PAC/exp_paperA_reproduce_ewclora/ewclora_reproduce_cifar100_t10_r10.yaml |
| final | CIFAR-100 | 1.000 | 5.613 | 15.556 | ewclora_youyue_fitarchitecture | sgd | 1993 | 10 | 0.0005 | 20 |  | 1.0e+07 |  | matched | config_exps_paper1_PAC/exp_paperA_reproduce_ewclora/ewclora_reproduce_cifar100_t10_r10.yaml |
| final | CIFAR-100 | 1.000 | 5.827 | 16.578 | ewclora_youyue_fitarchitecture | sgd | 0 | 10 | 0.0005 | 20 |  | 1.0e+07 |  | matched | config_exps_paper1_PAC/exp_paperA_reproduce_ewclora/ewclora_reproduce_cifar100_t10_r10.yaml |
| final | CUB200 | 0.937 | 1.944 | 3.262 | as1_normfisher_gam | gam | 0 | 16 | 0.01 | 40 | 1.0 | 5000 | 0.2 | matched:model_mismatch | config_exps_paper1_PAC/exp_paperA_newmethod/as1_normfisher_lam5000_cub200_t20.yaml |
| final | ImageNet-R | 0.739 | 2.064 | 4.402 | ewclora_youyue_fitarchitecture | sgd | 42 | 10 | 0.0005 | 50 |  | 1.0e+07 |  | matched | config_exps_paper1_PAC/exp_paperA_reproduce_ewclora/ewclora_reproduce_imagenetr_t10_r10.yaml |
| final | ImageNet-R | 0.739 | 3.418 | 9.673 | ewclora_youyue_fitarchitecture | sgd | 1993 | 10 | 0.0005 | 50 |  | 1.0e+07 |  | matched | config_exps_paper1_PAC/exp_paperA_reproduce_ewclora/ewclora_reproduce_imagenetr_t10_r10.yaml |
| final | ImageNet-A | 0.619 | 1.878 | 2.679 | ewclora_youyue_fitarchitecture | sgd | 42 | 10 | 0.0005 | 20 |  | 1.0e+07 |  | matched | config_exps_paper1_PAC/exp_paperA_reproduce_ewclora/ewclora_reproduce_imageneta_t10_r10.yaml |
| final | ImageNet-A | 0.619 | 1.400 | 1.486 | ewclora_youyue_fitarchitecture | sgd | 1993 | 10 | 0.0005 | 20 |  | 1.0e+07 |  | matched | config_exps_paper1_PAC/exp_paperA_reproduce_ewclora/ewclora_reproduce_imageneta_t10_r10.yaml |
| partial_derived | CIFAR-100 | 88.371 | 90.939 |  | ewclora_normfisher_gam | gam_sgd_lr001_e30_rho005_f100 | 0 | 10 | 0.01 | 30 | 1.0 | 2000 | 0.05 | matched | config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3_sgd_tuned/as1_normfisher_lam2000_cifar100_t10c10_r10_sgd_lr001_e30_rho005_f100.yaml |
| partial_derived | ImageNet-R | 77.133 | 80.564 |  | ewclora_normfisher_gam | gam_sgd_lr001_e50_rho005_f100 | 0 | 10 | 0.01 | 50 | 1.0 | 2000 | 0.05 | matched | config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3_sgd_tuned/as1_normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr001_e50_rho005_f100.yaml |
| partial_derived | DomainNet | 73.690 | 75.907 |  | ewclora_normfisher_gam | gam_sgd_lr001_e5_rho005_f100 | 0 | 30 | 0.01 | 5 | 1.0 | 2000 | 0.05 | matched | config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3_sgd_tuned/as1_normfisher_lam2000_domainnet_t5c69_r30_sgd_lr001_e5_rho005_f100.yaml |
| partial_derived | DomainNet | 26.130 | 26.130 |  | as1_normfisher_gam | gam_adam | 1993 | 30 | 0.0005 | 5 | 1.0 | 2000 | 0.2 | matched:model_mismatch | config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3/as1_normfisher_lam2000_domainnet_t5c69_r30.yaml |
| partial_derived | CIFAR-100 | 13.533 | 18.986 |  | as1_normfisher_gam | gam_adam | 1993 | 10 | 0.0005 | 20 | 1.0 | 2000 | 0.2 | matched:model_mismatch | config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3/as1_normfisher_lam2000_cifar100_t10c10_r10.yaml |
| partial_derived | ImageNet-R | 5.503 | 8.140 |  | as1_normfisher_gam | gam_adam | 1993 | 10 | 0.0005 | 50 | 1.0 | 2000 | 0.2 | matched:model_mismatch | config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3/as1_normfisher_lam2000_imagenetr_t10c20_r10.yaml |


## Current / Recent Partial Rows

| Dataset | Partial FAA | Partial AAA | Tasks | Matrix | Actual model | Actual optimizer | Seed | Prefix | mtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ImageNet-R | 77.133 | 80.564 | 9 | t08 | ewclora_normfisher_gam | gam_sgd_lr001_e50_rho005_f100 | 0 | paper4_as1normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr001_e50_rho005_f100_s0 | 2026-05-04 15:38:21 |
| CIFAR-100 | 88.371 | 90.939 | 7 | t06 | ewclora_normfisher_gam | gam_sgd_lr001_e30_rho005_f100 | 0 | paper4_as1normfisher_lam2000_cifar100_t10c10_r10_sgd_lr001_e30_rho005_f100_s0 | 2026-05-04 15:27:59 |
| DomainNet | 73.690 | 75.907 | 2 | t01 | ewclora_normfisher_gam | gam_sgd_lr001_e5_rho005_f100 | 0 | paper4_as1normfisher_lam2000_domainnet_t5c69_r30_sgd_lr001_e5_rho005_f100_s0 | 2026-05-04 13:48:43 |
| CIFAR-100 | 13.533 | 18.986 | 6 | t05 | as1_normfisher_gam | gam_adam | 1993 | paper4_as1normfisher_lam2000_cifar100_t10c10_r10_s0 | 2026-05-03 07:21:53 |
| ImageNet-R | 5.503 | 8.140 | 4 | t03 | as1_normfisher_gam | gam_adam | 1993 | paper4_as1normfisher_lam2000_imagenetr_t10c20_r10_s0 | 2026-05-03 05:00:09 |
| DomainNet | 26.130 | 26.130 | 1 | t00 | as1_normfisher_gam | gam_adam | 1993 | paper4_as1normfisher_lam2000_domainnet_t5c69_r30_s0 | 2026-05-03 04:13:32 |
| ImageNet-R | 89.030 | 89.637 | 2 | t01 | ewclora_youyue_fitarchitecture_gam | gam | 1993 | ewclora_fitarch_explicitgam_imagenetr_t20_r16 | 2026-04-24 09:44:33 |
| tiny_imagenetc | 83.900 | 87.220 | 4 | t03 | ewclora_youyue_fitarchitecture_gam | gam | 42 | ewclora_fitarch_explicitgam_imagenetc_t20_r8 | 2026-04-24 09:40:15 |
| CUB200 | 34.409 | 52.868 | 9 | t08 | inclora | gam | 0 | inclora_resnet50_cub200_gam_t20_rank16 | 2026-04-05 15:29:38 |
| CUB200 | 34.409 | 52.868 | 9 | t08 | olora | gam | 0 | olora_resnet50_cub200_gam_t20_rank16 | 2026-04-05 15:29:10 |
| CUB200 | 60.624 | 65.968 | 5 | t04 | sdlora | gam | 0 | sdlora_resnet50_cub200_gam_t20_rank16_allconv | 2026-04-05 15:27:51 |
| CUB200 | 46.339 | 55.574 | 7 | t06 | sdlora | sam | 0 | sdlora_resnet50_cub200_sam_t20_rank16_allconv | 2026-04-05 15:24:08 |
| CIFAR-100 | 45.256 | 51.353 | 9 | t08 | inclora | sam | 0 | inclora_resnet50_cifar100_sam_t10_rank16_allconv | 2026-04-05 15:22:25 |
| CIFAR-100 | 44.489 | 50.660 | 9 | t08 | olora | sam | 0 | olora_resnet50_cifar100_sam_t10_rank16_allconv | 2026-04-05 15:02:05 |
| Cars196 | 38.815 | 49.717 | 2 | t01 | olora | rwp_None | 0 | olora_inr_rwp_hyperfinaltry-lr002_cars196_t20-E40_rank16_lr005_train-eval-paer | 2026-03-29 18:27:00 |
| Cars196 | 1.218 | 2.351 | 4 | t03 | sdlora | gam | 0 | sdlora_inr_gam_hyper_cars196_t20-E40_rank16_lr005_train-eval-paer | 2026-03-29 01:34:19 |
| Flowers | 87.950 | 89.847 | 2 | t01 | seqlora | cflat | 0 | seqlora_inr_cflat_flowers_ep40_lr00025_t20_rank16 | 2026-03-28 10:29:05 |
| Flowers | 93.860 | 93.860 | 1 | t00 | olora | cflat | 0 | olora_inr_cflat_flowers_ep40_lr00025_t20_rank16 | 2026-03-28 10:24:38 |
| Flowers | 93.860 | 93.860 | 1 | t00 | inclora | cflat | 0 | inclora_inr_cflat_flowers_ep40_lr00025_t20_rank16 | 2026-03-28 10:04:07 |
| Flowers | 95.390 | 95.390 | 1 | t00 | inflora | cflat | 0 | inflora_inr_cflat_flowers_ep40_lr00025_t20_rank16 | 2026-03-28 10:03:25 |


## Top 30 Final Rows by CNN FAA

| Dataset | CNN FAA | CNN AAA | Actual model | Actual optimizer | Seed | Rank | LR | Epochs | Prefix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 | 97.880 | 97.880 | seqlora | taskwise_sam_delta_sgd | 1993 |  |  |  | exp_D_smoke_taskwise_samdelta_sgd |
| CIFAR-10 | 97.500 | 97.500 | seqlora | random_factor | 1993 |  |  |  | exp_C_smoke_random_factor |
| CIFAR-10 | 97.110 | 97.780 | seqlora | sam_random | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_sam_random |
| CIFAR-10 | 96.970 | 97.620 | seqlora | taskwise_sam_factor_sgd | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_sam_factor_sgd |
| CIFAR-10 | 96.960 | 97.667 | seqlora | sam_delta | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_sam_delta |
| CIFAR-10 | 96.940 | 97.600 | seqlora | taskwise_sam_factor_sam_factor | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_sam_factor_sam_factor |
| CIFAR-10 | 96.920 | 97.587 | seqlora | sam_factor | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_sam_factor |
| CIFAR-10 | 96.830 | 97.567 | seqlora | sam_full | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_sam_full |
| CIFAR-10 | 96.810 | 97.533 | seqlora | taskwise_random_factor_sgd | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_random_factor_sgd |
| CIFAR-10 | 96.810 | 97.533 | seqlora | taskwise_random_factor_random_factor | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_random_factor_random_factor |
| CIFAR-10 | 96.810 | 97.533 | seqlora | random_frozen | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_random_frozen |
| CIFAR-10 | 96.810 | 97.533 | seqlora | random_factor | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_random_factor |
| CIFAR-10 | 96.800 | 97.527 | seqlora | taskwise_sgd_random_factor | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_sgd_random_factor |
| CIFAR-10 | 96.800 | 97.527 | seqlora | random_all | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_random_all |
| CIFAR-10 | 96.800 | 97.527 | seqlora | taskwise_sgd_sgd | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_sgd_sgd |
| CIFAR-10 | 96.800 | 97.527 | seqlora | random_full | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_random_full |
| CIFAR-10 | 96.790 | 97.520 | seqlora | taskwise_sgd_sam_factor | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_sgd_sam_factor |
| CIFAR-10 | 96.790 | 97.520 | seqlora | random_delta | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_random_delta |
| CIFAR-10 | 96.780 | 97.520 | seqlora | sgd | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_sgd |
| CIFAR-10 | 96.330 | 97.253 | seqlora | sam_frozen | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_sam_frozen |
| ImageNet-R | 89.520 | 89.520 | seqlora | sgd | 521 |  |  |  | seqlora_curvloc_imagenetr_vitb16_r16_task1 |
| Flowers | 87.652 | 90.149 | inflora | gam | 0 | 16 | 0.0025 | 40 | inflora_inr_gam_flowers_ep40_lr00025_t20_rank16 |
| Flowers | 86.747 | 89.743 | ewclora_normfisher_gam | gam | 0 | 16 | 0.0025 | 40 | paperA_as1normfisher_lam2000_flowers_t10c10_r16_s0 |
| tiny_imagenetp | 86.083 | 88.137 | seqlora | gam | 1993 | 8 | 0.01 | 20 | seqlora_gam_imagenetp_t20_r8_1993 |
| CIFAR-100 | 85.900 | 88.244 | seqlora | gam | 0 |  |  |  | seqlora_inr_gam_cifar100_t10_rank16_eval |
| Het5 | 85.692 | 83.259 | inflora | gam | 0 | 16 | 0.02 | 40 | inflora_inr_gam_het5_t5_rank16_lr002_epoch40 |
| ImageNet-R | 85.490 | 85.490 | seqlora | sgd | 1993 |  |  |  | seqlora_showLossland |
| tiny_imagenetp | 85.091 | 87.255 | inclora | gam | 1993 | 8 | 0.01 | 20 | inclora_gam_imagenetp_t20_r8_1993 |
| CIFAR-100 | 84.980 | 87.860 | inclora | gam | 0 |  |  |  | inclora_inr_gam_cifar100_t10_rank16_eval |
| Flowers | 84.615 | 87.187 | sdlora | gam | 0 | 16 | 0.0025 | 40 | sdlora_inr_gam_flowers_ep40_lr00025_t20_rank16_eval2 |


## Top 30 Final Rows by CNN AAA

| Dataset | CNN FAA | CNN AAA | Actual model | Actual optimizer | Seed | Rank | LR | Epochs | Prefix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 | 97.880 | 97.880 | seqlora | taskwise_sam_delta_sgd | 1993 |  |  |  | exp_D_smoke_taskwise_samdelta_sgd |
| CIFAR-10 | 97.110 | 97.780 | seqlora | sam_random | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_sam_random |
| CIFAR-10 | 96.960 | 97.667 | seqlora | sam_delta | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_sam_delta |
| CIFAR-10 | 96.970 | 97.620 | seqlora | taskwise_sam_factor_sgd | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_sam_factor_sgd |
| CIFAR-10 | 96.940 | 97.600 | seqlora | taskwise_sam_factor_sam_factor | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_sam_factor_sam_factor |
| CIFAR-10 | 96.920 | 97.587 | seqlora | sam_factor | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_sam_factor |
| CIFAR-10 | 96.830 | 97.567 | seqlora | sam_full | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_sam_full |
| CIFAR-10 | 96.810 | 97.533 | seqlora | taskwise_random_factor_sgd | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_random_factor_sgd |
| CIFAR-10 | 96.810 | 97.533 | seqlora | taskwise_random_factor_random_factor | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_random_factor_random_factor |
| CIFAR-10 | 96.810 | 97.533 | seqlora | random_frozen | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_random_frozen |
| CIFAR-10 | 96.810 | 97.533 | seqlora | random_factor | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_random_factor |
| CIFAR-10 | 96.800 | 97.527 | seqlora | taskwise_sgd_random_factor | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_sgd_random_factor |
| CIFAR-10 | 96.800 | 97.527 | seqlora | random_all | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_random_all |
| CIFAR-10 | 96.800 | 97.527 | seqlora | taskwise_sgd_sgd | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_sgd_sgd |
| CIFAR-10 | 96.800 | 97.527 | seqlora | random_full | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_random_full |
| CIFAR-10 | 96.790 | 97.520 | seqlora | taskwise_sgd_sam_factor | 1993 |  |  |  | exp_D_taskwise_sam_trajectory_sgd_sam_factor |
| CIFAR-10 | 96.790 | 97.520 | seqlora | random_delta | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_random_delta |
| CIFAR-10 | 96.780 | 97.520 | seqlora | sgd | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_sgd |
| CIFAR-10 | 97.500 | 97.500 | seqlora | random_factor | 1993 |  |  |  | exp_C_smoke_random_factor |
| CIFAR-10 | 96.330 | 97.253 | seqlora | sam_frozen | 1993 |  |  |  | exp_C_cifar10_taskcond_seqlora_sam_frozen |
| Flowers | 87.652 | 90.149 | inflora | gam | 0 | 16 | 0.0025 | 40 | inflora_inr_gam_flowers_ep40_lr00025_t20_rank16 |
| Flowers | 86.747 | 89.743 | ewclora_normfisher_gam | gam | 0 | 16 | 0.0025 | 40 | paperA_as1normfisher_lam2000_flowers_t10c10_r16_s0 |
| ImageNet-R | 89.520 | 89.520 | seqlora | sgd | 521 |  |  |  | seqlora_curvloc_imagenetr_vitb16_r16_task1 |
| CIFAR-100 | 85.900 | 88.244 | seqlora | gam | 0 |  |  |  | seqlora_inr_gam_cifar100_t10_rank16_eval |
| tiny_imagenetp | 86.083 | 88.137 | seqlora | gam | 1993 | 8 | 0.01 | 20 | seqlora_gam_imagenetp_t20_r8_1993 |
| CIFAR-100 | 84.980 | 87.860 | inclora | gam | 0 |  |  |  | inclora_inr_gam_cifar100_t10_rank16_eval |
| tiny_imagenetp | 85.091 | 87.255 | inclora | gam | 1993 | 8 | 0.01 | 20 | inclora_gam_imagenetp_t20_r8_1993 |
| Flowers | 84.615 | 87.187 | sdlora | gam | 0 | 16 | 0.0025 | 40 | sdlora_inr_gam_flowers_ep40_lr00025_t20_rank16_eval2 |
| CIFAR-100 | 84.180 | 87.085 | olora | gam | 0 |  |  |  | olora_inr_gam_cifar100_t10_rank16_eval |
| CIFAR-100 | 83.400 | 86.936 | inclora | sam | 0 |  |  |  | inclora_inr_sam_cifar100_t10_rank16_eval |


## Grouped Parameter Statistics

The full grouped table is in the group stats CSV. The rows below show the first 40 groups sorted by best CNN FAA.

| Dataset | Actual model | Actual optimizer | Rank | LR | Epochs | lambda_flat | ewc_lambda | rho | Status | n | Mean FAA | Mean AAA | Best FAA | Best AAA | Best prefix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 | seqlora | taskwise_sam_delta_sgd |  |  |  |  |  |  | final | 1 | 97.880 | 97.880 | 97.880 | 97.880 | exp_D_smoke_taskwise_samdelta_sgd |
| CIFAR-10 | seqlora | random_factor |  |  |  |  |  |  | final | 2 | 97.155 | 97.517 | 97.500 | 97.500 | exp_C_smoke_random_factor |
| CIFAR-10 | seqlora | sam_random |  |  |  |  |  |  | final | 1 | 97.110 | 97.780 | 97.110 | 97.780 | exp_C_cifar10_taskcond_seqlora_sam_random |
| CIFAR-10 | seqlora | taskwise_sam_factor_sgd |  |  |  |  |  |  | final | 1 | 96.970 | 97.620 | 96.970 | 97.620 | exp_D_taskwise_sam_trajectory_sam_factor_sgd |
| CIFAR-10 | seqlora | sam_delta |  |  |  |  |  |  | final | 1 | 96.960 | 97.667 | 96.960 | 97.667 | exp_C_cifar10_taskcond_seqlora_sam_delta |
| CIFAR-10 | seqlora | taskwise_sam_factor_sam_factor |  |  |  |  |  |  | final | 1 | 96.940 | 97.600 | 96.940 | 97.600 | exp_D_taskwise_sam_trajectory_sam_factor_sam_factor |
| CIFAR-10 | seqlora | sam_factor |  |  |  |  |  |  | final | 1 | 96.920 | 97.587 | 96.920 | 97.587 | exp_C_cifar10_taskcond_seqlora_sam_factor |
| CIFAR-10 | seqlora | sam_full |  |  |  |  |  |  | final | 1 | 96.830 | 97.567 | 96.830 | 97.567 | exp_C_cifar10_taskcond_seqlora_sam_full |
| CIFAR-10 | seqlora | taskwise_random_factor_sgd |  |  |  |  |  |  | final | 1 | 96.810 | 97.533 | 96.810 | 97.533 | exp_D_taskwise_sam_trajectory_random_factor_sgd |
| CIFAR-10 | seqlora | taskwise_random_factor_random_factor |  |  |  |  |  |  | final | 1 | 96.810 | 97.533 | 96.810 | 97.533 | exp_D_taskwise_sam_trajectory_random_factor_random_factor |
| CIFAR-10 | seqlora | random_frozen |  |  |  |  |  |  | final | 1 | 96.810 | 97.533 | 96.810 | 97.533 | exp_C_cifar10_taskcond_seqlora_random_frozen |
| CIFAR-10 | seqlora | taskwise_sgd_random_factor |  |  |  |  |  |  | final | 1 | 96.800 | 97.527 | 96.800 | 97.527 | exp_D_taskwise_sam_trajectory_sgd_random_factor |
| CIFAR-10 | seqlora | random_all |  |  |  |  |  |  | final | 1 | 96.800 | 97.527 | 96.800 | 97.527 | exp_C_cifar10_taskcond_seqlora_random_all |
| CIFAR-10 | seqlora | taskwise_sgd_sgd |  |  |  |  |  |  | final | 1 | 96.800 | 97.527 | 96.800 | 97.527 | exp_D_taskwise_sam_trajectory_sgd_sgd |
| CIFAR-10 | seqlora | random_full |  |  |  |  |  |  | final | 1 | 96.800 | 97.527 | 96.800 | 97.527 | exp_C_cifar10_taskcond_seqlora_random_full |
| CIFAR-10 | seqlora | taskwise_sgd_sam_factor |  |  |  |  |  |  | final | 1 | 96.790 | 97.520 | 96.790 | 97.520 | exp_D_taskwise_sam_trajectory_sgd_sam_factor |
| CIFAR-10 | seqlora | random_delta |  |  |  |  |  |  | final | 1 | 96.790 | 97.520 | 96.790 | 97.520 | exp_C_cifar10_taskcond_seqlora_random_delta |
| CIFAR-10 | seqlora | sgd |  |  |  |  |  |  | final | 1 | 96.780 | 97.520 | 96.780 | 97.520 | exp_C_cifar10_taskcond_seqlora_sgd |
| CIFAR-10 | seqlora | sam_frozen |  |  |  |  |  |  | final | 1 | 96.330 | 97.253 | 96.330 | 97.253 | exp_C_cifar10_taskcond_seqlora_sam_frozen |
| Flowers | inflora | cflat | 16 | 0.0025 | 40 |  |  |  | partial_derived | 1 | 95.390 | 95.390 | 95.390 | 95.390 | inflora_inr_cflat_flowers_ep40_lr00025_t20_rank16 |
| Flowers | olora | cflat | 16 | 0.0025 | 40 |  |  |  | partial_derived | 1 | 93.860 | 93.860 | 93.860 | 93.860 | olora_inr_cflat_flowers_ep40_lr00025_t20_rank16 |
| Flowers | inclora | cflat | 16 | 0.0025 | 40 |  |  |  | partial_derived | 1 | 93.860 | 93.860 | 93.860 | 93.860 | inclora_inr_cflat_flowers_ep40_lr00025_t20_rank16 |
| tiny_imagenetc | inclora | sgd |  |  |  |  |  |  | partial_derived | 5 | 80.357 | 83.842 | 92.830 | 93.110 | inclora_sgd_imagenetc_t20_r8_cosine |
| ImageNet-R | gem | sgd |  |  |  |  |  |  | partial_derived | 1 | 92.450 | 92.450 | 92.450 | 92.450 | exps13_gem_noise_gem_sgd_gpu0 |
| ImageNet-R | fopng | sgd |  |  |  |  |  |  | partial_derived | 1 | 90.680 | 90.680 | 90.680 | 90.680 | exps8_ogd_fisher_fopng_sgd |
| ImageNet-R | finetune | sgd |  | 0.01 | 20 |  |  |  | partial_derived | 1 | 89.620 | 89.620 | 89.620 | 89.620 | train_eval_1 |
| ImageNet-R | seqlora | sgd |  |  |  |  |  |  | final | 27 | 65.071 | 69.433 | 89.520 | 89.520 | seqlora_curvloc_imagenetr_vitb16_r16_task1 |
| ImageNet-R | ewclora_youyue_fitarchitecture_gam | gam | 16 | 0.01 | 20 | 1.0 | 20.0 | 0.2 | partial_derived | 1 | 89.030 | 89.637 | 89.030 | 89.637 | ewclora_fitarch_explicitgam_imagenetr_t20_r16 |
| CIFAR-100 | ewclora_normfisher_gam | gam_sgd_lr001_e30_rho005_f100 | 10 | 0.01 | 30 | 1.0 | 2000 | 0.05 | partial_derived | 1 | 88.371 | 90.939 | 88.371 | 90.939 | paper4_as1normfisher_lam2000_cifar100_t10c10_r10_sgd_lr001_e30_rho005_f100_s0 |
| Flowers | seqlora | cflat | 16 | 0.0025 | 40 |  |  |  | partial_derived | 1 | 87.950 | 89.847 | 87.950 | 89.847 | seqlora_inr_cflat_flowers_ep40_lr00025_t20_rank16 |
| ImageNet-R | inclora | sam | 16 | 0.01 | 20 |  |  |  | partial_derived | 5 | 64.835 | 68.020 | 87.940 | 87.940 | inclora_inr_sam_imagenetr_t20_rank16 |
| Flowers | inflora | gam | 16 | 0.0025 | 40 |  |  | 0.05 | final | 1 | 87.652 | 90.149 | 87.652 | 90.149 | inflora_inr_gam_flowers_ep40_lr00025_t20_rank16 |
| Flowers | ewclora_normfisher_gam | gam | 16 | 0.0025 | 40 | 1.0 | 2000 |  | final | 1 | 86.747 | 89.743 | 86.747 | 89.743 | paperA_as1normfisher_lam2000_flowers_t10c10_r16_s0 |
| ImageNet-R | seqlora | sam |  |  |  |  |  |  | partial_derived | 4 | 73.639 | 75.393 | 86.380 | 86.380 | seqlora_inr_sam_imagenetr_t20_rank16 |
| tiny_imagenetp | seqlora | gam | 8 | 0.01 | 20 |  |  |  | final | 1 | 86.083 | 88.137 | 86.083 | 88.137 | seqlora_gam_imagenetp_t20_r8_1993 |
| CIFAR-100 | seqlora | gam |  |  |  |  |  |  | final | 1 | 85.900 | 88.244 | 85.900 | 88.244 | seqlora_inr_gam_cifar100_t10_rank16_eval |
| Het5 | inflora | gam | 16 | 0.02 | 40 |  |  |  | final | 1 | 85.692 | 83.259 | 85.692 | 83.259 | inflora_inr_gam_het5_t5_rank16_lr002_epoch40 |
| tiny_imagenetp | inclora | gam | 8 | 0.01 | 20 |  |  |  | final | 1 | 85.091 | 87.255 | 85.091 | 87.255 | inclora_gam_imagenetp_t20_r8_1993 |
| CIFAR-100 | inclora | gam |  |  |  |  |  |  | final | 1 | 84.980 | 87.860 | 84.980 | 87.860 | inclora_inr_gam_cifar100_t10_rank16_eval |
| Flowers | sdlora | gam | 16 | 0.0025 | 40 |  |  |  | final | 1 | 84.615 | 87.187 | 84.615 | 87.187 | sdlora_inr_gam_flowers_ep40_lr00025_t20_rank16_eval2 |


## Config Match Coverage

| Config directory | Matched result rows |
| --- | --- |
| config_exps_paper1_PAC/exp2_sam-rwp-flat_imageR_r16_t20 | 37 |
| config_exps_paper1_PAC/exps_inc_lora | 34 |
| config_exps_paper1_PAC/exp1_rebuttel_cub200 | 24 |
| config_exps_paper1_PAC/exp1_rebuttel_flower | 24 |
| config_exps_paper1_PAC/exps_inc | 22 |
| config_exps_paper1_PAC/exp1_rebuttel_oxfordPet | 21 |
| config_exps_paper1_PAC/exp3_dataset_imagec | 21 |
| config_exps_paper1_PAC/exp1_rebuttel_aircraft | 20 |
| config_exps_paper1_PAC/exp1_rebuttel_5datacombine | 20 |
| config_exps_paper1_PAC/exp1_rebuttel_cars196 | 20 |
| config_exps_paper1_PAC/exp5_full0lora/010 | 18 |
| config_exps_paper1_PAC/exp5_full0lora/005 | 18 |
| config_exps_paper1_PAC/exp3_dataset_imagep | 17 |
| config_exps_paper1_PAC/exp3_dataset_domainnet_imageac | 16 |
| config_exps_paper1_PAC/exp1_rebuttel_cub200_resnet | 12 |
| config_exps_paper1_PAC/exp5_full0lora/015 | 12 |
| config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction | 11 |
| config_exps_paper1_PAC/exp_1_sam-sgd_imageR_r16_t20 | 11 |
| config_exps_paper1_PAC/exp_weight | 10 |
| config_exps_paper1_PAC/exp_paperA_ewclora_paper4_method3 | 9 |
| config_exps_paper1_PAC/exp_paperA_reproduce_ewclora | 9 |
| config_exps_paper1_PAC/exp4_ablation_rank/rank8 | 9 |
| config_exps_paper1_PAC/exp4_ablation_length | 9 |
| config_exps_paper1_PAC/exp4_ablation_rank | 9 |
| config_exps_paper1_PAC/exp_paperA_newmethod | 8 |


## Notes

- `final` rows are complete metric artifacts and should be used for paper tables.

- `partial_derived` rows are useful for monitoring active or interrupted runs, but should not be mixed with final paper results without the status flag.

- `matched:model_mismatch` means the result prefix matched a YAML, but the actual result path model differs from the YAML model. This catches old invalid `as1_normfisher_gam` full-ViT artifacts.

- FAA and AAA here are CNN-only metrics. NME was intentionally excluded because the request asked for CNN results.
