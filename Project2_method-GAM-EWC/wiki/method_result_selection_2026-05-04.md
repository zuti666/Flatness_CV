# Method Result Selection: Latest Runs, as1_normfisher_gam Audit, and FitArchitecture Baselines

Generated: 2026-05-04 15:48:29 CEST

Source files:
- Full CNN FAA/AAA summary: `Project2_method-GAM-EWC/tables/cnn_faa_aaa_all_existing_results_2026-05-04.csv`
- Method description: `Project2_method-GAM-EWC/wiki/methods.md`
- Paper4 audit: `Project2_method-GAM-EWC/wiki/paper4_method3_config_summary_2026-05-03.md`
- Focus CSV: `Project2_method-GAM-EWC/tables/method_result_selection_2026-05-04.csv`

## Method Naming and Selection Rule

`Project2_method-GAM-EWC` describes three layers:

- Layer 1: `ewclora_youyue_fitarchitecture` = ΔW Fisher/EWC baseline. If used with GAM, GAM sees `L_task + Fisher` jointly, so flatness and Fisher are coupled.
- Layer 2: `ewclora_youyue_fitarchitecture_gam` = decoupled raw-Fisher GAM. It uses `g_clean + lambda_flat*(g_gam-g_clean) + g_fisher`.
- Layer 3/current method: `ewclora_normfisher_gam` = Layer 2 + trace-normalized Fisher, optionally dual ascent. This is the valid LoRA alias for the current AS^1+NormFisher method.

Important selection rule: old output paths with actual model `as1_normfisher_gam` should be treated as invalid/discard for method comparison. The paper4 audit says those runs did not trigger the LoRA backbone path because this code checks whether `model_name` contains `lora`; the corrected valid alias is `ewclora_normfisher_gam`.

## Latest Active Run: exp_paperA_ewclora_paper4_method3_sgd_tuned

| Dataset | Run state | PID | latest log phase | completed evals | CNN FAA | CNN AAA | summary status | matrix | log mtime | error hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-100 | running | 2809183 | task 7 epoch 9/30 | 7 | 88.371 | 90.939 | partial_derived | t06 | 2026-05-04 15:47:33 | 0 |
| ImageNet-R | running | 2809187 | task 9 epoch 7/50 | 9 | 77.133 | 80.564 | partial_derived | t08 | 2026-05-04 15:48:19 | 0 |
| DomainNet | running | 2809192 | task 2 epoch 3/5 | 2 | 73.690 | 75.907 | partial_derived | t01 | 2026-05-04 15:31:36 | 0 |
| ImageNet-A | complete |  | task 9 epoch 30/30 | 10 | 45.999 | 56.367 | final | final | 2026-05-04 01:57:50 | 0 |

Interpretation: the three active jobs are valid `ewclora_normfisher_gam` runs with SGD-base GAM tuning (`lr=0.01`, `rho=0.05`, `ewc_lambda=2000`, `lambda_flat=1.0`). ImageNet-A already has a final JSON; CIFAR-100/ImageNet-R/DomainNet are still partial and should not be mixed with final paper rows without the `status` flag.

## Current Method Results to Use

| Status | Dataset | CNN FAA | CNN AAA | Forget | Model | Optimizer | Seed | Tasks | Prefix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final | CIFAR-100 | 66.450 | 76.338 | 35.444 | ewclora_normfisher_gam | gam_adam | 0 | 10 | paper4_as1normfisher_lam2000_cifar100_t10c10_r10_s0 |
| partial_derived | CIFAR-100 | 88.371 | 90.939 |  | ewclora_normfisher_gam | gam_sgd_lr001_e30_rho005_f100 | 0 | 7 | paper4_as1normfisher_lam2000_cifar100_t10c10_r10_sgd_lr001_e30_rho005_f100_s0 |
| final | DomainNet | 57.246 | 65.361 | 33.655 | ewclora_normfisher_gam | gam_adam | 0 | 5 | paper4_as1normfisher_lam2000_domainnet_t5c69_r30_s0 |
| partial_derived | DomainNet | 73.690 | 75.907 |  | ewclora_normfisher_gam | gam_sgd_lr001_e5_rho005_f100 | 0 | 2 | paper4_as1normfisher_lam2000_domainnet_t5c69_r30_sgd_lr001_e5_rho005_f100_s0 |
| final | ImageNet-R | 73.143 | 76.266 | 9.605 | ewclora_normfisher_gam | gam | 1993 | 20 | paperA_as1normfisher_lam2000_imagenetr_t20c10_r16_s1993 |
| final | ImageNet-R | 9.143 | 16.684 | 91.797 | ewclora_normfisher_gam | gam_adam | 0 | 10 | paper4_as1normfisher_lam2000_imagenetr_t10c20_r10_s0 |
| partial_derived | ImageNet-R | 77.133 | 80.564 |  | ewclora_normfisher_gam | gam_sgd_lr001_e50_rho005_f100 | 0 | 9 | paper4_as1normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr001_e50_rho005_f100_s0 |
| final | ImageNet-A | 13.691 | 34.599 | 28.718 | ewclora_normfisher_gam | gam_adam | 0 | 10 | paper4_as1normfisher_lam2000_imageneta_t10c20_r10_s0 |
| final | ImageNet-A | 1.341 | 2.708 | 3.258 | ewclora_normfisher_gam | gam_sgd | 0 | 10 | paper4_as1normfisher_lam2000_imageneta_t10c20_r10_sgd_s0 |
| final | ImageNet-A | 45.999 | 56.367 | 8.300 | ewclora_normfisher_gam | gam_sgd_lr001_e30_rho005_f100 | 0 | 10 | paper4_as1normfisher_lam2000_imageneta_t10c20_r10_sgd_lr001_e30_rho005_f100_s0 |
| final | CUB200 | 73.881 | 74.207 | 12.039 | ewclora_normfisher_gam | gam | 0 | 10 | paperA_as1normfisher_dual_cub200_t20 |
| final | CUB200 | 74.224 | 74.332 | 11.709 | ewclora_normfisher_gam | gam | 0 | 10 | paperA_as1normfisher_lam2000_cub200_t20 |
| final | CUB200 | 74.181 | 74.255 | 11.667 | ewclora_normfisher_gam | gam | 0 | 10 | paperA_as1normfisher_lam5000_cub200_t20 |
| final | CUB200 | 73.931 | 74.241 | 11.986 | ewclora_normfisher_gam | gam | 0 | 10 | paperA_as1normfisher_lam500_cub200_t20 |
| final | Flowers | 86.747 | 89.743 | 4.584 | ewclora_normfisher_gam | gam | 0 | 10 | paperA_as1normfisher_lam2000_flowers_t10c10_r16_s0 |
| final | OxfordPet | 78.824 | 81.197 | 16.546 | ewclora_normfisher_gam | gam | 0 | 9 | paperA_as1normfisher_lam2000_oxfordpet_t9c4_r16_s0 |
| final | Cars196 | 51.741 | 52.494 | 12.321 | ewclora_normfisher_gam | gam | 0 | 10 | paperA_as1normfisher_lam2000_cars196_t10c20_r16_s0 |
| final | Aircraft | 46.660 | 43.862 | 15.567 | ewclora_normfisher_gam | gam | 0 | 10 | paperA_as1normfisher_lam2000_aircraft_t10c10_r16_s0 |

Main takeaway: for paper4 datasets, the tuned SGD run is the relevant current-method run. It already beats the old Adam paper4 run on ImageNet-A final and has strong partial curves on CIFAR-100/ImageNet-R/DomainNet. For the cross-dataset story, use the completed `ewclora_normfisher_gam/gam` rows.

## Old as1_normfisher_gam Rows

| Status | Dataset | CNN FAA | CNN AAA | Forget | Model | Optimizer | Seed | Tasks | Prefix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| partial_derived | CIFAR-100 | 13.533 | 18.986 |  | as1_normfisher_gam | gam_adam | 1993 | 6 | paper4_as1normfisher_lam2000_cifar100_t10c10_r10_s0 |
| partial_derived | DomainNet | 26.130 | 26.130 |  | as1_normfisher_gam | gam_adam | 1993 | 1 | paper4_as1normfisher_lam2000_domainnet_t5c69_r30_s0 |
| partial_derived | ImageNet-R | 5.503 | 8.140 |  | as1_normfisher_gam | gam_adam | 1993 | 4 | paper4_as1normfisher_lam2000_imagenetr_t10c20_r10_s0 |
| final | ImageNet-A | 1.827 | 2.136 | 11.021 | as1_normfisher_gam | gam_adam | 1993 | 10 | paper4_as1normfisher_lam2000_imageneta_t10c20_r10_s0 |
| final | CUB200 | 2.596 | 3.478 | 8.109 | as1_normfisher_gam | gam | 0 | 10 | paperA_as1normfisher_dual_cub200_t20 |
| final | CUB200 | 1.393 | 2.306 | 4.683 | as1_normfisher_gam | gam | 0 | 10 | paperA_as1normfisher_lam2000_cub200_t20 |
| final | CUB200 | 0.937 | 1.944 | 3.262 | as1_normfisher_gam | gam | 0 | 10 | paperA_as1normfisher_lam5000_cub200_t20 |
| final | CUB200 | 2.110 | 2.932 | 6.198 | as1_normfisher_gam | gam | 0 | 10 | paperA_as1normfisher_lam500_cub200_t20 |
| final | CUB200 | 1.073 | 1.959 | 3.193 | as1_normfisher_gam | gam | 0 | 10 | paperA_layer3_as1_normfisher_lam500_cub200_t20 |

These rows are kept only for audit/debugging. The `matched:model_mismatch` tag means the result prefix matched an AS^1+NormFisher config, but the actual output model was `as1_normfisher_gam` under `logs_inc`, not `logs_inc_lora`; this makes the LoRA/Fisher mechanism inactive or unreliable for the intended comparison.

## ewclora_youyue_fitarchitecture / FitArchitecture Results

| Status | Dataset | CNN FAA | CNN AAA | Forget | Model | Optimizer | Seed | Tasks | Prefix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final | CIFAR-100 | 1.000 | 5.827 | 16.578 | ewclora_youyue_fitarchitecture | sgd | 0 | 10 | reproduce_ewclora_cifar100_t10_r10 |
| final | CIFAR-100 | 1.000 | 5.613 | 15.556 | ewclora_youyue_fitarchitecture | sgd | 1993 | 10 | reproduce_ewclora_cifar100_t10_r10 |
| final | CIFAR-100 | 1.000 | 9.085 | 24.278 | ewclora_youyue_fitarchitecture | sgd | 42 | 10 | reproduce_ewclora_cifar100_t10_r10 |
| final | ImageNet-R | 22.949 | 26.096 | 7.508 | ewclora_youyue_fitarchitecture | sgd | 0 | 10 | reproduce_ewclora_imagenetr_t10_r10 |
| final | ImageNet-R | 0.739 | 3.418 | 9.673 | ewclora_youyue_fitarchitecture | sgd | 1993 | 10 | reproduce_ewclora_imagenetr_t10_r10 |
| final | ImageNet-R | 0.739 | 2.064 | 4.402 | ewclora_youyue_fitarchitecture | sgd | 42 | 10 | reproduce_ewclora_imagenetr_t10_r10 |
| final | ImageNet-R | 73.317 | 76.555 | 9.232 | ewclora_youyue_fitarchitecture_gam | gam | 1993 | 20 | ewclora_fitarch_gam_imagenetr_t20_r16 |
| final | ImageNet-A | 1.011 | 2.004 | 2.528 | ewclora_youyue_fitarchitecture | sgd | 0 | 10 | reproduce_ewclora_imageneta_t10_r10 |
| final | ImageNet-A | 0.619 | 1.400 | 1.486 | ewclora_youyue_fitarchitecture | sgd | 1993 | 10 | reproduce_ewclora_imageneta_t10_r10 |
| final | ImageNet-A | 0.619 | 1.878 | 2.679 | ewclora_youyue_fitarchitecture | sgd | 42 | 10 | reproduce_ewclora_imageneta_t10_r10 |
| final | CUB200 | 73.227 | 72.952 | 10.796 | ewclora_youyue_fitarchitecture | gam | 0 | 10 | paperA_layer1_mixed_gam_rawfisher_cub200_t20 |
| final | CUB200 | 74.054 | 74.307 | 11.896 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 10 | ewclora_fitarch_explicitgam_cub200_t10_r16 |
| final | CUB200 | 73.800 | 74.193 | 12.229 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 10 | ewclora_fitarch_explicitgam_cub200_t10_r16_mechanism |
| final | CUB200 | 73.844 | 74.246 | 12.087 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 10 | ewclora_fitarch_gam_cub200_t10_r16 |
| final | CUB200 | 74.220 | 74.284 | 11.563 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 10 | paperA_l2_rawfisher_flat1p0_ewc1000_cub200_t20 |
| final | CUB200 | 74.182 | 74.309 | 11.806 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 10 | paperA_l2_rawfisher_flat1p0_ewc100_cub200_t20 |
| final | CUB200 | 73.967 | 74.253 | 12.091 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 10 | paperA_l2_rawfisher_flat1p0_ewc20_cub200_t20 |
| final | CUB200 | 74.097 | 74.316 | 11.753 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 10 | paperA_l2_rawfisher_flat1p0_ewc500_cub200_t20 |
| final | CUB200 | 73.845 | 74.245 | 12.079 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 10 | paperA_layer2_explicit_gam_rawfisher_cub200_t20 |
| final | Cars196 | 0.882 | 1.466 | 0.000 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 10 | ewclora_fitarch_explicitgam_cars196_t10_r16 |
| final | Cars196 | 51.975 | 52.457 | 12.678 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 10 | ewclora_fitarch_gam_cars196_t10_r16 |
| final | Aircraft | 46.540 | 43.851 | 16.467 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 10 | ewclora_fitarch_explicitgam_aircraft_t10_r16 |
| final | Aircraft | 46.660 | 43.906 | 16.001 | ewclora_youyue_fitarchitecture_gam | gam | 0 | 10 | ewclora_fitarch_gam_aircraft_t10_r16 |
| final | tiny_imagenetc | 73.780 | 76.474 | 8.832 | ewclora_youyue_fitarchitecture_gam | gam | 42 | 20 | ewclora_fitarch_gam_imagenetc_t20_r8 |

Use these as baselines/ablations, not as the final method. The cleanest FitArchitecture ablation is CUB200: Layer1 mixed raw Fisher is `73.227/72.952`, while Layer2 decoupled raw Fisher reaches about `74.22/74.28` in the λ sweep. The EWC-LoRA reproduction rows with `ewclora_youyue_fitarchitecture + sgd` on CIFAR-100/ImageNet-A/ImageNet-R are very low in this summary and should be treated as an internal reproduction artifact until the reproduction pipeline is rechecked.

## Recommended Result Choice

For writing the method/result section now:

1. Use `ewclora_normfisher_gam` as the method name for our AS^1 + trace-normalized Fisher method.
2. Use `exp_paperA_ewclora_paper4_method3_sgd_tuned` for the latest paper4 benchmark run; mark CIFAR-100/ImageNet-R/DomainNet as partial until final JSONs are complete.
3. Exclude `as1_normfisher_gam` output rows from method comparison tables; keep them only in an audit note.
4. Use `ewclora_youyue_fitarchitecture` and `ewclora_youyue_fitarchitecture_gam` as Layer1/Layer2 baselines, especially on CUB200 where the comparison is clean.

The focus CSV contains all selected rows with config paths and metric paths so the paper table can be rebuilt from a smaller, auditable subset.
