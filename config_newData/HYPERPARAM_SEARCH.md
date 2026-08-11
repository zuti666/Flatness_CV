# Hyperparameter search

Base configs follow the tuned setup:

- ImageNet-R: `10x20`, `rank=10`, `seed=0`, `class_shuffle=false`, `50` epochs
- ImageNet-A: `10x20`, `rank=10`, `seed=0`, `class_shuffle=false`, `30` epochs
- CIFAR-100: `10x10`, `rank=10`, `seed=0`, `class_shuffle=false`, `30` epochs

Use the tuned optimizer/regularizer defaults as the fixed reference:

- `optimizer: sgd`
- `optimizer_type: gam`
- `init_lr/lrate: 0.01`
- `ewc_gamma: 0.9`
- `gam_grad_norm_rho: 0.1`
- `gam_grad_gamma: 0.1`

Recommended search:

1. Hold `ewc_lambda=2000`, sweep `gam_grad_rho in {0.05, 0.1, 0.2}`.
2. Fix the best `gam_grad_rho`, sweep `ewc_lambda in {1000, 2000, 4000}`.
3. Re-run the best pair on all three datasets.

Requested anchor:

- `ewc_lambda: 2000`
- `gam_grad_rho: 0.2`

Repo note: CIFAR-100 uses `dataset: "cifar224"` for the ViT path.
