# Rebuttal Summary Table: Fine-Grained + het5

**Metric:** raw continual-accuracy summaries reported in the supplementary experiments.  
**Backbone:** ViT-B/16 pretrained on ImageNet-21K (frozen).  
**Note:** `het5` is the heterogeneous cross-dataset stream `Aircraft -> Cars196 -> CUB200 -> Flowers -> OxfordPet`. `--` means the corresponding run is not yet available.

| Method | Optimizer | Aircraft | Cars196 | CUB200 | Flowers | OxfordPet | het5 |
|---|---|---:|---:|---:|---:|---:|---:|
| SeqLoRA | SGD | 39.26 | 33.39 | 65.48 | 60.82 | 69.94 | 66.30 |
| SeqLoRA | SAM | 40.54 | 38.16 | 69.19 | 62.97 | 73.14 | 72.79 |
| SeqLoRA | GAM | 43.86 | 51.65 | 73.54 | 85.80 | 83.03 | 78.15 |
| SeqLoRA | RWP | 39.63 | 35.52 | 66.56 | 57.31 | 68.69 | 69.10 |
| IncLoRA | SGD | 38.11 | 38.67 | 67.27 | 57.50 | 64.32 | 72.99 |
| IncLoRA | SAM | 38.37 | 41.80 | 70.16 | 61.04 | 71.53 | 74.42 |
| IncLoRA | GAM | 40.54 | 50.83 | 70.68 | 84.22 | 82.04 | 75.14 |
| IncLoRA | RWP | 27.97 | 21.44 | 68.29 | 80.01 | 74.74 | 67.93 |
| OLoRA | SGD | 36.87 | 36.75 | 66.55 | 57.15 | 64.29 | 70.98 |
| OLoRA | SAM | 38.31 | 39.81 | 69.95 | 60.20 | 70.16 | -- |
| OLoRA | GAM | 35.53 | 51.01 | 70.33 | 83.93 | 81.91 | -- |
| OLoRA | RWP | 20.12 | 19.46 | 68.30 | 79.84 | 74.66 | -- |
| InfLoRA | SGD | 40.38 | 37.31 | 66.52 | 67.31 | 67.57 | -- |
| InfLoRA | SAM | 40.90 | 42.43 | 71.35 | 73.09 | 73.43 | -- |
| InfLoRA | GAM | 42.15 | 27.77 | 68.31 | 90.15 | 83.09 | -- |
| InfLoRA | RWP | 38.68 | 41.05 | 76.19 | 67.09 | 68.85 | -- |
| SDLoRA | SGD | 24.70 | 37.93 | 65.94 | 57.47 | 70.47 | -- |
| SDLoRA | SAM | 37.91 | 41.01 | 67.95 | 61.05 | 71.53 | -- |
| SDLoRA | GAM | 36.31 | 41.04 | 73.55 | 87.19 | 82.13 | -- |
| SDLoRA | RWP | 37.46 | 39.26 | 67.55 | 55.11 | 67.49 | -- |
