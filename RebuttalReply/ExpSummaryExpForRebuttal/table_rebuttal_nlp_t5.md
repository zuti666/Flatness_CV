# Rebuttal Summary Table: T5-large / T5-small

**Metric:** raw FAA values from the final task row in `Main_Result`, reported for each task order.  
**Baseline:** Adam.  
**Note:** `GAM` and `RWP` are currently not available in the T5 setting, so they are shown as `--`.

| Model | Method | Optimizer | o1 | o2 | o3 | Avg |
|---|---|---|---:|---:|---:|---:|
| T5-large | SeqLoRA | Baseline (Adam) | 56.40 | 66.24 | 69.88 | 64.17 |
| T5-large | SeqLoRA | SAM | 65.43 | 68.09 | 73.35 | 68.96 |
| T5-large | SeqLoRA | GAM | -- | -- | -- | -- |
| T5-large | SeqLoRA | RWP | -- | -- | -- | -- |
| T5-large | IncLoRA | Baseline (Adam) | 48.01 | 77.39 | 70.37 | 65.26 |
| T5-large | IncLoRA | SAM | 71.00 | 69.99 | 73.27 | 71.42 |
| T5-large | IncLoRA | GAM | -- | -- | -- | -- |
| T5-large | IncLoRA | RWP | -- | -- | -- | -- |
| T5-large | OLoRA | Baseline (Adam) | 77.59 | 76.50 | 77.55 | 77.21 |
| T5-large | OLoRA | SAM | 78.51 | 78.60 | 76.26 | 77.79 |
| T5-large | OLoRA | GAM | -- | -- | -- | -- |
| T5-large | OLoRA | RWP | -- | -- | -- | -- |
| T5-small | SeqLoRA | Baseline (Adam) | 34.32 | 44.26 | 50.22 | 42.93 |
| T5-small | SeqLoRA | SAM | 43.07 | 49.61 | 50.12 | 47.60 |
| T5-small | SeqLoRA | GAM | -- | -- | -- | -- |
| T5-small | SeqLoRA | RWP | -- | -- | -- | -- |
| T5-small | IncLoRA | Baseline (Adam) | 48.01 | 57.84 | 55.46 | 53.77 |
| T5-small | IncLoRA | SAM | 50.92 | 56.99 | 45.56 | 51.16 |
| T5-small | IncLoRA | GAM | -- | -- | -- | -- |
| T5-small | IncLoRA | RWP | -- | -- | -- | -- |
| T5-small | OLoRA | Baseline (Adam) | 62.45 | 61.68 | 57.36 | 60.50 |
| T5-small | OLoRA | SAM | 61.67 | 62.88 | 58.11 | 60.89 |
| T5-small | OLoRA | GAM | -- | -- | -- | -- |
| T5-small | OLoRA | RWP | -- | -- | -- | -- |
