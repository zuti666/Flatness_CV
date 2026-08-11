# Supplementary Experimental Results

**Submission**: ICML 2026 #2173 — *Revisiting Sharpness in Low-rank Subspaces for Continual Learning*

This document provides three supplementary result sets referenced in the rebuttal:
1. ViT-B/16 on five fine-grained vision datasets (individual datasets)
2. ViT-B/16 on the het5 cross-dataset benchmark
3. T5-small / T5-large on an NLP sequential learning suite (three task orders)

---

## At-a-Glance Summary

Baseline = SGD for vision settings and Adam for NLP. The table below is a compact
summary for reviewer questions about dataset breadth, cross-dataset transfer,
and architecture/scale.

| Setting | Metric | Model | Method | Baseline | SAM | GAM | RWP | Delta (SAM-Baseline) |
|---------|--------|-------|--------|----------|-----|-----|-----|----------------------|
| Fine-grained 5 datasets | Avg over 5 dataset summaries | ViT-B/16 | SeqLoRA | 53.78 | 56.80 | 67.58 | 53.54 | +3.02 |
| Fine-grained 5 datasets | Avg over 5 dataset summaries | ViT-B/16 | IncLoRA | 53.17 | 56.58 | 65.66 | 54.49 | +3.41 |
| Fine-grained 5 datasets | Avg over 5 dataset summaries | ViT-B/16 | OLoRA   | 52.32 | 55.69 | 64.54 | 52.48 | +3.37 |
| Fine-grained 5 datasets | Avg over 5 dataset summaries | ViT-B/16 | InfLoRA | 55.82 | 60.24 | 62.29 | 58.37 | +4.42 |
| Fine-grained 5 datasets | Avg over 5 dataset summaries | ViT-B/16 | SDLoRA  | 51.30 | 55.89 | 64.04 | 53.37 | +4.59 |
| Heterogeneous het5 | 5-dataset aggregate | ViT-B/16 | SeqLoRA | 66.30 | 72.79 | 78.15 | 69.10 | +6.49 |
| Heterogeneous het5 | 5-dataset aggregate | ViT-B/16 | IncLoRA | 72.99 | 74.42 | 75.14 | 67.93 | +1.43 |
| NLP, 3 orders | Avg FAA over 3 orders | T5-large | SeqLoRA | 64.17 | 68.96 | -- | -- | +4.78 |
| NLP, 3 orders | Avg FAA over 3 orders | T5-large | IncLoRA | 65.26 | 71.42 | -- | -- | +6.16 |
| NLP, 3 orders | Avg FAA over 3 orders | T5-large | OLoRA   | 77.21 | 77.79 | -- | -- | +0.58 |
| NLP, 3 orders | Avg FAA over 3 orders | T5-small | SeqLoRA | 42.93 | 47.60 | -- | -- | +4.67 |
| NLP, 3 orders | Avg FAA over 3 orders | T5-small | IncLoRA | 53.77 | 51.16 | -- | -- | -2.61 |
| NLP, 3 orders | Avg FAA over 3 orders | T5-small | OLoRA   | 60.50 | 60.89 | -- | -- | +0.39 |

This compact table is only for fast inspection; the detailed per-dataset and
per-order tables remain in Settings 1-3 below.

---

## Setting 1: ViT-B/16 — Five Fine-Grained Vision Datasets

Sequential continual learning on Aircraft, Cars196, CUB200, Flowers102, OxfordPet.
Backbone: ViT-B/16 pretrained on ImageNet-21K (frozen).
Metric: per-dataset continual-accuracy summary (%) as reported below; full FAA/AAA
breakdowns are provided separately in `combine_CNN_detailed.*`.
GAM = geometry-aware SAM (AS(1) in the paper); SAM = isotropic adapter-scope SAM (AS(0)); RWP = random weight perturbation baseline.

| Method  | Optimizer | Aircraft | Cars196 | CUB200 | Flowers | OxfordPet |
|---------|-----------|----------|---------|--------|---------|-----------|
| SeqLoRA | SGD       | 39.26    | 33.39   | 65.48  | 60.82   | 69.94     |
|         | SAM       | 40.54    | 38.16   | 69.19  | 62.97   | 73.14     |
|         | GAM       | 43.86    | 51.65   | 73.54  | 85.80   | 83.03     |
|         | RWP       | 39.63    | 35.52   | 66.56  | 57.31   | 68.69     |
| IncLoRA | SGD       | 38.11    | 38.67   | 67.27  | 57.50   | 64.32     |
|         | SAM       | 38.37    | 41.80   | 70.16  | 61.04   | 71.53     |
|         | GAM       | 40.54    | 50.83   | 70.68  | 84.22   | 82.04     |
|         | RWP       | 27.97    | 21.44   | 68.29  | 80.01   | 74.74     |
| OLoRA   | SGD       | 36.87    | 36.75   | 66.55  | 57.15   | 64.29     |
|         | SAM       | 38.31    | 39.81   | 69.95  | 60.20   | 70.16     |
|         | GAM       | 35.53    | 51.01   | 70.33  | 83.93   | 81.91     |
|         | RWP       | 20.12    | 19.46   | 68.30  | 79.84   | 74.66     |
| InfLoRA | SGD       | 40.38    | 37.31   | 66.52  | 67.31   | 67.57     |
|         | SAM       | 40.90    | 42.43   | 71.35  | 73.09   | 73.43     |
|         | GAM       | 42.15    | 27.77   | 68.31  | 90.15   | 83.09     |
|         | RWP       | 38.68    | 41.05   | 76.19  | 67.09   | 68.85     |
| SDLoRA  | SGD       | 24.70    | 37.93   | 65.94  | 57.47   | 70.47     |
|         | SAM       | 37.91    | 41.01   | 67.95  | 61.05   | 71.53     |
|         | GAM       | 36.31    | 41.04   | 73.55  | 87.19   | 82.13     |
|         | RWP       | 37.46    | 39.26   | 67.55  | 55.11   | 67.49     |

**Summary**: SAM outperforms SGD on 5/5 datasets for all five methods (+3.0 to +4.6 pp avg).
GAM is the best variant on 5/5 datasets (SeqLoRA, IncLoRA), 4/5 (OLoRA, SDLoRA), 3/5 (InfLoRA).

---

## Setting 2: ViT-B/16 — Cross-Dataset Heterogeneous CL Benchmark (het5)

Sequential CL across five heterogeneous fine-grained datasets in a single stream:
Aircraft -> Cars196 -> CUB200 -> Flowers -> OxfordPet.
Metric: final average accuracy (%) after all tasks.

| Method  | Optimizer | 5-Dataset Avg |
|---------|-----------|---------------|
| SeqLoRA | SGD       | 66.30         |
|         | SAM       | 72.79         |
|         | GAM       | 78.15         |
|         | RWP       | 69.10         |
| IncLoRA | SGD       | 72.99         |
|         | SAM       | 74.42         |
|         | GAM       | 75.14         |
|         | RWP       | 67.93         |
| OLoRA   | SGD       | 70.98         |

Additional optimizer variants for OLoRA and results for InfLoRA/SDLoRA on this benchmark are being collected.

---

## Setting 3: T5-small and T5-large — NLP Sequential Learning

Sequential CL on an NLP task suite with T5-small and T5-large.
Three random task orders (o1, o2, o3). Optimizer: Adam (baseline) vs. adapter-scope SAM.
Metric: average accuracy (%) over all tasks after sequential training.

| Model    | Method  | Optimizer | Order 1 | Order 2 | Order 3 | Avg   |
|----------|---------|-----------|---------|---------|---------|-------|
| T5-small | SeqLoRA | Adam      | 34.32   | 44.26   | 50.22   | 42.93 |
|          |         | SAM       | 43.07   | 49.61   | 50.12   | 47.60 |
|          | IncLoRA | Adam      | 48.01   | 57.84   | 55.46   | 53.77 |
|          |         | SAM       | 50.92   | 56.99   | 45.56   | 51.16 |
|          | OLoRA   | Adam      | 62.45   | 61.68   | 57.36   | 60.50 |
|          |         | SAM       | 61.67   | 62.88   | 58.11   | 60.89 |
| T5-large | SeqLoRA | Adam      | 56.40   | 66.24   | 69.88   | 64.17 |
|          |         | SAM       | 65.43   | 68.09   | 73.35   | 68.96 |
|          | IncLoRA | Adam      | 48.01   | 77.39   | 70.37   | 65.26 |
|          |         | SAM       | 71.00   | 69.99   | 73.27   | 71.42 |
|          | OLoRA   | Adam      | 77.59   | 76.50   | 77.55   | 77.21 |
|          |         | SAM       | 78.51   | 78.60   | 76.26   | 77.79 |

**Summary**: On T5-large, adapter-scope SAM improves average FAA for all three methods: +4.8 pp (SeqLoRA), +6.2 pp (IncLoRA), +0.6 pp (OLoRA). On T5-small, gains are positive on average for SeqLoRA and OLoRA, while IncLoRA is mixed across orders.
