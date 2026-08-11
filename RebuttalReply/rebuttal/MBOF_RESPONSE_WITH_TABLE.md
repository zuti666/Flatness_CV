# mBoF Response Block With Settings and Summary Table

## Suggested Rebuttal Text

We agree that the current empirical section should not support a universal claim. Our intended claim is narrower: in frozen-backbone PECL, the relevant perturbation geometry is the adapter subspace. To broaden evidence, the supplement now adds three settings (full tables at [ANON_LINK]): (i) ViT-B/16 on five fine-grained datasets with five LoRA variants and four optimizers; (ii) a heterogeneous 5-dataset stream Aircraft→Cars196→CUB200→Flowers→OxfordPet; and (iii) T5-small/T5-large over three NLP task orders.

These additions directly address Q2-Q4. Across the five fine-grained datasets, SAM exceeds SGD for all five LoRA variants (+3.0–4.6 pp avg). On the heterogeneous 5-dataset benchmark, SeqLoRA gains +9.4 pp and IncLoRA +1.4 pp over SGD. On T5-large, average FAA improves by +4.8 pp (SeqLoRA), +6.2 pp (IncLoRA), and +0.6 pp (OLoRA). Thus the ordering persists across additional datasets, a cross-dataset stream, and a different architecture/scale.

## Experimental Settings

- Setting 1: `ViT-B/16` (ImageNet-21K pretrained, frozen backbone) on five fine-grained datasets: Aircraft, Cars196, CUB200, Flowers, OxfordPet. Methods: SeqLoRA, IncLoRA, OLoRA, InfLoRA, SDLoRA. Optimizers: SGD, SAM, GAM, RWP. Metric below: average over the five per-dataset continual-accuracy summaries reported in the supplement.
- Setting 2: heterogeneous `het5` cross-dataset stream `Aircraft -> Cars196 -> CUB200 -> Flowers -> OxfordPet`. Backbone: `ViT-B/16`. Completed methods with full optimizer comparison: SeqLoRA and IncLoRA. Metric: 5-dataset aggregate accuracy.
- Setting 3: `T5-small` and `T5-large` on three NLP task orders over DBpedia, Amazon, Yahoo, and AGNews. Methods: SeqLoRA, IncLoRA, OLoRA. Optimizers: Adam and adapter-scope SAM. Metric: average FAA over the three orders.

## One-Page Result Table

| Setting | Metric | Model | Method | Baseline | SAM | GAM | RWP | Delta (SAM-Baseline) |
|---|---|---|---|---:|---:|---:|---:|---:|
| Fine-grained 5 datasets | Avg over 5 dataset summaries | ViT-B/16 | SeqLoRA | 53.78 | 56.80 | 67.58 | 53.54 | +3.02 |
| Fine-grained 5 datasets | Avg over 5 dataset summaries | ViT-B/16 | IncLoRA | 53.17 | 56.58 | 65.66 | 54.49 | +3.41 |
| Fine-grained 5 datasets | Avg over 5 dataset summaries | ViT-B/16 | OLoRA | 52.32 | 55.69 | 64.54 | 52.48 | +3.37 |
| Fine-grained 5 datasets | Avg over 5 dataset summaries | ViT-B/16 | InfLoRA | 55.82 | 60.24 | 62.29 | 58.37 | +4.42 |
| Fine-grained 5 datasets | Avg over 5 dataset summaries | ViT-B/16 | SDLoRA | 51.30 | 55.89 | 64.04 | 53.37 | +4.59 |
| Heterogeneous het5 | 5-dataset aggregate | ViT-B/16 | SeqLoRA | 66.30 | 72.79 | 78.15 | 69.10 | +6.49 |
| Heterogeneous het5 | 5-dataset aggregate | ViT-B/16 | IncLoRA | 72.99 | 74.42 | 75.14 | 67.93 | +1.43 |
| NLP, 3 orders | Avg FAA over 3 orders | T5-large | SeqLoRA | 64.17 | 68.96 | -- | -- | +4.78 |
| NLP, 3 orders | Avg FAA over 3 orders | T5-large | IncLoRA | 65.26 | 71.42 | -- | -- | +6.16 |
| NLP, 3 orders | Avg FAA over 3 orders | T5-large | OLoRA | 77.21 | 77.79 | -- | -- | +0.58 |
| NLP, 3 orders | Avg FAA over 3 orders | T5-small | SeqLoRA | 42.93 | 47.60 | -- | -- | +4.67 |
| NLP, 3 orders | Avg FAA over 3 orders | T5-small | IncLoRA | 53.77 | 51.16 | -- | -- | -2.61 |
| NLP, 3 orders | Avg FAA over 3 orders | T5-small | OLoRA | 60.50 | 60.89 | -- | -- | +0.39 |

## Notes

- For the vision fine-grained setting, the table compresses the five per-dataset supplement tables into a single average for quick reading; the full per-dataset numbers remain in `ANON_REPO_README.md` and `combine_CNN_detailed.*`.
- For the NLP setting, `T5-large` is the cleaner scalability signal; `T5-small` is included for completeness and should be described as supplementary rather than headline evidence.
