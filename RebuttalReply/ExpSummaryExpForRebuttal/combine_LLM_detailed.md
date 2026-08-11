# Combined LLM Results: FAA Across All Task Orders

**Metric:** FAA (Final Average Accuracy, %) extracted from the final task row in `Main_Result`.

**Setup:** T5-large / T5-small · 4 NLP datasets per order · SeqLoRA / IncLoRA / OLoRA · Adam vs. SAM

| Order | Task sequence |
|---|---|
| o1 | DBpedia -> Amazon -> Yahoo -> AGNews |
| o2 | DBpedia -> Amazon -> AGNews -> Yahoo |
| o3 | Yahoo -> Amazon -> AGNews -> DBpedia |

| Model | Method | Optimizer | o1 | o2 | o3 | Avg |
|---|---|---|---|---|---|---|
| T5-large | SeqLoRA | Adam | 56.40 | 66.24 | 69.88 | 64.17 |
| T5-large | SeqLoRA | SAM | 65.43 | 68.09 | 73.35 | 68.96 |
| T5-large | IncLoRA | Adam | 48.01 | 77.39 | 70.37 | 65.26 |
| T5-large | IncLoRA | SAM | 71.00 | 69.99 | 73.27 | 71.42 |
| T5-large | OLoRA | Adam | 77.59 | 76.50 | 77.55 | 77.21 |
| T5-large | OLoRA | SAM | 78.51 | 78.60 | 76.26 | 77.79 |
| T5-small | SeqLoRA | Adam | 34.32 | 44.26 | 50.22 | 42.93 |
| T5-small | SeqLoRA | SAM | 43.07 | 49.61 | 50.12 | 47.60 |
| T5-small | IncLoRA | Adam | 48.01 | 57.84 | 55.46 | 53.77 |
| T5-small | IncLoRA | SAM | 50.92 | 56.99 | 45.56 | 51.16 |
| T5-small | OLoRA | Adam | 62.45 | 61.68 | 57.36 | 60.50 |
| T5-small | OLoRA | SAM | 61.67 | 62.88 | 58.11 | 60.89 |

**Observation:** On T5-large, SAM improves the average FAA for all three methods; on T5-small, gains are positive for SeqLoRA and OLoRA but mixed for IncLoRA.
