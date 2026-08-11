# Supplementary Results for ICML 2026 Rebuttal

This directory contains detailed result tables provided to facilitate verification
of all numbers reported in the paper.
All reported numbers are explicitly listed to support consistency checking.

In accordance with the double-blind review policy, only tabulated results are
released at this stage.

---

## 1. Vision Continual Learning (Fine-Grained Benchmarks)

Five fine-grained image classification datasets trained with LoRA-based continual
learning methods (SeqLoRA / IncLoRA / OLoRA / InfLoRA / SDLoRA) under four
optimizers (SGD / SAM / GAM / RWP). Backbone: ViT (pre-trained), LoRA rank=16,
40 epochs/task, seed=0.

### 1a. Combined Summary (FAA & AAA)

| File | Description |
|------|-------------|
| `combine_CNN_detailed.tex` | LaTeX table: FAA & AAA for all 5 methods × 4 optimizers across all 5 datasets |
| `combine_CNN_detailed.md`  | Markdown version of the same summary |
| `table_rebuttal_vision_onepage.tex` | Rebuttal-ready one-page table combining fine-grained datasets and het5 |
| `table_rebuttal_vision_onepage.md`  | Markdown version of the same one-page table |

### 1b. Full Task-wise Accuracy Matrices (lower-triangular, per dataset)

Each file contains 5 tables (one per method), each showing the complete
$A_{t,j}$ matrix (accuracy on task $j$ after training task $t$) for all 4 optimizers.

| Dataset | Config | `.tex` | `.md` |
|---------|--------|--------|-------|
| CUB-200 | ΔT=20, lr=1e-2 | `taskmatrix_cub200.tex` | `taskmatrix_cub200.md` |
| CARS-196 | ΔT=20, lr=5e-3 | `taskmatrix_cars196.tex` | `taskmatrix_cars196.md` |
| Aircraft | ΔT=10, lr=5e-3 | `taskmatrix_aircraft.tex` | `taskmatrix_aircraft.md` |
| Flowers | ΔT=10, lr=2.5e-4 | `taskmatrix_flowers.tex` | `taskmatrix_flowers.md` |
| Oxford-Pet | ΔT=4, lr=2.5e-4 | `taskmatrix_pets.tex` | `taskmatrix_pets.md` |

---

## 2. Vision Continual Learning (Cross-Dataset / Heterogeneous Setting)

Sequential learning across 5 heterogeneous datasets in a single training stream
(het5 benchmark, ΔT=1 dataset per task).

| File | Description |
|------|-------------|
| `Exp_Table1_5datasets—train-eval_het5_het5_inc1_rank16.xlsx` | Raw results for all methods and optimizers on the 5-dataset heterogeneous CL benchmark |

*(Located in `summaries/` relative to the project root.)*

---

## 3. LLM Continual Learning (NLP Benchmarks)

Sequential text classification on 4 NLP datasets (DBpedia / Amazon / Yahoo /
AGNews) using T5-large and T5-small backbones. Three task orders are provided to
verify order-robustness. Methods: SeqLoRA / IncLoRA / OLoRA, each with and
without SAM.

### 3a. Combined Summary (FAA across orders)

| File | Description |
|------|-------------|
| `combine_LLM_detailed.tex` | LaTeX table: FAA for T5-large / T5-small across all 3 orders |
| `combine_LLM_detailed.md`  | Markdown version of the same summary |
| `table_rebuttal_nlp_t5.tex` | Rebuttal-ready T5 table with Baseline / SAM / GAM / RWP rows |
| `table_rebuttal_nlp_t5.md`  | Markdown version of the same table |

### 3b. Full Task-wise Accuracy Matrices (per order)

Each file contains 6 tables (T5-large / T5-small x SeqLoRA / IncLoRA / OLoRA),
with Adam and SAM shown explicitly for every training step in the source
`Main_Result` sheet.

| Order | `.tex` | `.md` | Task sequence |
|-------|--------|-------|---------------|
| Order 1 | `taskmatrix_nlp_o1.tex` | `taskmatrix_nlp_o1.md` | DBpedia -> Amazon -> Yahoo -> AGNews |
| Order 2 | `taskmatrix_nlp_o2.tex` | `taskmatrix_nlp_o2.md` | DBpedia -> Amazon -> AGNews -> Yahoo |
| Order 3 | `taskmatrix_nlp_o3.tex` | `taskmatrix_nlp_o3.md` | Yahoo -> Amazon -> AGNews -> DBpedia |

### 3c. Source Training Logs

| File | Task Order | Tasks | Models |
|------|-----------|-------|--------|
| `Order1_Summary_redoanalyse2.xlsx` | Order 1: DBpedia→Amazon→Yahoo→AGNews | 4 | T5-large, T5-small |
| `SeqLoRA_Compare_redoanalyse.xlsx` | Order 2: DBpedia→Amazon→AGNews→Yahoo | 4 | T5-large, T5-small |
| `order3_Summary_Seqlora.xlsx`      | Order 3: Yahoo→Amazon→AGNews→DBpedia | 4 | T5-large, T5-small |

*(Located in `/Logs_Short_Paper/Order-{1,2,3}_8/` relative to the O-LoRA project root.)*

Each source Excel file contains:
- **`Main_Result` sheet**: raw task-wise accuracy matrix $A_{t,j}$ per method
- **`Metrics` sheet**: FAA, AAA summary per method

---

## Notes

- All `.tex` files compile directly with `booktabs` + `multirow` packages (ICML 2026 style).
- All `.md` files render in standard Markdown viewers (GitHub, VS Code, Obsidian).
- Every numeric value in the `.tex` and `.md` files has been automatically
  cross-verified against the source Excel sheets (tolerance < 0.02%).
- For NLP, `combine_LLM_detailed.*` is generated from the final task row of `Main_Result`
  in each Excel log; `taskmatrix_nlp_o{1,2,3}.*` mirrors the raw task-wise values.
