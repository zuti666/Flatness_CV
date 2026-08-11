# Order 3 - Task-wise Accuracy Matrix (Raw NLP Results)

**Task order:** Yahoo -> Amazon -> AGNews -> DBpedia  
**Backbones:** T5-large, T5-small  
**Methods:** SeqLoRA, IncLoRA, OLoRA  
**Optimizers:** Adam, SAM  

$A_{t,j}$ = raw task-wise accuracy from the source `Main_Result` sheet.  
**Mean** = `Model_Average` at step $t$; final row Mean = FAA.  
---
## T5-large / SeqLoRA

| Opt. | After t | yahoo | amazon | agnews | dbpedia | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-yahoo | 73.66 | --- | --- | --- | 73.66 |
| Adam | 2-amazon | 66.71 | 60.64 | --- | --- | 63.68 |
| Adam | 3-agnews | 48.51 | 57.17 | 89.62 | --- | 65.10 |
| Adam | 4-dbpedia | 62.08 | 46.29 | 72.89 | 98.25 | 69.88 |
| SAM | 1-yahoo | 72.34 | 0.00 | 51.34 | 9.05 | 33.18 |
| SAM | 2-amazon | 57.54 | 58.57 | 59.82 | 52.21 | 57.03 |
| SAM | 3-agnews | 42.07 | 50.33 | 89.53 | 37.49 | 54.85 |
| SAM | 4-dbpedia | 66.64 | 47.63 | 80.39 | 98.72 | 73.35 |

---
## T5-large / IncLoRA

| Opt. | After t | yahoo | amazon | agnews | dbpedia | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-yahoo | 73.66 | --- | --- | --- | 73.66 |
| Adam | 2-amazon | 67.09 | 59.74 | --- | --- | 63.41 |
| Adam | 3-agnews | 59.50 | 58.28 | 89.55 | --- | 69.11 |
| Adam | 4-dbpedia | 60.67 | 40.88 | 81.67 | 98.25 | 70.37 |
| SAM | 1-yahoo | 72.34 | --- | --- | --- | 72.34 |
| SAM | 2-amazon | 65.28 | 58.21 | --- | --- | 61.74 |
| SAM | 3-agnews | 61.79 | 57.70 | 89.62 | --- | 69.70 |
| SAM | 4-dbpedia | 57.82 | 53.83 | 82.93 | 98.50 | 73.27 |

---
## T5-large / OLoRA

| Opt. | After t | yahoo | amazon | agnews | dbpedia | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-yahoo | 73.66 | --- | --- | --- | 73.66 |
| Adam | 2-amazon | 72.04 | 55.24 | --- | --- | 63.64 |
| Adam | 3-agnews | 71.80 | 49.07 | 87.29 | --- | 69.39 |
| Adam | 4-dbpedia | 70.84 | 54.11 | 86.46 | 98.79 | 77.55 |
| SAM | 1-yahoo | 72.34 | --- | --- | --- | 72.34 |
| SAM | 2-amazon | 70.46 | 54.51 | --- | --- | 62.49 |
| SAM | 3-agnews | 71.71 | 51.12 | 89.53 | --- | 70.79 |
| SAM | 4-dbpedia | 69.70 | 49.41 | 87.38 | 98.57 | 76.26 |

---
## T5-small / SeqLoRA

| Opt. | After t | yahoo | amazon | agnews | dbpedia | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-yahoo | 63.53 | --- | --- | --- | 63.53 |
| Adam | 2-amazon | 11.08 | 43.12 | --- | --- | 27.10 |
| Adam | 3-agnews | 18.84 | 24.86 | 82.46 | --- | 42.05 |
| Adam | 4-dbpedia | 39.03 | 14.86 | 50.83 | 96.18 | 50.22 |
| SAM | 1-yahoo | 61.47 | --- | --- | --- | 61.47 |
| SAM | 2-amazon | 46.14 | 36.37 | --- | --- | 41.26 |
| SAM | 3-agnews | 20.96 | 35.34 | 80.51 | --- | 45.61 |
| SAM | 4-dbpedia | 37.53 | 27.66 | 38.83 | 96.49 | 50.12 |

---
## T5-small / IncLoRA

| Opt. | After t | yahoo | amazon | agnews | dbpedia | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-yahoo | 63.53 | --- | --- | --- | 63.53 |
| Adam | 2-amazon | 23.93 | 38.14 | --- | --- | 31.04 |
| Adam | 3-agnews | 21.45 | 35.32 | 83.13 | --- | 46.63 |
| Adam | 4-dbpedia | 40.20 | 25.92 | 59.96 | 95.78 | 55.46 |
| SAM | 1-yahoo | 61.47 | --- | --- | --- | 61.47 |
| SAM | 2-amazon | 51.01 | 35.72 | --- | --- | 43.37 |
| SAM | 3-agnews | 22.71 | 34.76 | 80.59 | --- | 46.02 |
| SAM | 4-dbpedia | 36.26 | 17.47 | 32.54 | 95.97 | 45.56 |

---
## T5-small / OLoRA

| Opt. | After t | yahoo | amazon | agnews | dbpedia | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-yahoo | 63.53 | --- | --- | --- | 63.53 |
| Adam | 2-amazon | 59.55 | 35.20 | --- | --- | 47.38 |
| Adam | 3-agnews | 49.70 | 31.87 | 71.75 | --- | 51.11 |
| Adam | 4-dbpedia | 50.01 | 29.36 | 55.57 | 94.49 | 57.36 |
| SAM | 1-yahoo | 61.47 | --- | --- | --- | 61.47 |
| SAM | 2-amazon | 56.29 | 33.68 | --- | --- | 44.99 |
| SAM | 3-agnews | 52.91 | 32.24 | 71.30 | --- | 52.15 |
| SAM | 4-dbpedia | 50.68 | 29.71 | 57.09 | 94.96 | 58.11 |

---
