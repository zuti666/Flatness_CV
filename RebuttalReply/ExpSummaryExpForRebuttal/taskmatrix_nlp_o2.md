# Order 2 - Task-wise Accuracy Matrix (Raw NLP Results)

**Task order:** DBpedia -> Amazon -> AGNews -> Yahoo  
**Backbones:** T5-large, T5-small  
**Methods:** SeqLoRA, IncLoRA, OLoRA  
**Optimizers:** Adam, SAM  

$A_{t,j}$ = raw task-wise accuracy from the source `Main_Result` sheet.  
**Mean** = `Model_Average` at step $t$; final row Mean = FAA.  
---
## T5-large / SeqLoRA

| Opt. | After t | dbpedia | amazon | agnews | yahoo | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-dbpedia | 98.74 | --- | --- | --- | 98.74 |
| Adam | 2-amazon | 96.18 | 58.99 | --- | --- | 77.59 |
| Adam | 3-agnews | 96.79 | 58.91 | 90.32 | --- | 82.00 |
| Adam | 4-yahoo | 58.83 | 53.84 | 79.36 | 72.92 | 66.24 |
| SAM | 1-dbpedia | 98.79 | --- | --- | --- | 98.79 |
| SAM | 2-amazon | 97.97 | 58.57 | --- | --- | 78.27 |
| SAM | 3-agnews | 96.50 | 57.39 | 90.17 | --- | 81.36 |
| SAM | 4-yahoo | 77.42 | 49.66 | 72.28 | 72.99 | 68.09 |

---
## T5-large / IncLoRA

| Opt. | After t | dbpedia | amazon | agnews | yahoo | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-dbpedia | 98.74 | --- | --- | --- | 98.74 |
| Adam | 2-amazon | 98.62 | 59.08 | --- | --- | 78.85 |
| Adam | 3-agnews | 98.25 | 60.12 | 89.17 | --- | 82.51 |
| Adam | 4-yahoo | 96.25 | 54.46 | 85.79 | 73.07 | 77.39 |
| SAM | 1-dbpedia | 98.79 | --- | --- | --- | 98.79 |
| SAM | 2-amazon | 96.97 | 58.47 | --- | --- | 77.72 |
| SAM | 3-agnews | 96.42 | 57.93 | 90.34 | --- | 81.57 |
| SAM | 4-yahoo | 80.76 | 49.37 | 77.38 | 72.45 | 69.99 |

---
## T5-large / OLoRA

| Opt. | After t | dbpedia | amazon | agnews | yahoo | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-dbpedia | 98.74 | --- | --- | --- | 98.74 |
| Adam | 2-amazon | 98.76 | 50.92 | --- | --- | 74.84 |
| Adam | 3-agnews | 98.70 | 49.82 | 88.53 | --- | 79.01 |
| Adam | 4-yahoo | 98.67 | 48.83 | 86.92 | 71.57 | 76.50 |
| SAM | 1-dbpedia | 98.79 | --- | --- | --- | 98.79 |
| SAM | 2-amazon | 98.59 | 58.26 | --- | --- | 78.43 |
| SAM | 3-agnews | 98.45 | 58.50 | 88.55 | --- | 81.83 |
| SAM | 4-yahoo | 98.54 | 55.86 | 87.09 | 72.89 | 78.60 |

---
## T5-small / SeqLoRA

| Opt. | After t | dbpedia | amazon | agnews | yahoo | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-dbpedia | 96.24 | --- | --- | --- | 96.24 |
| Adam | 2-amazon | 28.61 | 39.28 | --- | --- | 33.94 |
| Adam | 3-agnews | 41.74 | 32.37 | 82.46 | --- | 52.19 |
| Adam | 4-yahoo | 3.59 | 34.78 | 72.63 | 66.05 | 44.26 |
| SAM | 1-dbpedia | 95.47 | --- | --- | --- | 95.47 |
| SAM | 2-amazon | 64.82 | 37.38 | --- | --- | 51.10 |
| SAM | 3-agnews | 61.89 | 35.21 | 81.34 | --- | 59.48 |
| SAM | 4-yahoo | 39.25 | 35.28 | 60.01 | 63.91 | 49.61 |

---
## T5-small / IncLoRA

| Opt. | After t | dbpedia | amazon | agnews | yahoo | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-dbpedia | 96.24 | --- | --- | --- | 96.24 |
| Adam | 2-amazon | 29.43 | 37.96 | --- | --- | 33.70 |
| Adam | 3-agnews | 69.99 | 31.45 | 81.74 | --- | 61.06 |
| Adam | 4-yahoo | 56.92 | 30.95 | 78.38 | 65.12 | 57.84 |
| SAM | 1-dbpedia | 95.47 | --- | --- | --- | 95.47 |
| SAM | 2-amazon | 45.95 | 38.75 | --- | --- | 42.35 |
| SAM | 3-agnews | 76.37 | 35.21 | 79.68 | --- | 63.75 |
| SAM | 4-yahoo | 58.46 | 35.91 | 70.41 | 63.17 | 56.99 |

---
## T5-small / OLoRA

| Opt. | After t | dbpedia | amazon | agnews | yahoo | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-dbpedia | 96.24 | --- | --- | --- | 96.24 |
| Adam | 2-amazon | 79.74 | 36.74 | --- | --- | 58.24 |
| Adam | 3-agnews | 68.32 | 34.97 | 71.26 | --- | 58.18 |
| Adam | 4-yahoo | 90.43 | 35.04 | 61.83 | 59.42 | 61.68 |
| SAM | 1-dbpedia | 95.47 | --- | --- | --- | 95.47 |
| SAM | 2-amazon | 76.09 | 37.25 | --- | --- | 56.67 |
| SAM | 3-agnews | 81.84 | 35.49 | 73.88 | --- | 63.74 |
| SAM | 4-yahoo | 88.46 | 35.16 | 67.92 | 60.00 | 62.88 |

---
