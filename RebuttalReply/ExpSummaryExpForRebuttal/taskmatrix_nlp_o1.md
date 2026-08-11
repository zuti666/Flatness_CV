# Order 1 - Task-wise Accuracy Matrix (Raw NLP Results)

**Task order:** DBpedia -> Amazon -> Yahoo -> AGNews  
**Backbones:** T5-large, T5-small  
**Methods:** SeqLoRA, IncLoRA, OLoRA  
**Optimizers:** Adam, SAM  

$A_{t,j}$ = raw task-wise accuracy from the source `Main_Result` sheet.  
**Mean** = `Model_Average` at step $t$; final row Mean = FAA.  
---
## T5-large / SeqLoRA

| Opt. | After t | dbpedia | amazon | yahoo | agnews | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-dbpedia | 98.74 | --- | --- | --- | 98.74 |
| Adam | 2-amazon | 96.18 | 58.99 | --- | --- | 77.59 |
| Adam | 3-yahoo | 76.33 | 43.59 | 73.54 | --- | 64.49 |
| Adam | 4-agnews | 52.59 | 38.67 | 44.22 | 90.11 | 56.40 |
| SAM | 1-dbpedia | 98.79 | --- | --- | --- | 98.79 |
| SAM | 2-amazon | 97.97 | 58.57 | --- | --- | 78.27 |
| SAM | 3-yahoo | 88.84 | 51.58 | 73.55 | --- | 71.32 |
| SAM | 4-agnews | 73.49 | 39.87 | 58.16 | 90.21 | 65.43 |

---
## T5-large / IncLoRA

| Opt. | After t | dbpedia | amazon | yahoo | agnews | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-dbpedia | 96.24 | --- | --- | --- | 96.24 |
| Adam | 2-amazon | 29.43 | 37.96 | --- | --- | 33.70 |
| Adam | 3-yahoo | 65.88 | 33.47 | 63.50 | --- | 54.29 |
| Adam | 4-agnews | 35.78 | 31.78 | 39.75 | 84.74 | 48.01 |
| SAM | 1-dbpedia | 98.79 | --- | --- | --- | 98.79 |
| SAM | 2-amazon | 96.97 | 58.47 | --- | --- | 77.72 |
| SAM | 3-yahoo | 82.39 | 51.28 | 72.82 | --- | 68.83 |
| SAM | 4-agnews | 91.59 | 42.80 | 59.64 | 89.95 | 71.00 |

---
## T5-large / OLoRA

| Opt. | After t | dbpedia | amazon | yahoo | agnews | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-dbpedia | 98.74 | --- | --- | --- | 98.74 |
| Adam | 2-amazon | 98.76 | 50.92 | --- | --- | 74.84 |
| Adam | 3-yahoo | 98.63 | 53.86 | 72.76 | --- | 75.08 |
| Adam | 4-agnews | 98.54 | 51.37 | 70.97 | 89.46 | 77.59 |
| SAM | 1-dbpedia | 98.79 | --- | --- | --- | 98.79 |
| SAM | 2-amazon | 98.59 | 58.26 | --- | --- | 78.43 |
| SAM | 3-yahoo | 98.54 | 54.47 | 73.14 | --- | 75.39 |
| SAM | 4-agnews | 98.37 | 54.50 | 71.93 | 89.22 | 78.51 |

---
## T5-small / SeqLoRA

| Opt. | After t | dbpedia | amazon | yahoo | agnews | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-dbpedia | 96.24 | --- | --- | --- | 96.24 |
| Adam | 2-amazon | 28.61 | 39.28 | --- | --- | 33.94 |
| Adam | 3-yahoo | 24.00 | 33.46 | 64.99 | --- | 40.82 |
| Adam | 4-agnews | 0.93 | 29.17 | 23.99 | 83.20 | 34.32 |
| SAM | 1-dbpedia | 95.47 | --- | --- | --- | 95.47 |
| SAM | 2-amazon | 64.82 | 37.38 | --- | --- | 51.10 |
| SAM | 3-yahoo | 82.86 | 36.08 | 63.70 | --- | 60.88 |
| SAM | 4-agnews | 26.30 | 34.03 | 28.92 | 83.03 | 43.07 |

---
## T5-small / IncLoRA

| Opt. | After t | dbpedia | amazon | yahoo | agnews | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-dbpedia | 96.24 | --- | --- | --- | 96.24 |
| Adam | 2-amazon | 29.43 | 37.96 | --- | --- | 33.70 |
| Adam | 3-yahoo | 65.88 | 33.47 | 63.50 | --- | 54.29 |
| Adam | 4-agnews | 35.78 | 31.78 | 39.75 | 84.74 | 48.01 |
| SAM | 1-dbpedia | 95.47 | --- | --- | --- | 95.47 |
| SAM | 2-amazon | 45.95 | 38.75 | --- | --- | 42.35 |
| SAM | 3-yahoo | 70.34 | 36.74 | 61.13 | --- | 56.07 |
| SAM | 4-agnews | 53.91 | 33.13 | 33.46 | 83.18 | 50.92 |

---
## T5-small / OLoRA

| Opt. | After t | dbpedia | amazon | yahoo | agnews | Mean |
|---|---|---|---|---|---|---|
| Adam | 1-dbpedia | 96.24 | --- | --- | --- | 96.24 |
| Adam | 2-amazon | 79.74 | 36.74 | --- | --- | 58.24 |
| Adam | 3-yahoo | 91.55 | 34.59 | 58.92 | --- | 61.69 |
| Adam | 4-agnews | 82.92 | 33.01 | 54.91 | 78.95 | 62.45 |
| SAM | 1-dbpedia | 95.47 | --- | --- | --- | 95.47 |
| SAM | 2-amazon | 76.09 | 37.25 | --- | --- | 56.67 |
| SAM | 3-yahoo | 89.87 | 36.09 | 60.03 | --- | 62.00 |
| SAM | 4-agnews | 80.07 | 35.63 | 53.45 | 77.55 | 61.67 |

---
