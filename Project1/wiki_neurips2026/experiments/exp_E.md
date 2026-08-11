---
id: exp_E
name: Support × Direction Main Experiment
dataset: ImageNet-R r=16 T=20
method: SeqLoRA, ViT-B/16, no replay, seed 1993
status: complete
config: config_exps_paper1_PAC/exp_E_imagenetr_r16_t20_support_direction
results: outputs_logs/exp_E_imagenetr_r16_t20_support_direction_summary/
---

## Design

10 variants crossing perturbation support {factor, full, delta, all, frozen} × direction {SAM adversarial, Gaussian random}, plus SGD baseline.

## Key Results

| Variant | FAA | AAA | BWT | Forget |
|---------|-----|-----|-----|--------|
| sgd | 58.29 | 67.05 | -18.99 | 19.09 |
| sam_factor | **67.94** | **70.60** | **-6.81** | **7.10** |
| sam_full | 64.67 | 69.04 | -12.01 | 12.04 |
| sam_delta | 65.41 | 69.63 | -11.56 | 11.66 |
| sam_all | 63.19 | 67.55 | -12.28 | 12.28 |
| sam_frozen | 62.91 | 67.29 | -12.37 | 12.46 |
| random_factor | 58.79 | 67.12 | -18.47 | 18.57 |
| random_full | 58.15 | 67.01 | -19.24 | 19.34 |
| random_delta | 58.26 | 67.04 | -19.08 | 19.18 |
| random_all | 58.32 | 67.04 | -19.01 | 19.11 |
| random_frozen | 58.24 | 67.05 | -19.10 | 19.20 |

## Pairwise SAM vs Random by Support

| Support | SAM FAA | Random FAA | Delta FAA | SAM BWT | Random BWT | Delta BWT |
|---------|---------|------------|-----------|---------|------------|-----------|
| factor | 67.94 | 58.79 | **+9.14** | -6.81 | -18.47 | **+11.67** |
| full | 64.67 | 58.15 | +6.52 | -12.01 | -19.24 | +7.23 |
| delta | 65.41 | 58.26 | +7.15 | -11.56 | -19.08 | +7.52 |
| all | 63.19 | 58.32 | +4.87 | -12.28 | -19.01 | +6.72 |
| frozen | 62.91 | 58.24 | +4.67 | -12.37 | -19.10 | +6.73 |

## Interpretation

- For every support, SAM direction >> Gaussian random direction. Supports **C4**.
- random_* ≈ SGD baseline → generic noise injection does not explain the benefit.
- sam_factor is empirically strongest. Does NOT yet prove raw factor-space is theoretically correct — Exp G rescaling control needed (**C6**).
- sam_delta improves strongly, confirming effective adapter-update geometry is relevant.
- sam_frozen is the weakest SAM variant, confirming frozen-backbone directions are not the relevant support.

## Figures

- `figures/support_direction_faa_bwt.png` / `.pdf`
- `figures/support_direction_pairwise.csv`

## Supports Claims

- **C4** (SAM direction > random direction for every support)
- Partial **C3** (effective adapter-update space is effective; frozen-only is not)
