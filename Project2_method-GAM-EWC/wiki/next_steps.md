# Next Steps

## Priority 1 — Experiments in Progress (wait for results)

- **Cars196**: `logs_paperA/cross_datasets_method3/as1_normfisher_lam2000_cars196_t10c20_r16_gpu2.log`
- **ImageNet-R**: `logs_paperA/imagenetr_method3/as1_normfisher_lam2000_imagenetr_t20c10_r16_gpu0.log`
- **EWC-Yaoyue baseline**: `logs_paperA/ewc_yaoyue_sgd/` — run when GPUs free

---

## Priority 2 — Layer 2 / Layer 3 lambda_flat Sweep (most impactful)

**Problem:** flat/clean ≈ 4.3 on CUB200 (80 on OxfordPet). AS^1 dominates the update; Fisher never meaningfully constrains forgetting at any reasonable lambda.

**Experiment:** Layer2 grid on CUB200 t20:
```
lambda_flat ∈ {0.25, 0.5, 1.0}
ewc_lambda  ∈ {100, 500, 1000}
```
```bash
RUN_EWC_SCAN=0 RUN_FLAT_GRID=1 GRID_FLATS="0.5 0.25" GRID_EWCS="100 500 1000" \
GPUS_SWEEP="1 2 3 5 6 7" \
bash scripts/exp_paperA/run_layer2_rawfisher_sweep_cub200_t20.sh
```

**Success criterion:** CNN Forget < 11.5 while CNN FAA ≥ 76.0, fisher/clean in 0.02–0.10.

---

## Priority 3 — 2×2 Ablation (needed for paper §5.1)

```bash
RUN_THREE_METHODS=0 RUN_ABLATION=1 RUN_NORMFISHER_SWEEP=0 \
GPUS_ABL="0 1 2 3" \
bash scripts/exp_paperA/run_paperA_cub200_t20.sh
```

Must show: Full > Flat-only > Base; Full > Fisher-only > Base.

---

## Priority 4 — Dual Ascent Revisit

Current dual didn't activate because D^Fisher < δ=1e-4 (drift budget too generous).  
Retry with tighter budget:
```yaml
ewc_dual_delta: 1.0e-5   # was 1e-4
ewc_dual_alpha: 2000.0   # was 500
```
Add a config `as1_normfisher_dual_tighter_cub200_t20.yaml` and run.

---

## Priority 5 — Dataset-Specific Tuning (Aircraft / OxfordPet)

OxfordPet flat/clean=80 → try `lambda_flat=0.1, gam_rho=0.05`.  
Aircraft flat/clean=7.6 → try `lambda_flat=0.5, ewc_lambda=500`.

---

## Paper Writing Milestones

| Section | Status | Blocker |
|---------|--------|---------|
| §3 Method | Draft exists | — |
| §4 Orthogonality theorem | Informal claim written | Formal proof needed |
| §5.1 Ablation 2×2 | Config ready | Run ablation |
| §5.2 Mechanism stats (cos_flat_fisher table) | CUB200 done | Need ImageNet-R, Cars196 |
| §5.3 Pareto vs baselines | Pending | EWC-Yaoyue results needed |
| §5.4 Cross-dataset | Partial (4/6 datasets) | Cars196, ImageNet-R running |

---

## Open Questions

1. **Why does Layer1 have lower forgetting than Layer2?** Layer1 couples Fisher into GAM's perturbation, which might accidentally constrain ΔW drift. Needs mechanistic explanation.

2. **Does lower lambda_flat actually hurt FAA?** flat/clean=4.3 suggests AS^1 strongly helps current-task learning. Reducing lambda_flat may hurt FAA. Need to track both.

3. **Dual ascent with tighter δ**: Will it converge to a better trade-off, or oscillate?

4. **Does the orthogonality finding hold for larger LoRA rank (r=32)?** At larger rank, adapter subspace grows — Fisher and flatness directions may become less separated.
