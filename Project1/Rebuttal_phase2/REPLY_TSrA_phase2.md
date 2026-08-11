# Phase 2 Reply — Reviewer TSrA

## TSrA
> "Partially resolved. I will raise my score to 3, as I still find the motivation of the paper to be weak and insufficient to meet the acceptance standard."

We clarify the motivation directly. The paper is not about whether SAM can be applied to LoRA in general; it addresses a specific unresolved ambiguity in frozen-backbone PECL. In this setting, all future adaptation is confined to a low-rank admissible subspace, yet existing LoRA flatness prescriptions are not aligned: some works perturb only the trainable adapters, while Flat-LoRA argues that full-parameter perturbation is needed even with a frozen backbone. The question we study is therefore: once adaptation is subspace-confined, which sharpness notion is actually relevant to continual generalization?

Theorem 4.4 gives the PECL-specific answer. After specializing the sequential PAC-Bayesian analysis to frozen-backbone adaptation, both the KL term and the sharpness-smoothed empirical term reduce to quantities supported on $W_{\Delta,t}$. In other words, the bound-derived perturbation criterion is determined by directions the learner can actually update, rather than by curvature in frozen directions.

This is the contribution we want to isolate: identifying the sharpness object that matches the PECL update mechanism, rather than proposing another optimizer. The empirical results then become interpretable rather than merely observational: in the reported settings, adapter-scoped perturbation is preferable to full-scope perturbation, which is exactly the design choice the paper resolves.
