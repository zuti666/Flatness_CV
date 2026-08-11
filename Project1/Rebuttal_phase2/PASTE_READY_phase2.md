## TSrA

We clarify the motivation directly.

The paper is not about whether SAM can be applied to LoRA in general; it addresses a specific unresolved ambiguity in frozen-backbone PECL: which notion of sharpness is actually relevant to continual generalization, and why does that matter for forgetting?

What was missing was a PECL-specific criterion for this question. PECL methods already organize adaptation through explicit low-rank subspaces to control transfer and interference [1–3], while Flat-LoRA argues that full-parameter perturbation is needed even with a frozen backbone [4]. The missing point was therefore not another implementation comparison, but identifying which sharpness notion remains relevant once updates are structurally confined to the admissible subspace.

Theorem 4.4 gives the PECL-specific answer. After specializing the sequential PAC-Bayesian analysis to frozen-backbone adaptation, both the KL term and the sharpness-smoothed empirical term reduce to quantities supported on $W_{\Delta,t}$. This implies that the bound-derived perturbation criterion is determined by update-permissible directions. Since later adaptation in PECL is confined to those directions, curvature outside $W_{\Delta,t}$ is not part of the criterion derived by the bound.

Within this relevant support, the paper further analyzes perturbation type, showing that more curvature-aware types impose progressively stronger control on subspace sharpness, consistent with the observed ordering RS(0) < AS(0) < AS(1).

Our contribution is to identify the sharpness object that matches the PECL update mechanism, rather than to propose another optimizer. Under that criterion, the empirical results become interpretable rather than merely observational: full-scope perturbation is detrimental, while within the relevant support different perturbation types induce systematic differences in subspace curvature control.

[1] Orthogonal Subspace Learning for Language Model Continual Learning, EMNLP 2023.
[2] InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning, CVPR 2024.
[3] SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning, ICLR 2025.
[4] Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape, ICML 2025.

---

## mBoF

To directly address the concern about limited architectural coverage and scale variation, we further added decoder-only LLM experiments on Llama-3.2-1B and Llama-3.2-3B. The new setup extends the earlier NLP check from encoder-decoder LMs to decoder-only LLMs, and from T5-small / T5-large to Llama-3.2-1B / 3B, while keeping the same continual-learning protocol for a controlled comparison. The results are directionally consistent across all 1B and 3B comparisons: the average gains are +4.9 / +5.6 / +6.9 points on 1B and +5.7 / +5.6 / +5.4 points on 3B for SeqLoRA / IncLoRA / OLoRA, respectively. Together with the ViT and T5 results already added in Phase 1, this supports that the qualitative effect is not confined to a narrow ViT-to-T5 transition or a single scale point.

Table 1 (LA: Last task Acc per order)

| Model | Method | Opt. | o1 | o2 | o3 | avg |
| - | - | - | -: | -: | -: | -: |
| Llama-3.2-1B | SeqLoRA | Adam | 66.9 | 55.7 | 68.1 | 63.6 |
| | | AS(0) | 69.2 | 66.0 | 70.2 | 68.5 |
| | IncLoRA | Adam | 64.5 | 56.2 | 65.4 | 62.1 |
| | | AS(0) | 67.5 | 64.9 | 70.7 | 67.7 |
| | OLoRA | Adam | 58.6 | 53.1 | 58.4 | 56.7 |
| | | AS(0) | 66.0 | 61.5 | 63.3 | 63.6 |
| Llama-3.2-3B | SeqLoRA | Adam | 67.9 | 67.3 | 67.4 | 67.5 |
| | | AS(0) | 73.3 | 72.6 | 73.8 | 73.2 |
| | IncLoRA | Adam | 65.2 | 66.7 | 69.5 | 67.2 |
| | | AS(0) | 72.4 | 72.0 | 74.0 | 72.8 |
| | OLoRA | Adam | 63.4 | 63.0 | 63.9 | 63.4 |
| | | AS(0) | 71.1 | 69.5 | 65.7 | 68.8 |

The contribution is not simply to show that SAM can help LoRA empirically, but to establish that, in frozen-backbone PECL, the theoretically relevant sharpness and KL terms are supported on the admissible adapter subspace $W_{\Delta,t}$. This is precisely the role of Theorem 4.4. The novelty is therefore not the generic observation that flatness can help, which is already known in full-model training and LoRA fine-tuning, but identifying the sharpness object that matches the PECL update mechanism once future adaptation is restricted to the admissible low-rank subspace. The added experiments are therefore used only to test the qualitative consequence of this reduction across broader model families, not to define the contribution itself.

[1] Lfpt5: A unified framework for lifelong few-shot language learning based on prompt tuning of T5, ICLR 2022.

---

## Hwkz

The paper is not asking whether sharpness-aware training can help in general. It asks a narrower PECL-specific question: once the backbone is frozen and all future updates are confined to a low-rank adapter subspace, which sharpness notion is the theoretically relevant continual-learning criterion?

Standard practice alone does not determine the answer to that question. Full-space flatness and adapter-subspace flatness can recommend different perturbation scopes, and existing LoRA flatness prescriptions do not by themselves resolve which one is appropriate for frozen-backbone continual learning. Theorem 4.4 provides the missing criterion by deriving the relevant sharpness notion from the frozen-backbone reduction itself, rather than assuming the perturbation scope from standard definition.

The reported experiments support this criterion. In the reported settings, full-scope perturbations are detrimental, while within the admissible support we observe a stable ordering RS(0) < AS(0) < AS(1). The paper therefore does not stop at identifying the theoretically relevant PECL support for sharpness, but also shows that perturbation type induces systematic differences in subspace curvature control.

This also connects to your Phase 1 point about prior PAC-Bayesian CL work. Those papers are important background, but to our knowledge they do not derive the frozen-backbone adapter-subspace reduction that turns the bound into a perturbation-design criterion for LoRA-based PECL. Our contribution is therefore to derive the PECL-relevant sharpness criterion, not merely to confirm that adapter-only SAM works.

---

## AC

Dear AC,

We would like to briefly clarify the intended evaluation focus, not to add a new rebuttal.

The remaining disagreement appears to concern whether the paper is being read primarily as a theory-driven analysis of sharpness in frozen-backbone PECL, or instead as an empirical paper about whether adapter-only perturbations work well in practice.

The main contribution of the paper is to identify which notion of sharpness remains theoretically relevant once adaptation is restricted to the admissible low-rank subspace. This is the role of Theorem 4.4, which shows that both the KL complexity and the sharpness-smoothed empirical term reduce to quantities supported on the task-permissible subspace.

Thank you for your consideration.
