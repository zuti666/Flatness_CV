# Phase 2 Reply — Reviewer Hwkz

## Hwkz

> "Partially resolved. I might concur with Reviewer TSrA that the paper's motivation requires better justification."

We clarify the motivation directly.

The paper is not asking whether sharpness-aware training can help in general. It asks a narrower PECL-specific question: once the backbone is frozen and all future updates are confined to a low-rank adapter subspace, which sharpness notion is the right one to use as a continual-learning criterion? That distinction matters because full-space flatness and adapter-subspace flatness can recommend different perturbation scopes, and existing LoRA flatness prescriptions do not by themselves resolve which one is appropriate for frozen-backbone continual learning.

Theorem 4.4 is our answer to that question. Under frozen-backbone PECL, both the KL complexity and the sharpness-smoothed empirical term reduce to quantities supported on $W_{\Delta,t}$; via Lemma 4.2, the frozen directions contribute only degenerate factors and drop out of the reduction. This is why the paper focuses on adapter-subspace sharpness rather than full-model sharpness.

This also connects to your Phase 1 point about prior PAC-Bayesian CL work. Those papers are important background for sequential generalization, but to our knowledge they do not derive the frozen-backbone adapter-subspace reduction that turns the bound into a perturbation-design criterion for LoRA-based PECL. The reported experiments, together with the added Llama-3.2-1B/3B results, support the practical consequence of this distinction: in the settings we study, adapter-scoped perturbation is preferable to full-scope perturbation.
