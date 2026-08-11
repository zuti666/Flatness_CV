# Reviews — Verbatim

## Reviewer TSrA
**Date**: 12 Mar 2026 (modified: 24 Mar 2026)
**Overall**: 2: Reject
**Confidence**: 4
**Soundness**: 2: fair | **Presentation**: 2: fair | **Significance**: 1: poor | **Originality**: 2: fair

### Summary
This paper studies how task-subspace sharpness and full-parameter sharpness affect generalization in PECL, and investigates the use of SAM to improve PECL performance. The authors define subspace sharpness and derive a sequential hierarchical PAC-Bayes bound for this setting. Experiments are conducted to demonstrate the effectiveness of optimizing sharpness within the task subspace.

### Strengths
- In-depth analysis of subspace sharpness with several insights.
- Derives a PAC-Bayes bound tailored to the PECL setting.
- Experiments supporting subspace SAM effectiveness.

### Weaknesses
1. I am not convinced that practitioners would use full-parameter perturbation when applying SAM to LoRA. To my knowledge, this corresponds to a specific and somewhat unnatural implementation, since frozen parameters typically do not produce gradients. Therefore, the practical importance of the problem studied in this paper appears limited, which is the main weakness of the work.
2. The PAC-Bayes bound largely builds on prior work, and I did not see major technical challenges in the derivation.
3. The paper does not introduce a new method, but mainly validates the effectiveness of combining LoRA with different variants of SAM.
4. Experiments are conducted only in a single setting (ViT on ImageNet), and the scope and scale of evaluation are relatively limited.

### Key Questions
1. Does this paper use novel proof techniques? What are the main technical challenges in the theoretical derivations?
2. Could the authors provide experimental results at a larger scale and under more diverse settings?

---

## Reviewer ZRjU
**Date**: 12 Mar 2026 (modified: 24 Mar 2026)
**Overall**: 5: Accept
**Confidence**: 4
**Soundness**: 3: good | **Presentation**: 4: excellent | **Significance**: 3: good | **Originality**: 3: good

### Summary
This paper studies sharpness-aware optimization in PECL with LoRA adapters. The authors derive a sequential hierarchical PAC-Bayesian bound showing both sharpness penalty and KL complexity reduce to quantities defined on the task-permissible LoRA subspace. They argue perturbations should be restricted to this subspace, and that AS(1) most effectively suppresses dominant curvature directions. Experiments on ImageNet-R, ImageNet-C, and ImageNet-P support claims across three LoRA organizational strategies.

### Strengths
- Theoretical framework is well-motivated. Lemma 4.2's translation-invariance argument is sound, KL decomposition onto subspace is clean.
- Clearly written, logical progression from theory to experiments, appropriately referenced.
- Scope mismatch finding (perturbing frozen backbone hurts rather than helps) is a practically important and non-obvious result.
- Consistent advantage of AS(1) across all settings is a strong empirical finding.
- Extension of hierarchical PAC-Bayes to sequentially evolving hyperposterior with explicit drift penalty seems like a genuine contribution.

### Weaknesses / Key Questions
1. **Theorem 4.1 → Theorem 4.4 gap**: The classical bound in Theorem 4.1 contains a logm·logm term in the numerator inside the square root, which is absent in Theorem 4.4. Where does this term go in the sequential proof? Please provide a detailed accounting of this step.
2. **Lemma 4.2 support assumption**: The lemma assumes Qt△ ≪ Pt△ and implicitly requires the prior to be supported on the same subspace W△,t as the posterior. For IncLoRA and OLoRA, where subspaces grow or are constrained across tasks, is this assumption actually satisfied? If not, how does this affect the KL decomposition?
3. **Bound tightness**: PAC-Bayes bounds are often too loose to be numerically meaningful. Are the authors able to evaluate how tight Theorem 4.4 is in any of their experimental settings, or is the bound intended primarily as a conceptual motivation for the empirical investigation?
4. **Figure 4**: Arrives too late and is too technical to serve as an intuitive illustration of geometric differences between RS(0), AS(0), and AS(1).

---

## Reviewer mBoF
**Date**: 06 Mar 2026 (modified: 24 Mar 2026)
**Overall**: 3: Weak Reject
**Confidence**: 2
**Soundness**: 2: fair | **Presentation**: 3: good | **Significance**: 2: fair | **Originality**: 2: fair

### Summary
This paper investigates the relationship between subspace sharpness and generalization in LoRA-based PECL. The authors explore whether SAM benefits transfer to the low-rank, task-specific subspaces used by LoRA. Curvature diagnostics show lower subspace sharpness is related to higher continual accuracy and less forgetting.

### Strengths
- Provides theoretical justification for sharpness analysis in the PECL domain.
- Clearly written.

### Weaknesses
- **Limited generalization**: Only ViT-B/16 pretrained on ImageNet-21K; continual learning datasets limited to ImageNet-C/R/P. Single model type + single domain — insufficient to support claimed generalization.
- **Limited scalability**: Does this analysis hold regardless of model complexity or model size?
- **Incremental**: Observations appear somewhat incremental (just very similar to full fine-tuning).

### Key Questions
1. FlatLoRA [1] highlights that LoRA fine-tuning can lead to sharper loss landscapes in the full parameter space and proposes encouraging flatter minima during LoRA fine-tuning. Do the authors think their approach would influence the generalization bound derived in this work?
2. Could the authors provide additional empirical data covering various model architectures and scales?
3. Can the authors provide more empirical analysis using different datasets, including cross-dataset scenarios?
4. Is this approach affected by the domain gap between the pre-training dataset and the target dataset used for continual learning?

References: [1] Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape (ICML 2025)

---

## Reviewer Hwkz
**Date**: 01 Mar 2026 (modified: 24 Mar 2026)
**Overall**: 3: Weak Reject
**Confidence**: 4
**Soundness**: 2: fair | **Presentation**: 3: good | **Significance**: 2: fair | **Originality**: 2: fair

### Summary
This paper explores the interaction between sharpness-aware optimization and low-rank subspace constraints (LoRA) in PECL. Traditional sharpness-aware methods perturb the entire parameter space, creating a disconnect with PECL's dynamics. The paper introduces "Subspace Sharpness" and derives a novel sequential hierarchical PAC-Bayesian generalization bound. Theoretical analysis demonstrates that controlling sharpness within the subspace suffices for generalization guarantees when the backbone is frozen. Experiments validate that AS(1) in task-permissible subspaces significantly enhances continual learning accuracy and mitigates forgetting across multiple LoRA variants (SeqLoRA, IncLoRA, OLORA).

### Strengths
1. Sequential hierarchical PAC-Bayesian analysis framework providing theoretical guarantees.
2. Extensive experiments across multiple benchmarks demonstrating effectiveness.

### Weaknesses
1. **Motivation not convincing**: Applying AS(1) perturbation only to the LoRA adapter appears to be the standard way of using SAM, and this practice does not seem to introduce the claimed perturbation misalignment issue. Moreover, the empirical results presented in the paper suggest that this default strategy is already the most effective.
2. **Missing related work**: The paper should discuss prior work that applies PAC-Bayesian analysis to continual learning, such as [1] and [2].
3. **Additional PEFT approaches**: Additional PEFT approaches should be included to better validate the effectiveness of the proposed subspace-restricted updates.
4. **Undefined notation**: Some notations are used before being defined. For example, H in Line 101 (Page 2) is not formally introduced.

References:
[1] A PAC-Bayesian bound for Lifelong Learning. ICML 2014.
[2] PAC-Bayes bounds for cumulative loss in Continual Learning. ICLR 2026.
