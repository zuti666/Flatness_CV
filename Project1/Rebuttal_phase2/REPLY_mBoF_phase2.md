# Phase 2 Reply — Reviewer mBoF

## mBoF

> I still believe that scalability remains an issue, as the results only cover a limited set of architectures (ViT → T5) and limited scale variation.

This is not yet a full scaling-law study, but the Phase 2 additions materially broaden the evidence beyond the original ViT and T5 settings.

To broaden the evidence, we added decoder-only LLM experiments on Llama-3.2-1B and Llama-3.2-3B. We train four text classification datasets (DBpedia, Amazon Reviews, Yahoo Answers, AG News) sequentially under three task orders and compare Adam against adapter-scoped SAM for SeqLoRA, IncLoRA, and OLoRA. In each of the six model/method combinations, SAM improves the average accuracy:

| Model | Method | Adam avg | SAM avg | Gain |
| - | - | -: | -: | -: |
| Llama-3.2-1B | SeqLoRA | 63.6 | 68.5 | +4.9 |
| Llama-3.2-1B | IncLoRA | 62.1 | 67.7 | +5.6 |
| Llama-3.2-1B | OLoRA | 56.7 | 63.6 | +6.9 |
| Llama-3.2-3B | SeqLoRA | 67.5 | 73.2 | +5.7 |
| Llama-3.2-3B | IncLoRA | 67.2 | 72.8 | +5.6 |
| Llama-3.2-3B | OLoRA | 63.4 | 68.8 | +5.4 |

These additions do not remove the limitation entirely, but they substantially broaden the coverage: the same qualitative pattern now appears on an encoder vision backbone (ViT), an encoder-decoder LM family (T5), and decoder-only LLMs at 1B and 3B scales.

> I still think the insights provided in this paper also appear somewhat incremental.

The contribution is not simply to show that SAM can help LoRA empirically, but to establish that, in frozen-backbone PECL, the theoretically relevant sharpness and KL terms are supported on the admissible adapter subspace $W_{\Delta,t}$ (Theorem 4.4). This identifies the sharpness object that matches the PECL update mechanism, which is different from generic flatness arguments for full-model training or standard LoRA fine-tuning. The added Llama results support the qualitative consequence of this reduction across broader model families.
