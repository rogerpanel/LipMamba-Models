# Model Card — LipMamba

## Overview

LipMamba is a Lipschitz-constrained selective state-space language model
trained with a PAC-Bayesian adversarial objective.  Its distinguishing
feature is a *certified* defence against hidden-state-poisoning triggers
(Section 5 of the paper).

* **Architectures**: 130M, 370M, 1.3B parameters.
* **Pre-training corpora**: SlimPajama-627B (1.3B), The Pile (370M),
  WikiText-103 (130M).
* **Adversarial fine-tuning**: RoBench-25 (HiSPA family), AdvBench,
  HarmBench, JailbreakBench.
* **License**: MIT — see [`LICENSE`](../LICENSE).

## Intended Use

1. Research on certified robustness for selective state-space models.
2. As a defensive component in `robustidps.ai`-class IDPS deployments.
3. Pedagogical reference implementation for the LipMamba paper.

## Out-of-scope Use

LipMamba is not a finished safety-aligned LLM — it is a robustness
*primitive*.  It must be combined with the rest of the robustidps.ai stack
(content filters, RAG-poisoning defence, RLHF) for production use.

## Evaluation Summary

| Variant       | WikiText-103 PPL | HarmBench ASR | RoBench-25 ASR | Certified ε\* |
| ------------- | ---------------- | ------------- | -------------- | ------------- |
| LipMamba-130M | 19.6             |  6.1 %        |  4.0 %         | 0.18          |
| LipMamba-370M | 17.1             |  5.4 %        |  3.7 %         | 0.18          |
| LipMamba-1.3B | 14.2             |  4.6 %        |  3.0 %         | 0.18          |
| Mamba-130M (baseline) | 18.7    | 87.3 %        | 92.0 %         | 0.04          |

Numbers are reported in the manuscript and reproduced by
`scripts/evaluate.py` + `scripts/certify.py` + `scripts/attack.py`.

## Limitations

* The published Lipschitz bound is a worst-case estimate; the empirical
  Lipschitz constant is typically 30–60 % smaller.
* The certified radius ε\* applies to ℓ₂ embedding-space perturbations
  (GloroNet convention).  For discrete token-substitution attacks, see the
  greedy-search robustness numbers in Appendix C.3 of the paper.
* Pre-training data inherits all biases of SlimPajama / The Pile; users
  should run dedicated bias evaluations before downstream deployment.

## Ethical Considerations

The HiSPA attack toolkit is included to enable defensive research and
red-teaming of *your own* systems.  Do not use it against systems you do
not have explicit permission to test.

## Citing

```bibtex
@article{anaedevha2026lipmamba,
  author  = {Anaedevha, Roger Nick},
  title   = {LipMamba: Lipschitz-Constrained Selective State-Space Models with
             PAC-Bayesian Certificates for Certified Robustness Against Hidden
             State Poisoning in Language Models},
  year    = {2026},
  url     = {https://github.com/rogerpanel/CV/tree/main/lipmamba}
}
```
