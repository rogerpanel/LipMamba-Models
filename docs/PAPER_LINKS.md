# Paper ↔ Code Cross-Reference

Source manuscripts (LaTeX + PDF) live in
<https://github.com/rogerpanel/LipMamba-Models>.  This file maps every
section of those manuscripts to the implementation in this repository so
reviewers can follow the code while reading the paper.

| Manuscript section | Equation / Algorithm | Code location |
| --- | --- | --- |
| §3.1 Spectral parameterisation | Eq. 3 | `src/lipmamba/models/spectral_norm.py` |
| §3.2 Eigenvalue reparameterisation | Eq. 4 | `src/lipmamba/models/eigen_reparam.py` |
| §3.3 Clipped discretisation | Eq. 5 | `src/lipmamba/models/clipped_delta.py` |
| §3.4 Selective scan | Eq. 6 | `src/lipmamba/models/selective_ssm.py` |
| §4.1 Theorem 1 (Lipschitz bound) | Thm. 1 | `src/lipmamba/certificates/lipschitz.py` |
| §4.2 Theorem 2 (Poisoning immunity) | Thm. 2 | `src/lipmamba/certificates/poisoning_immunity.py` |
| §4.3 Theorem 3 (PAC-Bayes) | Thm. 3 | `src/lipmamba/certificates/pac_bayes.py` |
| §4.4 GloroNet certified radius | Eq. 15 | `src/lipmamba/models/glorot_head.py`, `certificates/certified_radius.py` |
| §4.5 Margin-augmented logit | Eq. 16 | `models/glorot_head.py::margin_augmented` |
| Algorithm 1 — Forward pass | — | `src/lipmamba/models/lipmamba_block.py` |
| Algorithm 2 — Adversarial PAC-Bayes training | — | `src/lipmamba/training/trainer.py` |
| §5.1 RoBench-25 | — | `src/lipmamba/data/robench.py`, `scripts/run_robench.py` |
| §5.2 HarmBench / JailbreakBench | — | `src/lipmamba/data/safety.py`, `attacks/jailbreak.py` |
| §5.3 IDS evaluation | — | `src/lipmamba/data/ids.py`, `configs/ids_cic2017.yaml` |
| §5.4 Ablation: Δ_max sweep | Table 3 | `configs/lipmamba_*.yaml` (override `delta_max`) |
| Appendix B.1 SiLU constant | — | `certificates/lipschitz.py::L_SILU` |
| Appendix B.2 HiPPO init | — | `models/hippo.py` |
