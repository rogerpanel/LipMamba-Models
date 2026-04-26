# LipMamba

**Lipschitz-Constrained Selective State-Space Models with PAC-Bayesian Certificates for Certified Robustness Against Hidden-State Poisoning in Language Models**

Reference implementation accompanying the manuscript:

> Roger Nick Anaedevha. *LipMamba: Lipschitz-Constrained Selective State-Space Models with PAC-Bayesian Certificates for Certified Robustness Against Hidden State Poisoning in Language Models.* 2026.

Source manuscripts: <https://github.com/rogerpanel/LipMamba-Models>
Reproducibility mirror: <https://github.com/rogerpanel/CV/tree/main/lipmamba>

---

## Overview

LipMamba is the first certified architectural defense against **hidden-state
poisoning** of selective state-space language models (Mamba-style). It combines
three ideas:

1. **Spectral parameterization** of all selective projections (`B`, `C`, `Δ`)
   via single-step power-iteration spectral normalization.
2. **Eigenvalue reparameterization** of the state matrix `A` so that
   `λᵢ(A) ∈ [-λ_max, -λ_min]` is enforced by construction.
3. **Clipped discretization** `Δ_t = Δ_max · tanh(softplus(W_Δx_t + τ)/Δ_max)`
   so that the discrete recurrence radius `ρ_max = exp(-Δ_min λ_min) < 1`.

These ingredients yield a closed-form per-layer Lipschitz bound (Theorem 1),
an exponential lower bound on post-trigger hidden-state norms (Theorem 2,
"certified poisoning immunity"), and a PAC-Bayesian generalization /
adversarial bound (Theorem 3). A GloroNet-style certification head produces a
per-input certified radius `ε*(x)`.

The codebase reproduces:

* LipMamba 130M / 370M / 1.3B configurations.
* Training with the PAC-Bayes adversarial objective.
* HiSPA hidden-state poisoning attack + RoBench-25/26, HarmBench, JailbreakBench evaluation.
* Network-intrusion application aligned with the `robustidps.ai` deployment
  (CIC-IDS2017, Edge-IIoTset, UNSW-NB15, TON_IoT, NSL-KDD, CIC-IoT-2023,
  CIC-DDoS-2019, post-quantum traffic).

## Repository Layout

```
lipmamba/
├── src/lipmamba/
│   ├── models/         Selective SSM block, spectral norm, eigenvalue reparam, GloroNet head
│   ├── certificates/   Lipschitz tracking, PAC-Bayes bound, certified radius, poisoning immunity
│   ├── attacks/        HiSPA poisoning, PGD, jailbreak, discrete trigger search
│   ├── data/           Dataset loaders + registry (LM, safety, IDS)
│   ├── training/       Trainer, AdamW + cosine, PAC-Bayes objective, prior fitting
│   ├── evaluation/     Clean/Certified/PACC accuracy, perplexity, benchmark runner
│   └── utils/          Logging, seeding, checkpoints
├── configs/            YAML configs (130M, 370M, 1.3B, IDS, certificate, attack)
├── scripts/            CLI entry points (train, evaluate, certify, attack, pretrain, finetune, download_datasets)
├── tests/              Unit tests for math invariants
├── examples/           Minimal end-to-end demos
└── docs/               THEORY, METHODOLOGY, DATASETS, HYPERPARAMETERS, REPRODUCIBILITY, ROBUSTIDPS_INTEGRATION
```

## Installation

```bash
git clone https://github.com/rogerpanel/CV.git
cd CV/lipmamba
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

GPU training requires PyTorch ≥ 2.1 with CUDA 12. Optional kernels: install
`mamba-ssm` and `causal-conv1d` for parity-fast scans (the included reference
scan is pure PyTorch so the codebase runs on CPU as well for testing).

## Quick Start

```bash
# 1. Sanity-check the math
pytest -q tests

# 2. Train a small LipMamba on WikiText-103 with the PAC-Bayes objective
python scripts/train.py --config configs/lipmamba_130m.yaml

# 3. Compute certified radii on a held-out split
python scripts/certify.py --config configs/certificate.yaml \
    --checkpoint runs/lipmamba_130m/best.pt

# 4. Attack with HiSPA / RoBench-25
python scripts/attack.py --config configs/attack_robench25.yaml \
    --checkpoint runs/lipmamba_130m/best.pt

# 5. End-to-end intrusion-detection demo (robustidps.ai integration)
python scripts/train.py --config configs/ids_cic2017.yaml
```

## Datasets

All datasets used in the paper are listed in
[`docs/DATASETS.md`](docs/DATASETS.md) with download links and citation
information. The downloader script

```bash
python scripts/download_datasets.py --datasets wikitext103 robench25 cicids2017
```

automates retrieval where licensing allows (most are gated behind a click-wrap
agreement and must be downloaded manually; the script prints the canonical
URL when automatic download is not possible).

| Domain | Dataset | URL |
| --- | --- | --- |
| Pre-training | The Pile | <https://pile.eleuther.ai/> |
| Pre-training | SlimPajama-627B | <https://huggingface.co/datasets/cerebras/SlimPajama-627B> |
| Pre-training | C4 | <https://huggingface.co/datasets/allenai/c4> |
| LM eval | WikiText-103 | <https://huggingface.co/datasets/wikitext> |
| Safety | HarmBench | <https://www.harmbench.org/> |
| Safety | JailbreakBench | <https://github.com/JailbreakBench/jailbreakbench> |
| Safety | AdvBench | <https://github.com/llm-attacks/llm-attacks> |
| Safety | WildJailbreak | <https://huggingface.co/datasets/allenai/wildjailbreak> |
| SSM-poisoning | RoBench-25 / RoBench-26 | <https://github.com/HiSPA-robench> |
| IDS | CIC-IDS2017 | <https://www.unb.ca/cic/datasets/ids-2017.html> |
| IDS | Edge-IIoTset | <https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot> |
| IDS | UNSW-NB15 | <https://research.unsw.edu.au/projects/unsw-nb15-dataset> |
| IDS | TON_IoT | <https://research.unsw.edu.au/projects/toniot-datasets> |
| IDS | NSL-KDD | <https://www.unb.ca/cic/datasets/nsl.html> |
| IDS | CIC-IoT-2023 | <https://www.unb.ca/cic/datasets/iotdataset-2023.html> |
| IDS | CIC-DDoS-2019 | <https://www.unb.ca/cic/datasets/ddos-2019.html> |
| IDS | PQC traffic | <https://doi.org/10.34740/kaggle/dsv/15424420> |

## Reproducibility Checklist

* Hyperparameters: [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md).
* Theorems and proof sketches: [`docs/THEORY.md`](docs/THEORY.md).
* Algorithms (Algorithm 1: forward; Algorithm 2: training):
  [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).
* Step-by-step run instructions:
  [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
* Integration with `robustidps.ai` operational deployment:
  [`docs/ROBUSTIDPS_INTEGRATION.md`](docs/ROBUSTIDPS_INTEGRATION.md).

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

## License

MIT — see [LICENSE](LICENSE).
