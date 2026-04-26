# Reproducibility Guide

This document walks through the exact command sequence used to reproduce the
LipMamba paper's headline numbers (Section 5).  It assumes a standard Linux
environment with Python 3.10 +, CUDA 12, and 4 × A100 80 GB GPUs for the
larger models; smaller variants can be reproduced on a single 24 GB GPU.

## 1. Environment

```bash
git clone https://github.com/rogerpanel/CV.git
cd CV/lipmamba
python -m venv .venv && source .venv/bin/activate
pip install -e ".[training]"
# Optional fast kernels — may not be available on all platforms
pip install -e ".[fast_kernels]" || true
```

Verify the math:

```bash
pytest -q tests
```

## 2. Data preparation

```bash
mkdir -p data_cache
python scripts/download_datasets.py --datasets wikitext103 robench25 cicids2017
# manually download CIC-IDS2017 if the script requests it, then:
python -c "from lipmamba.data import IDSDataset, IDSDatasetConfig; \
            IDSDataset(IDSDatasetConfig(name='cicids2017', csv_path='data_cache/cicids2017_clean.parquet'))"
```

For the language model, use any tokeniser to produce flat ``int32`` arrays:

```python
from transformers import AutoTokenizer
import numpy as np
tok = AutoTokenizer.from_pretrained("gpt2")
ids = np.asarray(tok.encode(open("wikitext103.txt").read()), dtype=np.int32)
ids.tofile("data_cache/wikitext103_train.bin")
```

## 3. Pre-training (LipMamba-130M)

```bash
python scripts/train.py --config configs/lipmamba_130m.yaml
```

Expected reference numbers (Section 5, Table 2):

| Metric | Mamba | LipMamba-130M |
| --- | --- | --- |
| WikiText-103 PPL | 18.7 | 19.6 |
| RoBench-25 ASR | 92 % | 4 % |
| Certified ε* | 0.04 | 0.18 |
| Clean accuracy | 89.7 % | 90.8 % |

## 4. Certifying robustness

```bash
python scripts/certify.py --config configs/certificate.yaml
```

Outputs ``runs/lipmamba_130m/certified.json`` containing the certified-radius
curve and the certified poisoning-immunity bound from Theorem 2.

## 5. Attacking with HiSPA / RoBench-25

```bash
python scripts/attack.py --config configs/attack_robench25.yaml
```

The reported attack success rate should drop below 5% for LipMamba whereas a
baseline Mamba checkpoint typically exceeds 90 %.

## 6. Network-intrusion fine-tuning

```bash
python scripts/train.py --config configs/ids_cic2017.yaml
```

Reproduces the IDS column of the paper (LipMamba achieves 95.9% F1 with
0.5 ms latency on a single A100 80 GB) — see
[`ROBUSTIDPS_INTEGRATION.md`](ROBUSTIDPS_INTEGRATION.md) for the full
deployment recipe.

## 7. Benchmark sweep

```bash
for cfg in configs/lipmamba_130m.yaml configs/lipmamba_370m.yaml; do
  python scripts/train.py    --config $cfg
  python scripts/certify.py  --config configs/certificate.yaml
  python scripts/attack.py   --config configs/attack_robench25.yaml
done
```

## 8. Reporting

All evaluation outputs are persisted as JSON under ``runs/.../*.json``.  Use
your favourite plotting tool (matplotlib / seaborn) to reproduce Figures 2
and 3 of the paper.
