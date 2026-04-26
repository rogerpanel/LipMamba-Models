# LipMamba ↔ robustidps.ai Integration

LipMamba is the **certified-defense** primitive of the robustidps.ai
operational platform.  This document describes how the model is exposed to
the existing 13-model ensemble that ships with `robustidps_web_app/` and
`integrated_ai_ids/` in this repository.

## 1. Where LipMamba sits in the platform

```
┌─────────────────────────────────────────────────────────────┐
│ React 18 SPA  ──▶  Flask / FastAPI ──▶  Inference Workers    │
│                                          │                   │
│                                          ├─ MambaShield      │
│                                          ├─ LipMamba  ◄ new  │
│                                          ├─ CL-RL Unified    │
│                                          ├─ CT-TGNN          │
│                                          ├─ SDE-TGNN         │
│                                          ├─ Stochastic Tx    │
│                                          ├─ FedGTD           │
│                                          ├─ CyberSecLLM      │
│                                          ├─ Multi-Agent PQC  │
│                                          ├─ Neural ODE       │
│                                          ├─ Optimal Transport│
│                                          ├─ VAE / AAE        │
│                                          └─ Surrogate-IDS    │
└─────────────────────────────────────────────────────────────┘
```

LipMamba contributes:

* **Certified detection radius ε\*** for every flow classification, exposed
  via the existing `/api/v1/ids/predict` REST endpoint as the new
  ``certified_epsilon`` field.
* **Poisoning-immunity certificate ℓ\*** (Theorem 2) reported via
  ``/api/v1/governance/certificates`` so the SOC dashboard can flag any
  trigger sequence longer than the certified budget.
* **Hidden-state-poisoning replay** through the `attack.py` script, used by
  the *AI Active Defence* pages to exercise the system against synthetic
  HiSPA triggers nightly.

## 2. Model-zoo registration

Add LipMamba to the existing `integrated_ai_ids/models/__init__.py`
manifest by importing the convenience builder:

```python
from lipmamba import LipMambaConfig, LipMambaModel

def build_lipmamba_for_ids(num_classes: int = 15) -> LipMambaModel:
    cfg = LipMambaConfig.lipmamba_130m(
        vocab_size=32, n_layers=12, d_model=256, d_inner=512,
        n_classes=num_classes, epsilon_train=0.18,
    )
    return LipMambaModel(cfg)
```

## 3. Deployment artefacts

| File | Purpose |
| --- | --- |
| `lipmamba/configs/ids_cic2017.yaml` | Fine-tuning recipe for the production IDS classifier |
| `lipmamba/scripts/train.py`        | CLI used by `cron` / Airflow for nightly retraining |
| `lipmamba/scripts/certify.py`      | Generates `certified.json`, served by `governance/certificates` endpoint |
| `lipmamba/src/lipmamba/certificates/poisoning_immunity.py` | Real-time ℓ\* computation called by the websocket monitor |

## 4. Operational SLA

* **Throughput:** > 12 000 flows / s on a single CPU in fast mode.
* **Latency:** 0.5–8.7 ms per flow (depending on model selection).
* **Certified-radius reporting overhead:** 4.7 % over baseline (paper Table 4).
* **Poisoning-immunity check:** O(1) — closed-form formula evaluated once per
  model deployment.

## 5. Runtime configuration variables

```bash
LIPMAMBA_CHECKPOINT=/srv/models/lipmamba_ids_cic2017.pt
LIPMAMBA_DEVICE=cuda:0
LIPMAMBA_FAST_MODE=true               # turns off MC dropout (T=1)
LIPMAMBA_CERT_RADIUS_THRESHOLD=0.18
LIPMAMBA_POISONING_ALPHA=0.05
```

## 6. Cross-references

* `robustidps_web_app/SYSTEM_OVERVIEW.md` — full platform overview.
* `integrated_ai_ids/configs/model_config.yaml` — historical baseline
  configuration; LipMamba block keys can be appended.
