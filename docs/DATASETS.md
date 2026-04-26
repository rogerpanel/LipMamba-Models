# Datasets

All datasets referenced in the LipMamba paper and the robustidps.ai
deployment, with canonical download URLs and licence notes.  These are also
exposed programmatically in :mod:`lipmamba.data.registry`.

## Pre-training / language modelling

| Dataset | Tokens | License | Link |
| --- | --- | --- | --- |
| The Pile | 825 GiB | various per subset | <https://pile.eleuther.ai/> |
| SlimPajama-627B | 627 B | various | <https://huggingface.co/datasets/cerebras/SlimPajama-627B> |
| C4 (en) | ~365 B | ODC-BY | <https://huggingface.co/datasets/allenai/c4> |
| WikiText-103 | 103 M | CC BY-SA 3.0 | <https://huggingface.co/datasets/wikitext> |

## Safety / hidden-state poisoning

| Dataset | Size | License | Link |
| --- | --- | --- | --- |
| HarmBench | 510 behaviours × 18 methods | research only | <https://www.harmbench.org/> |
| JailbreakBench | 100 misuse + 100 benign | MIT | <https://github.com/JailbreakBench/jailbreakbench> |
| AdvBench | 520 prompts | MIT | <https://github.com/llm-attacks/llm-attacks> |
| WildJailbreak | 2 210 prompts | ODC-BY | <https://huggingface.co/datasets/allenai/wildjailbreak> |
| RoBench-25 | 1 050 triggers / 9 families | research only | <https://github.com/HiSPA-robench> |

## Network intrusion detection (robustidps.ai)

| Dataset | Size | License | Link |
| --- | --- | --- | --- |
| CIC-IDS2017 | 2.8 M flows | research only | <https://www.unb.ca/cic/datasets/ids-2017.html> |
| CIC-IoT-2023 | 1.2 M flows | research only | <https://www.unb.ca/cic/datasets/iotdataset-2023.html> |
| CIC-DDoS-2019 | 80 M+ flows | research only | <https://www.unb.ca/cic/datasets/ddos-2019.html> |
| Edge-IIoTset | 20.8 M flows | CC BY-SA 4.0 | <https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot> |
| UNSW-NB15 | 2.5 M flows | research only | <https://research.unsw.edu.au/projects/unsw-nb15-dataset> |
| TON_IoT | telemetry + flows | research only | <https://research.unsw.edu.au/projects/toniot-datasets> |
| NSL-KDD | 148 K records | open | <https://www.unb.ca/cic/datasets/nsl.html> |
| PQC traffic | post-quantum captures | CC0 | <https://doi.org/10.34740/kaggle/dsv/15424420> |

## Downloading

```bash
python scripts/download_datasets.py --datasets wikitext103 robench25 cicids2017
```

For datasets gated by a click-wrap agreement (CIC family, UNSW family, PQC
DOI) the script prints the canonical URL and stops; download manually and
place the files inside ``data_cache/<dataset_name>/``.

## License Notice

All datasets retain their original licences.  The LipMamba codebase **does
not** redistribute any dataset content; users must obtain each dataset from
the linked source and respect its licence terms.
