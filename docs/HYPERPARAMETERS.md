# Hyperparameters Reference

Every hyperparameter used by LipMamba, mapped to the location in the codebase
where it is consumed.

## Architecture

| Symbol     | Default | Where it lives in the code |
| ---------- | ------- | -------------------------- |
| `vocab_size` | 50257 | `LipMambaConfig.vocab_size` (`models/lipmamba_model.py`) |
| `n_layers`  | 24    | per-variant override; see `configs/*.yaml` |
| `d_model`   | 768   | as above |
| `d_inner`   | 1536  | as above |
| `state_dim` (N) | 16 | `SSMConfig.state_dim` (`models/selective_ssm.py`) |
| `conv_kernel` | 4   | `LipMambaBlockConfig.conv_kernel` (`models/lipmamba_block.py`) |
| `s_B` | 1.0 | `SSMConfig.s_b` |
| `s_C` | 1.0 | `SSMConfig.s_c` |
| `s_Δ` | 0.5 | `SSMConfig.s_delta` |
| `s_out` | 1.0 | `SSMConfig.s_out` |
| `Δ_max` | 0.5 | `SSMConfig.delta_max` |
| `λ_min` | 0.05 | `EigenReparamA.lambda_min` |
| `λ_max` | 1.0 | `EigenReparamA.lambda_max` |
| `n_power_iters` | 1 | `SpectralNormLinear.n_power_iters` |
| `track_lipschitz` | True | `SSMConfig.track_lipschitz` |

### Variant table

| Variant | Layers | d_model | d_inner | L_SSM cap |
| ---     | ---    | ---     | ---     | --- |
| LipMamba-130M | 24 | 768 | 1536 | 5.0 |
| LipMamba-370M | 48 | 1024 | 2048 | 8.0 |
| LipMamba-1.3B | 64 | 2048 | 4096 | 12.0 |

## PAC-Bayes / certificates

| Symbol | Default | Where it lives |
| --- | --- | --- |
| `ε_train` | 0.18 | `TrainerConfig.epsilon_train`, `GloroNetHead.epsilon_train` |
| `δ` | 0.05 | `PACBayesConfig.delta` |
| `σ_post` | 0.05 | `PACBayesConfig.sigma_post` |
| `σ_prior` | 0.10 | `PACBayesConfig.sigma_prior` |
| `β` | 1.0 | `PACBayesConfig.beta` |

## Optimiser / schedule

| Symbol | Default | Where it lives |
| --- | --- | --- |
| Optimiser | AdamW | `training/optim.build_optimizer` |
| Learning rate | 2e-4 | `TrainerConfig.lr` |
| Weight decay | 0.1 | `TrainerConfig.weight_decay` |
| Betas | (0.9, 0.95) | `optim.build_optimizer` |
| LR scheduler | Cosine + linear warmup | `optim.CosineWithWarmup` |
| Warmup steps | 1 000 | `TrainerConfig.warmup_steps` |
| Max steps | 100 000 (130M) | `TrainerConfig.max_steps` |
| Min LR ratio | 0.1 | `optim.CosineWithWarmup.min_lr_ratio` |
| Gradient clip | 1.0 | `TrainerConfig.grad_clip` |

## Adversarial / attack defaults

| Field | Default | Where it lives |
| --- | --- | --- |
| `epsilon_train` | 0.18 | `TrainerConfig`, `PACBayesConfig` |
| PGD steps | 20 (sweep ∈ {10, 20, 40}) | `attacks/pgd.PGDConfig.n_steps` |
| PGD step size | ε / 4 | `PGDConfig.step_size` |
| HiSPA trigger length | 16 | `HiSPAConfig.trigger_length` |
| HiSPA n_steps | 200 | `HiSPAConfig.n_steps` |
| HiSPA target α | 0.05 | `HiSPAConfig.target_alpha` |

## Operational (robustidps.ai)

| Symbol | Default | Description |
| --- | --- | --- |
| MC dropout T | 20 (1 in fast mode) | `unified_model.py` |
| EWC β | 0.7 | Fisher information sharing |
| RAG poisoning τ | 0.82 | embedding similarity threshold |
| Detection throughput target | 12 000 flows/sec | per-CPU goal |
| Per-flow latency | 0.5–8.7 ms | SLA spec |

The full operational hyperparameter set is documented in
[`ROBUSTIDPS_INTEGRATION.md`](ROBUSTIDPS_INTEGRATION.md).
