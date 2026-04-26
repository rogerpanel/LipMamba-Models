# LipMamba — Methodology

This file expands on the algorithms used during training and evaluation,
written so that the reader can follow each step in the source code.

## Algorithm 1 — Lipschitz-Constrained Selective Forward Pass

```
Inputs : x_t ∈ R^{d_model},  hidden state h_{t-1} ∈ R^{N},
         spectral budgets (s_B, s_C, s_Δ, s_out),
         eigenvalue interval [λ_min, λ_max], Δ_max.
Output : y_t ∈ R^{d_model},  updated hidden state h_t.

1.  W̄_•  ← W_• · min(1, s_• / σ̂_max(W_•))    for • ∈ {B, C, Δ, out}      (spectral_norm.py)
2.  α    ← logits ;  λ ← λ_min + (λ_max - λ_min) · σ(α)                   (eigen_reparam.py)
3.  A    ← -diag(λ)
4.  Δ_t  ← Δ_max · tanh( softplus(W̄_Δ x_t + τ) / Δ_max )                  (clipped_delta.py)
5.  Ā_t  ← exp(Δ_t ⊙ A) ;  B̄_t  ← Δ_t ⊙ (W̄_B x_t)
6.  h_t  ← Ā_t ⊙ h_{t-1} + B̄_t · x_t                                      (selective_ssm.py)
7.  y_t  ← W̄_out · ( SiLU(C̄_t^T h_t) )
8.  Update online Lipschitz tracker  L_t ← ρ_t · L_{t-1} + s_C s_out L_SiLU (β_t + s_B)
```

## Algorithm 2 — PAC-Bayesian Adversarial Training

```
Inputs : data S = {(x_i, y_i)}_{i=1}^n, epochs E, ε_train, β, δ.
1.  Fit data-dependent prior θ_prior on a 5% clean held-out split (no
    adversarial perturbation, no Lipschitz penalty)                        (prior_fitting.py)
2.  Initialise posterior parameters θ from θ_prior, set σ_post (default 0.05).
3.  for epoch = 1, …, E:
       for batch (x, y) in DataLoader(S):
         (a) compute logits z(x) (forward pass)
         (b) margin-augmented loss
                z̃_K = max_{k≠ŷ} z_k(x) + √2 · L_net · ε_train             (glorot_head.py)
                L̂_S^{adv} = CE([z, z̃_K], y)
         (c) Lipschitz penalty   L_lip      = L_SSM(θ) · ε_train / 2
         (d) PAC-Bayes complexity L_complex = β · sqrt((KL(Q‖P) + ln(2√n/δ)) / 2n)
         (e) total loss          L = L̂_S^{adv} + L_lip + L_complex        (pac_objective.py)
         (f) backprop, AdamW step, gradient clipped to ‖·‖ ≤ 1.0
         (g) refresh σ̂ via single-step power iteration                     (spectral_norm.py)
4. return checkpoint.
```

## Hyperparameters Used in the Paper

The full table is in [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md); summary:

| Symbol | Value | Description |
| --- | --- | --- |
| `s_B`         | 1.0  | spectral budget on `W_B` |
| `s_C`         | 1.0  | spectral budget on `W_C` |
| `s_Δ`         | 0.5  | spectral budget on `W_Δ` |
| `s_out`       | 1.0  | spectral budget on output projection |
| `Δ_max`       | 0.5  | clipping ceiling for Δ_t |
| `λ_max`       | 1.0  | upper eigenvalue bound |
| `λ_min`       | 0.05 | lower eigenvalue bound |
| `ε_train`     | 0.18 | adversarial radius used in margin term |
| `δ`           | 0.05 | PAC-Bayes confidence |
| `σ_post`      | 0.05 | posterior std |
| `σ_prior`     | 0.10 | prior std |
| `β`           | 1.0  | weight on PAC-Bayes complexity term |
| Optimiser     | AdamW | lr = 2e-4, wd = 0.1 |
| LR schedule   | Cosine, 100 epochs | linear warm-up |
| Grad clip     | 1.0  | norm-clipping |
| Hardware      | 4 × A100 80GB | for 130M / 370M / 1.3B |
