# LipMamba — Theoretical Foundations

This document records the three central theorems of the LipMamba paper and the
quantities the code base computes to enforce / verify them. Notation follows
the manuscript (`lipmamba_injoit.tex`).

## 1. Lipschitz-Bounded Selective SSM Block (Theorem 1)

For a single LipMamba block with spectral budgets `(s_B, s_C, s_Δ, s_out)`,
clipped step-size `Δ_max`, and bounded eigenvalue interval
`λᵢ(A) ∈ [-λ_max, -λ_min]`, the block-level Lipschitz constant satisfies

```
L_block ≤ s_out · L_SiLU · [
    s_C · (s_B · Δ_max) / (1 - ρ_max)
  + s_C · ‖h‖_∞ · s_Δ · Δ_max / (1 - ρ_max)
]
```

with `ρ_max = exp(-Δ_min · λ_min) < 1` and `L_SiLU ≈ 1.0998`.
The *network* Lipschitz constant `L_net` is the product of per-block bounds.
The online tracker in `certificates/lipschitz.py` updates a running estimate

```
L_t = ρ_t · L_{t-1} + s_C · s_out · L_SiLU · (β_t + s_B)
```

where `ρ_t = ‖Ā_t‖₂` and `β_t = ‖B̄_t‖₂` are computed from the discretised
matrices each step.

## 2. Certified Hidden-State Poisoning Immunity (Theorem 2)

Let `τ = (τ₁,…,τ_ℓ)` be a trigger sequence. The poisoned hidden state obeys

```
‖h_{t₀+ℓ}‖₂ ≥ ρ_min^ℓ · ‖h_{t₀}‖₂
            − (B̄_max · X_max) · (1 − ρ_min^ℓ) / (1 − ρ_min)
```

so an `(α, ℓ)`-poisoning attack with α ≪ 1 is impossible whenever the lower
bound exceeds α. The maximum trigger length tolerated is

```
ℓ* ≤ log( α_min + B̄_max·X_max / ((1 − ρ_min) · ‖h_{t₀}‖) ) / log(ρ_min).
```

`certificates/poisoning_immunity.py` evaluates this for any trained model and
returns the per-position certified trigger budget.

## 3. PAC-Bayesian Adversarial Bound (Theorem 3)

Posterior `Q = N(θ, σ²I)` over the constrained parameters, prior
`P = N(θ_prior, σ₀²I)` fitted on a clean held-out split:

```
E_{θ~Q}[L_adv(θ; ε)] ≤ E_{θ~Q}[L̂_S^{adv}(θ; ε)]
                     + L_SSM(θ) · ε / 2
                     + sqrt( ( KL(Q‖P) + ln(2√n / δ) ) / (2n) ).
```

Training jointly minimises empirical adversarial risk, the Lipschitz term
(implicitly through margin-augmented training), and the Gaussian KL.

## Margin-Augmented Logit (Eq. 16)

For multiclass classification with predicted class `ŷ` and logits `z(x)`,
training uses

```
z̃_K = max_{k ≠ ŷ} z_k(x) + sqrt(2) · L_net · ε_train
```

and the corresponding cross-entropy. At inference, the per-input certified
radius (GloroNet-style) is

```
ε*(x) = ( z_{ŷ} − max_{k ≠ ŷ} z_k ) / ( sqrt(2) · L_net ).
```

Implementation: `models/glorot_head.py`, `certificates/certified_radius.py`.

## Spectral Normalisation (Power Iteration)

Each constrained projection is rescaled every step:

```
W̄_•  ← W_• · min(1, s_• / σ̂_max(W_•)),    • ∈ {B, C, Δ, out}.
```

`σ̂_max` is the one-step power-iteration estimate with running-average
smoothing — see `models/spectral_norm.py`.

## Eigenvalue Reparameterisation

```
A = -diag( λ_min + (λ_max − λ_min) · σ(α) ).
```

Sigmoid keeps every diagonal entry strictly inside `(λ_min, λ_max)`. See
`models/eigen_reparam.py`.

## Clipped Discretisation

```
Δ_t = Δ_max · tanh( softplus(W̄_Δ x_t + τ) / Δ_max ).
```

`softplus` keeps `Δ_t > 0`; `tanh` smooth-saturates to `Δ_max`. Implementation:
`models/clipped_delta.py`.
