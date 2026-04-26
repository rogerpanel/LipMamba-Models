"""PAC-Bayesian generalisation / adversarial bound (Theorem 3).

Given posterior ``Q = N(θ, σ²I)`` and prior ``P = N(θ₀, σ₀² I)`` over the
constrained parameters, with ``n`` training samples and confidence ``δ``:

    E_{θ~Q}[L_adv(θ; ε)] ≤ E_{θ~Q}[ L̂_S^{adv}(θ; ε) ]
                         + L_SSM(θ) · ε / 2
                         + sqrt( ( KL(Q‖P) + ln(2√n / δ) ) / (2n) )

This module provides:

* :func:`gaussian_kl_divergence` — analytic KL between two diagonal Gaussians.
* :func:`pac_bayes_complexity` — the right-most square root term.
* :func:`pac_bayes_training_term` — the trainable surrogate combining KL with
  the Lipschitz penalty.
* :func:`pac_bayes_bound` — the full RHS of Theorem 3 for evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn as nn


@dataclass
class PACBayesConfig:
    """Hyper-parameters for the PAC-Bayes objective."""

    delta: float = 0.05            # confidence parameter
    sigma_post: float = 0.05       # σ for posterior Q
    sigma_prior: float = 0.10      # σ₀ for prior P
    epsilon_train: float = 0.18    # adversarial radius used during training
    beta: float = 1.0              # weight on the complexity term
    n_train: int = 1               # number of training samples (set at runtime)


def gaussian_kl_divergence(
    posterior_mean: torch.Tensor,
    prior_mean: torch.Tensor,
    sigma_post: float,
    sigma_prior: float,
) -> torch.Tensor:
    """KL between two diagonal isotropic Gaussians of equal dimension.

    KL(N(μ, σ² I) ‖ N(μ₀, σ₀² I))
        = 0.5 · sum_i ( σ²/σ₀² + (μᵢ − μ₀ᵢ)² / σ₀² − 1 + 2·ln(σ₀/σ) ).
    """
    if posterior_mean.shape != prior_mean.shape:
        raise ValueError("posterior and prior must share dimensionality")
    diff_sq = (posterior_mean - prior_mean).pow(2).sum()
    n_params = posterior_mean.numel()
    var_p = sigma_post ** 2
    var_q = sigma_prior ** 2
    return 0.5 * (
        n_params * (var_p / var_q)
        + diff_sq / var_q
        - n_params
        + 2.0 * n_params * math.log(sigma_prior / sigma_post)
    )


def pac_bayes_complexity(kl: torch.Tensor, n: int, delta: float) -> torch.Tensor:
    """Right-most term: sqrt( (KL + ln(2√n/δ)) / (2n) )."""
    if n <= 0:
        raise ValueError("n must be positive")
    log_term = math.log(2.0 * math.sqrt(n) / delta)
    return torch.sqrt((kl + log_term) / (2.0 * n))


def pac_bayes_training_term(
    posterior_params: torch.Tensor,
    prior_params: torch.Tensor,
    l_ssm: torch.Tensor,
    cfg: PACBayesConfig,
) -> dict[str, torch.Tensor]:
    """Compute the additive PAC-Bayes surrogate used during training.

    The trainer adds ``loss + lipschitz_term + cfg.beta * complexity_term`` to
    the empirical adversarial loss.  Returning the components separately lets
    us log them and inspect their relative scale.
    """
    kl = gaussian_kl_divergence(
        posterior_mean=posterior_params,
        prior_mean=prior_params,
        sigma_post=cfg.sigma_post,
        sigma_prior=cfg.sigma_prior,
    )
    complexity = pac_bayes_complexity(kl, cfg.n_train, cfg.delta)
    lipschitz_term = l_ssm * cfg.epsilon_train / 2.0
    return {
        "kl": kl,
        "complexity": complexity,
        "lipschitz_term": lipschitz_term,
        "total": lipschitz_term + cfg.beta * complexity,
    }


def pac_bayes_bound(
    empirical_adv_loss: torch.Tensor,
    posterior_params: torch.Tensor,
    prior_params: torch.Tensor,
    l_ssm: torch.Tensor,
    cfg: PACBayesConfig,
) -> torch.Tensor:
    """Evaluate the full Theorem 3 RHS — used at evaluation time."""
    pieces = pac_bayes_training_term(posterior_params, prior_params, l_ssm, cfg)
    return empirical_adv_loss + pieces["lipschitz_term"] + pieces["complexity"]


def flatten_constrained_parameters(model: nn.Module) -> torch.Tensor:
    """Concatenate all SSM/projection/head parameters into one flat vector.

    The PAC-Bayes posterior is defined only over the constrained parameters
    (the ones that participate in the Lipschitz bound).  We identify them by
    the ``SpectralNormLinear`` module class plus the ``EigenReparamA.alpha``
    parameter.
    """
    from ..models.eigen_reparam import EigenReparamA
    from ..models.spectral_norm import SpectralNormLinear

    chunks: list[torch.Tensor] = []
    for m in model.modules():
        if isinstance(m, SpectralNormLinear):
            chunks.append(m.weight.detach().reshape(-1))
        elif isinstance(m, EigenReparamA):
            chunks.append(m.alpha.detach().reshape(-1))
    if not chunks:
        return torch.zeros(1)
    return torch.cat(chunks)
