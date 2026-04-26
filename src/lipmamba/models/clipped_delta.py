"""Clipped discretisation step ``Δ_t``.

Implements

    Δ_t = Δ_max · tanh( softplus( W̄_Δ x_t + τ ) / Δ_max )

so that Δ_t ∈ (0, Δ_max) with smooth saturation, ensuring the discrete
recurrence radius ``ρ = exp(-Δ · λ_min)`` stays strictly below one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_norm import SpectralNormLinear


class ClippedDelta(nn.Module):
    """Spectrally-bounded, clipped Δ_t projection."""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        delta_max: float = 0.5,
        s_delta: float = 0.5,
        n_power_iters: int = 1,
    ) -> None:
        super().__init__()
        self.delta_max = float(delta_max)
        self.s_delta = float(s_delta)
        self.proj = SpectralNormLinear(
            d_model, d_inner, s=s_delta, bias=True, n_power_iters=n_power_iters
        )
        # Bias offset τ — initialise so that Δ_t starts near Δ_max/2.
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Δ_t from input ``x`` of shape ``(B, T, d_model)``."""
        z = self.proj(x)
        return self.delta_max * torch.tanh(F.softplus(z) / self.delta_max)

    @property
    def delta_min(self) -> float:
        """Lower bound on Δ_t in the asymptotic regime (≈ 0)."""
        return 0.0
