"""Eigenvalue reparameterisation for the state matrix ``A``.

The LipMamba state matrix is constrained to be diagonal with strictly
negative eigenvalues bounded inside ``[-λ_max, -λ_min]``::

    A = -diag( λ_min + (λ_max − λ_min) · σ(α) )

where ``σ`` is the logistic sigmoid and ``α`` is an unconstrained learnable
parameter.  This makes the state operator a strict contraction whose discrete
counterpart ``Ā = exp(Δ ⊙ A)`` satisfies ``‖Ā‖₂ ≤ exp(-Δ_min · λ_min) < 1``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EigenReparamA(nn.Module):
    """Diagonal contractive state matrix with bounded eigenvalues.

    Parameters
    ----------
    state_dim : int
        Number of latent SSM states ``N``.
    n_channels : int
        Independent SSM channels (``D``).  Each channel owns its own ``A``.
    lambda_min, lambda_max : float
        Lower / upper bounds for ``|λᵢ(A)|``.
    """

    def __init__(
        self,
        state_dim: int,
        n_channels: int = 1,
        lambda_min: float = 0.05,
        lambda_max: float = 1.0,
    ) -> None:
        super().__init__()
        if not (0.0 < lambda_min < lambda_max):
            raise ValueError("require 0 < lambda_min < lambda_max")
        self.state_dim = state_dim
        self.n_channels = n_channels
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        # initialise α so that σ(α) ≈ 0.5 and λ ≈ midpoint of the interval
        self.alpha = nn.Parameter(torch.zeros(n_channels, state_dim))

    @property
    def lambdas(self) -> torch.Tensor:
        """Return ``|λᵢ(A)| ∈ [λ_min, λ_max]`` per channel and state."""
        return self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(self.alpha)

    def forward(self) -> torch.Tensor:
        """Return ``A`` of shape ``(n_channels, state_dim)`` (diagonal entries)."""
        return -self.lambdas

    def discretise(self, delta: torch.Tensor) -> torch.Tensor:
        """Return ``Ā = exp(Δ ⊙ A)`` for input ``Δ`` of broadcastable shape.

        ``delta`` may be ``(B, T, D)`` while ``A`` is ``(D, N)``.  Result has
        shape ``(B, T, D, N)`` with the diagonal of ``Ā``.
        """
        a = self.forward()  # (D, N)
        # delta: (B, T, D) -> (B, T, D, 1); a: (D, N) -> (1, 1, D, N)
        return torch.exp(delta.unsqueeze(-1) * a.unsqueeze(0).unsqueeze(0))

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, n_channels={self.n_channels}, "
            f"lambda_min={self.lambda_min}, lambda_max={self.lambda_max}"
        )
