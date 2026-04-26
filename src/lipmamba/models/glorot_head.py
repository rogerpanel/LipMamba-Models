"""GloroNet-style certification head.

For a classification model with logits ``z(x) ∈ R^K`` and predicted class
``ŷ = argmax z(x)``, the GloroNet certified radius is

    ε*(x) = ( z_{ŷ} − max_{k ≠ ŷ} z_k ) / ( sqrt(2) · L_net ).

During training the head computes a *margin-augmented logit* (Eq. 16 of the
paper)::

    z̃_K = max_{k ≠ ŷ} z_k(x) + sqrt(2) · L_net · ε_train

so that minimising cross-entropy on ``[z, z̃_K]`` directly enlarges the
certified margin.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_norm import SpectralNormLinear


class GloroNetHead(nn.Module):
    """Spectrally-bounded classification head with GloroNet certification."""

    def __init__(
        self,
        d_model: int,
        n_classes: int,
        s_head: float = 1.0,
        epsilon_train: float = 0.18,
    ) -> None:
        super().__init__()
        self.classifier = SpectralNormLinear(d_model, n_classes, s=s_head)
        self.s_head = float(s_head)
        self.epsilon_train = float(epsilon_train)

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Logits ``(B, K)`` from features ``(B, d_model)``."""
        return self.classifier(features)

    # ------------------------------------------------------------------ #
    # Certification                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def certified_radius(
        logits: torch.Tensor,
        l_net: torch.Tensor | float,
    ) -> torch.Tensor:
        """Per-sample certified radius ε*(x).

        Returns ``-inf`` for samples whose top-1 has been overtaken (negative
        margin).  The convention matches GloroNet: ``ε* > 0`` means
        ``argmax z(x + δ) = ŷ`` for all ``‖δ‖ ≤ ε*``.
        """
        z_hat, hat_idx = logits.max(dim=-1)
        masked = logits.clone()
        masked.scatter_(1, hat_idx.unsqueeze(-1), float("-inf"))
        z_runner = masked.max(dim=-1).values
        margin = z_hat - z_runner
        l = torch.as_tensor(l_net, dtype=logits.dtype, device=logits.device)
        return margin / (math.sqrt(2.0) * (l + 1e-12))

    def margin_augmented(
        self,
        logits: torch.Tensor,
        l_net: torch.Tensor | float,
        epsilon: float | None = None,
    ) -> torch.Tensor:
        """Return logits with the runner-up bumped by ``√2 · L · ε`` (Eq. 16)."""
        eps = self.epsilon_train if epsilon is None else float(epsilon)
        z_hat, hat_idx = logits.max(dim=-1)
        masked = logits.clone()
        masked.scatter_(1, hat_idx.unsqueeze(-1), float("-inf"))
        z_runner = masked.max(dim=-1).values
        l = torch.as_tensor(l_net, dtype=logits.dtype, device=logits.device)
        bump = math.sqrt(2.0) * l * eps
        # build new logits: keep z_hat, replace runner-up with bumped value.
        bumped = logits.clone()
        # Find the runner-up index per row.
        runner_idx = masked.argmax(dim=-1)
        bumped[torch.arange(logits.size(0)), runner_idx] = z_runner + bump
        return bumped

    def margin_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        l_net: torch.Tensor | float,
        epsilon: float | None = None,
    ) -> torch.Tensor:
        """Cross-entropy on margin-augmented logits."""
        bumped = self.margin_augmented(logits, l_net=l_net, epsilon=epsilon)
        return F.cross_entropy(bumped, targets)
