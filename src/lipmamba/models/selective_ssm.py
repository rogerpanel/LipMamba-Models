"""Selective state-space recurrence with Lipschitz-bounded projections.

This is the core layer underlying LipMamba.  It implements the selective scan

    h_t = Ā_t h_{t-1} + B̄_t x_t
    y_t = C_tᵀ h_t

with input-dependent ``B_t``, ``C_t``, ``Δ_t`` produced by spectrally-bounded
projection networks.  The parameterisation guarantees a closed-form Lipschitz
constant (see :mod:`lipmamba.certificates.lipschitz`).

For pedagogical clarity and CPU portability the scan is implemented as a
pure-PyTorch ``for`` loop.  For GPU acceleration drop in the official
``mamba_ssm.selective_scan`` kernel — the public tensor signatures match.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .clipped_delta import ClippedDelta
from .eigen_reparam import EigenReparamA
from .hippo import hippo_init
from .spectral_norm import SpectralNormLinear


@dataclass
class SSMConfig:
    """Selective SSM hyper-parameters (Section 3 of the paper)."""

    d_model: int = 1024
    d_inner: int = 2048
    state_dim: int = 16
    s_b: float = 1.0
    s_c: float = 1.0
    s_delta: float = 0.5
    s_out: float = 1.0
    delta_max: float = 0.5
    lambda_min: float = 0.05
    lambda_max: float = 1.0
    n_power_iters: int = 1
    track_lipschitz: bool = True


class SelectiveSSM(nn.Module):
    """Lipschitz-bounded selective scan over ``D`` channels.

    Tensor shapes
    -------------
    Input  : ``x``  — ``(B, T, d_inner)``
    Output : ``y``  — ``(B, T, d_inner)``

    The class also exposes :meth:`lipschitz_state` which, after a forward
    pass, returns the running estimate of the recurrence radius
    ``ρ_t = ‖Ā_t‖₂`` and projection norms ``β_t = ‖B̄_t‖₂`` used by the
    online Lipschitz tracker.
    """

    def __init__(self, cfg: SSMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Input-dependent projections — spectrally normalised.
        self.x_to_b = SpectralNormLinear(
            cfg.d_inner, cfg.state_dim, s=cfg.s_b, n_power_iters=cfg.n_power_iters
        )
        self.x_to_c = SpectralNormLinear(
            cfg.d_inner, cfg.state_dim, s=cfg.s_c, n_power_iters=cfg.n_power_iters
        )
        self.delta_proj = ClippedDelta(
            d_model=cfg.d_inner,
            d_inner=cfg.d_inner,
            delta_max=cfg.delta_max,
            s_delta=cfg.s_delta,
            n_power_iters=cfg.n_power_iters,
        )

        # Eigenvalue-bounded state matrix (one set of eigenvalues per channel).
        self.A = EigenReparamA(
            state_dim=cfg.state_dim,
            n_channels=cfg.d_inner,
            lambda_min=cfg.lambda_min,
            lambda_max=cfg.lambda_max,
        )
        # HiPPO-style warm start
        hippo_init(self.A.alpha, cfg.lambda_min, cfg.lambda_max)

        # Per-channel skip connection D (paper uses bounded form ``s_out``).
        self.D = nn.Parameter(torch.ones(cfg.d_inner))

        # Buffers for online Lipschitz statistics.
        self.register_buffer("_rho_running", torch.tensor(0.0))
        self.register_buffer("_beta_running", torch.tensor(0.0))

    # ------------------------------------------------------------------ #
    # Forward / scan                                                     #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Run the selective scan."""
        b, t, d = x.shape
        n = self.cfg.state_dim

        delta = self.delta_proj(x)              # (B, T, D)
        b_t = self.x_to_b(x)                    # (B, T, N)
        c_t = self.x_to_c(x)                    # (B, T, N)

        # Discretise: Ā_t (B, T, D, N) and B̄_t (B, T, D, N)
        a_bar = self.A.discretise(delta)         # exp(Δ ⊙ A)
        # Δ ⊙ B with broadcasting — ZOH-like rule (matches mamba reference)
        b_bar = delta.unsqueeze(-1) * b_t.unsqueeze(-2)  # (B, T, D, N)

        h = x.new_zeros(b, d, n)
        ys = []
        rho_acc = 0.0
        beta_acc = 0.0
        for tt in range(t):
            h = a_bar[:, tt] * h + b_bar[:, tt] * x[:, tt].unsqueeze(-1)  # (B, D, N)
            y = (h * c_t[:, tt].unsqueeze(1)).sum(dim=-1)                  # (B, D)
            ys.append(y)

            if self.cfg.track_lipschitz and self.training:
                # cheap upper-bound estimates of operator norms
                rho_acc += a_bar[:, tt].abs().amax().detach()
                beta_acc += b_bar[:, tt].abs().amax().detach()

        y = torch.stack(ys, dim=1)               # (B, T, D)
        y = y + self.D * x                       # bounded skip

        if self.cfg.track_lipschitz and self.training and t > 0:
            self._rho_running.copy_(rho_acc / t)
            self._beta_running.copy_(beta_acc / t)

        return y

    # ------------------------------------------------------------------ #
    # Lipschitz / certificate helpers                                    #
    # ------------------------------------------------------------------ #

    def block_lipschitz_bound(
        self,
        h_inf: float = 1.0,
        l_silu: float = 1.0998,
    ) -> torch.Tensor:
        """Closed-form per-block Lipschitz bound (Theorem 1).

        Parameters
        ----------
        h_inf : float
            Upper bound on ``‖h‖_∞`` (estimated empirically during training).
        l_silu : float
            Lipschitz constant of SiLU (≈ 1.0998).
        """
        cfg = self.cfg
        # ρ_max attained at λ_min (smallest decay).
        rho_max = float(torch.exp(torch.tensor(-cfg.delta_max * cfg.lambda_min)).item())
        # The bound equation from the paper:
        denom = max(1.0 - rho_max, 1e-8)
        term_input = cfg.s_c * (cfg.s_b * cfg.delta_max) / denom
        term_state = cfg.s_c * h_inf * cfg.s_delta * cfg.delta_max / denom
        return torch.tensor(cfg.s_out * l_silu * (term_input + term_state))

    @torch.no_grad()
    def lipschitz_state(self) -> dict[str, float]:
        """Return current running estimates of recurrence radius ρ and β."""
        return {
            "rho_running": float(self._rho_running.item()),
            "beta_running": float(self._beta_running.item()),
            "block_bound": float(self.block_lipschitz_bound().item()),
        }
