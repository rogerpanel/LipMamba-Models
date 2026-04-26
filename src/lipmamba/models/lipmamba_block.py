"""LipMamba block — selective SSM wrapped with bounded gating + projections.

Architecture (Section 3, Algorithm 1):

    x_in  ── LayerNorm
       │
       ├── linear↑(d_model → d_inner) ──► Conv1d ──► SiLU ──► SelectiveSSM ──┐
       │                                                                     │
       └── linear↑(d_model → d_inner) ──► SiLU ──── (gate)──────────────────►⊙
                                                                             │
                                                              linear↓(d_inner → d_model) ──► residual
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .selective_ssm import SelectiveSSM, SSMConfig
from .spectral_norm import SpectralNormLinear


@dataclass
class LipMambaBlockConfig:
    """Block-level hyper-parameters."""

    d_model: int = 1024
    d_inner: int = 2048
    state_dim: int = 16
    conv_kernel: int = 4
    s_b: float = 1.0
    s_c: float = 1.0
    s_delta: float = 0.5
    s_out: float = 1.0
    delta_max: float = 0.5
    lambda_min: float = 0.05
    lambda_max: float = 1.0
    n_power_iters: int = 1
    track_lipschitz: bool = True


class LipMambaBlock(nn.Module):
    """A single LipMamba layer — see Algorithm 1 / Figure 1 of the paper."""

    def __init__(self, cfg: LipMambaBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)

        self.in_proj_x = SpectralNormLinear(
            cfg.d_model, cfg.d_inner, s=1.0, n_power_iters=cfg.n_power_iters, bias=False
        )
        self.in_proj_z = SpectralNormLinear(
            cfg.d_model, cfg.d_inner, s=1.0, n_power_iters=cfg.n_power_iters, bias=False
        )

        # Causal conv with bounded operator norm (depth-wise so conv norm ≤ ‖kernel‖₂).
        self.conv = nn.Conv1d(
            cfg.d_inner,
            cfg.d_inner,
            kernel_size=cfg.conv_kernel,
            groups=cfg.d_inner,
            padding=cfg.conv_kernel - 1,
            bias=True,
        )

        self.ssm = SelectiveSSM(
            SSMConfig(
                d_model=cfg.d_model,
                d_inner=cfg.d_inner,
                state_dim=cfg.state_dim,
                s_b=cfg.s_b,
                s_c=cfg.s_c,
                s_delta=cfg.s_delta,
                s_out=cfg.s_out,
                delta_max=cfg.delta_max,
                lambda_min=cfg.lambda_min,
                lambda_max=cfg.lambda_max,
                n_power_iters=cfg.n_power_iters,
                track_lipschitz=cfg.track_lipschitz,
            )
        )
        self.out_proj = SpectralNormLinear(
            cfg.d_inner, cfg.d_model, s=cfg.s_out, n_power_iters=cfg.n_power_iters, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass with residual connection."""
        residual = x
        x_n = self.norm(x)

        x_proj = self.in_proj_x(x_n)
        z_proj = self.in_proj_z(x_n)

        # Causal convolution (B, T, D) -> (B, D, T)
        b, t, d = x_proj.shape
        xc = x_proj.transpose(1, 2)
        xc = self.conv(xc)[..., :t]
        xc = xc.transpose(1, 2)
        xc = F.silu(xc)

        ys = self.ssm(xc)
        ys = ys * F.silu(z_proj)

        out = self.out_proj(ys)
        return residual + out

    # ------------------------------------------------------------------ #
    # Lipschitz reporting                                                #
    # ------------------------------------------------------------------ #

    def block_lipschitz_bound(self, h_inf: float = 1.0) -> torch.Tensor:
        """Closed-form Lipschitz bound for this block (Theorem 1)."""
        return self.ssm.block_lipschitz_bound(h_inf=h_inf)

    @torch.no_grad()
    def lipschitz_state(self) -> dict[str, float]:
        return self.ssm.lipschitz_state()
