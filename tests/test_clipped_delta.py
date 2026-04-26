"""Clipped Δ_t bounds."""
from __future__ import annotations

import torch

from lipmamba.models.clipped_delta import ClippedDelta


def test_delta_within_zero_max() -> None:
    layer = ClippedDelta(d_model=16, d_inner=16, delta_max=0.5, s_delta=0.5)
    layer.eval()
    x = torch.randn(4, 32, 16) * 5.0
    delta = layer(x)
    assert (delta >= 0).all()
    assert (delta <= 0.5 + 1e-5).all()
