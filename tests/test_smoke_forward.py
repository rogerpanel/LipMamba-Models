"""End-to-end smoke test: a tiny LipMambaModel runs and produces logits."""
from __future__ import annotations

import torch

from lipmamba import LipMambaConfig, LipMambaModel


def test_tiny_forward_returns_logits() -> None:
    cfg = LipMambaConfig(
        vocab_size=100, n_layers=2, d_model=32, d_inner=64,
        state_dim=8, conv_kernel=3, n_classes=4,
    )
    model = LipMambaModel(cfg)
    ids = torch.randint(0, 100, (2, 16))
    out = model(ids)
    assert out["lm_logits"].shape == (2, 16, 100)
    assert out["cls_logits"].shape == (2, 4)


def test_network_lipschitz_bound_reports_finite_value() -> None:
    cfg = LipMambaConfig(
        vocab_size=50, n_layers=2, d_model=16, d_inner=32, state_dim=4
    )
    model = LipMambaModel(cfg)
    bound = model.network_lipschitz_bound().item()
    assert bound > 0
    assert bound < 1e6
