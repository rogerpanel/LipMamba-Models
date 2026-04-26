"""Lipschitz tracker / closed-form bound."""
from __future__ import annotations

from lipmamba.certificates.lipschitz import (
    LipschitzTracker,
    layer_lipschitz_bound,
    network_lipschitz,
)


def test_layer_bound_finite_and_positive() -> None:
    bound = layer_lipschitz_bound(
        s_b=1.0, s_c=1.0, s_delta=0.5, s_out=1.0,
        delta_max=0.5, lambda_min=0.05,
    )
    assert bound > 0
    assert bound < 1e3  # sanity


def test_network_lipschitz_with_residuals() -> None:
    block_bound = layer_lipschitz_bound(
        s_b=1.0, s_c=1.0, s_delta=0.5, s_out=1.0,
        delta_max=0.5, lambda_min=0.05,
    )
    full = network_lipschitz([block_bound] * 4, head_factor=1.0)
    assert full > (1.0 + block_bound) ** 3


def test_tracker_monotone_with_constant_rho() -> None:
    tr = LipschitzTracker(s_b=1.0, s_c=1.0, s_out=1.0)
    seen = []
    for _ in range(10):
        seen.append(tr.update(rho_t=0.9, beta_t=0.5))
    assert all(seen[i] <= seen[i + 1] + 1e-9 for i in range(len(seen) - 1))
