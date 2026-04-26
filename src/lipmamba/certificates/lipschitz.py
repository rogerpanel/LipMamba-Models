"""Lipschitz tracking utilities.

Two complementary code paths are exposed:

* :func:`layer_lipschitz_bound` — the *closed-form* upper bound from
  Theorem 1, valid given the spectral budgets and clipped Δ.
* :class:`LipschitzTracker` — an *online* estimate updated on the fly using

      L_t  =  ρ_t · L_{t-1} + s_C · s_out · L_SiLU · ( β_t + s_B )

  where ``ρ_t = ‖Ā_t‖₂``, ``β_t = ‖B̄_t‖₂``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

L_SILU: float = 1.0998  # Lipschitz constant of SiLU (numerical, see paper App. B.1)


def layer_lipschitz_bound(
    s_b: float,
    s_c: float,
    s_delta: float,
    s_out: float,
    delta_max: float,
    lambda_min: float,
    h_inf: float = 1.0,
    l_silu: float = L_SILU,
) -> float:
    """Closed-form per-block Lipschitz bound (Theorem 1).

    ``ρ_max = exp(-Δ_max · λ_min)``.  We use ``Δ_max`` for the worst case.
    """
    rho_max = math.exp(-delta_max * lambda_min)
    denom = max(1.0 - rho_max, 1e-8)
    inp_term = s_c * (s_b * delta_max) / denom
    state_term = s_c * h_inf * s_delta * delta_max / denom
    return s_out * l_silu * (inp_term + state_term)


def network_lipschitz(
    block_bounds: list[float],
    head_factor: float = 1.0,
    use_residual_inflation: bool = True,
) -> float:
    """Multiplicative network Lipschitz bound.

    With residual connections the per-block contribution is ``1 + L_block``;
    set ``use_residual_inflation=False`` to drop the +1 (only valid when no
    residuals are used).
    """
    factor = 1.0
    for l in block_bounds:
        factor *= (1.0 + l) if use_residual_inflation else l
    return head_factor * factor


class LipschitzTracker:
    """Streaming tracker: updates ``L_t`` token by token.

    Use it during training to log the *empirical* network Lipschitz constant
    and compare it with the closed-form bound from Theorem 1.
    """

    def __init__(
        self,
        s_b: float,
        s_c: float,
        s_out: float,
        l_silu: float = L_SILU,
    ) -> None:
        self.s_b = float(s_b)
        self.s_c = float(s_c)
        self.s_out = float(s_out)
        self.l_silu = float(l_silu)
        self.value = 0.0

    def reset(self) -> None:
        self.value = 0.0

    def update(self, rho_t: float, beta_t: float) -> float:
        """Single recurrence step update."""
        increment = self.s_c * self.s_out * self.l_silu * (beta_t + self.s_b)
        self.value = rho_t * self.value + increment
        return self.value


@torch.no_grad()
def empirical_network_lipschitz(
    model: nn.Module,
    h_inf: float = 1.0,
) -> float:
    """Iterate over all :class:`LipMambaBlock` instances and aggregate.

    Designed to walk arbitrary nested ``nn.Module`` trees so the same helper
    works for the LM and the IDS classifier.
    """
    bounds: list[float] = []
    for module in model.modules():
        if hasattr(module, "block_lipschitz_bound"):
            bounds.append(float(module.block_lipschitz_bound(h_inf=h_inf).item()))
    return network_lipschitz(bounds, head_factor=1.0)
