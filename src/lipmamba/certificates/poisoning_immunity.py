"""Hidden-state poisoning immunity (Theorem 2).

For a trigger of length ℓ the post-trigger hidden state norm is bounded
below by

    ‖h_{t₀+ℓ}‖₂ ≥ ρ_min^ℓ · ‖h_{t₀}‖₂
                − B̄_max · X_max · ( 1 − ρ_min^ℓ ) / ( 1 − ρ_min ).

This rules out (α, ℓ)-poisoning attacks for trigger lengths

    ℓ* ≤ log( α_min + B̄_max·X_max / ((1 − ρ_min) · ‖h_{t₀}‖) ) / log(ρ_min).

We expose both quantities and a convenience function that returns the
maximum trigger length tolerated for a given attack threshold ``α``.
"""
from __future__ import annotations

import math


def poisoning_immunity_lower_bound(
    rho_min: float,
    h0_norm: float,
    b_bar_max: float,
    x_max: float,
    ell: int,
) -> float:
    """Lower bound on ``‖h_{t₀+ℓ}‖₂``."""
    if not (0.0 < rho_min < 1.0):
        raise ValueError("rho_min must lie in (0, 1)")
    decay = rho_min ** ell
    drift = b_bar_max * x_max * (1.0 - decay) / (1.0 - rho_min)
    return decay * h0_norm - drift


def max_certified_trigger_length(
    rho_min: float,
    h0_norm: float,
    b_bar_max: float,
    x_max: float,
    alpha: float,
) -> int:
    """Largest ℓ such that the lower bound stays ≥ ``alpha · ‖h₀‖``.

    Returns ``∞`` if the parameters guarantee no length is unsafe (e.g. the
    bound is trivially above ``α · h_0_norm`` for all ℓ); we cap returns at
    1024 for practical purposes.
    """
    if not (0.0 < rho_min < 1.0):
        raise ValueError("rho_min must lie in (0, 1)")
    threshold = alpha * h0_norm
    drift_factor = b_bar_max * x_max / max(1.0 - rho_min, 1e-12)
    target = threshold + drift_factor
    base = h0_norm + drift_factor
    if target <= 0:
        return 1024
    # ρ_min^ℓ · base ≥ target  ⇒  ℓ ≤ log(target/base) / log(ρ_min)
    if base <= 0:
        return 0
    quotient = target / base
    if quotient >= 1.0:
        return 0
    ell_star = math.log(quotient) / math.log(rho_min)
    return int(max(0, math.floor(ell_star)))


def certified_immunity_summary(
    *,
    delta_min: float,
    lambda_min: float,
    s_b: float,
    delta_max: float,
    x_max: float,
    h0_norm: float,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Convenience wrapper computing all quantities used in the paper."""
    rho_min = math.exp(-delta_min * lambda_min)
    b_bar_max = s_b * delta_max
    ell_star = max_certified_trigger_length(
        rho_min=rho_min,
        h0_norm=h0_norm,
        b_bar_max=b_bar_max,
        x_max=x_max,
        alpha=alpha,
    )
    return {
        "rho_min": rho_min,
        "b_bar_max": b_bar_max,
        "ell_star": ell_star,
        "lower_bound_at_ell_star": poisoning_immunity_lower_bound(
            rho_min=rho_min,
            h0_norm=h0_norm,
            b_bar_max=b_bar_max,
            x_max=x_max,
            ell=ell_star,
        ),
    }
