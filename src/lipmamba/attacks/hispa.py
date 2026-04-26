"""HiSPA — Hidden-State Poisoning Attack.

The threat model from Section 2 of the paper.  An ``(α, ℓ)``-poisoning
attack appends a trigger sequence ``τ = (τ₁, …, τ_ℓ)`` so that the post-
trigger hidden-state norm collapses by a factor ``α ≪ 1``.

We implement two flavours:

* *Continuous* HiSPA — gradient-based optimisation over input embeddings to
  drive the SSM hidden state toward zero.
* *Discrete* HiSPA — token-level greedy search via :mod:`trigger_search`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HiSPAConfig:
    """HiSPA hyper-parameters."""

    trigger_length: int = 16
    n_steps: int = 200
    lr: float = 5e-2
    norm_budget: float = 1.0
    target_alpha: float = 0.05
    init: str = "gaussian"      # one of {gaussian, uniform, zeros}


class HiSPAAttack:
    """Continuous, embedding-space hidden-state poisoning attack.

    Given a model with token embeddings ``E`` of shape ``(V, d_model)``, we
    optimise a perturbation ``δ ∈ R^{ℓ × d_model}`` (added to a frozen
    benign prefix's embeddings) so that the resulting trajectory minimises
    ``‖h_{T}‖₂`` while ``‖δ‖₂ ≤ norm_budget``.
    """

    def __init__(self, model: nn.Module, cfg: HiSPAConfig) -> None:
        self.model = model
        self.cfg = cfg

    @torch.enable_grad()
    def attack(
        self,
        prefix_ids: torch.Tensor,
        target_position: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the attack and return the optimised perturbation.

        Parameters
        ----------
        prefix_ids : (B, T_p) token ids, the benign prompt.
        target_position : index of the hidden state to minimise (default: last).
        """
        cfg = self.cfg
        device = prefix_ids.device
        b = prefix_ids.size(0)

        prefix_embed = self.model.embed_tokens(prefix_ids).detach()
        d_model = prefix_embed.size(-1)

        if cfg.init == "gaussian":
            delta = torch.randn(b, cfg.trigger_length, d_model, device=device) * 0.01
        elif cfg.init == "uniform":
            delta = (torch.rand(b, cfg.trigger_length, d_model, device=device) - 0.5) * 0.1
        else:
            delta = torch.zeros(b, cfg.trigger_length, d_model, device=device)
        delta.requires_grad_(True)

        optim = torch.optim.Adam([delta], lr=cfg.lr)

        for step in range(cfg.n_steps):
            optim.zero_grad(set_to_none=True)
            embeds = torch.cat([prefix_embed, delta], dim=1)
            h = embeds
            for blk in self.model.blocks:
                h = blk(h)
            h = self.model.norm_f(h)
            tp = target_position if target_position is not None else h.size(1) - 1
            target_norm = h[:, tp].norm(dim=-1)
            # Minimise the post-trigger norm.
            loss = target_norm.mean()
            loss.backward()
            optim.step()

            with torch.no_grad():
                # Project δ back onto the L2 ball.
                norms = delta.flatten(1).norm(dim=-1, keepdim=True)
                scale = (cfg.norm_budget / (norms + 1e-12)).clamp(max=1.0)
                delta.mul_(scale.unsqueeze(-1))

        with torch.no_grad():
            embeds = torch.cat([prefix_embed, delta], dim=1)
            h = embeds
            for blk in self.model.blocks:
                h = blk(h)
            h = self.model.norm_f(h)
            tp = target_position if target_position is not None else h.size(1) - 1
            final_norm = h[:, tp].norm(dim=-1).mean().item()
            clean_norm = self.model.encode(prefix_ids)[:, -1].norm(dim=-1).mean().item()

        info = {
            "final_norm": final_norm,
            "clean_norm": clean_norm,
            "alpha": final_norm / max(clean_norm, 1e-12),
            "success": (final_norm / max(clean_norm, 1e-12)) <= cfg.target_alpha,
        }
        return delta.detach(), info
