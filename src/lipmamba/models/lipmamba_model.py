"""Full LipMamba language / classification model.

Stack of :class:`LipMambaBlock` layers with a token-embedding, optional
language-modelling head, and the GloroNet certification head used for
classification fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .glorot_head import GloroNetHead
from .lipmamba_block import LipMambaBlock, LipMambaBlockConfig


@dataclass
class LipMambaConfig:
    """Top-level model configuration.

    The three reference points from the paper:

    ============   ========   =========   ========   ========
    Variant        n_layers   d_model     d_inner    L_SSM cap
    ============   ========   =========   ========   ========
    LipMamba-130M  24         768         1536       5.0
    LipMamba-370M  48         1024        2048       8.0
    LipMamba-1.3B  64         2048        4096       12.0
    ============   ========   =========   ========   ========
    """

    vocab_size: int = 50257
    n_layers: int = 24
    d_model: int = 768
    d_inner: int = 1536
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

    # Heads
    n_classes: int = 0          # 0 ⇒ language-modelling head only
    s_head: float = 1.0
    epsilon_train: float = 0.18

    # Network-level cap on the certified Lipschitz constant L_SSM
    l_ssm_cap: float = 5.0
    extras: dict = field(default_factory=dict)

    @classmethod
    def lipmamba_130m(cls, **overrides) -> "LipMambaConfig":
        cfg = cls(n_layers=24, d_model=768, d_inner=1536, l_ssm_cap=5.0)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def lipmamba_370m(cls, **overrides) -> "LipMambaConfig":
        cfg = cls(n_layers=48, d_model=1024, d_inner=2048, l_ssm_cap=8.0)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def lipmamba_1300m(cls, **overrides) -> "LipMambaConfig":
        cfg = cls(n_layers=64, d_model=2048, d_inner=4096, l_ssm_cap=12.0)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


class LipMambaModel(nn.Module):
    """Stacked LipMamba blocks with optional LM and classification heads."""

    def __init__(self, cfg: LipMambaConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)

        block_cfg = LipMambaBlockConfig(
            d_model=cfg.d_model,
            d_inner=cfg.d_inner,
            state_dim=cfg.state_dim,
            conv_kernel=cfg.conv_kernel,
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
        self.blocks = nn.ModuleList(LipMambaBlock(block_cfg) for _ in range(cfg.n_layers))
        self.norm_f = nn.LayerNorm(cfg.d_model)

        # Language-modelling head — tied to the embedding for parameter efficiency.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # weight tying

        # Optional classification head with GloroNet certification.
        if cfg.n_classes > 0:
            self.cls_head: GloroNetHead | None = GloroNetHead(
                d_model=cfg.d_model,
                n_classes=cfg.n_classes,
                s_head=cfg.s_head,
                epsilon_train=cfg.epsilon_train,
            )
        else:
            self.cls_head = None

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token IDs → final hidden states ``(B, T, d_model)``."""
        h = self.embed_tokens(input_ids)
        for blk in self.blocks:
            h = blk(h)
        return self.norm_f(h)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_logits: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Standard forward pass.

        Returns a dict with the keys:

        * ``hidden_states`` — ``(B, T, d_model)``
        * ``lm_logits``     — ``(B, T, vocab)`` (only when ``return_logits``)
        * ``cls_logits``    — ``(B, n_classes)`` if a classification head exists
        """
        h = self.encode(input_ids)
        out = {"hidden_states": h}
        if return_logits:
            out["lm_logits"] = self.lm_head(h)
        if self.cls_head is not None:
            # Use the final-position representation for classification.
            pooled = h[:, -1]
            out["cls_logits"] = self.cls_head(pooled)
        return out

    # ------------------------------------------------------------------ #
    # Certificates                                                        #
    # ------------------------------------------------------------------ #

    def network_lipschitz_bound(self, h_inf: float = 1.0) -> torch.Tensor:
        """Multiplicative network Lipschitz bound (Theorem 1, network level).

        ``L_net = ∏_blocks L_block`` (residual pathways inflate by +1 — see
        certificates.lipschitz for the exact formula used during training).
        """
        l = torch.tensor(1.0)
        for blk in self.blocks:
            l = l * (1.0 + blk.block_lipschitz_bound(h_inf=h_inf))
        if self.cls_head is not None:
            l = l * float(self.cls_head.s_head)
        return l

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
