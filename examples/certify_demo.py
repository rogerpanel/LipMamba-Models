"""Certified-radius demonstration."""
from __future__ import annotations

import torch

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.certificates.certified_radius import certified_radius_batch
from lipmamba.certificates.lipschitz import empirical_network_lipschitz
from lipmamba.certificates.poisoning_immunity import certified_immunity_summary
from lipmamba.utils import set_seed


def main() -> None:
    set_seed(0)
    cfg = LipMambaConfig(
        vocab_size=128, n_layers=2, d_model=32, d_inner=64,
        state_dim=4, conv_kernel=3, n_classes=4,
    )
    model = LipMambaModel(cfg).eval()
    ids = torch.randint(0, 128, (8, 16))
    out = model(ids)

    l_net = empirical_network_lipschitz(model)
    eps = certified_radius_batch(out["cls_logits"], l_net=l_net)
    print("L_net          =", l_net)
    print("certified ε(x) =", eps.tolist())
    print("immunity bound =", certified_immunity_summary(
        delta_min=0.05, lambda_min=0.05, s_b=1.0, delta_max=0.5,
        x_max=1.0, h0_norm=1.0, alpha=0.05,
    ))


if __name__ == "__main__":
    main()
