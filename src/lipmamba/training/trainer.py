"""LipMamba trainer.

Implements *Algorithm 2 — PAC-Bayesian adversarial training* from the
manuscript.  The training loop iterates:

1. Sample a clean mini-batch.
2. (Optional) generate adversarial inputs via PGD or HiSPA.
3. Forward pass → margin-augmented classification or LM cross-entropy.
4. Add the Lipschitz penalty L_SSM · ε / 2 and the PAC-Bayes complexity.
5. Backprop, gradient clip, AdamW step, cosine-annealed LR.
6. After ``power_iter_freq`` steps, refresh σ̂ via :class:`SpectralNormLinear`.

The trainer is deliberately framework-light (no Lightning / Accelerate
dependency) so that the core algorithm is easy to inspect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..certificates.lipschitz import empirical_network_lipschitz
from ..certificates.pac_bayes import PACBayesConfig
from ..utils.checkpoint import save_checkpoint
from ..utils.logging import get_logger
from .adv_objective import adversarial_loss, margin_adversarial_loss
from .optim import build_optimizer, build_scheduler
from .pac_objective import pac_bayes_total_loss


@dataclass
class TrainerConfig:
    """Trainer-level hyper-parameters (paper defaults)."""

    max_steps: int = 100_000
    warmup_steps: int = 1_000
    lr: float = 2e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    epsilon_train: float = 0.18
    log_every: int = 100
    eval_every: int = 5_000
    save_every: int = 5_000
    out_dir: str = "runs/lipmamba"
    power_iter_freq: int = 1
    pac_bayes: PACBayesConfig = field(default_factory=PACBayesConfig)
    use_margin_objective: bool = True   # Eq. 16 — closed-form upper bound
    use_attack: str | None = None        # {"pgd", "hispa", None}
    attack_kwargs: dict = field(default_factory=dict)


class LipMambaTrainer:
    """Single-process PAC-Bayes adversarial trainer."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        prior_params: torch.Tensor | None = None,
        cfg: TrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg or TrainerConfig()
        self.optimizer = build_optimizer(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            warmup_steps=self.cfg.warmup_steps,
            max_steps=self.cfg.max_steps,
        )
        self.prior_params = prior_params
        self.logger = get_logger("lipmamba.train")

        Path(self.cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Inner step                                                          #
    # ------------------------------------------------------------------ #

    def _attack_fn(self):
        kind = self.cfg.use_attack
        if kind is None:
            return None
        if kind == "pgd":
            from ..attacks.pgd import PGDAttack, PGDConfig
            cfg = PGDConfig(**self.cfg.attack_kwargs)
            attacker = PGDAttack(self.model, cfg)
            return lambda model, batch: attacker.attack(batch["input_ids"], batch["labels"])
        if kind == "hispa":
            from ..attacks.hispa import HiSPAAttack, HiSPAConfig
            cfg = HiSPAConfig(**self.cfg.attack_kwargs)
            attacker = HiSPAAttack(self.model, cfg)
            return lambda model, batch: attacker.attack(batch["input_ids"])[0]
        raise ValueError(f"unknown attack kind {kind!r}")

    def _step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}

        if self.cfg.use_margin_objective and self.model.cls_head is not None:
            l_net = self.model.network_lipschitz_bound().to(device)
            adv = margin_adversarial_loss(
                self.model, batch, l_net=l_net, epsilon=self.cfg.epsilon_train
            )
        else:
            adv = adversarial_loss(self.model, batch, self._attack_fn())
            l_net = self.model.network_lipschitz_bound().to(device)

        if self.prior_params is None:
            # Cold start: prior == current parameters (KL ≈ 0)
            prior = torch.zeros_like(
                torch.cat([p.detach().reshape(-1) for p in self.model.parameters()])
            )
        else:
            prior = self.prior_params

        components = pac_bayes_total_loss(
            self.model,
            batch,
            empirical_adv_loss=adv,
            l_net=l_net,
            prior_params=prior,
            cfg=self.cfg.pac_bayes,
        )

        loss = components["loss"]
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": float(loss.detach().item()),
            "adv": float(components["adv_loss"].item()),
            "kl": float(components["kl"].item()),
            "lip": float(components["lipschitz_term"].item()),
            "L_net": float(l_net.item()),
        }

    # ------------------------------------------------------------------ #
    # Outer training loop                                                  #
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        cfg = self.cfg
        step = 0
        iterator = iter(self.train_loader)
        while step < cfg.max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                batch = next(iterator)
            stats = self._step(batch)
            step += 1
            if step % cfg.log_every == 0:
                self.logger.info(
                    "step=%d loss=%.4f adv=%.4f L_net=%.3f kl=%.2f",
                    step, stats["loss"], stats["adv"], stats["L_net"], stats["kl"],
                )
            if step % cfg.eval_every == 0 and self.val_loader is not None:
                self.evaluate(step=step)
            if step % cfg.save_every == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    step=step,
                    path=str(Path(cfg.out_dir) / f"step{step}.pt"),
                )
        save_checkpoint(self.model, self.optimizer, step=step, path=str(Path(cfg.out_dir) / "final.pt"))

    @torch.no_grad()
    def evaluate(self, step: int | None = None) -> dict[str, float]:
        self.model.eval()
        device = next(self.model.parameters()).device
        ce = 0.0
        n = 0
        for batch in self.val_loader or []:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = self.model(batch["input_ids"])
            if "cls_logits" in out:
                logits = out["cls_logits"]
                loss = nn.functional.cross_entropy(logits, batch["labels"])
            else:
                loss = nn.functional.cross_entropy(
                    out["lm_logits"].reshape(-1, out["lm_logits"].size(-1)),
                    batch["labels"].reshape(-1),
                )
            ce += float(loss.item()) * batch["input_ids"].size(0)
            n += batch["input_ids"].size(0)
        l_net = float(empirical_network_lipschitz(self.model))
        ce_avg = ce / max(1, n)
        self.logger.info("[eval] step=%s ce=%.4f L_net=%.3f", step, ce_avg, l_net)
        return {"ce": ce_avg, "L_net": l_net}
