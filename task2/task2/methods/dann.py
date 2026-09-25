# =============================================================================
# task2/methods/dann.py  — DANN: adversarial marginal alignment (Step 3)
# -----------------------------------------------------------------------------
# ML CONCEPT — DANN (Ganin et al. 2016):
#   Replace MMD's fixed kernel distance with a LEARNED adversary. A binary domain
#   discriminator D tries to classify each 512-d feature as source (0) or target
#   (1). A Gradient Reversal Layer (GRL) sits between the backbone and D, so when
#   we minimize D's loss, the reversed gradient trains the backbone to MAXIMIZE
#   domain confusion — making features domain-invariant. This is still MARGINAL
#   alignment (D sees only the feature, not the class), but learned adversarially.
#
# WHO CONTRIBUTES WHICH LOSS (spec):
#   "Only source examples contribute to the class loss, while both source and
#    target examples contribute to the domain loss." We enforce that below:
#   classification loss uses source only; domain loss uses source+target.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from methods.base_method import BaseMethod
    from models.domain_discriminator import DomainDiscriminator, grad_reverse
except ImportError:
    from .base_method import BaseMethod
    from ..models.domain_discriminator import DomainDiscriminator, grad_reverse


class DANN(BaseMethod):
    def __init__(self, cfg, feature_dim: int):
        super().__init__()
        self.cfg = cfg
        self.domain_loss_weight = cfg["method"]["domain_loss_weight"]  # unit=1.0
        # Whether to L2-normalise the feature before the discriminator. On the raw
        # (unbounded) 512-d feature the discriminator overpowers F and training
        # collapses; normalising bounds the discriminator input and stabilises the
        # adversarial game (same fix that keeps CDAN stable). Not a spec
        # hyperparameter — only rescales the discriminator's input.
        self.normalize_disc_input = cfg["method"].get("normalize_disc_input", True)
        # Discriminator on the 512-d feature (in_dim = feature_dim).
        self.discriminator = DomainDiscriminator(
            in_dim=feature_dim,
            hidden=cfg["method"]["disc_hidden"],
            dropout=cfg["method"]["disc_dropout"],
        )
        self.domain_ce = nn.CrossEntropyLoss()         # 2-class domain loss

    def extra_modules(self):
        # The discriminator's parameters must be optimized too, so train.py adds
        # them to the optimizer via this hook.
        return [self.discriminator]

    def compute_loss(self, source_feats, source_logits, source_labels,
                     target_feats, target_logits, alpha):
        # 1) Classification loss: SOURCE ONLY.
        loss_cls = self.classification_loss(source_logits, source_labels)

        # 2) Domain loss: SOURCE + TARGET. We stack both feature sets, pass them
        #    through the GRL (so the backbone gets the reversed gradient), then
        #    through the discriminator.
        feats = torch.cat([source_feats, target_feats], dim=0)   # (Ns+Nt, 512)
        # Optional L2 normalisation of the discriminator input (per example). This
        # bounds the feature magnitude the discriminator sees so it cannot race
        # ahead and collapse the extractor. Gradients still flow to the backbone
        # through the normalised feature, so the adversarial objective is intact.
        if self.normalize_disc_input:
            feats = F.normalize(feats, p=2, dim=1)               # unit-norm per example
        feats_rev = grad_reverse(feats, alpha)                    # GRL applied
        domain_logits = self.discriminator(feats_rev)            # (Ns+Nt, 2)

        # Domain labels: source=0, target=1.
        ns, nt = source_feats.shape[0], target_feats.shape[0]
        domain_labels = torch.cat([
            torch.zeros(ns, dtype=torch.long, device=feats.device),
            torch.ones(nt, dtype=torch.long, device=feats.device),
        ])
        loss_domain = self.domain_ce(domain_logits, domain_labels)

        # 3) Total: class loss + weighted domain loss. Because of the GRL, the
        #    domain term simultaneously trains D to discriminate and F to confuse.
        loss = loss_cls + self.domain_loss_weight * loss_domain

        # Diagnostic: domain-discriminator accuracy (spec 'What to watch for':
        # near-chance = 50% may mean successful confusion OR a weak discriminator).
        with torch.no_grad():
            dom_pred = domain_logits.argmax(dim=1)
            dom_acc = (dom_pred == domain_labels).float().mean().item()

        logs = {"loss_total": loss.item(), "loss_cls": loss_cls.item(),
                "loss_align": loss_domain.item(), "domain_acc": dom_acc,
                "alpha": alpha}
        return loss, logs
