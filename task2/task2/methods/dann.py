

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
       
        self.normalize_disc_input = cfg["method"].get("normalize_disc_input", True)
        self.discriminator = DomainDiscriminator(
            in_dim=feature_dim,
            hidden=cfg["method"]["disc_hidden"],
            dropout=cfg["method"]["disc_dropout"],
        )
        self.domain_ce = nn.CrossEntropyLoss()         # 2-class domain loss

    def extra_modules(self):
        
        return [self.discriminator]

    def compute_loss(self, source_feats, source_logits, source_labels,
                     target_feats, target_logits, alpha):
        loss_cls = self.classification_loss(source_logits, source_labels)

        
        feats = torch.cat([source_feats, target_feats], dim=0)   # (Ns+Nt, 512)
        
        if self.normalize_disc_input:
            feats = F.normalize(feats, p=2, dim=1)               # unit-norm per example
        feats_rev = grad_reverse(feats, alpha)                    # GRL applied
        domain_logits = self.discriminator(feats_rev)            # (Ns+Nt, 2)

        ns, nt = source_feats.shape[0], target_feats.shape[0]
        domain_labels = torch.cat([
            torch.zeros(ns, dtype=torch.long, device=feats.device),
            torch.ones(nt, dtype=torch.long, device=feats.device),
        ])
        loss_domain = self.domain_ce(domain_logits, domain_labels)

        
        loss = loss_cls + self.domain_loss_weight * loss_domain

        
        with torch.no_grad():
            dom_pred = domain_logits.argmax(dim=1)
            dom_acc = (dom_pred == domain_labels).float().mean().item()

        logs = {"loss_total": loss.item(), "loss_cls": loss_cls.item(),
                "loss_align": loss_domain.item(), "domain_acc": dom_acc,
                "alpha": alpha}
        return loss, logs
