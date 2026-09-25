

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from methods.base_method import BaseMethod
    from models.domain_discriminator import DomainDiscriminator, grad_reverse
except ImportError:
    from .base_method import BaseMethod
    from ..models.domain_discriminator import DomainDiscriminator, grad_reverse


def multilinear_map(feats, probs):
   
    N, d = feats.shape
    k = probs.shape[1]

    outer = torch.einsum("nd,nk->ndk", feats, probs)
    return outer.reshape(N, d * k)


class CDAN(BaseMethod):
    def __init__(self, cfg, feature_dim: int, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.domain_loss_weight = cfg["method"]["domain_loss_weight"]
        
        self.normalize_multilinear = cfg["method"].get("normalize_multilinear", True)

        disc_in = feature_dim * num_classes
        self.discriminator = DomainDiscriminator(
            in_dim=disc_in,
            hidden=cfg["method"]["disc_hidden"],
            dropout=cfg["method"]["disc_dropout"],
        )
        self.domain_ce = nn.CrossEntropyLoss()

    def extra_modules(self):
        return [self.discriminator]

    def compute_loss(self, source_feats, source_logits, source_labels,
                     target_feats, target_logits, alpha):

        loss_cls = self.classification_loss(source_logits, source_labels)

        
        source_probs = F.softmax(source_logits, dim=1)     # (Ns, k)
        target_probs = F.softmax(target_logits, dim=1)     # (Nt, k)
        g_source = multilinear_map(source_feats, source_probs)   # (Ns, 512*k)
        g_target = multilinear_map(target_feats, target_probs)   # (Nt, 512*k)

        g = torch.cat([g_source, g_target], dim=0)
        
        if self.normalize_multilinear:
            g = F.normalize(g, p=2, dim=1)                        # unit-norm per example
        g_rev = grad_reverse(g, alpha)
        domain_logits = self.discriminator(g_rev)

        ns, nt = source_feats.shape[0], target_feats.shape[0]
        domain_labels = torch.cat([
            torch.zeros(ns, dtype=torch.long, device=g.device),
            torch.ones(nt, dtype=torch.long, device=g.device),
        ])
        loss_domain = self.domain_ce(domain_logits, domain_labels)

        loss = loss_cls + self.domain_loss_weight * loss_domain

        with torch.no_grad():
            dom_acc = (domain_logits.argmax(1) == domain_labels).float().mean().item()

        logs = {"loss_total": loss.item(), "loss_cls": loss_cls.item(),
                "loss_align": loss_domain.item(), "domain_acc": dom_acc,
                "alpha": alpha}
        return loss, logs
