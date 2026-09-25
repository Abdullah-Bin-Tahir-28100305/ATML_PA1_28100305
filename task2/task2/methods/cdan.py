# =============================================================================
# task2/methods/cdan.py  — CDAN: class-conditional adversarial alignment (Step 4)
# -----------------------------------------------------------------------------
# ML CONCEPT — CDAN (Long et al. 2018):
#   DANN aligns only the marginal feature distribution, so it can accidentally
#   line up DIFFERENT classes across domains. CDAN fixes this by conditioning the
#   discriminator on the classifier's prediction as well as the feature, through
#   the MULTILINEAR MAP:
#         g(x) = vec( f  (x)  p ),
#   the flattened OUTER PRODUCT of the 512-d feature f = F(x) and the 7-d softmax
#   probability p = softmax(C(f)). This couples 'where in feature space' with
#   'which class', so aligning g pushes the model to match SEMANTICALLY
#   corresponding regions (dog-to-dog), not just the overall cloud. Result: the
#   discriminator input has dimension 512 * num_classes.
#
# STRICT SPEC REQUIREMENTS (enforced here):
#   "Do not use entropy conditioning or detach f or p in the required
#    implementation." So:
#     - NO entropy weighting of examples;
#     - gradients flow through BOTH f and p (we do NOT call .detach()).
#   Everything else (hidden width, ReLU, dropout, GRL schedule, unit loss weight)
#   MIRRORS DANN, as the spec demands.
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


def multilinear_map(feats, probs):
    """Compute g = vec(f  p): the flattened outer product per example.

    feats: (N, d) 512-d features.  probs: (N, k) softmax class probabilities.
    For each example i we form the outer product f_i p_i^T (a d x k matrix) and
    flatten it to a (d*k,) vector. Batched via einsum, then reshaped.
    NOTE: we deliberately do NOT detach feats or probs, so gradients propagate
    into BOTH the backbone (via f) and the classifier (via p), as required.
    """
    N, d = feats.shape
    k = probs.shape[1]
    # 'nd,nk->ndk' gives the per-example outer product; reshape to (N, d*k).
    outer = torch.einsum("nd,nk->ndk", feats, probs)
    return outer.reshape(N, d * k)


class CDAN(BaseMethod):
    def __init__(self, cfg, feature_dim: int, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.domain_loss_weight = cfg["method"]["domain_loss_weight"]
        # Whether to L2-normalize the multilinear map per example (stability).
        # Not entropy conditioning and not a detach, so the required
        # implementation is preserved; it only rescales the conditioning vector.
        self.normalize_multilinear = cfg["method"].get("normalize_multilinear", True)
        # Discriminator input dimension is 512 * num_classes (the multilinear map).
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
        # 1) Classification loss: SOURCE ONLY (same as DANN).
        loss_cls = self.classification_loss(source_logits, source_labels)

        # 2) Build the multilinear conditioning inputs for source and target.
        #    p = softmax over class logits (the classifier's soft prediction).
        source_probs = F.softmax(source_logits, dim=1)     # (Ns, k)
        target_probs = F.softmax(target_logits, dim=1)     # (Nt, k)
        g_source = multilinear_map(source_feats, source_probs)   # (Ns, 512*k)
        g_target = multilinear_map(target_feats, target_probs)   # (Nt, 512*k)

        # 3) Domain loss on the conditioned features, via the GRL (same as DANN).
        g = torch.cat([g_source, g_target], dim=0)
        # Optional per-example L2 normalization of the multilinear map. This keeps
        # the discriminator input in a stable magnitude range (the raw outer
        # product of unscaled ResNet features can be large and caused the run to
        # diverge). Gradients still flow through f and p (no detach), and no
        # entropy weighting is used, so the required CDAN construction is intact.
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
