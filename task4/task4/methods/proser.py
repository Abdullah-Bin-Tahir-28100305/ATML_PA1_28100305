# =============================================================================
# task4/methods/proser.py  — PROSER: classifier + data placeholders (Step 4)
# -----------------------------------------------------------------------------
# ML CONCEPT — PROSER (Zhou et al. 2021, "Learning Placeholders for Open-Set
# Recognition"):
#   Take a trained closed-set model and append K DUMMY output units ("classifier
#   placeholders"). Two losses teach those dummies to represent "unknown-like"
#   responses using ONLY known data:
#
#   (A) CLASSIFIER-PLACEHOLDER loss (weight beta): for a known example (x, y),
#       * the TRUE class should stay the largest overall response
#         -> CE over [known logits ; max dummy logit] with target y;
#       * once the TRUE class is REMOVED, a DUMMY should be the strongest
#         remaining response
#         -> CE over [known logits with y masked to -inf ; max dummy logit] with
#            target = "the dummy slot". This carves out a reject region next to
#            each known class.
#
#   (B) DATA-PLACEHOLDER loss (weight gamma): synthesize PROXY unknowns by
#       manifold mixup of two DIFFERENT-class examples (after layer2, before
#       layer3, lambda~Beta(2,2)); train those mixed points toward the DUMMIES
#       (target = dummy slot) rather than either original class. This provides
#       examples of "in-between, not clearly known" regions to reject.
#
#   Spec: split each mini-batch in half -> first half for (A), second half for
#   (B). beta=1, gamma=0.1. At test time, the placeholder-based DETECTION score
#   combines the strongest dummy response with the known-class responses.
#
# We follow the paper's combination. The exact dummy handling: we collapse the K
# dummy logits to their MAX (the paper treats the placeholder response as the
# strongest dummy), giving an augmented (num_known + 1) logit vector for the CE
# terms. Known-class classification (CSA) always uses ONLY the 10 known logits.
#
# LINKS: models/resnet_cifar (phi_pre/phi_post for mixup), manifold_mixup helper.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from methods.manifold_mixup import (sample_lambda, make_different_class_pairs,
                                        manifold_mixup_features)
except ImportError:
    from .manifold_mixup import (sample_lambda, make_different_class_pairs,
                                 manifold_mixup_features)


class ProserModel(nn.Module):
    """Wraps a CifarResNet18 and adds K dummy classifier heads on the 512-d feature.

    - known logits come from the ORIGINAL classifier (net.fc), 10 outputs;
    - dummy logits come from a NEW linear layer with K outputs.
    Both read the SAME penultimate feature f(x). Known-class prediction uses only
    the 10 known logits; the dummies are used only for the reject signal.
    """
    def __init__(self, backbone, num_known, num_dummy):
        super().__init__()
        self.backbone = backbone                       # a CifarResNet18
        self.num_known = num_known
        self.num_dummy = num_dummy
        self.dummy = nn.Linear(backbone.feature_dim, num_dummy)  # placeholder heads

    def forward(self, x):
        """Return (known_logits (N,10), dummy_logits (N,K)) from raw images."""
        feats = self.backbone.forward_features(x)
        known = self.backbone.forward_from_features(feats)   # (N,10)
        dummy = self.dummy(feats)                            # (N,K)
        return known, dummy

    def logits_from_features(self, feats):
        return self.backbone.forward_from_features(feats), self.dummy(feats)

    def known_logits(self, x):
        """Only the 10 known-class logits (used for CSA + MLS comparison)."""
        # A single forward pass through the backbone, then the known classifier.
        feats = self.backbone.forward_features(x)
        return self.backbone.forward_from_features(feats)


class PROSER:
    name = "proser"

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_known = cfg["known"]["num_classes"]
        self.num_dummy = cfg["method"]["num_dummy"]
        self.beta = cfg["method"]["beta"]              # classifier-placeholder weight
        self.gamma = cfg["method"]["gamma"]            # data-placeholder weight
        self.alpha = cfg["method"]["mixup_alpha"]      # Beta(alpha,alpha)
        self.ce = nn.CrossEntropyLoss()

    # ---- (A) classifier-placeholder loss on real known examples ----
    def classifier_placeholder_loss(self, model, images, labels):
        """CE(keep-true-class) + CE(true-class-masked -> dummy)."""
        known, dummy = model(images)                   # (N,10), (N,K)
        dummy_max = dummy.max(dim=1, keepdim=True).values          # (N,1) strongest dummy
        # augmented logits: [10 known ; strongest dummy] -> (N, 11)
        aug = torch.cat([known, dummy_max], dim=1)
        dummy_slot = self.num_known                     # index of the dummy column (=10)

        # Term 1: true class should remain the largest among [known ; dummy].
        loss_keep = self.ce(aug, labels)

        # Term 2: with the TRUE class masked out, a dummy should win.
        masked = aug.clone()
        masked[torch.arange(masked.size(0)), labels] = float("-inf")  # remove true class
        target_dummy = torch.full_like(labels, dummy_slot)            # target = dummy slot
        loss_reject = self.ce(masked, target_dummy)

        return loss_keep + loss_reject

    # ---- (B) data-placeholder loss on manifold-mixed proxy unknowns ----
    def data_placeholder_loss(self, model, images, labels):
        """Manifold-mixup two different-class examples; train mix toward the dummy."""
        device = images.device
        backbone = model.backbone
        # h = phi_pre(x) : activations AFTER layer2 (spec's mixup point).
        h = backbone.phi_pre(images)                    # (N, C, H, W)
        partner, valid = make_different_class_pairs(labels)
        partner_t = torch.as_tensor(partner, device=device, dtype=torch.long)
        lam = sample_lambda(self.alpha, size=h.size(0), device=device)  # ~Beta(2,2)
        h_mixed = manifold_mixup_features(h, partner_t, lam)            # (N,C,H,W)

        # continue through the rest of the net -> penultimate feature -> logits.
        feats_mixed = backbone.phi_post(h_mixed)        # (N,512)
        known_mixed = backbone.forward_from_features(feats_mixed)       # (N,10)
        dummy_mixed = model.dummy(feats_mixed)                          # (N,K)
        dummy_max = dummy_mixed.max(dim=1, keepdim=True).values
        aug = torch.cat([known_mixed, dummy_max], dim=1)               # (N,11)

        # mixed points (between two classes) should be predicted as the DUMMY slot.
        target_dummy = torch.full((aug.size(0),), self.num_known,
                                  dtype=torch.long, device=device)
        # only count pairs that are genuinely different-class (valid mask).
        valid_t = torch.as_tensor(valid, device=device)
        if valid_t.sum() == 0:
            return aug.new_zeros(())
        return self.ce(aug[valid_t], target_dummy[valid_t])

    def loss(self, model, images, labels):
        """Total PROSER loss on a mini-batch, split in half (spec).

        First half -> classifier-placeholder loss (beta), second half -> data-
        placeholder loss (gamma). This keeps the two objectives on disjoint
        examples each step, as the assignment specifies.
        """
        n = images.size(0)
        half = n // 2
        # split the mini-batch into two equal parts.
        x_a, y_a = images[:half], labels[:half]         # classifier placeholders
        x_b, y_b = images[half:], labels[half:]         # data placeholders

        loss_cls = self.classifier_placeholder_loss(model, x_a, y_a)
        loss_data = self.data_placeholder_loss(model, x_b, y_b)
        total = self.beta * loss_cls + self.gamma * loss_data
        return total, {"loss_total": total.item(),
                       "loss_classifier_placeholder": loss_cls.item(),
                       "loss_data_placeholder": float(loss_data.item())}


# -----------------------------------------------------------------------------
# PROSER placeholder-based DETECTION score (for evaluation).
# -----------------------------------------------------------------------------
def proser_detection_score(known_logits, dummy_logits):
    """Placeholder-based unknownness score combining dummies with known logits.

    ML CONCEPT: an input is 'unknown-like' when a DUMMY response is strong
    relative to the known responses. Following the reference combination, we use
    the strongest dummy minus the strongest known logit as the reject signal:
        u_PROSER(x) = max_k dummy_k(x) - max_c known_c(x)   (larger => more novel)
    A large value means a placeholder outresponds every known class.
    """
    import numpy as np
    known = np.asarray(known_logits, dtype=np.float64)
    dummy = np.asarray(dummy_logits, dtype=np.float64)
    return dummy.max(axis=1) - known.max(axis=1)        # (N,) unknownness
