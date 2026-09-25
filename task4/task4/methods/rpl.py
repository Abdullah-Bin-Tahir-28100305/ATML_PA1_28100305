# =============================================================================
# task4/methods/rpl.py  — Reciprocal Point Learning (Step 5, OPTIONAL)
# -----------------------------------------------------------------------------
# ML CONCEPT — RPL (Chen et al. 2020, "Learning Open Set Network with
# Discriminative Reciprocal Points"):
#   Ordinary classifiers learn a prototype for what each class IS. RPL instead
#   learns, for each class k, a RECIPROCAL POINT P_k representing what class k is
#   NOT (the "otherness" / extra-class space for k). A feature f(x) of class k is
#   trained to be FAR from P_k, so:
#       larger distance to P_k  =>  stronger evidence for class k.
#   An OPEN-SPACE regularization term keeps features within a bounded radius R of
#   the reciprocal points so unknowns cannot hide arbitrarily far out.
#
# The four required clarifications (spec):
#   1. HOW reciprocal points are represented/learned: as learnable vectors P_k in
#      the 512-d feature space (one per class here), trained jointly with the net.
#   2. HOW distance -> known-class score: the logit for class k is the distance
#      d(f, P_k) (so being far from P_k => high score for k); classification is
#      argmax_k d(f, P_k), trained with cross-entropy over these distance-logits.
#   3. HOW open-space regularization constrains the space: penalize
#      (d(f, P_{true}) - R)^2 so the true-class distance stays near a radius R,
#      bounding how far known features can drift (keeps open space compact).
#   4. WHICH score rejects unknowns: the UNKNOWNNESS score is the negative max
#      distance-logit, u_RPL(x) = - max_k d(f(x), P_k) — an input close to every
#      reciprocal point (small max distance) is likely unknown.
#
# Trained on CIFAR-10 only (no CIFAR-100 influence). Uses the base SGD recipe.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class RPLModel(nn.Module):
    """CifarResNet18 backbone + learnable reciprocal points, one per class.

    We use squared Euclidean distance d(f, P_k) = ||f - P_k||^2 as the per-class
    'evidence' logit. Classification maximizes the distance to the correct class's
    reciprocal point; rejection looks at whether the max distance is large enough.
    """
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        d = backbone.feature_dim
        # Reciprocal points P_k as learnable parameters, initialized small/random.
        self.reciprocal = nn.Parameter(torch.randn(num_classes, d) * 0.1)

    def distances(self, x):
        """Return (features f(x), distance-logits d(f,P_k)) for all classes.

        distance-logits[:,k] = ||f - P_k||^2  (larger => more evidence for k).
        """
        feats = self.backbone.forward_features(x)       # (N,512)
        # pairwise squared distances between each feature and each reciprocal point
        # ||f - P||^2 = ||f||^2 + ||P||^2 - 2 f.P
        f_sq = (feats * feats).sum(1, keepdim=True)     # (N,1)
        p_sq = (self.reciprocal * self.reciprocal).sum(1).unsqueeze(0)  # (1,C)
        cross = feats @ self.reciprocal.t()             # (N,C)
        d2 = (f_sq + p_sq - 2.0 * cross).clamp(min=0.0) # (N,C) squared distances
        return feats, d2


class RPL:
    name = "rpl"

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_classes = cfg["known"]["num_classes"]
        self.lambda_open = cfg["method"]["lambda_open"]     # open-space weight
        self.R = cfg["method"]["open_space_radius"]         # target radius
        self.ce = nn.CrossEntropyLoss()

    def loss(self, model, images, labels):
        """CE over distance-logits + open-space regularization.

        - CE(distance-logits, labels): pushes f AWAY from its own reciprocal point
          (maximizes d(f,P_true) relative to others).
        - open-space term: (d(f,P_true) - R)^2 keeps the true-class distance near
          R, bounding the feature space so unknowns cannot drift out freely.
        """
        feats, d2 = model.distances(images)             # (N,512),(N,C)
        loss_cls = self.ce(d2, labels)                  # distances act as logits
        # true-class distance for the open-space regularizer
        d_true = d2[torch.arange(d2.size(0)), labels]   # (N,)
        loss_open = ((d_true - self.R) ** 2).mean()
        total = loss_cls + self.lambda_open * loss_open
        return total, {"loss_total": total.item(), "loss_cls": loss_cls.item(),
                       "loss_open": loss_open.item()}


def rpl_unknownness(distance_logits):
    """u_RPL(x) = - max_k d(f(x), P_k). Larger => more novel.

    An unknown tends to be close to ALL reciprocal points (small distances), so
    its max distance-logit is small and its unknownness (negative of it) is large.
    """
    import numpy as np
    d = np.asarray(distance_logits, dtype=np.float64)
    return -d.max(axis=1)
