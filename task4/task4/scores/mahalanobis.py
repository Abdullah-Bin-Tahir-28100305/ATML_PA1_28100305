# =============================================================================
# task4/scores/mahalanobis.py  — Mahalanobis feature-distance novelty score
# -----------------------------------------------------------------------------
# ML CONCEPT — MAHALANOBIS DISTANCE OOD (Lee et al. 2018 style):
#   Model each known class as a Gaussian in feature space. Estimate a class mean
#   mu_c for each class and ONE shared covariance (here DIAGONAL, per spec) from
#   the UNAUGMENTED CIFAR-10 TRAIN features. The novelty of a test feature f(x)
#   is its distance to the NEAREST class Gaussian:
#       u_Mah(x) = min_c ( f(x) - mu_c )^T Sigma^{-1} ( f(x) - mu_c )
#   Large distance from every known cluster => likely unknown. Unlike the logit-
#   based scores, this is a pure FEATURE-SPACE geometry signal, so it can catch
#   unknowns the classifier is confidently wrong about.
#
# SPEC DETAILS (implemented exactly):
#   * class means mu_c and ONE SHARED DIAGONAL covariance Sigma;
#   * estimated from UNAUGMENTED CIFAR-10 TRAIN features;
#   * add 1e-6 to every diagonal entry (regularize the inverse).
#
# This score is fit ONCE (fit_mahalanobis) using training features and then
# applied to any features (test knowns / near / far). It reads the SAME saved
# features as everything else.
# =============================================================================

import numpy as np


def fit_mahalanobis(train_features, train_labels, num_classes, eps=1e-6):
    """Estimate per-class means + one shared diagonal covariance (inverse).

    Args:
      train_features: (N,512) UNAUGMENTED CIFAR-10 train features.
      train_labels:   (N,) integer class labels.
      eps:            added to every diagonal entry of the covariance (spec 1e-6).
    Returns a dict with the class means and the inverse diagonal variance vector.
    """
    X = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_labels)
    d = X.shape[1]

    # 1) per-class means mu_c
    means = np.zeros((num_classes, d), dtype=np.float64)
    for c in range(num_classes):
        means[c] = X[y == c].mean(axis=0)

    # 2) ONE SHARED covariance from the pooled, mean-centered features. We keep
    #    only the DIAGONAL (per-feature variance), as the spec specifies a shared
    #    DIAGONAL covariance. Centering uses each sample's OWN class mean (the
    #    within-class scatter), which is the standard tied-covariance estimate.
    centered = X - means[y]                              # subtract class mean
    var = (centered ** 2).mean(axis=0)                  # (d,) diagonal variances
    var = var + eps                                     # spec: +1e-6 on diagonal
    inv_var = 1.0 / var                                 # Sigma^{-1} diagonal

    return {"means": means, "inv_var": inv_var, "num_classes": num_classes}


def score(logits, features=None, params=None):
    """u_Mah(x) = min_c (f-mu_c)^T diag(inv_var) (f-mu_c). Larger = more novel.

    `logits` is accepted for a uniform score interface but unused. `params` is the
    dict returned by fit_mahalanobis; it MUST be provided (fit on train features).
    """
    if params is None:
        raise ValueError("Mahalanobis score requires fitted params "
                         "(call fit_mahalanobis on train features first).")
    F = np.asarray(features, dtype=np.float64)          # (N,512) test features
    means = params["means"]                             # (C,512)
    inv_var = params["inv_var"]                         # (512,)

    # For each class c, squared Mahalanobis distance with diagonal Sigma^{-1}:
    #   sum_j inv_var[j] * (f_j - mu_c_j)^2 . Vectorized over classes.
    N = F.shape[0]; C = means.shape[0]
    dists = np.empty((N, C), dtype=np.float64)
    for c in range(C):
        diff = F - means[c]                             # (N,512)
        dists[:, c] = (diff * diff * inv_var).sum(axis=1)
    return dists.min(axis=1)                            # (N,) nearest-class distance
