# =============================================================================
# task4/methods/manifold_mixup.py
# -----------------------------------------------------------------------------
# PURPOSE: the MANIFOLD MIXUP helper used by PROSER's data-placeholder branch.
#
# ML CONCEPT — MANIFOLD MIXUP as a source of PROXY UNKNOWNS (Zhou et al. 2021):
#   We have no real unknown examples during training. So we synthesize "unknown-
#   like" points by mixing the INTERMEDIATE FEATURES of two examples from
#   DIFFERENT known classes:
#       h_i = phi_pre(x_i),  h_j = phi_pre(x_j)
#       lambda ~ Beta(2,2)
#       h_tilde = lambda * h_i + (1 - lambda) * h_j     (only for y_i != y_j)
#   Passing h_tilde through the rest of the network yields a feature that lies
#   BETWEEN two class regions — a plausible "not clearly any known class" point.
#   PROSER trains these mixed points toward the DUMMY classifiers, teaching the
#   model to flag such in-between regions as unknown. Spec: mix AFTER layer2 and
#   BEFORE layer3, with lambda ~ Beta(2,2).
#
# LINKS: proser.py calls mix_pairs_different_class + the model's phi_pre/phi_post.
# =============================================================================

import numpy as np
import torch


def sample_lambda(alpha, size, device):
    """Sample lambda ~ Beta(alpha, alpha) (alpha=2 -> Beta(2,2)) as a tensor."""
    lam = np.random.beta(alpha, alpha, size=size).astype("float32")
    return torch.from_numpy(lam).to(device)


def make_different_class_pairs(labels):
    """Pair each example i with a partner j of a DIFFERENT class (spec: y_i != y_j).

    Returns an index array `partner` such that labels[partner[i]] != labels[i]
    whenever possible. We build it by a random permutation and, for any position
    whose partner happens to share the class, reselect a valid partner. If a batch
    somehow contained a single class (not the case with domain-balanced CIFAR
    batches), those positions are left paired with themselves and filtered out by
    the caller via the returned validity mask.
    """
    labels = labels.detach().cpu().numpy()
    n = len(labels)
    rng = np.random.default_rng()
    partner = rng.permutation(n)
    valid = labels[partner] != labels
    # Try to fix invalid (same-class) pairings by searching for a different class.
    for i in np.where(~valid)[0]:
        candidates = np.where(labels != labels[i])[0]
        if len(candidates) > 0:
            partner[i] = rng.choice(candidates)
            valid[i] = True
    return partner, valid


def manifold_mixup_features(h, partner, lam):
    """Mix intermediate activations: h_tilde = lam*h_i + (1-lam)*h_partner.

    h:       (N, C, H, W) layer2 activations (from model.phi_pre).
    partner: (N,) index of the partner example for each i.
    lam:     (N,) mixing coefficients in [0,1].
    Broadcasting reshapes lam to (N,1,1,1) so it scales each example's whole map.
    """
    lam_view = lam.view(-1, 1, 1, 1)                    # broadcast over C,H,W
    h_partner = h[partner]                             # partner activations
    return lam_view * h + (1.0 - lam_view) * h_partner
