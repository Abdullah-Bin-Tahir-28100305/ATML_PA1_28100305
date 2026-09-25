# =============================================================================
# task2/evaluation/domain_separability.py
# -----------------------------------------------------------------------------
# PURPOSE: measure how much RESIDUAL DOMAIN INFORMATION remains in a model's
# features after adaptation, by trying to classify a feature as source-vs-target
# with a simple logistic regression.
#
# ML CONCEPT — DOMAIN SEPARABILITY:
#   Freeze the backbone, extract features for equal numbers of source-validation
#   and target examples, then train a balanced logistic-regression classifier
#   (C=1) on a 70/30 split to predict domain (source=0/target=1). Its HELD-OUT
#   accuracy is the 'domain separability' score:
#     * ~50% (chance) => features carry little domain info (well aligned);
#     * high          => source and target are still easy to tell apart.
#   Spec 'What to watch for': lower separability means domain info is harder to
#   recover, NOT automatically that class info is preserved — so we compare this
#   score WITH target recognition rather than assuming lower is better.
#
# WHY LOGISTIC REGRESSION (not the adversarial discriminator):
#   It is a fixed, simple, reproducible probe evaluated AFTER training. It gives
#   an objective, method-independent read of residual domain information, whereas
#   the training-time discriminator is entangled with the adversarial game.
#
# LINKS:
#   - Called by evaluate_final.py once every checkpoint is fixed.
#   - Uses seed 6304 and a 70/30 split, per spec.
# =============================================================================

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@torch.no_grad()
def _extract_features(backbone, loader, device, max_features):
    """Collect up to `max_features` feature vectors from a loader (no labels)."""
    feats = []
    count = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        f = backbone(imgs).cpu().numpy()
        feats.append(f)
        count += f.shape[0]
        if count >= max_features:
            break
    feats = np.concatenate(feats, axis=0)
    return feats[:max_features]


def compute_domain_separability(backbone, source_val_loader, target_loader,
                                cfg, device):
    """Train a source-vs-target logistic regression on frozen features.

    Returns a dict with the held-out separability accuracy and the group sizes.
    Steps:
      1) Extract EQUAL numbers of source-val and target features (spec: "equal
         numbers of source-validation and target features").
      2) Label source=0, target=1; standardize is unnecessary for LogReg here.
      3) 70/30 stratified split (seed 6304), fit balanced LogReg with C=1.
      4) Report held-out accuracy = domain separability score.
    """
    max_per = cfg["domain_separability"]["max_features_per_group"]

    # 1) equal-sized feature groups.
    src = _extract_features(backbone, source_val_loader, device, max_per)
    tgt = _extract_features(backbone, target_loader, device, max_per)
    n = min(src.shape[0], tgt.shape[0])                # enforce equal counts
    src, tgt = src[:n], tgt[:n]

    # 2) build X, y with domain labels.
    X = np.concatenate([src, tgt], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)

    # 3) 70/30 split, stratified by domain label, deterministic via seed 6304.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["domain_separability"]["test_frac"],
        random_state=cfg["seed"], stratify=y)

    clf = LogisticRegression(
        C=cfg["domain_separability"]["C"],             # inverse regularization
        class_weight=cfg["domain_separability"]["class_weight"],  # 'balanced'
        max_iter=2000)                                 # ensure convergence
    clf.fit(X_tr, y_tr)

    # 4) held-out accuracy = domain separability. 0.5 == chance.
    score = float(clf.score(X_te, y_te))
    return {"domain_separability": score,
            "n_per_group": int(n),
            "chance_level": 0.5}
