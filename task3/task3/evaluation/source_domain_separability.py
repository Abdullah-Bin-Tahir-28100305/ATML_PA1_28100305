# =============================================================================
# task3/evaluation/source_domain_separability.py
# -----------------------------------------------------------------------------
# PURPOSE: measure how much SOURCE-DOMAIN information remains in a model's
# features, by trying to classify a feature into WHICH of the three source
# domains it came from (Photo / Art Painting / Cartoon).
#
# ML CONCEPT — SOURCE-DOMAIN SEPARABILITY (the DG analogue of Task 2's probe):
#   Freeze the backbone, extract balanced features from the three source-VAL
#   sets, and train a MULTINOMIAL (3-way) logistic regression (C=1) on a 70/30
#   split (seed 6304) to predict the domain. Held-out accuracy is the
#   separability score:
#     * ~33.3% (chance for 3 classes) => the three sources are hard to tell apart
#       (strong cross-source invariance);
#     * high => domains remain easily distinguishable.
#   Spec caution ('What to watch for'): lower separability indicates stronger
#   invariance across observed sources, but does NOT by itself establish that
#   class information or unseen-domain (Sketch) performance improved. So we report
#   it alongside Sketch accuracy rather than assuming lower is better.
#
# NOTE: this uses only SOURCE features — it never touches Sketch, so it is a
# legitimate source-side diagnostic that can be computed before final evaluation.
# =============================================================================

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@torch.no_grad()
def _extract_features(backbone, loader, device, max_features):
    """Collect up to max_features feature vectors from a loader (no labels)."""
    feats, count = [], 0
    for imgs, _ in loader:
        f = backbone(imgs.to(device)).cpu().numpy()
        feats.append(f); count += f.shape[0]
        if count >= max_features:
            break
    feats = np.concatenate(feats, axis=0)
    return feats[:max_features]


def compute_source_domain_separability(backbone, source_val_loaders, cfg, device):
    """3-way logistic regression predicting the source domain from features.

    source_val_loaders: dict {domain_name: DataLoader over that source's val set}.
    Returns dict with the held-out separability accuracy and the group size.
    Steps:
      1) extract EQUAL numbers of features per source domain (balanced),
      2) label each by its domain index (0/1/2),
      3) 70/30 stratified split (seed 6304), fit balanced multinomial LogReg C=1,
      4) held-out accuracy = source-domain separability (chance = 1/3).
    """
    max_per = cfg["domain_separability"]["max_features_per_group"]
    domains = list(source_val_loaders.keys())

    # 1) balanced feature groups, one per source domain.
    feats_per_domain = [_extract_features(backbone, source_val_loaders[d], device, max_per)
                        for d in domains]
    n = min(f.shape[0] for f in feats_per_domain)      # equalize counts
    feats_per_domain = [f[:n] for f in feats_per_domain]

    # 2) stack features + domain labels (0,1,2).
    X = np.concatenate(feats_per_domain, axis=0)
    y = np.concatenate([np.full(n, i) for i in range(len(domains))]).astype(int)

    # 3) stratified 70/30 split, deterministic via seed 6304.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["domain_separability"]["test_frac"],
        random_state=cfg["seed"], stratify=y)

    # A softmax (multinomial) logistic regression over the 3 domain classes.
    # NOTE: scikit-learn >= 1.7 removed the explicit `multi_class` argument and
    # uses multinomial by default for multiclass problems, so we simply omit it;
    # on older versions the default 'auto' also selects multinomial for LBFGS.
    # This keeps the code correct across sklearn versions.
    clf = LogisticRegression(
        C=cfg["domain_separability"]["C"],
        class_weight=cfg["domain_separability"]["class_weight"],
        max_iter=2000)
    clf.fit(X_tr, y_tr)

    return {"source_domain_separability": float(clf.score(X_te, y_te)),
            "n_per_group": int(n),
            "chance_level": 1.0 / len(domains)}        # 0.333 for 3 sources
