

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@torch.no_grad()
def _extract_features(backbone, loader, device, max_features):

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
