
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@torch.no_grad()
def _extract_features(backbone, loader, device, max_features):

    feats, count = [], 0
    for imgs, _ in loader:
        f = backbone(imgs.to(device)).cpu().numpy()
        feats.append(f); count += f.shape[0]
        if count >= max_features:
            break
    feats = np.concatenate(feats, axis=0)
    return feats[:max_features]


def compute_source_domain_separability(backbone, source_val_loaders, cfg, device):
   
    max_per = cfg["domain_separability"]["max_features_per_group"]
    domains = list(source_val_loaders.keys())

    feats_per_domain = [_extract_features(backbone, source_val_loaders[d], device, max_per)
                        for d in domains]
    n = min(f.shape[0] for f in feats_per_domain)      # equalize counts
    feats_per_domain = [f[:n] for f in feats_per_domain]

    X = np.concatenate(feats_per_domain, axis=0)
    y = np.concatenate([np.full(n, i) for i in range(len(domains))]).astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["domain_separability"]["test_frac"],
        random_state=cfg["seed"], stratify=y)

   
    clf = LogisticRegression(
        C=cfg["domain_separability"]["C"],
        class_weight=cfg["domain_separability"]["class_weight"],
        max_iter=2000)
    clf.fit(X_tr, y_tr)

    return {"source_domain_separability": float(clf.score(X_te, y_te)),
            "n_per_group": int(n),
            "chance_level": 1.0 / len(domains)}        # 0.333 for 3 sources
