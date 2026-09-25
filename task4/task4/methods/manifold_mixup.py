

import numpy as np
import torch


def sample_lambda(alpha, size, device):
    lam = np.random.beta(alpha, alpha, size=size).astype("float32")
    return torch.from_numpy(lam).to(device)


def make_different_class_pairs(labels):
    
    labels = labels.detach().cpu().numpy()
    n = len(labels)
    rng = np.random.default_rng()
    partner = rng.permutation(n)
    valid = labels[partner] != labels
    for i in np.where(~valid)[0]:
        candidates = np.where(labels != labels[i])[0]
        if len(candidates) > 0:
            partner[i] = rng.choice(candidates)
            valid[i] = True
    return partner, valid


def manifold_mixup_features(h, partner, lam):
  
    lam_view = lam.view(-1, 1, 1, 1)                    
    h_partner = h[partner]                             
    return lam_view * h + (1.0 - lam_view) * h_partner
