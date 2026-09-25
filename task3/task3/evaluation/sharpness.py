
import numpy as np
import torch
import torch.nn as nn


def _fixed_source_batch(val_sets, cfg, device):
   
    per = cfg["sharpness"]["batch_per_source"]         
    rng = np.random.default_rng(cfg["seed"])
    imgs, labels = [], []
    for domain, ds in val_sets.items():
        n = len(ds)
        idx = rng.permutation(n)[:per]                 
        for i in idx:
            x, y = ds[int(i)]
            imgs.append(x); labels.append(int(y))
    X = torch.stack(imgs).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    return X, y


def compute_sharpness_proxy(backbone, head, val_sets, cfg, device):
   
    rho = cfg["sharpness"]["rho"]                       
    X, y = _fixed_source_batch(val_sets, cfg, device)
    criterion = nn.CrossEntropyLoss()

    
    backbone.eval(); head.eval()
    params = [p for p in list(backbone.parameters()) + list(head.parameters())
              if p.requires_grad]

    logits = head(backbone(X))
    loss1 = criterion(logits, y)
    grads = torch.autograd.grad(loss1, params)         

    grad_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
    scale = rho / (grad_norm + 1e-12)
    eps = [g * scale for g in grads]

    with torch.no_grad():
        for p, e in zip(params, eps):
            p.add_(e)                                  

        loss2 = criterion(head(backbone(X)), y)
        delta = (loss2 - loss1).item()
        for p, e in zip(params, eps):
            p.sub_(e)                                  

    return {"delta_sharp": float(delta),
            "loss_base": float(loss1.item()),
            "loss_perturbed": float(loss2.item()),
            "rho": rho}
