# =============================================================================
# task3/evaluation/sharpness.py
# -----------------------------------------------------------------------------
# PURPOSE: the COMMON local-sharpness proxy applied to all three models (ERM,
# DAN-DG, SAM), so their local stability can be compared on one fixed yardstick.
#
# ML CONCEPT — THE SHARPNESS PROXY (spec):
#   Take a FIXED validation batch of 32 examples per source (seed 6304), put the
#   model in EVAL mode, and measure how much the cross-entropy RISES after one
#   normalized gradient-ASCENT step of radius rho=0.05:
#       Delta_sharp = L_val(theta + eps) - L_val(theta),   eps = rho * g/||g||,
#   where g = grad_theta L_val(theta). A LARGER Delta_sharp means the loss climbs
#   faster in the worst local direction => a SHARPER (less locally stable) point;
#   a SMALLER value => flatter/more stable. SAM optimizes for flatness, so we
#   expect (but must verify) SAM to have the smallest Delta_sharp.
#
# IMPORTANT CAVEATS (spec 'What to watch for'):
#   * This is a STANDARDIZED LOCAL diagnostic under ONE specified perturbation,
#     not a proof that a model's entire loss landscape is globally flatter.
#   * The batch, seed, and rho are fixed and identical across models so the
#     comparison is fair. We use eval mode so BN/dropout are deterministic.
#
# NOTE: it uses only SOURCE validation data (no Sketch), so it is a legitimate
# source-side diagnostic.
# =============================================================================

import numpy as np
import torch
import torch.nn as nn


def _fixed_source_batch(val_sets, cfg, device):
    """Build ONE fixed batch of `batch_per_source` images from EACH source.

    Deterministic selection via seed 6304: we take the first N indices from a
    seeded permutation of each source-val subset, so the batch is identical every
    run and across models (spec: "Select a fixed validation batch ... using seed
    6304").
    """
    per = cfg["sharpness"]["batch_per_source"]         # 32
    rng = np.random.default_rng(cfg["seed"])
    imgs, labels = [], []
    for domain, ds in val_sets.items():
        n = len(ds)
        idx = rng.permutation(n)[:per]                 # deterministic subset
        for i in idx:
            x, y = ds[int(i)]
            imgs.append(x); labels.append(int(y))
    X = torch.stack(imgs).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    return X, y


def compute_sharpness_proxy(backbone, head, val_sets, cfg, device):
    """Return Delta_sharp for the given model on the fixed source-val batch.

    Steps:
      1) build the fixed 32-per-source batch (seed 6304),
      2) EVAL mode; compute L_val(theta) and its gradient g w.r.t. all params,
      3) eps = rho * g/||g||; apply eps to the parameters (ascent),
      4) recompute L_val(theta+eps); Delta_sharp = L2 - L1; restore theta.
    """
    rho = cfg["sharpness"]["rho"]                       # 0.05
    X, y = _fixed_source_batch(val_sets, cfg, device)
    criterion = nn.CrossEntropyLoss()

    # 2) EVAL mode so BN/dropout are deterministic (spec: "place the model in
    #    evaluation mode"). We DO need gradients, so no torch.no_grad() here.
    backbone.eval(); head.eval()
    params = [p for p in list(backbone.parameters()) + list(head.parameters())
              if p.requires_grad]

    # base loss L_val(theta)
    logits = head(backbone(X))
    loss1 = criterion(logits, y)
    grads = torch.autograd.grad(loss1, params)         # g = grad L_val(theta)

    # 3) normalized ascent perturbation eps = rho * g / ||g||
    grad_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
    scale = rho / (grad_norm + 1e-12)
    eps = [g * scale for g in grads]

    with torch.no_grad():
        for p, e in zip(params, eps):
            p.add_(e)                                  # theta -> theta + eps
        # 4) perturbed loss L_val(theta+eps)
        loss2 = criterion(head(backbone(X)), y)
        delta = (loss2 - loss1).item()
        for p, e in zip(params, eps):
            p.sub_(e)                                  # restore theta

    return {"delta_sharp": float(delta),
            "loss_base": float(loss1.item()),
            "loss_perturbed": float(loss2.item()),
            "rho": rho}
