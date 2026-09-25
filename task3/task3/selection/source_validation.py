# =============================================================================
# task3/selection/source_validation.py
# -----------------------------------------------------------------------------
# PURPOSE: the SOURCE-ONLY model-selection logic. In DG we may NOT look at the
# target (Sketch) when choosing checkpoints, so selection is driven entirely by
# source-validation performance:
#   * per-source-domain accuracy and macro-F1,
#   * their MEAN (mean-source performance) — the checkpoint-selection signal,
#   * their WORST (worst-source performance) — a diagnostic that reveals whether
#     one environment is being neglected.
#
# ML CONCEPT — WHY SOURCE-ONLY SELECTION MATTERS (Gulrajani & Lopez-Paz 2021):
#   Peeking at the target to pick a checkpoint is 'target leakage' and inflates
#   DG results. Honest DG selects using only observed sources, then evaluates the
#   unseen domain once. This file is that firewall: it never sees Sketch.
#
# LINKS: metrics reused by train.py (early stopping) and evaluate_sketch.py.
# =============================================================================

import numpy as np
import torch
from sklearn.metrics import f1_score


def top1_accuracy(preds, labels):
    preds = np.asarray(preds); labels = np.asarray(labels)
    return float((preds == labels).mean())


def macro_f1(preds, labels, num_classes):
    preds = np.asarray(preds); labels = np.asarray(labels)
    return float(f1_score(labels, preds, average="macro",
                          labels=list(range(num_classes)), zero_division=0))


@torch.no_grad()
def collect_predictions(backbone, head, loader, device):
    """Run backbone+head over a loader; return (preds, labels) numpy arrays."""
    preds, labels = [], []
    for imgs, ys in loader:
        imgs = imgs.to(device)
        logits = head(backbone(imgs))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(ys.numpy())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(labels)


@torch.no_grad()
def evaluate_source_val(backbone, head, val_loaders, device, num_classes):
    """Evaluate all source-VAL domains; return per-domain metrics + mean + worst.

    Returns:
      {
        "per_domain": {domain: {"top1":..,"macro_f1":..}},
        "mean_top1":.., "mean_macro_f1":..,
        "worst_top1":.., "worst_macro_f1":..,
      }
    'mean_macro_f1' is the checkpoint-selection signal (spec). 'worst_*' is the
    worst-source diagnostic.
    """
    per_domain = {}
    for d, ld in val_loaders.items():
        preds, labels = collect_predictions(backbone, head, ld, device)
        per_domain[d] = {"top1": top1_accuracy(preds, labels),
                         "macro_f1": macro_f1(preds, labels, num_classes)}
    top1s = [v["top1"] for v in per_domain.values()]
    f1s = [v["macro_f1"] for v in per_domain.values()]
    return {
        "per_domain": per_domain,
        "mean_top1": float(np.mean(top1s)),
        "mean_macro_f1": float(np.mean(f1s)),
        "worst_top1": float(np.min(top1s)),         # lowest-performing source
        "worst_macro_f1": float(np.min(f1s)),
    }
