# =============================================================================
# task2/evaluation/metrics.py
# -----------------------------------------------------------------------------
# PURPOSE: the scoring utilities used everywhere: top-1 accuracy, macro-F1, and a
# reusable routine to run a model over a loader and collect predictions/labels.
#
# ML CONCEPTS:
#   * TOP-1 ACCURACY: fraction of correctly classified examples. Headline metric.
#   * MACRO-F1: mean of per-class F1, weighting every class equally. PACS classes
#     are not perfectly balanced across domains and adaptation can help some
#     classes while hurting others, so macro-F1 exposes per-class imbalance that
#     plain accuracy hides. The spec selects checkpoints by MEAN source-validation
#     macro-F1, so this metric also drives model selection.
#
# LINKS:
#   - train.py calls evaluate_loader() on source-val loaders for early stopping.
#   - evaluate_final.py calls it on the target for final reporting.
#   - class_analysis.py reuses the collected predictions.
# =============================================================================

import numpy as np
import torch
from sklearn.metrics import f1_score


def top1_accuracy(preds, labels):
    """Fraction of examples where the predicted class equals the true label."""
    preds = np.asarray(preds); labels = np.asarray(labels)
    return float((preds == labels).mean())


def macro_f1(preds, labels, num_classes):
    """Macro-averaged F1 across all classes (each class weighted equally).

    We pass an explicit label set so classes absent from a particular batch still
    count (as 0 recall), keeping the macro average comparable across domains.
    """
    preds = np.asarray(preds); labels = np.asarray(labels)
    return float(f1_score(labels, preds, average="macro",
                          labels=list(range(num_classes)), zero_division=0))


@torch.no_grad()
def collect_predictions(backbone, head, loader, device):
    """Run backbone+head over a loader; return (preds, labels) numpy arrays.

    IMPORTANT: the caller is responsible for putting the model in the correct
    mode. For pure evaluation we set full eval mode; for source-val checks during
    training we still evaluate deterministically. Here we simply run forward in
    no-grad; mode is controlled outside so BN policy stays consistent.
    """
    preds, labels = [], []
    for imgs, ys in loader:
        imgs = imgs.to(device)
        feats = backbone(imgs)
        logits = head(feats)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(ys.numpy())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(labels)


@torch.no_grad()
def evaluate_loader(backbone, head, loader, device, num_classes):
    """Evaluate one loader; return dict with top1 and macro_f1 (+ raw arrays)."""
    preds, labels = collect_predictions(backbone, head, loader, device)
    if preds.size == 0:
        return {"top1": float("nan"), "macro_f1": float("nan"),
                "preds": preds, "labels": labels}
    return {"top1": top1_accuracy(preds, labels),
            "macro_f1": macro_f1(preds, labels, num_classes),
            "preds": preds, "labels": labels}


def mean_source_val_macro_f1(per_domain_results):
    """Average macro-F1 across the three source-validation domains.

    Spec: "select checkpoints using the mean macro-F1 across the three source
    validation splits." This is the SINGLE scalar early stopping monitors — and
    critically it uses SOURCE labels only, never target labels.
    """
    vals = [r["macro_f1"] for r in per_domain_results.values()]
    return float(np.mean(vals)) if vals else float("nan")
