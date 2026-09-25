# =============================================================================
# task3/evaluation/domain_metrics.py
# -----------------------------------------------------------------------------
# PURPOSE: final-evaluation metrics, including the ones that DO use the unseen
# target (Sketch) — but ONLY here, after all training/selection is frozen:
#   * top-1 accuracy and macro-F1 on a loader (reused for Sketch),
#   * per-class accuracy on the target,
#   * per-class change vs ERM, and dominant confusions.
#
# ML CONCEPT — WHY PER-CLASS ANALYSIS ON THE UNSEEN DOMAIN:
#   An aggregate Sketch accuracy can hide class-level failures: a method may lift
#   the average while collapsing one class (e.g. mapping most 'horse' sketches to
#   'dog'). Breaking Sketch accuracy down per class, and comparing each method to
#   ERM class by class, exposes those hidden failures (spec Required Evidence:
#   "Per-class Sketch changes and selected failures").
#
# LINKS: used by evaluate_sketch.py; the plain metrics reuse the selection ones.
# =============================================================================

import numpy as np
import torch

try:
    from selection.source_validation import top1_accuracy, macro_f1, collect_predictions
except ImportError:
    from ..selection.source_validation import top1_accuracy, macro_f1, collect_predictions


@torch.no_grad()
def evaluate_target(backbone, head, target_loader, device, num_classes):
    """Top-1 + macro-F1 + raw preds/labels on the target (Sketch)."""
    preds, labels = collect_predictions(backbone, head, target_loader, device)
    return {"top1": top1_accuracy(preds, labels) if preds.size else float("nan"),
            "macro_f1": macro_f1(preds, labels, num_classes) if preds.size else float("nan"),
            "preds": preds, "labels": labels}


def per_class_accuracy(preds, labels, num_classes):
    """dict class_index -> accuracy on that class's examples (NaN if class absent)."""
    preds = np.asarray(preds); labels = np.asarray(labels)
    out = {}
    for c in range(num_classes):
        mask = (labels == c)
        out[c] = float((preds[mask] == c).mean()) if mask.sum() else float("nan")
    return out


def compare_to_baseline(method_pca, baseline_pca, class_names):
    """Per-class accuracy CHANGE (method - ERM), sorted worst-first.

    Directly answers 'does an aggregate Sketch gain hide class-level failures?'.
    """
    rows = []
    for c, name in enumerate(class_names):
        m = method_pca.get(c, float("nan"))
        b = baseline_pca.get(c, float("nan"))
        rows.append({"class_index": c, "class_name": name,
                     "erm_acc": b, "method_acc": m, "change": m - b})
    rows.sort(key=lambda r: (float("-inf") if np.isnan(r["change"]) else r["change"]))
    return rows


def dominant_confusions(preds, labels, num_classes, class_names, top_k=3):
    """For each true class, the classes it is most often misclassified as."""
    preds = np.asarray(preds); labels = np.asarray(labels)
    out = {}
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            out[class_names[c]] = []; continue
        wrong = preds[mask][preds[mask] != c]
        if wrong.size == 0:
            out[class_names[c]] = []; continue
        vals, counts = np.unique(wrong, return_counts=True)
        order = np.argsort(-counts)
        out[class_names[c]] = [
            {"predicted_as": class_names[int(vals[i])], "count": int(counts[i])}
            for i in order[:top_k]]
    return out
