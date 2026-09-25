# =============================================================================
# task2/evaluation/class_analysis.py
# -----------------------------------------------------------------------------
# PURPOSE: the per-class target analysis required by the spec: per-class target
# accuracy, the classes that improved/degraded most vs Source-only, and their
# dominant confusions.
#
# ML CONCEPT — WHY AGGREGATE NUMBERS CAN LIE (spec):
#   "Use target labels only at this final analysis stage to compute per-class
#    accuracy. Identify the classes with the largest improvement and degradation
#    relative to Source-only ... This establishes whether an aggregate gain hides
#    class-specific negative transfer."
#   A method can raise overall target accuracy while actually HURTING some class
#   (negative transfer on that class), which is masked by the average. Breaking
#   accuracy down per class, and comparing each method to Source-only class by
#   class, exposes exactly that.
#
# NOTE ON TARGET LABELS: target labels are used ONLY here, at the very end, after
# every checkpoint and setting is frozen — never during training or selection.
#
# LINKS:
#   - Consumes predictions/labels produced by metrics.collect_predictions.
#   - Called by evaluate_final.py to build the per-class evidence tables.
# =============================================================================

import numpy as np


def per_class_accuracy(preds, labels, num_classes):
    """Return a dict class_index -> accuracy on that class's examples.

    For class c: (predictions correct among examples whose true label is c).
    Classes absent from the target return NaN so they are visibly not measured.
    """
    preds = np.asarray(preds); labels = np.asarray(labels)
    out = {}
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            out[c] = float("nan")
        else:
            out[c] = float((preds[mask] == c).mean())
    return out


def compare_to_baseline(method_pca, baseline_pca, class_names):
    """Compute per-class accuracy CHANGE (method - baseline) and rank it.

    Returns a list of dicts sorted by change, so the largest improvements and the
    largest degradations (most-negative changes) are easy to read off. This is
    the direct evidence for 'does an aggregate gain hide class-level negative
    transfer?'.
    """
    rows = []
    for c, name in enumerate(class_names):
        m = method_pca.get(c, float("nan"))
        b = baseline_pca.get(c, float("nan"))
        change = m - b
        rows.append({"class_index": c, "class_name": name,
                     "baseline_acc": b, "method_acc": m, "change": change})
    rows.sort(key=lambda r: (float("-inf") if np.isnan(r["change"]) else r["change"]))
    return rows                                        # ascending: worst first


def dominant_confusions(preds, labels, num_classes, class_names, top_k=3):
    """For each true class, list the classes it is most often MISclassified as.

    ML CONCEPT — CONFUSION STRUCTURE: when a class degrades, it usually leaks into
    a specific other class (e.g. horse -> dog). Reporting the top wrong
    predictions per class turns 'accuracy dropped' into an interpretable failure
    mode, which the spec asks us to inspect for the most-changed classes.
    """
    preds = np.asarray(preds); labels = np.asarray(labels)
    out = {}
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            out[class_names[c]] = []
            continue
        wrong = preds[mask][preds[mask] != c]          # misclassified examples
        if wrong.size == 0:
            out[class_names[c]] = []
            continue
        vals, counts = np.unique(wrong, return_counts=True)
        order = np.argsort(-counts)                    # most frequent first
        out[class_names[c]] = [
            {"predicted_as": class_names[int(vals[i])], "count": int(counts[i])}
            for i in order[:top_k]
        ]
    return out
