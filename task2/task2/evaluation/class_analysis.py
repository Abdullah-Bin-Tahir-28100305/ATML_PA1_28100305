

import numpy as np


def per_class_accuracy(preds, labels, num_classes):
    
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
