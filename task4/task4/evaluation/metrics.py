

import numpy as np
from sklearn.metrics import roc_auc_score


def closed_set_accuracy(known_logits, known_labels):
    preds = np.asarray(known_logits).argmax(axis=1)
    return float((preds == np.asarray(known_labels)).mean())


def auroc_known_vs_unknown(known_scores, unknown_scores):
   
    y = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    s = np.concatenate([np.asarray(known_scores), np.asarray(unknown_scores)])
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def acceptance_rejection_rates(known_scores_test, unknown_scores, tau):
   
    known_scores_test = np.asarray(known_scores_test)
    unknown_scores = np.asarray(unknown_scores)
    known_acc = float((known_scores_test <= tau).mean())
    unknown_rej = float((unknown_scores > tau).mean())
    fpr = float((unknown_scores <= tau).mean())         
    return {"known_acceptance_rate": known_acc,
            "unknown_rejection_rate": unknown_rej,
            "fpr_at_95tpr": fpr}
