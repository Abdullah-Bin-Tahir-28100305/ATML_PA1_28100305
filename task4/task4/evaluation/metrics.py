# =============================================================================
# task4/evaluation/metrics.py
# -----------------------------------------------------------------------------
# PURPOSE: the OSR scoring metrics — AUROC (threshold-free ranking quality) and
# the validation-calibrated operating-point metrics (acceptance/rejection rates,
# FPR@95TPR) — plus closed-set accuracy (CSA).
#
# ML CONCEPTS:
#   * AUROC (Area Under the ROC Curve): treats OSR as a binary detection problem
#     — KNOWN (positive) vs UNKNOWN (negative) — and measures how well the
#     unknownness score RANKS unknowns above knowns across ALL thresholds. 0.5 =
#     chance, 1.0 = perfect separation. We report it for Known-vs-Near,
#     Known-vs-Far, and Known-vs-All (spec).
#   * FPR@95TPR / validation-calibrated rejection: AUROC ignores the operating
#     point; in practice we pick ONE threshold. Spec: threshold tau = 95th
#     percentile of unknownness on the KNOWN validation set, accept when
#     u(x) <= tau. By construction ~95% of knowns are accepted (TPR~=0.95); the
#     fraction of UNKNOWNS wrongly accepted at that tau is FPR@95TPR. Lower is
#     better. Reporting BOTH (spec 'What to watch for') matters: AUROC is ranking
#     over all thresholds, this is behavior at one chosen point.
#   * CSA (Closed-Set Accuracy): ordinary known-class accuracy BEFORE rejection,
#     computed from the 10 known logits only (spec).
# =============================================================================

import numpy as np
from sklearn.metrics import roc_auc_score


def closed_set_accuracy(known_logits, known_labels):
    """Top-1 accuracy on known TEST examples using the 10 known logits (CSA)."""
    preds = np.asarray(known_logits).argmax(axis=1)
    return float((preds == np.asarray(known_labels)).mean())


def auroc_known_vs_unknown(known_scores, unknown_scores):
    """AUROC treating higher unknownness as 'more likely unknown'.

    We label knowns as 0 and unknowns as 1 and rank by the unknownness score, so a
    higher score for unknowns yields AUROC > 0.5. Equivalent to the standard
    'AUROC for detecting unknowns'.
    """
    y = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    s = np.concatenate([np.asarray(known_scores), np.asarray(unknown_scores)])
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def acceptance_rejection_rates(known_scores_test, unknown_scores, tau):
    """At threshold tau (accept when u <= tau): known acceptance + unknown rejection.

    Returns:
      known_acceptance_rate: fraction of known TEST examples accepted (~TPR).
      unknown_rejection_rate: fraction of unknowns correctly rejected.
      fpr_at_95tpr: fraction of unknowns wrongly ACCEPTED (= 1 - rejection rate),
                    which under the 95th-percentile tau is FPR@95TPR.
    """
    known_scores_test = np.asarray(known_scores_test)
    unknown_scores = np.asarray(unknown_scores)
    known_acc = float((known_scores_test <= tau).mean())
    unknown_rej = float((unknown_scores > tau).mean())
    fpr = float((unknown_scores <= tau).mean())         # unknowns wrongly accepted
    return {"known_acceptance_rate": known_acc,
            "unknown_rejection_rate": unknown_rej,
            "fpr_at_95tpr": fpr}
