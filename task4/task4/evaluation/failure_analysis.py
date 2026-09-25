# =============================================================================
# task4/evaluation/failure_analysis.py
# -----------------------------------------------------------------------------
# PURPOSE: the qualitative failure analysis required by the spec: using the
# VANILLA MLS threshold, find UNKNOWNS that were wrongly ACCEPTED, and for each
# record the true unknown class, the predicted CIFAR-10 class, the score, and the
# threshold — so plausible confusions (e.g. wolf->dog) can be told apart from
# surprising ones (spec Required Evidence: >=3 near + >=3 far failures).
#
# ML CONCEPT — WHY INSPECT ACCEPTED UNKNOWNS:
#   A wrongly-accepted unknown is a false negative for the detector. Its predicted
#   known class reveals WHICH known concept 'absorbed' it. If a near unknown like
#   'wolf' is accepted as 'dog', that is a semantically plausible failure (they
#   look alike); if a far unknown like 'keyboard' is accepted as 'ship', that is a
#   surprising failure worth flagging. This connects the numbers to interpretable
#   model behavior.
#
# LINKS: consumes cached logits/scores + the unknown class names from
# cifar100_unknowns.unknown_class_names; called by evaluate_osr.py.
# =============================================================================

import numpy as np


def collect_accepted_unknown_failures(unknown_logits, unknown_scores, tau,
                                      unknown_names, cifar10_classes, k=3):
    """Return up to k unknowns wrongly ACCEPTED (u <= tau) with their details.

    For each accepted unknown we report:
      * true unknown class (CIFAR-100 fine name),
      * predicted CIFAR-10 class (argmax of the 10 known logits),
      * the unknownness score and the threshold tau.
    Sorted by score ASCENDING (most confidently-accepted first), i.e. the model
    was surest these were known.
    """
    logits = np.asarray(unknown_logits)
    scores = np.asarray(unknown_scores)
    accepted = np.where(scores <= tau)[0]               # wrongly accepted unknowns
    # order by how far below the threshold they are (most 'known-looking' first)
    accepted = accepted[np.argsort(scores[accepted])]
    out = []
    for i in accepted[:k]:
        pred = int(logits[i].argmax())
        out.append({
            "true_unknown_class": unknown_names[i],
            "predicted_cifar10_class": cifar10_classes[pred],
            "score": float(scores[i]),
            "threshold": float(tau),
        })
    return out
