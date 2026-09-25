# =============================================================================
# task4/evaluation/thresholds.py
# -----------------------------------------------------------------------------
# PURPOSE: compute the validation-calibrated rejection threshold.
#
# ML CONCEPT — WHY CALIBRATE THE THRESHOLD ON KNOWN VALIDATION DATA:
#   OSR must pick ONE operating threshold tau to convert the continuous
#   unknownness score into an accept/reject decision. Spec: tau = the 95th
#   PERCENTILE of the unknownness score computed on the KNOWN CIFAR-10 VALIDATION
#   set, and we ACCEPT when u(x) <= tau. Choosing the 95th percentile means ~95%
#   of known validation examples fall below tau and are accepted -> a target true-
#   positive rate of 0.95. Crucially this uses KNOWN data ONLY, so no unknown
#   ever influences the threshold (spec: unknowns are evaluation-only).
#
# LINKS: evaluate_osr.py fits tau per (model, score) on the val unknownness
# scores, then applies it to test knowns + near/far unknowns via metrics.py.
# =============================================================================

import numpy as np


def calibrate_threshold(val_unknownness_scores, tpr_target=0.95):
    """tau = the (100*tpr_target)-th percentile of the validation unknownness.

    With accept-when-u<=tau, this makes ~tpr_target of known VAL examples accepted.
    """
    s = np.asarray(val_unknownness_scores, dtype=np.float64)
    # percentile at tpr_target (e.g. 95) so ~95% of knowns have u <= tau.
    return float(np.percentile(s, 100.0 * tpr_target))
