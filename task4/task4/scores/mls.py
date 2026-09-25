# =============================================================================
# task4/scores/mls.py  — MLS: Maximum Logit Score novelty score
# -----------------------------------------------------------------------------
# ML CONCEPT — MLS (Vaze et al. 2022):
#   Instead of softmax confidence, use the raw MAXIMUM LOGIT. Softmax throws away
#   the absolute scale of the logits; MLS keeps it. Vaze et al. show that a good
#   closed-set classifier produces larger maximum logits for knowns than for
#   unknowns, making MLS a strong, simple OSR signal. As an UNKNOWNNESS score
#   (larger => more novel) we negate the maximum logit:
#       u_MLS(x) = - max_k z_k(x)
#   MLS is the COMMON score used to compare Vanilla, GCSC, and PROSER (spec).
# =============================================================================

import numpy as np


def score(logits, features=None):
    """u_MLS(x) = - max_k z_k(x). Larger = more novel."""
    logits = np.asarray(logits, dtype=np.float64)
    return -logits.max(axis=1)                          # (N,) unknownness
