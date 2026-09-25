# =============================================================================
# task4/scores/msp.py  — MSP: Maximum Softmax Probability novelty score
# -----------------------------------------------------------------------------
# ML CONCEPT — MSP (Hendrycks & Gimpel 2017):
#   The most basic OOD/novelty baseline. Softmax-normalize the logits to a
#   probability distribution p_k(x) = exp(z_k)/sum_j exp(z_j), then the model's
#   confidence is max_k p_k(x). Unknown inputs are EXPECTED to be less confident,
#   so the UNKNOWNNESS score is:
#       u_MSP(x) = 1 - max_k p_k(x)          (larger => more novel)
#   Weakness: softmax discards absolute logit magnitude and often stays confident
#   on near unknowns — which is exactly why MLS/Energy/Mahalanobis exist.
#
# All four scores read the SAME saved logits/features (spec), so they operate on
# numpy arrays produced once by extract_outputs.py.
# =============================================================================

import numpy as np


def _softmax(logits):
    """Numerically stable row-wise softmax of a (N, C) logit array."""
    z = logits - logits.max(axis=1, keepdims=True)     # subtract max for stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def score(logits, features=None):
    """u_MSP(x) = 1 - max_k softmax(z)_k. `features` unused (uniform interface)."""
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    return 1.0 - probs.max(axis=1)                      # (N,) unknownness
