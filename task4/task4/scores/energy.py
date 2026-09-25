# =============================================================================
# task4/scores/energy.py  — Energy novelty score
# -----------------------------------------------------------------------------
# ML CONCEPT — ENERGY (Liu et al. 2020):
#   The free energy of the logits is  E(x) = -logsumexp_k z_k(x). It aggregates
#   ALL logits (not just the max), so it reflects the total evidence the model
#   assigns to any known class. Knowns tend to have LOW energy (high total
#   evidence); unknowns HIGH energy. The spec's unknownness score is:
#       u_Energy(x) = - log sum_k exp(z_k(x))  =  - logsumexp_k z_k(x)  =  E(x)
#   (larger => more novel). Unlike MLS it uses every logit, so it can differ from
#   MLS when the evidence is spread across several classes.
# =============================================================================

import numpy as np


def score(logits, features=None):
    """u_Energy(x) = -logsumexp_k z_k(x). Larger = more novel."""
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max(axis=1, keepdims=True)               # stability shift
    lse = (m.squeeze(1) + np.log(np.exp(logits - m).sum(axis=1)))  # logsumexp
    return -lse                                         # (N,) unknownness
