
import numpy as np


def score(logits, features=None):
    logits = np.asarray(logits, dtype=np.float64)
    return -logits.max(axis=1)                          
