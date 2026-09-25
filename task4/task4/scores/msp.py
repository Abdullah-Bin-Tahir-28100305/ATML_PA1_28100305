

import numpy as np


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)     
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def score(logits, features=None):
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    return 1.0 - probs.max(axis=1)                      
