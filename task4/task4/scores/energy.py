

import numpy as np


def score(logits, features=None):
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max(axis=1, keepdims=True)               
    lse = (m.squeeze(1) + np.log(np.exp(logits - m).sum(axis=1)))  
    return -lse                                         
