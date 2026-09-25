
import numpy as np


def calibrate_threshold(val_unknownness_scores, tpr_target=0.95):
    
    s = np.asarray(val_unknownness_scores, dtype=np.float64)
    return float(np.percentile(s, 100.0 * tpr_target))
