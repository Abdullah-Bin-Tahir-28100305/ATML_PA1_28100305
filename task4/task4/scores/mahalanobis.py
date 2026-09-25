

import numpy as np


def fit_mahalanobis(train_features, train_labels, num_classes, eps=1e-6):
    
    X = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_labels)
    d = X.shape[1]

    means = np.zeros((num_classes, d), dtype=np.float64)
    for c in range(num_classes):
        means[c] = X[y == c].mean(axis=0)

    
    centered = X - means[y]                              
    var = (centered ** 2).mean(axis=0)                  
    var = var + eps                                     
    inv_var = 1.0 / var                                 

    return {"means": means, "inv_var": inv_var, "num_classes": num_classes}


def score(logits, features=None, params=None):
   
    if params is None:
        raise ValueError("Mahalanobis score requires fitted params "
                         "(call fit_mahalanobis on train features first).")
    F = np.asarray(features, dtype=np.float64)          # (N,512) test features
    means = params["means"]                             # (C,512)
    inv_var = params["inv_var"]                         # (512,)

    
    N = F.shape[0]; C = means.shape[0]
    dists = np.empty((N, C), dtype=np.float64)
    for c in range(C):
        diff = F - means[c]                             
        dists[:, c] = (diff * diff * inv_var).sum(axis=1)
    return dists.min(axis=1)                            
