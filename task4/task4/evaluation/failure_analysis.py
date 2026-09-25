
import numpy as np


def collect_accepted_unknown_failures(unknown_logits, unknown_scores, tau,
                                      unknown_names, cifar10_classes, k=3):
    
    logits = np.asarray(unknown_logits)
    scores = np.asarray(unknown_scores)
    accepted = np.where(scores <= tau)[0]               
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
