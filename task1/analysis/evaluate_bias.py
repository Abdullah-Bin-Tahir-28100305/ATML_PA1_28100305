

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score      



def logits_to_pred_and_conf(logits: torch.Tensor):
    
    probs = F.softmax(logits, dim=1)          
    conf, pred = probs.max(dim=1)            
    return pred.cpu(), conf.cpu()


def top1_accuracy(pred, labels):

    pred = torch.as_tensor(pred)
    labels = torch.as_tensor(labels)
    return (pred == labels).float().mean().item()


def macro_f1(pred, labels):
    
    pred = np.asarray(pred)
    labels = np.asarray(labels)
    return f1_score(labels, pred, average="macro")


def mean_max_confidence(conf):

    return float(np.mean(np.asarray(conf)))


def prediction_consistency(pred_clean, pred_transformed):
   
    pred_clean = torch.as_tensor(pred_clean)
    pred_transformed = torch.as_tensor(pred_transformed)
    return (pred_clean == pred_transformed).float().mean().item()



def standard_metrics(logits, labels):
   
    pred, conf = logits_to_pred_and_conf(logits)
    return {
        "top1": top1_accuracy(pred, labels),
        "macro_f1": macro_f1(pred, labels),
        "mean_max_conf": mean_max_confidence(conf),
    }, pred, conf



def shape_bias_and_coverage(preds, meta):
    
    preds = np.asarray(preds)
    n_shape = n_texture = n_other = 0
    for i, m in enumerate(meta):
        p = int(preds[i])
        if p == m["shape_label"]:
            n_shape += 1                    
        elif p == m["texture_label"]:
            n_texture += 1                  
        else:
            n_other += 1                    
    n_total = len(meta)
    decisive = n_shape + n_texture          
    shape_bias = (100.0 * n_shape / decisive) if decisive > 0 else float("nan")
    coverage = (100.0 * decisive / n_total) if n_total > 0 else float("nan")
    return {
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_other": n_other,
        "n_total": n_total,
        "shape_bias_pct": shape_bias,
        "coverage_pct": coverage,
    }


def collect_conflict_examples(preds, meta, class_names, k=6):
   
    preds = np.asarray(preds)
    shape_cases, texture_cases, other_cases = [], [], []
    for i, m in enumerate(meta):
        p = int(preds[i])
        rec = {
            "index": i,
            "shape": m["shape_name"],
            "texture": m["texture_name"],
            "predicted": class_names[p],
        }
        if p == m["shape_label"] and len(shape_cases) < k:
            shape_cases.append(rec)
        elif p == m["texture_label"] and len(texture_cases) < k:
            texture_cases.append(rec)
        elif len(other_cases) < k:
            other_cases.append(rec)
    return {"followed_shape": shape_cases,
            "followed_texture": texture_cases,
            "followed_other": other_cases}
