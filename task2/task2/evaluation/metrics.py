

import numpy as np
import torch
from sklearn.metrics import f1_score


def top1_accuracy(preds, labels):

    preds = np.asarray(preds); labels = np.asarray(labels)
    return float((preds == labels).mean())


def macro_f1(preds, labels, num_classes):
    
    preds = np.asarray(preds); labels = np.asarray(labels)
    return float(f1_score(labels, preds, average="macro",
                          labels=list(range(num_classes)), zero_division=0))


@torch.no_grad()
def collect_predictions(backbone, head, loader, device):
   
    preds, labels = [], []
    for imgs, ys in loader:
        imgs = imgs.to(device)
        feats = backbone(imgs)
        logits = head(feats)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(ys.numpy())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(labels)


@torch.no_grad()
def evaluate_loader(backbone, head, loader, device, num_classes):

    preds, labels = collect_predictions(backbone, head, loader, device)
    if preds.size == 0:
        return {"top1": float("nan"), "macro_f1": float("nan"),
                "preds": preds, "labels": labels}
    return {"top1": top1_accuracy(preds, labels),
            "macro_f1": macro_f1(preds, labels, num_classes),
            "preds": preds, "labels": labels}


def mean_source_val_macro_f1(per_domain_results):
    
    vals = [r["macro_f1"] for r in per_domain_results.values()]
    return float(np.mean(vals)) if vals else float("nan")
