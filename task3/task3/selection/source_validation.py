
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
        logits = head(backbone(imgs))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(ys.numpy())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(labels)


@torch.no_grad()
def evaluate_source_val(backbone, head, val_loaders, device, num_classes):
    
    per_domain = {}
    for d, ld in val_loaders.items():
        preds, labels = collect_predictions(backbone, head, ld, device)
        per_domain[d] = {"top1": top1_accuracy(preds, labels),
                         "macro_f1": macro_f1(preds, labels, num_classes)}
    top1s = [v["top1"] for v in per_domain.values()]
    f1s = [v["macro_f1"] for v in per_domain.values()]
    return {
        "per_domain": per_domain,
        "mean_top1": float(np.mean(top1s)),
        "mean_macro_f1": float(np.mean(f1s)),
        "worst_top1": float(np.min(top1s)),         
        "worst_macro_f1": float(np.min(f1s)),
    }
