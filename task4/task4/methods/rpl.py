

import torch
import torch.nn as nn
import torch.nn.functional as F


class RPLModel(nn.Module):
   
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        d = backbone.feature_dim
        self.reciprocal = nn.Parameter(torch.randn(num_classes, d) * 0.1)

    def distances(self, x):
       
        feats = self.backbone.forward_features(x)       # (N,512)
        
        f_sq = (feats * feats).sum(1, keepdim=True)     # (N,1)
        p_sq = (self.reciprocal * self.reciprocal).sum(1).unsqueeze(0)  # (1,C)
        cross = feats @ self.reciprocal.t()             # (N,C)
        d2 = (f_sq + p_sq - 2.0 * cross).clamp(min=0.0) # (N,C) squared distances
        return feats, d2


class RPL:
    name = "rpl"

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_classes = cfg["known"]["num_classes"]
        self.lambda_open = cfg["method"]["lambda_open"]     # open-space weight
        self.R = cfg["method"]["open_space_radius"]         # target radius
        self.ce = nn.CrossEntropyLoss()

    def loss(self, model, images, labels):
       
        feats, d2 = model.distances(images)             # (N,512),(N,C)
        loss_cls = self.ce(d2, labels)                  # distances act as logits
        d_true = d2[torch.arange(d2.size(0)), labels]   # (N,)
        loss_open = ((d_true - self.R) ** 2).mean()
        total = loss_cls + self.lambda_open * loss_open
        return total, {"loss_total": total.item(), "loss_cls": loss_cls.item(),
                       "loss_open": loss_open.item()}


def rpl_unknownness(distance_logits):
   
    import numpy as np
    d = np.asarray(distance_logits, dtype=np.float64)
    return -d.max(axis=1)
