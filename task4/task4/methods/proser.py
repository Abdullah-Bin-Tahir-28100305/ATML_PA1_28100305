

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from methods.manifold_mixup import (sample_lambda, make_different_class_pairs,
                                        manifold_mixup_features)
except ImportError:
    from .manifold_mixup import (sample_lambda, make_different_class_pairs,
                                 manifold_mixup_features)


class ProserModel(nn.Module):
    
    def __init__(self, backbone, num_known, num_dummy):
        super().__init__()
        self.backbone = backbone                       
        self.num_known = num_known
        self.num_dummy = num_dummy
        self.dummy = nn.Linear(backbone.feature_dim, num_dummy)  # placeholder heads

    def forward(self, x):
        
        feats = self.backbone.forward_features(x)
        known = self.backbone.forward_from_features(feats)   
        dummy = self.dummy(feats)                            
        return known, dummy

    def logits_from_features(self, feats):
        return self.backbone.forward_from_features(feats), self.dummy(feats)

    def known_logits(self, x):
        feats = self.backbone.forward_features(x)
        return self.backbone.forward_from_features(feats)


class PROSER:
    name = "proser"

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_known = cfg["known"]["num_classes"]
        self.num_dummy = cfg["method"]["num_dummy"]
        self.beta = cfg["method"]["beta"]              # classifier-placeholder weight
        self.gamma = cfg["method"]["gamma"]            # data-placeholder weight
        self.alpha = cfg["method"]["mixup_alpha"]      # Beta(alpha,alpha)
        self.ce = nn.CrossEntropyLoss()

    def classifier_placeholder_loss(self, model, images, labels):
        known, dummy = model(images)                   # (N,10), (N,K)
        dummy_max = dummy.max(dim=1, keepdim=True).values          # (N,1) strongest dummy
        aug = torch.cat([known, dummy_max], dim=1)
        dummy_slot = self.num_known                     # index of the dummy column (=10)

        loss_keep = self.ce(aug, labels)

        masked = aug.clone()
        masked[torch.arange(masked.size(0)), labels] = float("-inf")  # remove true class
        target_dummy = torch.full_like(labels, dummy_slot)            # target = dummy slot
        loss_reject = self.ce(masked, target_dummy)

        return loss_keep + loss_reject

    def data_placeholder_loss(self, model, images, labels):
        device = images.device
        backbone = model.backbone
        h = backbone.phi_pre(images)                    # (N, C, H, W)
        partner, valid = make_different_class_pairs(labels)
        partner_t = torch.as_tensor(partner, device=device, dtype=torch.long)
        lam = sample_lambda(self.alpha, size=h.size(0), device=device)  # ~Beta(2,2)
        h_mixed = manifold_mixup_features(h, partner_t, lam)            # (N,C,H,W)

        feats_mixed = backbone.phi_post(h_mixed)        # (N,512)
        known_mixed = backbone.forward_from_features(feats_mixed)       # (N,10)
        dummy_mixed = model.dummy(feats_mixed)                          # (N,K)
        dummy_max = dummy_mixed.max(dim=1, keepdim=True).values
        aug = torch.cat([known_mixed, dummy_max], dim=1)               # (N,11)

        target_dummy = torch.full((aug.size(0),), self.num_known,
                                  dtype=torch.long, device=device)
        valid_t = torch.as_tensor(valid, device=device)
        if valid_t.sum() == 0:
            return aug.new_zeros(())
        return self.ce(aug[valid_t], target_dummy[valid_t])

    def loss(self, model, images, labels):
       
        n = images.size(0)
        half = n // 2
        x_a, y_a = images[:half], labels[:half]         # classifier placeholders
        x_b, y_b = images[half:], labels[half:]         # data placeholders

        loss_cls = self.classifier_placeholder_loss(model, x_a, y_a)
        loss_data = self.data_placeholder_loss(model, x_b, y_b)
        total = self.beta * loss_cls + self.gamma * loss_data
        return total, {"loss_total": total.item(),
                       "loss_classifier_placeholder": loss_cls.item(),
                       "loss_data_placeholder": float(loss_data.item())}



def proser_detection_score(known_logits, dummy_logits):
   
    import numpy as np
    known = np.asarray(known_logits, dtype=np.float64)
    dummy = np.asarray(dummy_logits, dtype=np.float64)
    return dummy.max(axis=1) - known.max(axis=1)        # (N,) unknownness
