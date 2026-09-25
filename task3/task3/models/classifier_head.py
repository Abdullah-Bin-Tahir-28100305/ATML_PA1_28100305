# =============================================================================
# task3/models/classifier_head.py
# -----------------------------------------------------------------------------
# The 7-class linear classifier head C mapping the 512-d ResNet-18 feature to
# class logits. Identical to Task 2 (spec: same seven-class classifier head).
#
# ML CONCEPT — the full model is C(F(x)). All three methods share this same
# head; they differ only in the training objective applied on top:
#   ERM     -> cross-entropy on sources;
#   DAN-DG  -> cross-entropy + pairwise source MMD on F's features;
#   SAM     -> cross-entropy optimized with sharpness-aware two-step updates.
# =============================================================================

import torch.nn as nn


class ClassifierHead(nn.Module):
    """A single linear layer: 512-d feature -> num_classes logits."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, feats):
        return self.fc(feats)                          # (N, num_classes) logits
