# =============================================================================
# task2/models/classifier_head.py
# -----------------------------------------------------------------------------
# PURPOSE: the 7-class classifier head C that maps the 512-d ResNet-18 feature to
# class logits. Spec: "Replace its ImageNet classifier with a seven-class linear
# classifier head and fine-tune the complete network."
#
# ML CONCEPT — CLASSIFIER ON TOP OF A SHARED FEATURE:
#   The full model is C(F(x)): the backbone F produces a feature, the head C
#   produces class scores. In DA, only SOURCE examples get a classification loss
#   (we have no target labels), so C is trained purely on source supervision,
#   while alignment losses shape F. CDAN additionally uses the head's SOFTMAX
#   output p = softmax(C(F(x))) inside its multilinear conditioning map.
# =============================================================================

import torch.nn as nn


class ClassifierHead(nn.Module):
    """A single linear layer: 512-d feature -> num_classes logits."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        # A plain affine classifier (weights + bias). Kept linear so the model is
        # standard ResNet-18 + linear head, matching the spec exactly.
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, feats):
        return self.fc(feats)                          # (N, num_classes) logits
