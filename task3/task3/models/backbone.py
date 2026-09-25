# =============================================================================
# task3/models/backbone.py
# -----------------------------------------------------------------------------
# The shared FEATURE EXTRACTOR F: torchvision ResNet-18 (ImageNet pretrained)
# with its 1000-way classifier removed, outputting the 512-d feature that the
# 7-class head and the DAN-DG alignment loss consume.
#
# This is IDENTICAL to Task 2's backbone (spec requires "exactly the same
# ResNet-18 initialization" and the same frozen-BatchNorm policy), so ERM here
# matches Task 2's Source-only model.
#
# TWO REQUIRED details (same as Task 2):
#   1. The whole network is FINE-TUNED (not frozen).
#   2. BatchNorm RUNNING STATISTICS are FROZEN at ImageNet values, while the BN
#      scale/shift (gamma, beta) stay trainable.
#
# ML CONCEPT — WHY FREEZE BN RUNNING STATS:
#   In DG we compare ERM vs DAN-DG vs SAM. Letting BN running stats adapt would
#   inject an uncontrolled, implicit form of adaptation and confound the
#   comparison. Freezing them keeps the study focused on the explicit objectives.
#   (For SAM this also matters: both the ascent and descent passes must use the
#   same frozen stats, or the two passes would 'see' different normalizations.)
# =============================================================================

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18Backbone(nn.Module):
    """ResNet-18 up to the global-average-pooled 512-d feature (classifier removed)."""

    def __init__(self):
        super().__init__()
        net = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_dim = net.fc.in_features          # 512
        net.fc = nn.Identity()                         # output the feature, not logits
        self.net = net                                 # all params trainable (fine-tune)

    def forward(self, x):
        return self.net(x)                             # (N, 512)

    def set_bn_eval(self):
        """Put every BatchNorm module in EVAL mode so its running stats are FROZEN.

        Called after model.train() every step. eval() on a BN module stops it
        updating running_mean/var and makes it normalize with the stored
        (ImageNet) values, while gamma/beta keep receiving gradients.
        """
        for m in self.net.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
