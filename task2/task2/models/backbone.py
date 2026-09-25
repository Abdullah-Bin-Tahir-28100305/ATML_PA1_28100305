# =============================================================================
# task2/models/backbone.py
# -----------------------------------------------------------------------------
# PURPOSE: the shared FEATURE EXTRACTOR F: a torchvision ResNet-18 (ImageNet
# pretrained) with its 1000-way classifier removed, so it outputs the
# 512-dimensional feature that every alignment loss and the 7-class head consume.
#
# Two REQUIRED, easy-to-miss details from the spec are implemented here:
#   1. The whole network is FINE-TUNED (not frozen) — DA methods update F.
#   2. BatchNorm RUNNING STATISTICS are FROZEN at their ImageNet values, while
#      the BN scale/shift parameters (gamma, beta) stay trainable.
#
# ML CONCEPT — WHY FREEZE BN RUNNING STATS (recap):
#   BatchNorm normalizes each activation using a running mean/variance. If those
#   updated on adaptation batches (which mix source + target), the normalization
#   itself would drift toward the target distribution — an implicit, uncontrolled
#   adaptation that would contaminate the comparison of the *explicit* alignment
#   losses (MMD/adversarial). Freezing running stats keeps the study clean.
#   gamma/beta remain trainable because they are ordinary learnable affine
#   parameters, not distribution statistics.
#
# LINKS:
#   - classifier_head.py sits on top of this 512-d feature.
#   - methods/*.py call .forward() to get features for their losses.
#   - train.py calls freeze_bn_running_stats() after every model.train().
# =============================================================================

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18Backbone(nn.Module):
    """ResNet-18 up to the global-average-pooled 512-d feature (classifier removed)."""

    def __init__(self):
        super().__init__()
        # IMAGENET1K_V1 is the weight set the spec names for ResNet-18.
        net = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_dim = net.fc.in_features          # 512 for ResNet-18
        # Replace the 1000-way ImageNet head with Identity so forward() returns
        # the pre-classifier feature (after global average pooling + flatten).
        net.fc = nn.Identity()
        self.net = net
        # NOTE: we do NOT freeze parameters — the spec says "fine-tune the
        # complete network for every method." All conv/BN-affine weights train.

    def forward(self, x):
        # x: (N,3,224,224) already normalized by the dataset transform.
        return self.net(x)                             # (N, 512)

    def set_bn_eval(self):
        """Put every BatchNorm module in EVAL mode so its running stats are FROZEN.

        Spec: "after calling model.train(), place only the BatchNorm modules in
        evaluation mode so their running statistics are not updated; do not place
        the complete model in evaluation mode." Calling .eval() on a BN module
        stops it from updating running_mean/running_var and makes it use the
        stored (ImageNet) values for normalization — while gamma/beta still
        receive gradients and keep training. Dropout and the rest of the network
        stay in train mode.
        """
        for m in self.net.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()                               # freeze running stats
                # (We intentionally leave m.weight/m.bias = gamma/beta trainable;
                #  eval() does not affect their requires_grad.)
