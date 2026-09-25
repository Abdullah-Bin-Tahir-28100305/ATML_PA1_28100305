# =============================================================================
# task4/models/resnet_cifar.py
# -----------------------------------------------------------------------------
# PURPOSE: the CIFAR-appropriate ResNet-18 used throughout Task 4, plus the
# hooks PROSER needs for manifold mixup.
#
# CIFAR STEM MODIFICATION (spec): the standard ImageNet ResNet-18 starts with a
# 7x7 stride-2 conv + 3x3 stride-2 maxpool, which downsamples a 32x32 CIFAR image
# far too aggressively. We REPLACE conv1 with a 3x3 stride-1 conv and REMOVE the
# maxpool, so the network keeps enough spatial resolution on 32x32 inputs. This
# is the standard "CIFAR ResNet-18".
#
# WHAT THE MODEL EXPOSES (needed by scores + methods):
#   * forward(x)            -> logits (N, num_classes)
#   * forward_features(x)   -> penultimate feature f(x) (N, 512) [for Mahalanobis]
#   * forward_split(x)      -> (phi_pre up to layer2, then rest) [for PROSER mixup]
#
# ML CONCEPT — PENULTIMATE FEATURE vs LOGITS:
#   f(x) is the 512-d global-average-pooled feature just before the linear
#   classifier; z(x) are the class logits. MSP/MLS/Energy use logits; Mahalanobis
#   uses f(x). Exposing both lets all four scores read from ONE forward pass.
#
# LINKS: methods/*.py build on this; PROSER uses forward_split for manifold mixup
# after layer2 and before layer3 (spec).
# =============================================================================

import torch
import torch.nn as nn
from torchvision.models import resnet18


class CifarResNet18(nn.Module):
    """ResNet-18 with a CIFAR stem (3x3 stride-1 conv1, no maxpool)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Start from the torchvision ResNet-18 ARCHITECTURE (weights=None: we
        # train from scratch on CIFAR-10, per spec "from random initialization").
        net = resnet18(weights=None, num_classes=num_classes)
        # --- CIFAR stem surgery ---
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()                     # remove the initial maxpool
        self.net = net
        self.feature_dim = net.fc.in_features           # 512

    # ----- standard forward paths -----
    def _stem(self, x):
        n = self.net
        x = n.relu(n.bn1(n.conv1(x)))                   # CIFAR stem (no maxpool)
        x = n.maxpool(x)                                # Identity -> no-op
        return x

    def forward_features(self, x):
        """Return the 512-d penultimate feature f(x) (post global-avg-pool)."""
        n = self.net
        x = self._stem(x)
        x = n.layer1(x); x = n.layer2(x); x = n.layer3(x); x = n.layer4(x)
        x = n.avgpool(x)                                # (N,512,1,1)
        return torch.flatten(x, 1)                      # (N,512)

    def forward(self, x):
        """Return class logits z(x) (N, num_classes)."""
        feats = self.forward_features(x)
        return self.net.fc(feats)

    def forward_from_features(self, feats):
        """Apply only the final linear classifier to a given feature (N,512)->logits."""
        return self.net.fc(feats)

    # ----- split forward for PROSER manifold mixup -----
    def phi_pre(self, x):
        """Network UP TO layer2 (inclusive): h = phi_pre(x). Spec's mixup point."""
        n = self.net
        x = self._stem(x)
        x = n.layer1(x)
        x = n.layer2(x)                                 # <-- mix AFTER layer2
        return x

    def phi_post(self, h):
        """Network AFTER layer2 -> penultimate feature, given h from phi_pre.

        Spec: manifold mixup happens 'after layer2 and before layer3', so the mix
        is applied to h, then h continues through layer3, layer4, pool.
        """
        n = self.net
        x = n.layer3(h)                                 # <-- continue BEFORE layer3
        x = n.layer4(x)
        x = n.avgpool(x)
        return torch.flatten(x, 1)                      # (N,512) penultimate feature

    def forward_split_logits(self, h):
        """From a (possibly mixed) layer2 activation h -> logits."""
        return self.net.fc(self.phi_post(h))
