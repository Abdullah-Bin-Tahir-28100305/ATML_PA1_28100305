# =============================================================================
# task4/methods/vanilla.py  — Vanilla closed-set training (Step 1)
# -----------------------------------------------------------------------------
# ML CONCEPT — VANILLA:
#   Ordinary cross-entropy training of the CIFAR ResNet-18 on the 10 known
#   classes. No open-set-specific objective — the OSR ability comes entirely from
#   post-hoc scores read off its frozen logits/features. Also the init for PROSER.
#
# This module exposes a single train_step-style loss so the shared train.py loop
# can treat Vanilla, GCSC (same loss, different augmentation) and RPL uniformly.
# =============================================================================

import torch.nn as nn


class Vanilla:
    """Plain cross-entropy method. `randaugment` controls GCSC vs Vanilla data."""
    name = "vanilla"
    uses_randaugment = False                            # Vanilla: base aug only

    def __init__(self, cfg):
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss()

    def loss(self, model, images, labels):
        """Standard supervised cross-entropy on known-class logits."""
        logits = model(images)                          # (N, 10)
        loss = self.ce(logits, labels)
        return loss, {"loss_total": loss.item(), "loss_cls": loss.item()}
