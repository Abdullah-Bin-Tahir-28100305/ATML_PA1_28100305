# =============================================================================
# task4/methods/gcsc.py  — GCSC training (Step 3)
# -----------------------------------------------------------------------------
# ML CONCEPT — GCSC:
#   Identical to Vanilla EXCEPT the training data uses stronger positive
#   augmentation (RandAugment). Everything else — architecture, optimizer,
#   schedule, seed, checkpoint rule — is unchanged, so this is a CONTROLLED test
#   of whether better closed-set training (Vaze et al. 2022) also improves
#   open-set rejection, particularly for NEAR unknowns. Evaluated with MLS so it
#   is directly comparable to Vanilla and PROSER.
#
# The only difference from Vanilla is `uses_randaugment = True`, which tells the
# data layer to add RandAugment to the TRAIN transform. The loss is the same
# cross-entropy.
# =============================================================================

import torch.nn as nn


class GCSC:
    name = "gcsc"
    uses_randaugment = True                             # add RandAugment to train aug

    def __init__(self, cfg):
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss()

    def loss(self, model, images, labels):
        logits = model(images)
        loss = self.ce(logits, labels)
        return loss, {"loss_total": loss.item(), "loss_cls": loss.item()}
