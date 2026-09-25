

import torch.nn as nn


class Vanilla:
    name = "vanilla"
    uses_randaugment = False                            

    def __init__(self, cfg):
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss()

    def loss(self, model, images, labels):
        """Standard supervised cross-entropy on known-class logits."""
        logits = model(images)                          # (N, 10)
        loss = self.ce(logits, labels)
        return loss, {"loss_total": loss.item(), "loss_cls": loss.item()}
