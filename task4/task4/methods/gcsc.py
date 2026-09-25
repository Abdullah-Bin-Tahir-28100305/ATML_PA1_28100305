

import torch.nn as nn


class GCSC:
    name = "gcsc"
    uses_randaugment = True                             

    def __init__(self, cfg):
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss()

    def loss(self, model, images, labels):
        logits = model(images)
        loss = self.ce(logits, labels)
        return loss, {"loss_total": loss.item(), "loss_cls": loss.item()}
