

try:
    from methods.base_method import BaseMethod
except ImportError:
    from .base_method import BaseMethod


class SourceOnly(BaseMethod):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def compute_loss(self, source_feats, source_logits, source_labels,
                     target_feats, target_logits, alpha):

        loss_cls = self.classification_loss(source_logits, source_labels)

        logs = {"loss_total": loss_cls.item(), "loss_cls": loss_cls.item(),
                "loss_align": 0.0}
        return loss_cls, logs
