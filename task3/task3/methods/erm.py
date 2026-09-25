

try:
    from methods.base_method import BaseMethod
except ImportError:
    from .base_method import BaseMethod


class ERM(BaseMethod):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def compute_loss(self, source_feats_by_domain, source_logits, source_labels):
        # source_feats_by_domain is unused by ERM (only DAN-DG needs per-domain
        # features); it is accepted for a uniform interface.
        loss_cls = self.classification_loss(source_logits, source_labels)
        logs = {"loss_total": loss_cls.item(), "loss_cls": loss_cls.item(),
                "loss_align": 0.0}
        return loss_cls, logs
