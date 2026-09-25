# =============================================================================
# task2/methods/source_only.py  — Source-only ERM (Step 1)
# -----------------------------------------------------------------------------
# ML CONCEPT — ERM BASELINE:
#   No alignment at all. The loss is purely cross-entropy on the pooled labeled
#   source domains. This measures how far ordinary supervised learning transfers
#   to the unseen target with NO adaptation. Every other method is compared to
#   this baseline; an adaptation method that does WORSE than this exhibits
#   'negative transfer'.
#
# This method ignores the target batch entirely (it is passed for interface
# uniformity but contributes nothing to the loss). That is exactly what makes it
# the no-adaptation control.
# =============================================================================

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
        # Only the source classification loss. Target arguments are unused.
        loss_cls = self.classification_loss(source_logits, source_labels)
        # logs: components we want to plot later (spec requires loss curves).
        logs = {"loss_total": loss_cls.item(), "loss_cls": loss_cls.item(),
                "loss_align": 0.0}
        return loss_cls, logs
