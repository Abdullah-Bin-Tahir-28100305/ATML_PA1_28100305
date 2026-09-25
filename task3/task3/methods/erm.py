# =============================================================================
# task3/methods/erm.py  — ERM baseline (Step 1)
# -----------------------------------------------------------------------------
# ML CONCEPT — ERM for domain generalization:
#   Pure cross-entropy over the three pooled, domain-balanced source domains.
#   No alignment, no sharpness term. Because batches are domain-balanced
#   (8 per source), each environment contributes equally, realizing
#   L_ERM = (1/3) sum_e R_e(theta). This is the strong DG baseline the other two
#   methods are compared against, and the SAME model as Task 2's Source-only.
# =============================================================================

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
