# =============================================================================
# task2/methods/base_method.py
# -----------------------------------------------------------------------------
# PURPOSE: the COMMON interface every alignment method implements, so the single
# training loop in train.py can drive all four methods identically (spec: "all
# methods must run through the same training and evaluation pipeline").
#
# The contract: each method's compute_loss() receives
#   - source features, source labels (for the classification loss),
#   - source logits (already computed by the shared head),
#   - target features, target logits (for alignment losses),
#   - the current GRL alpha (for adversarial methods),
# and returns (total_loss, logs_dict). Non-adversarial methods simply ignore
# the arguments they do not need. This keeps train.py method-agnostic.
# =============================================================================

import torch.nn as nn


class BaseMethod(nn.Module):
    """Base class holding the shared cross-entropy classification loss.

    ML CONCEPT — SOURCE-ONLY SUPERVISION:
      In UDA we have NO target labels, so the classification loss is computed on
      SOURCE examples only. Every method includes this term; the methods differ
      only in the ADDITIONAL alignment term they add on top.
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()                # standard multiclass loss

    def classification_loss(self, source_logits, source_labels):
        """Cross-entropy on labeled source examples (shared by all methods)."""
        return self.ce(source_logits, source_labels)

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError                      # each method overrides this

    def extra_modules(self):
        """Return any method-owned submodules (e.g. a discriminator) whose
        parameters must be added to the optimizer. Default: none."""
        return []
