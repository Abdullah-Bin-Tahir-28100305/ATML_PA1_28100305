# =============================================================================
# task3/methods/base_method.py
# -----------------------------------------------------------------------------
# Common interface so the single training loop in train.py can drive ERM,
# DAN-DG, and SAM uniformly (spec: all methods share one pipeline).
#
# Contract: compute_loss() receives per-source features, the pooled source
# logits, and the pooled source labels, and returns (total_loss, logs). ERM and
# DAN-DG use this directly. SAM reuses ERM's loss but changes HOW the optimizer
# steps (two passes), which train.py handles via a flag the SAM method sets.
# =============================================================================

import torch.nn as nn


class BaseMethod(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        # SAM sets this True so train.py runs the two-pass sharpness-aware update.
        self.uses_sam = False

    def classification_loss(self, source_logits, source_labels):
        """Cross-entropy on labeled SOURCE examples (shared by all methods)."""
        return self.ce(source_logits, source_labels)

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def extra_modules(self):
        """Method-owned submodules to add to the optimizer (none by default)."""
        return []
