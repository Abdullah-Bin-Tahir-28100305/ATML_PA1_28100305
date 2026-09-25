# =============================================================================
# task2/models/domain_discriminator.py
# -----------------------------------------------------------------------------
# PURPOSE: the adversarial components shared by DANN and CDAN:
#   1. GradientReversalLayer (GRL) — identity forward, sign-flipped-and-scaled
#      gradient backward;
#   2. DomainDiscriminator — an MLP that classifies a feature as source vs target.
#
# ML CONCEPT — ADVERSARIAL DOMAIN ALIGNMENT:
#   We want the backbone F to produce features a discriminator D CANNOT use to
#   tell source from target (domain-invariant features). That is a min-max game:
#       min_F max_D  [ - domain_classification_loss ]  (schematically).
#   The clean trick (Ganin et al. 2016) is the GRL. In the FORWARD pass GRL(x)=x,
#   so D trains normally to classify domains. In the BACKWARD pass GRL multiplies
#   the incoming gradient by -alpha, so the gradient that reaches F is the
#   NEGATIVE of what would minimize D's loss — i.e. F is pushed to MAXIMIZE
#   domain confusion. One ordinary backward() call thus trains D to discriminate
#   and F to fool it simultaneously.
#
# LINKS:
#   - methods/dann.py feeds the raw 512-d feature through GRL -> D.
#   - methods/cdan.py feeds the multilinear map g(x)=vec(f x p) through GRL -> D.
# =============================================================================

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    """The autograd Function implementing gradient reversal.

    forward:  returns the input unchanged (identity).
    backward: returns -alpha * grad, so upstream modules receive a reversed,
              scaled gradient. `alpha` controls the strength of the adversarial
              signal and is ramped up during training (see the schedule below).
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha                              # stash alpha for backward
        return x.view_as(x)                            # identity (no-op view)

    @staticmethod
    def backward(ctx, grad_output):
        # Flip the sign and scale by alpha. The second return (for `alpha`) is
        # None because alpha is not a tensor we differentiate w.r.t.
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha: float):
    """Convenience wrapper to apply the gradient-reversal Function."""
    return _GradReverse.apply(x, alpha)


def grl_alpha(progress: float, gamma: float, max_alpha: float) -> float:
    """DANN's standard alpha schedule: alpha(p) = (2 / (1 + exp(-gamma*p)) - 1)*max.

    Spec: alpha(p) = 2 / (1 + exp(-10 p)) - 1, with p in [0,1] training progress.
    ML CONCEPT — WHY RAMP alpha FROM 0:
      Early in training the features are still forming, so strong adversarial
      pressure would fight the class task. Starting alpha near 0 lets the network
      learn useful features first; alpha then rises toward its cap so domain
      confusion is enforced increasingly as training proceeds. `max_alpha` is the
      controlled-study knob for DANN (spec allows varying the max GRL strength).
    """
    import math
    base = 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0   # in [0,1)
    return base * max_alpha


class DomainDiscriminator(nn.Module):
    """MLP domain classifier: input -> hidden(ReLU, dropout) -> 2 logits.

    Spec (DANN): "256-unit hidden layer, ReLU, dropout of 0.5, and a two-class
    output layer." CDAN reuses the same architecture but with a larger input
    dimension (the multilinear map). `in_dim` therefore differs by method:
      - DANN: in_dim = 512 (the feature itself)
      - CDAN: in_dim = 512 * num_classes (vec(f x p))
    """
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),                 # project to hidden width
            nn.ReLU(inplace=True),                     # nonlinearity
            nn.Dropout(dropout),                       # regularize the discriminator
            nn.Linear(hidden, 2),                      # 2-class output: source/target
        )

    def forward(self, x):
        return self.net(x)                             # (N, 2) domain logits
