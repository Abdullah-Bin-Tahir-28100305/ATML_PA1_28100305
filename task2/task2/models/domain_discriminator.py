

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
   
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha                              # stash alpha for backward
        return x.view_as(x)                            # identity (no-op view)

    @staticmethod
    def backward(ctx, grad_output):
        
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha: float):
    return _GradReverse.apply(x, alpha)


def grl_alpha(progress: float, gamma: float, max_alpha: float) -> float:
   
    import math
    base = 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0   # in [0,1)
    return base * max_alpha


class DomainDiscriminator(nn.Module):
 
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
