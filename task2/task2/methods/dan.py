

import torch

try:
    from methods.base_method import BaseMethod
except ImportError:
    from .base_method import BaseMethod


def _pairwise_squared_distances(x, y):
    
    x_sq = (x * x).sum(dim=1, keepdim=True)            # (n,1)
    y_sq = (y * y).sum(dim=1, keepdim=True).t()        # (1,m)
    cross = x @ y.t()                                  # (n,m) dot products
    d2 = x_sq + y_sq - 2.0 * cross                     # (n,m) squared distances
    return d2.clamp(min=0.0)


def multi_kernel_mmd2(source, target, bandwidth_multipliers):
    
    n, m = source.shape[0], target.shape[0]
    # 1) pairwise squared distances for the three blocks.
    d_ss = _pairwise_squared_distances(source, source)     # (n,n)
    d_tt = _pairwise_squared_distances(target, target)     # (m,m)
    d_st = _pairwise_squared_distances(source, target)     # (n,m)

    with torch.no_grad():
        all_d = torch.cat([d_ss.reshape(-1), d_tt.reshape(-1), d_st.reshape(-1)])
        median = all_d.median()

        median = torch.clamp(median, min=1e-8)

    mmd2 = source.new_zeros(())                             # scalar accumulator
    for mult in bandwidth_multipliers:
        bandwidth = mult * median                           # this kernel's width
        k_ss = torch.exp(-d_ss / bandwidth)                 # RBF on source-source
        k_tt = torch.exp(-d_tt / bandwidth)                 # RBF on target-target
        k_st = torch.exp(-d_st / bandwidth)                 # RBF on source-target
        mmd2 = mmd2 + k_ss.mean() - 2.0 * k_st.mean() + k_tt.mean()
    return mmd2


class DAN(BaseMethod):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lambda_mmd = cfg["method"]["lambda_mmd"]                  # e.g. 1.0
        self.mults = cfg["method"]["rbf_bandwidth_multipliers"]       # [0.5,1,2]

    def compute_loss(self, source_feats, source_logits, source_labels,
                     target_feats, target_logits, alpha):
        # Classification loss on labeled source examples.
        loss_cls = self.classification_loss(source_logits, source_labels)
        # Alignment loss: multi-kernel MMD^2 between source and target FEATURES
        # (spec: applied to the 512-d feature immediately before the head).
        mmd2 = multi_kernel_mmd2(source_feats, target_feats, self.mults)
        loss = loss_cls + self.lambda_mmd * mmd2
        logs = {"loss_total": loss.item(), "loss_cls": loss_cls.item(),
                "loss_align": mmd2.item()}
        return loss, logs
