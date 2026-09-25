

import itertools
import torch

try:
    from methods.base_method import BaseMethod
except ImportError:
    from .base_method import BaseMethod


def _pairwise_squared_distances(x, y):
    
    x_sq = (x * x).sum(dim=1, keepdim=True)            # (n,1)
    y_sq = (y * y).sum(dim=1, keepdim=True).t()        # (1,m)
    cross = x @ y.t()                                  # (n,m)
    return (x_sq + y_sq - 2.0 * cross).clamp(min=0.0)


def multi_kernel_mmd2(source, target, bandwidth_multipliers):
   
    d_ss = _pairwise_squared_distances(source, source)
    d_tt = _pairwise_squared_distances(target, target)
    d_st = _pairwise_squared_distances(source, target)

    with torch.no_grad():
        all_d = torch.cat([d_ss.reshape(-1), d_tt.reshape(-1), d_st.reshape(-1)])
        median = torch.clamp(all_d.median(), min=1e-8)

    mmd2 = source.new_zeros(())
    for mult in bandwidth_multipliers:
        bw = mult * median
        k_ss = torch.exp(-d_ss / bw)
        k_tt = torch.exp(-d_tt / bw)
        k_st = torch.exp(-d_st / bw)
        mmd2 = mmd2 + k_ss.mean() - 2.0 * k_st.mean() + k_tt.mean()
    return mmd2


class DANDG(BaseMethod):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lambda_dg = cfg["method"]["lambda_dg"]                    # e.g. 1.0
        self.mults = cfg["method"]["rbf_bandwidth_multipliers"]       # [0.5,1,2]
       
        self.warmup_steps = int(cfg["method"].get("mmd_warmup_steps", 0))
        self.align_scale = 1.0            # updated each step by the loop when warming up

    def set_align_scale(self, s):
        self.align_scale = float(s)

    def compute_loss(self, source_feats_by_domain, source_logits, source_labels):
        
        loss_cls = self.classification_loss(source_logits, source_labels)

        domains = list(source_feats_by_domain.keys())
        pair_mmds = []
        for e, e2 in itertools.combinations(domains, 2):               # {P,A},{P,C},{A,C}
            pair_mmds.append(
                multi_kernel_mmd2(source_feats_by_domain[e],
                                  source_feats_by_domain[e2], self.mults))
        align = torch.stack(pair_mmds).sum() / len(pair_mmds)

        
        loss = loss_cls + (self.lambda_dg * self.align_scale) * align
        logs = {"loss_total": loss.item(), "loss_cls": loss_cls.item(),
                "loss_align": align.item(), "align_scale": self.align_scale}
        return loss, logs
