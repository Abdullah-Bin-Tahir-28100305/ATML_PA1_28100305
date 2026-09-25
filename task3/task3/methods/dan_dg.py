# =============================================================================
# task3/methods/dan_dg.py  — DAN-DG: pairwise SOURCE-domain MMD (Step 2)
# -----------------------------------------------------------------------------
# ML CONCEPT — MMD (recap) and its DG adaptation:
#   MMD^2 between two feature sets, in an RKHS with kernel k, is
#       MMD^2 = E[k(s,s')] - 2 E[k(s,t)] + E[k(t,t')],
#   zero iff the two distributions match under that kernel family. DAN (Task 2)
#   used it between SOURCE and TARGET. DAN-DG uses the SAME estimator but between
#   every pair of the three SOURCE domains, never touching the target:
#       L_DAN-DG = L_ERM + (lambda_DG/3) * sum_{e<e'} MMD^2(F(X_e), F(X_e')).
#   With sources {P,A,C} there are 3 unordered pairs {P,A},{P,C},{A,C}, hence /3.
#
# IMPORTANT (spec): "Use the same MMD implementation and kernel construction as
# Task 2." So the kernel code below is line-for-line the Task 2 estimator:
# a SUM of three RBF kernels at 0.5/1/2 x the median pairwise squared distance
# of the combined pair, computed with the kernel trick (no explicit feature map).
# =============================================================================

import itertools
import torch

try:
    from methods.base_method import BaseMethod
except ImportError:
    from .base_method import BaseMethod


def _pairwise_squared_distances(x, y):
    """Squared Euclidean distances between rows of x (n,d) and y (m,d) -> (n,m).

    ||xi - yj||^2 = ||xi||^2 + ||yj||^2 - 2 xi.yj, vectorized; clamped >= 0.
    """
    x_sq = (x * x).sum(dim=1, keepdim=True)            # (n,1)
    y_sq = (y * y).sum(dim=1, keepdim=True).t()        # (1,m)
    cross = x @ y.t()                                  # (n,m)
    return (x_sq + y_sq - 2.0 * cross).clamp(min=0.0)


def multi_kernel_mmd2(source, target, bandwidth_multipliers):
    """Multi-kernel (summed RBF) squared MMD — identical to Task 2's estimator.

    'source'/'target' here are just the two feature sets of a domain PAIR (the
    names are kept from the MMD definition; both are source domains in DG).
    Base bandwidth = median of ALL pairwise squared distances in the combined
    pair (median heuristic); accumulate the RBF-MMD over the three multipliers.
    """
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
        # MMD WARM-UP (stability, keeps lambda_DG=1 as the final weight).
        # `align_scale` in [0,1] multiplies the MMD term this step; the training
        # loop ramps it from 0 to 1 over the first `warmup_steps` updates so the
        # CLASSIFIER learns first and the alignment pressure engages only once
        # useful features exist (same logic as DANN's alpha-ramp). At/after
        # warm-up, align_scale=1, so the effective penalty is exactly
        # (lambda_DG/3)*sum MMD^2 with lambda_DG=1 — the prescribed objective.
        # Set warmup_steps=0 (default) to reproduce the original no-warmup run.
        self.warmup_steps = int(cfg["method"].get("mmd_warmup_steps", 0))
        self.align_scale = 1.0            # updated each step by the loop when warming up

    def set_align_scale(self, s):
        """Called by the training loop to set the current MMD warm-up multiplier."""
        self.align_scale = float(s)

    def compute_loss(self, source_feats_by_domain, source_logits, source_labels):
        # 1) ERM classification loss on ALL source labels (spec: class loss uses
        #    all source labels; MMD only shapes the marginal feature distributions).
        loss_cls = self.classification_loss(source_logits, source_labels)

        # 2) Average MMD^2 over the 3 unordered source-domain PAIRS.
        domains = list(source_feats_by_domain.keys())
        pair_mmds = []
        for e, e2 in itertools.combinations(domains, 2):               # {P,A},{P,C},{A,C}
            pair_mmds.append(
                multi_kernel_mmd2(source_feats_by_domain[e],
                                  source_feats_by_domain[e2], self.mults))
        # sum over pairs then divide by number of pairs (=3) -> the (1/3) factor.
        align = torch.stack(pair_mmds).sum() / len(pair_mmds)

        # Apply the warm-up multiplier (=1 outside warm-up, so the prescribed
        # lambda_DG=1 objective is recovered once warm-up completes).
        loss = loss_cls + (self.lambda_dg * self.align_scale) * align
        logs = {"loss_total": loss.item(), "loss_cls": loss_cls.item(),
                "loss_align": align.item(), "align_scale": self.align_scale}
        return loss, logs
