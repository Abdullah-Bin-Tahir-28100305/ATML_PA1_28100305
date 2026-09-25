# =============================================================================
# task2/methods/dan.py  — DAN: multi-kernel MMD alignment (Step 2)
# -----------------------------------------------------------------------------
# ML CONCEPT — MAXIMUM MEAN DISCREPANCY (MMD):
#   MMD measures the distance between two distributions by comparing their means
#   in a Reproducing Kernel Hilbert Space (RKHS). For a kernel k, the squared MMD
#   between source S and target T is:
#       MMD^2 = E[k(s,s')] - 2 E[k(s,t)] + E[k(t,t')].
#   If MMD^2 is 0, the two feature distributions are indistinguishable under that
#   kernel family. DAN adds lambda * MMD^2 to the loss so training pulls the
#   source and target FEATURE distributions together — WITHOUT using target
#   labels (it only sees the target features as an unlabeled cloud). This is
#   MARGINAL alignment: it matches P(F(X)) overall, ignoring class identity.
#
# THE 'KERNEL TRICK' (spec: "evaluates similarities directly rather than
#   constructing phi"): we never build the RKHS feature map phi; we only evaluate
#   the kernel k on pairs, which is all MMD needs.
#
# MULTI-KERNEL RBF + MEDIAN HEURISTIC (spec):
#   We use a SUM of three RBF (Gaussian) kernels whose bandwidths are 0.5, 1, and
#   2 times the MEDIAN pairwise squared distance in the current combined batch.
#   Using several bandwidths makes MMD sensitive at multiple scales; the median
#   heuristic picks a sensible, data-driven base bandwidth each batch.
# =============================================================================

import torch

try:
    from methods.base_method import BaseMethod
except ImportError:
    from .base_method import BaseMethod


def _pairwise_squared_distances(x, y):
    """Return the matrix of squared Euclidean distances between rows of x and y.

    ||xi - yj||^2 = ||xi||^2 + ||yj||^2 - 2 xi.yj, computed in a vectorized way.
    Shapes: x (n,d), y (m,d) -> (n,m). Clamped at 0 to avoid tiny negatives from
    floating-point error.
    """
    x_sq = (x * x).sum(dim=1, keepdim=True)            # (n,1)
    y_sq = (y * y).sum(dim=1, keepdim=True).t()        # (1,m)
    cross = x @ y.t()                                  # (n,m) dot products
    d2 = x_sq + y_sq - 2.0 * cross                     # (n,m) squared distances
    return d2.clamp(min=0.0)


def multi_kernel_mmd2(source, target, bandwidth_multipliers):
    """Compute the multi-kernel (summed RBF) squared MMD between two feature sets.

    Steps:
      1) Build the combined pairwise squared-distance blocks (ss, tt, st).
      2) Set the base bandwidth to the MEDIAN of ALL pairwise squared distances
         in the combined batch (the median heuristic).
      3) For each multiplier m in {0.5,1,2}, form an RBF kernel
         k(a,b)=exp(-||a-b||^2 / (m * median)) and accumulate.
      4) MMD^2 = mean(k_ss) - 2 mean(k_st) + mean(k_tt), summed over kernels.
    """
    n, m = source.shape[0], target.shape[0]
    # 1) pairwise squared distances for the three blocks.
    d_ss = _pairwise_squared_distances(source, source)     # (n,n)
    d_tt = _pairwise_squared_distances(target, target)     # (m,m)
    d_st = _pairwise_squared_distances(source, target)     # (n,m)

    # 2) median heuristic over the COMBINED batch's pairwise distances.
    with torch.no_grad():
        all_d = torch.cat([d_ss.reshape(-1), d_tt.reshape(-1), d_st.reshape(-1)])
        median = all_d.median()
        # guard against a degenerate (near-zero) median to avoid div-by-zero.
        median = torch.clamp(median, min=1e-8)

    mmd2 = source.new_zeros(())                             # scalar accumulator
    for mult in bandwidth_multipliers:
        bandwidth = mult * median                           # this kernel's width
        k_ss = torch.exp(-d_ss / bandwidth)                 # RBF on source-source
        k_tt = torch.exp(-d_tt / bandwidth)                 # RBF on target-target
        k_st = torch.exp(-d_st / bandwidth)                 # RBF on source-target
        # Unbiased-ish estimate: plain means of each block (standard in DAN impls)
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
