# =============================================================================
# task3/methods/sam.py  — SAM: Sharpness-Aware Minimization (Step 3)
# -----------------------------------------------------------------------------
# ML CONCEPT — SAM (Foret et al. 2021):
#   Minimize the WORST-CASE loss in a small neighborhood of the weights:
#       min_theta  max_{||eps||<=rho}  L(theta + eps).
#   Solved approximately with a two-step update per batch:
#     (1) ASCENT: at theta, compute g = grad L(theta); move to the worst-case
#         nearby point  theta_adv = theta + eps,  eps = rho * g / ||g||.
#     (2) DESCENT: compute grad L(theta_adv), restore theta, and let the base
#         optimizer (AdamW) apply THAT gradient to the original theta.
#   The result favours FLAT minima, which tend to generalize better — here, to
#   the unseen Sketch domain. It needs TWO forward/backward passes per step.
#
# This file provides:
#   * SAM: an optimizer wrapper implementing first_step()/second_step();
#   * SAMMethod: a thin method object that computes the ERM loss and flags
#     uses_sam=True so train.py runs the two-pass schedule.
#
# The base optimizer is AdamW with the SAME lr/weight decay as ERM (spec).
# The frozen-BN policy is applied by train.py on BOTH passes (spec).
# =============================================================================

import torch

try:
    from methods.base_method import BaseMethod
except ImportError:
    from .base_method import BaseMethod


class SAM(torch.optim.Optimizer):
    """SAM wrapper around a base optimizer (here AdamW).

    Usage per batch (driven by train.py):
        loss1 = criterion(model(x), y); loss1.backward()
        optimizer.first_step()     # ascend to theta+eps, zero grads
        loss2 = criterion(model(x), y); loss2.backward()
        optimizer.second_step()    # restore theta, AdamW step with grad@theta+eps
    """
    def __init__(self, params, base_optimizer_cls, rho=0.05, **base_kwargs):
        assert rho >= 0, "rho must be non-negative"
        defaults = dict(rho=rho, **base_kwargs)
        super().__init__(params, defaults)
        # The actual parameter updates are delegated to this base optimizer,
        # which owns the AdamW moment buffers etc.
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def _grad_norm(self):
        """Global L2 norm of the current gradients across all parameters."""
        # Collect the norm over every param that has a gradient.
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        """ASCENT: perturb each parameter by eps = rho * g / ||g|| (worst-case)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            # scale so that the overall perturbation has L2 norm rho.
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale                    # per-parameter perturbation
                p.add_(e_w)                             # move to theta + eps
                self.state[p]["e_w"] = e_w              # remember to undo it later
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        """DESCENT: undo the perturbation, then base-optimizer step with grad@adv."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])            # restore original theta
        self.base_optimizer.step()                      # AdamW update at theta
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        # SAM requires the explicit two-step protocol above; a single step() is
        # not meaningful, so we require a closure if this is ever called directly.
        raise RuntimeError("Use first_step()/second_step() for SAM (see train.py).")


class SAMMethod(BaseMethod):
    """Method object for SAM: ERM loss + a flag telling train.py to use SAM."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.rho = cfg["method"]["rho"]
        self.uses_sam = True                            # train.py checks this flag

    def compute_loss(self, source_feats_by_domain, source_logits, source_labels):
        # SAM optimizes the SAME objective as ERM (cross-entropy on sources);
        # only the optimizer's update rule differs.
        loss_cls = self.classification_loss(source_logits, source_labels)
        logs = {"loss_total": loss_cls.item(), "loss_cls": loss_cls.item(),
                "loss_align": 0.0}
        return loss_cls, logs
