

import torch

try:
    from methods.base_method import BaseMethod
except ImportError:
    from .base_method import BaseMethod


class SAM(torch.optim.Optimizer):

    def __init__(self, params, base_optimizer_cls, rho=0.05, **base_kwargs):
        assert rho >= 0, "rho must be non-negative"
        defaults = dict(rho=rho, **base_kwargs)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def _grad_norm(self):
       
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
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])            # restore original theta
        self.base_optimizer.step()                      # AdamW update at theta
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        
        raise RuntimeError("Use first_step()/second_step() for SAM (see train.py).")


class SAMMethod(BaseMethod):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.rho = cfg["method"]["rho"]
        self.uses_sam = True                            # train.py checks this flag

    def compute_loss(self, source_feats_by_domain, source_logits, source_labels):
        
        loss_cls = self.classification_loss(source_logits, source_labels)
        logs = {"loss_total": loss_cls.item(), "loss_cls": loss_cls.item(),
                "loss_align": 0.0}
        return loss_cls, logs
