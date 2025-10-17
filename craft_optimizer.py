# craft_optimizer.py
# Minimal reference implementation of CraftSelf Optimizer (PyTorch)
# This is a minimal, copy-paste-ready file for priority posting.

import torch
from torch.optim import Optimizer

class CraftSelfOptimizer(Optimizer):
    """
    Minimal CraftSelf Optimizer (hybrid rule).
    This is a reference implementation for demonstration and research use.
    """

    def __init__(self, params, a=0.0, k=1e-3, lr=1.0, beta=0.9, eps=1e-8, max_scale=10.0):
        defaults = dict(a=a, k=k, lr=lr, beta=beta, eps=eps, max_scale=max_scale)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            a = group['a']; k = group['k']; lr = group['lr']
            beta = group['beta']; eps = group['eps']; max_scale = group['max_scale']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'ema_g' not in state:
                    state['ema_g'] = torch.zeros_like(p)
                ema_g = state['ema_g']
                ema_g.mul_(beta).add_((1 - beta) * g.abs())

                p_exp = a + 1
                # target magnitude follows Craft +1 scaling
                target = k * (p.abs() ** (p_exp - 1)).clamp_min(eps)
                scale = (target / (ema_g + eps)).clamp_max(max_scale)
                update = - lr * scale * g
                p.add_(update)
        return loss
