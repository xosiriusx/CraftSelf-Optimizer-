# CraftSelfOptimizerv2.py
# CraftSelfOptimizer v2 with hybrid fallback, adaptive tuning, cooldown, and optional LR scheduling

import torch
from torch.optim import Optimizer

class CraftSelfOptimizerv2(Optimizer):
    """
    CraftSelfOptimizerv2:
    - Adaptive tuning of `a` and `k`
    - EMA gradient scaling
    - Hybrid optimizer fallback with cooldown
    - Optional learning rate scheduling
    - Optional debug logging
    """
    def __init__(self, params, a=0.5, k=1e-3, lr=0.01, beta=0.9, eps=1e-8,
                 max_scale=10.0, min_scale=1e-4, auto_tune=True,
                 hybrid_optimizer=None, hybrid_cooldown=5,
                 clip_grad=1.0, scheduler=None, debug=False):
        defaults = dict(a=a, k=k, lr=lr, beta=beta, eps=eps,
                        max_scale=max_scale, min_scale=min_scale,
                        auto_tune=auto_tune, clip_grad=clip_grad)
        super().__init__(params, defaults)
        self.hybrid_optimizer = hybrid_optimizer
        self.hybrid_cooldown = hybrid_cooldown
        self.last_switch_epoch = -hybrid_cooldown
        self.current_epoch = 0
        self.loss_history = []
        self.switch_triggered = False
        self.scheduler = scheduler
        self.debug = debug

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update loss history
        if loss is not None:
            self.loss_history.append(loss.item())
            if len(self.loss_history) > 10:  # keep last 10 for analysis
                self.loss_history = self.loss_history[-10:]

        # Check hybrid fallback
        if self.hybrid_optimizer and not self.switch_triggered:
            if len(self.loss_history) >= 5:
                loss_std = torch.tensor(self.loss_history[-5:]).std().item()
                if loss_std < 1e-4 and (self.current_epoch - self.last_switch_epoch >= self.hybrid_cooldown):
                    self.switch_triggered = True
                    self.last_switch_epoch = self.current_epoch
                    if self.debug:
                        print(f"[CraftSelfOptimizerv2] Hybrid switch triggered at epoch {self.current_epoch}")

        # Use hybrid optimizer if triggered
        if self.switch_triggered and self.hybrid_optimizer:
            self.hybrid_optimizer.step(closure)
            return

        # CraftSelf parameter update
        for group in self.param_groups:
            a = group['a']
            k = group['k']
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']
            max_scale = group['max_scale']
            min_scale = group['min_scale']
            clip_grad = group['clip_grad']

            # Auto-tune a and k
            if group['auto_tune'] and len(self.loss_history) >= 2:
                param_stds = [p.std() for p in group['params'] if p.numel() > 1]
                if param_stds:
                    avg_std = torch.stack(param_stds).mean().item()
                    group['a'] = float(min(0.5, max(0.0, avg_std * 0.5)))

                loss_ratio = self.loss_history[-1] / (self.loss_history[-2] + eps)
                group['k'] = float(k * min(2.0, max(0.5, loss_ratio)))

                if self.debug:
                    print(f"[CraftSelfOptimizerv2] Epoch {self.current_epoch}: a={group['a']:.4f}, k={group['k']:.6f}, loss_ratio={loss_ratio:.4f}")

            # Gradient update
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                g = g.clamp(-clip_grad, clip_grad)

                state = self.state[p]
                if 'ema_g' not in state:
                    state['ema_g'] = torch.zeros_like(p)
                ema_g = state['ema_g']
                ema_g.mul_(beta).add_((1 - beta) * g.abs())

                p_exp = group['a'] + 1
                target = (group['k'] * (p.abs() ** (p_exp - 1))).clamp_min(eps)
                scale = (target / (ema_g + eps)).clamp(min_scale, max_scale)
                p.add_(-lr * scale * g)

        self.current_epoch += 1

        # Step learning rate scheduler if provided
        if self.scheduler is not None:
            self.scheduler.step()

        # Reset hybrid switch if cooldown expired
        if self.switch_triggered and (self.current_epoch - self.last_switch_epoch >= self.hybrid_cooldown):
            self.switch_triggered = False
            if self.debug:
                print(f"[CraftSelfOptimizerv2] Hybrid switch reset at epoch {self.current_epoch}")
