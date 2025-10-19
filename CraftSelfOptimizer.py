# mCraftSelfOptimizer.py
# Optimized CraftSelfOptimizer for maximum efficiency
# Builds on Crafts Self-Referential Law with adaptive tuning and hybrid switching

import torch
from torch.optim import Optimizer

class MaxEfficiencyCraftSelfOptimizer(Optimizer):
    """
    Highly efficient CraftSelfOptimizer with adaptive tuning and hybrid switching.
    Maximizes convergence speed and stability while aligning with Crafts Self-Referential Law.
    """
    def __init__(self, params, a=0.5, k=1e-3, lr=0.01, beta=0.9, eps=1e-8, max_scale=10.0,
                 auto_tune=True, hybrid_optimizer=None, clip_grad=1.0, min_scale=1e-4):
        defaults = dict(a=a, k=k, lr=lr, beta=beta, eps=eps, max_scale=max_scale,
                        auto_tune=auto_tune, clip_grad=clip_grad, min_scale=min_scale)
        super().__init__(params, defaults)
        self.hybrid_optimizer = hybrid_optimizer
        self.current_epoch = 0
        self.loss_history = []
        self.switch_triggered = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Hybrid switch logic: detect loss plateau (std of last 5 losses < 1e-4)
        if self.hybrid_optimizer and not self.switch_triggered:
            if loss is not None:
                self.loss_history.append(loss.item())
                if len(self.loss_history) >= 5:
                    self.loss_history = self.loss_history[-5:]
                    loss_std = torch.tensor(self.loss_history).std().item()
                    if loss_std < 1e-4 and self.current_epoch > 10:  # Ensure min epochs
                        self.switch_triggered = True

        if self.switch_triggered and self.hybrid_optimizer:
            self.hybrid_optimizer.step(closure)
            return

        # CraftSelf step
        for group in self.param_groups:
            a = group['a']
            k = group['k']
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']
            max_scale = group['max_scale']
            clip_grad = group['clip_grad']
            min_scale = group['min_scale']

            # Auto-tune a and k every epoch
            if group['auto_tune']:
                param_stds = [p.std() for p in group['params'] if p.numel() > 1]
                if param_stds:
                    avg_std = torch.stack(param_stds).mean().item()
                    group['a'] = min(0.5, max(0.0, avg_std * 0.5))  # Scale with param variance
                if len(self.loss_history) >= 2:
                    loss_ratio = self.loss_history[-1] / (self.loss_history[-2] + eps)
                    group['k'] = k * min(2.0, max(0.5, loss_ratio))  # Adjust k based on loss trend

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                # Gradient clipping for stability
                g = g.clamp(-clip_grad, clip_grad)
                state = self.state[p]
                if 'ema_g' not in state:
                    state['ema_g'] = torch.zeros_like(p)
                ema_g = state['ema_g']
                ema_g.mul_(beta).add_((1 - beta) * g.abs())

                p_exp = a + 1
                target = k * (p.abs() ** (p_exp - 1)).clamp_min(eps)
                scale = (target / (ema_g + eps)).clamp(min_scale, max_scale)
                update = -lr * scale * g
                p.add_(update)

        self.current_epoch += 1

# Example usage (can be added as max_efficiency_train_example.py)
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from max_efficiency_craft_optimizer import MaxEfficiencyCraftSelfOptimizer

# Synthetic dataset
x = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = list(zip(x, y))
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

model = nn.Linear(10, 1)
loss_fn = nn.MSELoss()
optimizer = MaxEfficiencyCraftSelfOptimizer(
    model.parameters(),
    a=0.5, k=1e-3, lr=0.01, beta=0.9, eps=1e-8, max_scale=10.0,
    auto_tune=True,  # Enable adaptive a and k
    hybrid_optimizer=Adam(model.parameters(), lr=0.001),  # Switch to Adam on plateau
    clip_grad=1.0, min_scale=1e-4
)

for epoch in range(100):
    for data, target in dataloader:
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} loss: {loss.item():.6f}")
"""
