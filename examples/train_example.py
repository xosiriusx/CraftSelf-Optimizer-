# examples/train_example.py
# Tiny example to demonstrate import and use

import torch
from torch import nn
from torch.utils.data import DataLoader
from craft_optimizer import CraftSelfOptimizer

# tiny synthetic dataset
x = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = list(zip(x, y))
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

model = nn.Linear(10, 1)
loss_fn = nn.MSELoss()
optimizer = CraftSelfOptimizer(model.parameters(), a=0.0, k=1e-3, lr=0.01)

for epoch in range(3):
    for data, target in dataloader:
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item():.6f}")
