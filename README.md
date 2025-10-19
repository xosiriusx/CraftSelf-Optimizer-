# CraftSelf Optimizer

Implementation of the **Crafts Self-Referential Law** applied to neural network optimization.

## Overview

The Crafts Self-Referential Law states:

> For a system whose state scales as sₙ ~ n^(a+1), the natural increment scales as Δsₙ ~ n^a.

This repository provides a minimal reference implementation of the CraftSelf Optimizer — a hybrid optimizer that enforces exponent-structured update magnitudes derived from the Crafts Self-Referential Law.

## Priority / Reference

This work was first publicly deposited and timestamped here:

🔗 **Zenodo DOI:** https://doi.org/10.5281/zenodo.17206305

If you use this optimizer in academic work, **please cite the DOI above**.

## Licensing (Dual License)

- ✅ **Free for research, educational, or personal use.**  
- 💼 **Commercial users (companies, startups, or monetized projects) must obtain a commercial license.**  
  Please contact either:
  - xosiriusx@gmail.com  
  - ccraft0013@yahoo.com

**Suggested starting commercial licensing discussion fee:** USD 100,000 (negotiable depending on use case). Contact above for terms.

## Quick Usage (PyTorch)

```python
# CraftSelfOptimizer hybrid demo 
import torch
from torch import nn
from torch.optim import Adam
from mCraftSelfOptimizer import MaxEfficiencyCraftSelfOptimizer as CraftSelfOptimizer
from torch.utils.data import DataLoader

# Larger synthetic dataset
x = torch.randn(2000, 20)  # more samples, higher dimensionality
y = torch.randn(2000, 1)
dataset = list(zip(x, y))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Slightly bigger model
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
loss_fn = nn.MSELoss()

# CraftSelfOptimizer with hybrid Adam fallback
optimizer = CraftSelfOptimizer(
    model.parameters(),
    a=0.5, k=1e-3, lr=0.01,
    auto_tune=True,
    hybrid_optimizer=Adam(model.parameters(), lr=0.001)
)

# Training loop
for epoch in range(30):  # more epochs to show adaptive behavior
    epoch_loss = 0
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * data.size(0)
    avg_loss = epoch_loss / len(dataset)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")
