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
# Quick Use Example for MaxEfficiencyCraftSelfOptimizer
# Works out of the box

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

# Import your optimizer
# Make sure the file max_efficiency_craft_optimizer.py is in the same folder
from max_efficiency_craft_optimizer import MaxEfficiencyCraftSelfOptimizer

# Create a simple model
model = nn.Linear(10, 1)

# Create a synthetic dataset
x = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = list(zip(x, y))
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Define loss function
loss_fn = nn.MSELoss()

# Instantiate the optimizer
optimizer = MaxEfficiencyCraftSelfOptimizer(
    model.parameters(),
    a=0.5,
    k=1e-3,
    lr=0.01,
    beta=0.9,
    eps=1e-8,
    max_scale=10.0,
    auto_tune=True,
    hybrid_optimizer=Adam(model.parameters(), lr=0.001),  # Optional hybrid
    clip_grad=1.0,
    min_scale=1e-4
)

# Training loop
for epoch in range(10):  # reduced epochs for quick testing
    for data, target in dataloader:
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item():.6f}")
