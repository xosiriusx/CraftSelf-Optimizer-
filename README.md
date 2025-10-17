# CraftSelf Optimizer

Implementation of the **Craft +1 Law** applied to neural network optimization.

## Overview

The Crafts Self-Referential Law states:

> For a system whose state scales as sₙ ~ n^(a+1), the natural increment scales as Δsₙ ~ n^a.

This repository provides a minimal reference implementation of the CraftSelf Optimizer — a hybrid optimizer that enforces exponent-structured update magnitudes derived from the Craft +1 Law.

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
from craft_optimizer import CraftOptimizer
# model = ... your model ...
optimizer = CraftOptimizer(model.parameters(), a=0.0, k=1e-3, lr=1.0)
