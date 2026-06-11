"""Online optimizers for the spectral filtering predictor.

The paper trains with optax.contrib.cocob; we use the original author's PyTorch
implementation (parameterfree.COCOB). OGD = SGD with eta_t = c / sqrt(t+1).
"""

from __future__ import annotations

import torch
from parameterfree import COCOB


def make_optimizer(params, kind: str = "cocob", lr: float = 1e-2):
    if kind == "cocob":
        return COCOB(params), None
    if kind == "ogd":
        opt = torch.optim.SGD(params, lr=lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 1.0 / (t + 1) ** 0.5)
        return opt, sched
    raise ValueError(f"unknown optimizer kind: {kind}")
