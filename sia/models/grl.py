"""Gradient reversal (Ganin & Lempitsky, 2015) with the standard annealing schedule.

The layer is the identity on the forward pass and scales the backward gradient
by -lambda_e, with  lambda_e = 2 / (1 + exp(-10 e / E)) - 1  over epochs
e = 0, ..., E-1 (Section 3.7).  Because lambda_e starts at 0, the early epochs
train an essentially unconstrained encoder -- the root of the checkpoint-
selection failure analysed in Section 4.6.
"""
from __future__ import annotations

import math

import torch


class _ReverseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = 0.0

    def forward(self, x):
        return _ReverseGrad.apply(x, self.lam)


def grl_lambda(epoch: int, n_epochs: int, gamma: float = 10.0) -> float:
    p = epoch / max(n_epochs, 1)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
