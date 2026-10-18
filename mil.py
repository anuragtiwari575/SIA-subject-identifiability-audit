"""Attention-based multiple-instance learning head (Ilse et al., 2018).

Instance embeddings h_k of one subject are aggregated into
    z_bag = sum_k alpha_k h_k,   alpha_k = softmax_k( w^T tanh(V h_k) ),
and a linear classifier predicts the subject-level label from z_bag.
Aggregation governs how features are combined, not what they encode
(Section 3.7), which is exactly what the MIL intervention tests.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    def __init__(self, dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``h``: (K, d) instance embeddings of one bag -> (logits (n_classes,), attention (K,))."""
        a = torch.softmax(self.w(torch.tanh(self.V(h))).squeeze(-1), dim=0)
        z_bag = (a.unsqueeze(-1) * h).sum(dim=0)
        return self.classifier(z_bag), a
