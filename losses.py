"""Objectives used by the intervention strategies (Section 3.7)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectExcludedSupCon(nn.Module):
    """Supervised contrastive loss whose admissible positives exclude same-group images:

        P(i) = { j != i : y_j = y_i  and  g_j != g_i },

    so identity alone cannot satisfy the objective.  Projections must be L2-normalised.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.t = temperature

    def forward(self, proj: torch.Tensor, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        n = proj.shape[0]
        if n < 2:
            return proj.sum() * 0.0
        sim = proj @ proj.t() / self.t
        self_mask = torch.eye(n, dtype=torch.bool, device=proj.device)
        sim = sim.masked_fill(self_mask, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos = (y[:, None] == y[None, :]) & (g[:, None] != g[None, :]) & ~self_mask
        n_pos = pos.sum(dim=1)
        valid = n_pos > 0
        if not valid.any():
            return proj.sum() * 0.0
        mean_log_prob_pos = (log_prob.masked_fill(~pos, 0.0).sum(dim=1)[valid] / n_pos[valid])
        return -mean_log_prob_pos.mean()


def confusion_loss(adv_logits: torch.Tensor) -> torch.Tensor:
    """Bounded confusion term (Tzeng et al., 2015): cross-entropy of q(g | z) against the UNIFORM
    distribution over training groups,

        L_conf = -(1/N) sum_i (1/|G_tr|) sum_g log q(g | z_i),

    which drives the adversary's posterior towards uniform without the instability of
    maximising its cross-entropy.
    """
    return -F.log_softmax(adv_logits, dim=1).mean(dim=1).mean()
