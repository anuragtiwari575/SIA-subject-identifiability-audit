"""The audited encoder (Section 3.6).

An ImageNet-pretrained ResNet-18 (or EfficientNet-B0) truncated before the
classifier, followed by a CBAM-style *spatial* attention module

    A = sigmoid(Conv7x7([Mean_c(F); Max_c(F)])),   F_hat = F (.) A,

and global average pooling, giving a d-dimensional embedding z (d = 512 for
ResNet-18, 1280 for EfficientNet-B0).  Three heads share z:

    h_cls  linear classifier                  (L_cls)
    h_proj normalised projection for SupCon   (L_ctr)
    h_adv  linear identity head over training groups (L_adv / L_conf)

The encoder is a controlled experimental backbone, not a contribution: every
configuration has identical representational capacity.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

log = logging.getLogger("sia")


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(s))


def _backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return (feature extractor producing a B x d x h x w map, d)."""
    if name == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.resnet18(weights=weights)
        return nn.Sequential(*list(net.children())[:-2]), 512
    if name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.efficientnet_b0(weights=weights)
        return net.features, 1280
    raise ValueError(f"unknown backbone {name}")


class AuditedEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        try:
            self.features, self.dim = _backbone(backbone, pretrained)
        except Exception as e:  # e.g. no internet on Kaggle -> weights cannot be downloaded
            if not pretrained:
                raise
            log.warning("could not load ImageNet weights (%s); falling back to random initialisation", e)
            self.features, self.dim = _backbone(backbone, False)
        self.attention = SpatialAttention(7)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = self.attention(f)
        return torch.flatten(self.pool(f), 1)


class AuditModel(nn.Module):
    """Encoder + the three heads.  ``n_groups`` is the number of *training* groups (adversary classes)."""

    def __init__(self, n_classes: int, n_groups: int, backbone: str = "resnet18",
                 pretrained: bool = True, proj_dim: int = 128):
        super().__init__()
        self.encoder = AuditedEncoder(backbone, pretrained)
        d = self.encoder.dim
        self.h_cls = nn.Linear(d, n_classes)
        self.h_proj = nn.Linear(d, proj_dim)
        self.h_adv = nn.Linear(d, max(n_groups, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {"z": z, "logits": self.h_cls(z), "proj": F.normalize(self.h_proj(z), dim=1)}


def build_model(n_classes: int, n_groups: int, backbone: str = "resnet18", pretrained: bool = True) -> AuditModel:
    return AuditModel(n_classes, n_groups, backbone, pretrained)
