"""Frozen-encoder feature extraction (input to Stage 3) with timing and peak-memory accounting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.dataset import SliceDataset
from ..utils import Timer


@torch.no_grad()
def extract_features(encoder: torch.nn.Module, df: pd.DataFrame, device, img_size: int = 128,
                     channels: str = "all", batch_size: int = 128, num_workers: int = 2) -> dict:
    """Return {"Z": (N, d) float32, "groups": (N,), "labels": (N,), "seconds": float, "peak_mem_mb": float}."""
    ds = SliceDataset(df, img_size=img_size, train=False, channels=channels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    encoder = encoder.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    feats = []
    with Timer(device) as t:
        for x, _, _, _ in dl:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                z = encoder(x.to(device, non_blocking=True))
            feats.append(z.float().cpu())
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
    return {"Z": torch.cat(feats).numpy(), "groups": df["group"].to_numpy(), "labels": df["label"].to_numpy(),
            "seconds": t.seconds, "peak_mem_mb": peak}
