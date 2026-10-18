"""PyTorch datasets: single slices (classification / probing) and per-subject bags (MIL)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int = 128, train: bool = False):
    """Light affine augmentation + horizontal flips for training; resize only for evaluation (Section 4.1)."""
    ops = [T.Resize((img_size, img_size))]
    if train:
        ops += [T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.RandomHorizontalFlip()]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def load_image(path: str, channels: str = "all") -> Image.Image:
    """Load a slice as a 3-channel PIL image.

    ``channels='all'``   : keep the stored channels (grey images are replicated to RGB).
    ``channels='flair'`` : for TCGA-LGG .tif files whose channels are
                           (pre-contrast, FLAIR, post-contrast), keep the FLAIR channel only.
    """
    img = Image.open(path)
    if channels == "flair" and img.mode == "RGB":
        g = img.split()[1]
        return Image.merge("RGB", (g, g, g))
    return img.convert("RGB")


class SliceDataset(Dataset):
    """One item per image: (tensor, label, group_index, row_index)."""

    def __init__(self, df: pd.DataFrame, img_size: int = 128, train: bool = False,
                 channels: str = "all", group_to_idx: dict | None = None):
        self.df = df.reset_index(drop=True)
        self.paths = self.df["path"].tolist()
        self.labels = self.df["label"].to_numpy(dtype=np.int64)
        self.channels = channels
        self.tf = build_transforms(img_size, train)
        if group_to_idx is None:
            group_to_idx = {g: i for i, g in enumerate(sorted(self.df["group"].unique()))}
        self.group_to_idx = group_to_idx
        self.groups = np.array([group_to_idx.get(g, -1) for g in self.df["group"]], dtype=np.int64)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = self.tf(load_image(self.paths[i], self.channels))
        return x, int(self.labels[i]), int(self.groups[i]), i


class BagDataset(Dataset):
    """Bags of ``bag_size`` slices from one subject, for attention-based MIL (Section 3.7).

    For training, each subject contributes ceil(n_g / bag_size) random bags per
    epoch; for evaluation the subject's slices are chunked deterministically so
    that every slice is used once.
    """

    def __init__(self, df: pd.DataFrame, bag_size: int = 8, img_size: int = 128, train: bool = False,
                 channels: str = "all", seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.bag_size = bag_size
        self.channels = channels
        self.train = train
        self.tf = build_transforms(img_size, train)
        self.rng = np.random.default_rng(seed)
        self.bags: list[tuple[np.ndarray, int, str]] = []
        self.resample()

    def resample(self):
        self.bags = []
        for g, sub in self.df.groupby("group", sort=True):
            idx = sub.index.to_numpy()
            label = int(sub["label"].value_counts().idxmax())
            n_bags = max(1, int(np.ceil(len(idx) / self.bag_size)))
            if self.train:
                for _ in range(n_bags):
                    take = self.rng.choice(idx, size=min(self.bag_size, len(idx)), replace=len(idx) < self.bag_size)
                    self.bags.append((take, label, g))
            else:
                order = idx[np.argsort(sub["slice_idx"].to_numpy())]
                for chunk in np.array_split(order, n_bags):
                    self.bags.append((chunk, label, g))

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, i):
        idx, label, g = self.bags[i]
        x = torch.stack([self.tf(load_image(self.df.at[j, "path"], self.channels)) for j in idx])
        return x, label, g


def bag_collate(batch):
    """Bags may have different sizes: return a list of tensors plus label/group arrays."""
    xs, ys, gs = zip(*batch)
    return list(xs), torch.tensor(ys, dtype=torch.long), list(gs)
