"""Image-level and group-disjoint partitions (Definitions 1 and 2).

Both protocols produce 60:20:20 train/validation/test partitions of the *same*
image index and differ only in the unit that is assigned to a part:

* image-level   -- images are assigned independently (stratified by label), so
                   a subject's slices can straddle train and test;
* group-disjoint -- whole groups are assigned, G_tr, G_va, G_te are pairwise
                   disjoint, and every test image comes from an unseen subject.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .index import group_labels


@dataclass
class Partition:
    protocol: str            # "image" or "group"
    seed: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> dict:
        out = {"protocol": self.protocol, "seed": self.seed}
        for name, part in (("train", self.train), ("val", self.val), ("test", self.test)):
            out[name] = {"n_images": int(len(part)), "n_groups": int(part["group"].nunique())}
        tr, te = set(self.train["group"]), set(self.test["group"])
        out["groups_shared_train_test"] = int(len(tr & te))
        return out


def _three_way(items, strat, seed, fracs):
    """Split ``items`` into train/val/test with the given fractions, stratifying when possible."""
    f_tr, f_va, f_te = fracs
    try:
        tr, rest, _, strat_rest = train_test_split(items, strat, test_size=f_va + f_te, random_state=seed, stratify=strat)
        va, te = train_test_split(rest, test_size=f_te / (f_va + f_te), random_state=seed, stratify=strat_rest)
    except ValueError:  # a class with too few members to stratify
        tr, rest = train_test_split(items, test_size=f_va + f_te, random_state=seed)
        va, te = train_test_split(rest, test_size=f_te / (f_va + f_te), random_state=seed)
    return tr, va, te


def image_level_split(df: pd.DataFrame, seed: int, fracs=(0.6, 0.2, 0.2)) -> Partition:
    idx = np.arange(len(df))
    tr, va, te = _three_way(idx, df["label"].to_numpy(), seed, fracs)
    return Partition("image", seed, df.iloc[np.sort(tr)].reset_index(drop=True),
                     df.iloc[np.sort(va)].reset_index(drop=True), df.iloc[np.sort(te)].reset_index(drop=True))


def group_disjoint_split(df: pd.DataFrame, seed: int, fracs=(0.6, 0.2, 0.2), max_tries: int = 50) -> Partition:
    """Group-disjoint partition; retried with a derived seed until every part holds every class."""
    g_lab = group_labels(df)
    groups = g_lab.index.to_numpy()
    strat = g_lab.to_numpy()
    n_classes = df["label"].nunique()
    for t in range(max_tries):
        s = seed + 10_000 * t
        tr, va, te = _three_way(groups, strat, s, fracs)
        parts = [df[df["group"].isin(p)] for p in (tr, va, te)]
        if all(p["label"].nunique() == n_classes for p in parts):
            break
    else:
        import logging
        logging.getLogger("sia").warning("group-disjoint split (seed %d): some class is absent from a part", seed)
    return Partition("group", seed, *[p.reset_index(drop=True) for p in parts])


def make_partition(df: pd.DataFrame, protocol: str, seed: int) -> Partition:
    if protocol == "image":
        return image_level_split(df, seed)
    if protocol == "group":
        return group_disjoint_split(df, seed)
    raise ValueError(protocol)
