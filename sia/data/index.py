"""Operations on an image index (DataFrame with columns path, group, label, slice_idx, ...)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cap_slices_per_group(df: pd.DataFrame, k: int | None) -> pd.DataFrame:
    """Keep at most ``k`` evenly spaced slices per acquisition group (Section 4.1).

    Slices are ordered by ``slice_idx`` and the retained indices are
    ``round(linspace(0, n-1, k))``, so the kept slices span the whole volume.
    ``k=None`` returns the full benchmark (rho = 249 on OASIS).
    """
    if k is None:
        return df.reset_index(drop=True)
    keep = []
    for _, sub in df.groupby("group", sort=False):
        sub = sub.sort_values("slice_idx")
        n = len(sub)
        if n <= k:
            keep.append(sub)
        else:
            idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
            keep.append(sub.iloc[idx])
    return pd.concat(keep).reset_index(drop=True)


def group_labels(df: pd.DataFrame) -> pd.Series:
    """One label per group (the majority label; identical to the single label on OASIS)."""
    return df.groupby("group")["label"].agg(lambda s: int(s.value_counts().idxmax()))


def subsample_groups(df: pd.DataFrame, n_groups: int, seed: int, min_per_class: int = 3) -> pd.DataFrame:
    """Keep ``n_groups`` groups, allocated to classes proportionally (Section 4.4 budget sweep).

    Each class keeps at least ``min_per_class`` groups (when it has that many)
    so that a 60:20:20 group-disjoint split still contains every class.
    """
    g_lab = group_labels(df)
    if n_groups >= len(g_lab):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    counts = g_lab.value_counts().sort_index()
    alloc = {c: int(round(n_groups * n / len(g_lab))) for c, n in counts.items()}
    for c, n in counts.items():
        alloc[c] = min(int(n), max(alloc[c], min_per_class))
    # fix rounding so that the allocation sums to n_groups
    diff = n_groups - sum(alloc.values())
    order = sorted(alloc, key=lambda c: -counts[c])
    i = 0
    while diff != 0 and i < 10_000:
        c = order[i % len(order)]
        if diff > 0 and alloc[c] < counts[c]:
            alloc[c] += 1; diff -= 1
        elif diff < 0 and alloc[c] > min(min_per_class, counts[c]):
            alloc[c] -= 1; diff += 1
        i += 1
    chosen = []
    for c, n in alloc.items():
        pool = g_lab[g_lab == c].index.to_numpy()
        chosen.extend(rng.choice(pool, size=n, replace=False).tolist())
    return df[df["group"].isin(chosen)].reset_index(drop=True)


def describe_index(df: pd.DataFrame) -> dict:
    """Quick summary used in logs and Stage 1 tables."""
    g_lab = group_labels(df)
    return {
        "n_images": int(len(df)),
        "n_groups": int(df["group"].nunique()),
        "images_per_class": {int(c): int(n) for c, n in df["label"].value_counts().sort_index().items()},
        "groups_per_class": {int(c): int(n) for c, n in g_lab.value_counts().sort_index().items()},
    }
