"""Stage 3: the Subject Identifiability Score (Section 3.4, Definition 6).

Given a *frozen* encoder h_theta and the held-out test groups G_te (groups the
encoder never saw during training):

1. the images of every group g in G_te are split into two DISJOINT image
   subsets, a probe-fitting set F_g and an evaluation set Xi_g (class-balanced
   stratification; groups with fewer than ``min_per_group`` images are dropped
   -> SIS is UNDEFINED for them, a statement about benchmark redundancy);
2. a multinomial logistic probe q_phi(g | z) is fitted on F by maximum
   likelihood and evaluated on Xi:

        SIS = (1/|Xi|) sum_{i in Xi} 1[ argmax_g q_phi(g | z_i) = g_i ];

3. the control task refits the identical probe on *permuted* group labels,
   giving SIS_ctrl and the selectivity  sigma = SIS - SIS_ctrl  (Hewitt & Liang);
4. because chance  c = 1/|G_Xi|  depends on the number of probe classes, the
   chance-adjusted score  SIS~ = (SIS - c) / (1 - c)  (Eq. 5) is the quantity
   used for all cross-configuration comparisons.  SIS/c is reported only to
   show that it is misleading (its ceiling 1/c moves with |G_Xi|).

Note that the two disjointness requirements act at different levels: groups
are disjoint from encoder training, images are disjoint within the probe.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def probe_split(groups: np.ndarray, labels: np.ndarray, seed: int, min_per_group: int = 4,
                fit_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray, list]:
    """Return (fit_idx, eval_idx, kept_groups).  Within each group the split is stratified by label."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"g": groups, "y": labels, "i": np.arange(len(groups))})
    fit, ev, kept = [], [], []
    for g, sub in df.groupby("g", sort=True):
        if len(sub) < min_per_group:
            continue
        f_idx, e_idx = [], []
        for _, cls in sub.groupby("y", sort=True):
            idx = rng.permutation(cls["i"].to_numpy())
            n_fit = int(round(fit_fraction * len(idx)))
            n_fit = min(max(n_fit, 1), len(idx) - 1) if len(idx) >= 2 else n_fit
            f_idx.extend(idx[:n_fit]); e_idx.extend(idx[n_fit:])
        if not f_idx or not e_idx:  # a group whose per-class pieces are singletons: split without stratification
            idx = rng.permutation(sub["i"].to_numpy())
            half = len(idx) // 2
            f_idx, e_idx = idx[:half].tolist(), idx[half:].tolist()
        fit.extend(f_idx); ev.extend(e_idx); kept.append(g)
    return np.array(sorted(fit), dtype=int), np.array(sorted(ev), dtype=int), kept


def _make_probe(C: float = 1.0, max_iter: int = 2000):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))


def fit_and_score(Z_fit, g_fit, Z_eval, g_eval, C: float = 1.0) -> float:
    probe = _make_probe(C)
    probe.fit(Z_fit, g_fit)
    return float((probe.predict(Z_eval) == g_eval).mean())


def subject_identifiability(Z: np.ndarray, groups: np.ndarray, labels: np.ndarray, seed: int = 0,
                            min_per_group: int = 4, C: float = 1.0, with_control: bool = True) -> dict:
    """Compute SIS, the control accuracy, selectivity, chance and the chance-adjusted score."""
    Z = np.asarray(Z, dtype=np.float32)
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    fit_idx, eval_idx, kept = probe_split(groups, labels, seed, min_per_group)
    if len(kept) < 2:
        return {"sis": None, "undefined": True, "reason": "fewer than two groups with enough images",
                "n_groups": len(kept)}
    t0 = time.perf_counter()
    sis = fit_and_score(Z[fit_idx], groups[fit_idx], Z[eval_idx], groups[eval_idx], C)
    t_fit = time.perf_counter() - t0

    n_g = len(kept)
    chance = 1.0 / n_g
    out = {
        "sis": sis, "undefined": False,
        "n_groups": n_g, "n_fit": int(len(fit_idx)), "n_eval": int(len(eval_idx)),
        "chance": chance, "ratio_to_chance": sis / chance, "ratio_ceiling": 1.0 / chance,
        "sis_adjusted": (sis - chance) / (1 - chance),      # Eq. 5
        "probe_seconds": t_fit,
    }
    if with_control:
        rng = np.random.default_rng(seed + 1)
        g_perm = groups.copy()
        g_perm[fit_idx] = rng.permutation(groups[fit_idx])   # destroy the z -> g correspondence
        ctrl = fit_and_score(Z[fit_idx], g_perm[fit_idx], Z[eval_idx], groups[eval_idx], C)
        out.update({"sis_control": ctrl, "selectivity": sis - ctrl})
    return out
