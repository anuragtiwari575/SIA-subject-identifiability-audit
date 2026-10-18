"""Statistical resolution of the benchmark (Section 4.7).

Paired comparisons across group-disjoint partitions (t-test, Wilcoxon, Cohen's
d_z, Pearson / Spearman correlation of per-partition changes) and the power
analysis

    n = (z_{1-alpha/2} + z_{1-beta})^2 (SD / delta)^2

giving the number of independent partitions required to resolve a margin
``delta`` with paired standard deviation ``SD``.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_comparison(a: np.ndarray, b: np.ndarray) -> dict:
    """Statistics of the paired difference ``a - b`` (same partitions)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    out = {"n": int(n), "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if n > 1 else float("nan"),
           "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum())}
    if n > 1 and out["sd_diff"] > 0:
        t, p = stats.ttest_rel(a, b)
        out.update({"t": float(t), "p_ttest": float(p), "cohen_dz": float(d.mean() / out["sd_diff"])})
        try:
            w = stats.wilcoxon(a, b)
            out.update({"wilcoxon_W": float(w.statistic), "p_wilcoxon": float(w.pvalue)})
        except ValueError:
            pass
    return out


def correlation(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {"pearson_r": float(pr), "p_pearson": float(pp), "spearman_rho": float(sr), "p_spearman": float(sp)}


def required_partitions(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> float:
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) ** 2 * (sd / delta) ** 2)
