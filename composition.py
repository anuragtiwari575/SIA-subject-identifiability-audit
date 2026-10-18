"""Stage 1: composition analysis (Section 3.2).

For a benchmark D = {(x_i, y_i, g_i)} with M groups:

    rho = N / M                                  group redundancy (Definition 3)
    eta = mean_{g: n_g >= 2}  sum_c n_gc (n_gc - 1) / (n_g (n_g - 1))
                                                 within-group label homogeneity (Definition 4)
    1/C                                          reference level under label-independent groups
    M_c = |{g : n_gc > 0}|                       class coverage; K-fold group-disjoint evaluation
                                                 is feasible only if M_c >= K for every class.

Proposition 1: identity can act as a shortcut only if rho > 1 AND eta > 1/C.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def homogeneity(df: pd.DataFrame) -> float:
    """eta: probability that two observations drawn from the same group share a label."""
    vals = []
    for _, sub in df.groupby("group", sort=False):
        n = len(sub)
        if n < 2:
            continue
        counts = sub["label"].value_counts().to_numpy(dtype=float)
        vals.append(float((counts * (counts - 1)).sum() / (n * (n - 1))))
    return float(np.mean(vals)) if vals else float("nan")


def composition_analysis(df: pd.DataFrame, k_folds: int = 3, class_names: list[str] | None = None) -> dict:
    n = int(len(df))
    m = int(df["group"].nunique())
    n_classes = int(df["label"].nunique())
    rho = n / m
    eta = homogeneity(df)
    ref = 1.0 / n_classes
    per_class = []
    for c, sub in df.groupby("label", sort=True):
        m_c = int(sub["group"].nunique())
        per_class.append({
            "label": int(c),
            "class_name": class_names[int(c)] if class_names else str(c),
            "n_images": int(len(sub)),
            "n_groups": m_c,
            "evaluable_K": bool(m_c >= k_folds),
        })
    return {
        "N": n, "M": m, "C": n_classes,
        "rho": rho, "eta": eta, "one_over_C": ref, "eta_minus_ref": eta - ref,
        "shortcut_possible": bool(rho > 1 and eta > ref),      # Proposition 1
        "k_folds": k_folds,
        "statistically_evaluable": all(pc["evaluable_K"] for pc in per_class),
        "per_class": per_class,
    }


def composition_table(results: dict[str, dict]) -> str:
    """Markdown rendering of Table 1 (Stage 1 composition)."""
    lines = ["| Benchmark | N | M | rho | eta | 1/C | eta - 1/C |", "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['N']:,} | {r['M']} | {r['rho']:.1f} | {r['eta']:.3f} | "
                     f"{r['one_over_C']:.3f} | {r['eta_minus_ref']:+.3f} |")
    return "\n".join(lines)
