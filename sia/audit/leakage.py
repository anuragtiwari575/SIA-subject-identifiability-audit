"""Stage 2: paired protocol ablation and the Leakage Index (Section 3.3).

    LI = ( A(pi_img) - A(pi_grp) ) / ( A(pi_img) - c0 ),   c0 = 1/C,   A(pi_img) > c0,

the fraction of above-chance balanced accuracy that does not survive
group-disjoint evaluation.  LI is UNDEFINED when the image-level protocol
itself performs at (or below) chance.

Confidence intervals are percentile bootstraps over *test-set predictions*:
for every bootstrap draw, the test predictions of each paired partition are
resampled with replacement under both protocols, balanced accuracy is
recomputed, and LI / the drop are averaged over partitions.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def balanced_accuracy(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def per_class_recall(y_true, y_pred, n_classes: int) -> list[float]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.diag(cm) / cm.sum(axis=1)
    return [float(r) if np.isfinite(r) else float("nan") for r in rec]


def row_normalised_confusion(y_true, y_pred, n_classes: int) -> list[list[float]]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm = cm / cm.sum(axis=1, keepdims=True)
    return np.nan_to_num(cm).tolist()


def leakage_index(acc_img: float, acc_grp: float, n_classes: int) -> float | None:
    c0 = 1.0 / n_classes
    if acc_img <= c0:
        return None  # UNDEFINED
    return float((acc_img - acc_grp) / (acc_img - c0))


def bootstrap_leakage(pairs: list[dict], n_classes: int, n_boot: int = 2000, seed: int = 0) -> dict:
    """``pairs``: one dict per partition with keys y_img, p_img, y_grp, p_grp (test labels / predictions).

    Returns point estimates (mean over partitions) and 95% percentile CIs for the
    accuracy drop (in percentage points) and LI (in percent).
    """
    rng = np.random.default_rng(seed)
    pairs = [{k: np.asarray(v) for k, v in p.items()} for p in pairs]

    def stats_from(resample: bool):
        drops, lis = [], []
        for p in pairs:
            ii = rng.integers(0, len(p["y_img"]), len(p["y_img"])) if resample else slice(None)
            ig = rng.integers(0, len(p["y_grp"]), len(p["y_grp"])) if resample else slice(None)
            a_img = balanced_accuracy(p["y_img"][ii], p["p_img"][ii])
            a_grp = balanced_accuracy(p["y_grp"][ig], p["p_grp"][ig])
            drops.append(a_img - a_grp)
            li = leakage_index(a_img, a_grp, n_classes)
            lis.append(np.nan if li is None else li)
        return float(np.mean(drops)), float(np.nanmean(lis)) if np.isfinite(lis).any() else np.nan

    drop0, li0 = stats_from(False)
    boots = np.array([stats_from(True) for _ in range(n_boot)])
    drop_ci = np.nanpercentile(boots[:, 0], [2.5, 97.5])
    li_ci = np.nanpercentile(boots[:, 1], [2.5, 97.5])
    return {
        "drop_pp": 100 * drop0, "drop_ci95_pp": (100 * drop_ci).tolist(),
        "LI_percent": None if np.isnan(li0) else 100 * li0,
        "LI_ci95_percent": None if np.isnan(li0) else (100 * li_ci).tolist(),
        "n_partitions": len(pairs), "n_boot": n_boot,
    }
