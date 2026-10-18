"""Statistical resolution of the benchmark (Table 7; Section 4.7).

Paired comparison of bounded confusion against standard training on the
partitions they share, correlation of the per-partition changes, and the
power analysis giving the number of partitions needed to resolve a margin.
"""
from __future__ import annotations

import logging

import numpy as np

from ..audit.stats import correlation, paired_comparison, required_partitions
from ..utils import load_json
from .common import RunSpec, write_results

log = logging.getLogger("sia")


def run_stats(spec: RunSpec, dataset: str = "oasis", treated: str = "confusion", baseline: str = "standard",
              margins_points=(5.0, 3.0, 2.0, 0.14)) -> dict:
    path = spec.out_dir / f"interventions_{dataset}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `interventions` first")
    res = load_json(path)
    by_seed = lambda m: {r["seed"]: r for r in res["runs"].get(m, [])}
    a, b = by_seed(treated), by_seed(baseline)
    defined = lambda r: r.get("sis") and r["sis"].get("sis") is not None
    seeds = sorted(s for s in set(a) & set(b) if defined(a[s]) and defined(b[s]))
    if len(seeds) < 2:
        raise SystemExit("need at least two shared partitions (with a defined SIS) between the treated and baseline methods")

    def pick(runs, key):
        return np.array([runs[s]["sis"]["sis"] if key == "sis" else runs[s]["test_bal_acc"] for s in seeds], float)

    d_sis = paired_comparison(pick(a, "sis"), pick(b, "sis"))
    d_acc = paired_comparison(pick(a, "acc"), pick(b, "acc"))
    corr = correlation(pick(a, "sis") - pick(b, "sis"), pick(a, "acc") - pick(b, "acc"))
    sd = d_acc["sd_diff"]
    power = {f"{m}_points": required_partitions(sd, m / 100) for m in margins_points}
    out = {"dataset": dataset, "treated": treated, "baseline": baseline, "shared_seeds": seeds,
           "delta_sis": d_sis, "delta_bal_acc": d_acc, "correlation_dSIS_dAcc": corr,
           "paired_sd_bal_acc": sd, "required_partitions": power}
    write_results(out, spec, f"stats_{dataset}.json")
    print(f"\nDelta SIS  = {d_sis['mean_diff']:+.4f} +- {d_sis['sd_diff']:.4f}  t={d_sis.get('t', float('nan')):.2f} "
          f"p={d_sis.get('p_ttest', float('nan')):.4f} Wilcoxon p={d_sis.get('p_wilcoxon', float('nan')):.4f} "
          f"d_z={d_sis.get('cohen_dz', float('nan')):.2f}  ({d_sis['n_negative']}/{d_sis['n']} partitions reduced)")
    print(f"Delta Acc  = {d_acc['mean_diff']:+.4f} +- {d_acc['sd_diff']:.4f}  t={d_acc.get('t', float('nan')):.2f} "
          f"p={d_acc.get('p_ttest', float('nan')):.4f} d_z={d_acc.get('cohen_dz', float('nan')):.2f}")
    print(f"Pearson r = {corr['pearson_r']:.3f} (p={corr['p_pearson']:.4f}); Spearman rho = {corr['spearman_rho']:.3f} (p={corr['p_spearman']:.4f})")
    print("Partitions required (alpha=0.05, power=0.8): " + ", ".join(f"{k}: {v:.0f}" for k, v in power.items()))
    return out
