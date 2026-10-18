"""Redundancy sweep at fixed data budget (Table 4, Fig. 3; Section 4.4).

Retained slices per subject k and the number of subjects M are varied while the
total budget N ~ 3,500 is held fixed: (M, k) in {(347,10), (174,20), (87,40), (44,80)}.
Because the budget is fixed, k and M are coupled by construction; the M-fixed
comparison (M = 347, k = 10 -> 20) is the only confound-free redundancy result
and is reported from this sweep together with Stage 2 (k = 20, M = 347).
"""
from __future__ import annotations

import logging

import numpy as np

from ..audit.leakage import bootstrap_leakage, leakage_index
from ..data import cap_slices_per_group, describe_index, index_oasis, subsample_groups
from ..data.oasis import OASIS_CLASSES_3
from .common import RunSpec, run_and_probe, summarise, write_results

log = logging.getLogger("sia")

DEFAULT_SWEEP = [(347, 10), (174, 20), (87, 40), (44, 80)]


def run_budget(spec: RunSpec, sweep=DEFAULT_SWEEP, partitions: int | None = None) -> dict:
    if not spec.oasis_root:
        raise SystemExit("--oasis-root is required")
    full = index_oasis(spec.oasis_root)
    n_classes = len(OASIS_CLASSES_3)
    partitions = partitions or spec.partitions
    rows = []
    for m, k in sweep:
        df = cap_slices_per_group(subsample_groups(full, m, seed=spec.seed_base), k)
        log.info("budget point M=%d k=%d -> %s", m, k, describe_index(df))
        sub_spec = RunSpec(**{**spec.__dict__, "k": k, "max_groups": m})
        img, grp, pairs = [], [], []
        for i in range(partitions):
            seed = spec.seed_base + i
            r_i = run_and_probe("oasis", df, "image", seed, sub_spec, probe=False, tag=f"budget_M{m}_k{k}")
            r_g = run_and_probe("oasis", df, "group", seed, sub_spec, probe=True, tag=f"budget_M{m}_k{k}")
            img.append(r_i); grp.append(r_g)
            pairs.append({"y_img": r_i["y_true"], "p_img": r_i["y_pred"], "y_grp": r_g["y_true"], "p_grp": r_g["y_pred"]})
        a_i, a_g = summarise(img, "test_bal_acc"), summarise(grp, "test_bal_acc")
        sis = [r["sis"] for r in grp if r.get("sis") and not r["sis"].get("undefined")]
        li = leakage_index(a_i["mean"], a_g["mean"], n_classes)
        row = {"M": m, "k": k, "N": int(len(df)), "image_level": a_i, "group_disjoint": a_g,
               "LI_percent": None if li is None else 100 * li,
               "bootstrap": bootstrap_leakage(pairs, n_classes, n_boot=1000, seed=spec.seed_base),
               "sis": {key: summarise(sis, key) for key in ("sis", "sis_adjusted", "ratio_to_chance", "ratio_ceiling",
                                                           "chance", "n_groups", "selectivity")},
               "runs": {"image": img, "group": grp}}
        rows.append(row)
        print(f"M={m:>4} k={k:>3} N={row['N']:>5} | image {a_i['mean']:.3f}+-{a_i['std']:.3f} | group {a_g['mean']:.3f}"
              f"+-{a_g['std']:.3f} | LI {row['LI_percent']} | |G_Xi| {row['sis']['n_groups']['mean']} | "
              f"SIS {row['sis']['sis']['mean']} | SIS~ {row['sis']['sis_adjusted']['mean']} | SIS/c {row['sis']['ratio_to_chance']['mean']}")
    out = {"sweep": rows, "partitions": partitions}
    write_results(out, spec, "budget_sweep.json")
    return out
