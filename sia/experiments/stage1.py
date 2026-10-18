"""Stage 1 -- composition analysis of both benchmarks (Table 1, Fig. 1)."""
from __future__ import annotations

import logging

from ..audit.composition import composition_analysis, composition_table
from ..data import OASIS_CLASSES_3, OASIS_CLASSES_4, index_oasis, index_tcga
from .common import RunSpec, TCGA_CLASSES, write_results

log = logging.getLogger("sia")


def run_stage1(spec: RunSpec, k_folds: int = 3) -> dict:
    results = {}
    if spec.oasis_root:
        df = index_oasis(spec.oasis_root)
        df4 = df.assign(label=df["label4"])
        results["OASIS (4-class, as distributed)"] = composition_analysis(df4, k_folds, OASIS_CLASSES_4)
        results["OASIS (3-class, Moderate merged)"] = composition_analysis(df, k_folds, OASIS_CLASSES_3)
        subj_sessions = df.groupby("subject")["session"].nunique()
        results["OASIS (3-class, Moderate merged)"]["subjects_with_multiple_sessions"] = int((subj_sessions > 1).sum())
    if spec.tcga_root:
        results["TCGA-LGG"] = composition_analysis(index_tcga(spec.tcga_root), k_folds, TCGA_CLASSES)
    if not results:
        raise SystemExit("give --oasis-root and/or --tcga-root")
    print(composition_table(results))
    for name, r in results.items():
        print(f"\n{name}: shortcut possible (rho>1 and eta>1/C) = {r['shortcut_possible']}; "
              f"K={k_folds}-fold group-disjoint evaluable = {r['statistically_evaluable']}")
        for pc in r["per_class"]:
            print(f"  {pc['class_name']:<24s} images {pc['n_images']:>7,}  groups {pc['n_groups']:>4}  M_c>=K: {pc['evaluable_K']}")
    write_results(results, spec, "stage1_composition.json")
    return results
