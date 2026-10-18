"""Stage 2 -- paired protocol ablation (Tables 2-3, Fig. 2) and Stage 3 on the standard encoder.

For each partition seed the SAME capped image index is split once at the image
level and once at the group level; one standard run is trained under each.
The Leakage Index and the accuracy drop are reported with bootstrap CIs, and
the group-disjoint encoders are probed for subject identity (SIS, control,
selectivity, chance-adjusted score) -- the audit-cost figures of Section 4.9
come from the same runs.
"""
from __future__ import annotations

import logging

import numpy as np

from ..audit.leakage import bootstrap_leakage, leakage_index
from .common import RunSpec, load_benchmark, run_and_probe, summarise, write_results

log = logging.getLogger("sia")


def run_stage2(spec: RunSpec, dataset: str = "oasis", k: int | None = "default", backbone: str | None = None,
               tag: str | None = None, partitions: int | None = None, probe: bool = True) -> dict:
    df, classes = load_benchmark(dataset, spec, k=k)
    n_classes = len(classes)
    k_eff = spec.k if k == "default" else k
    backbone = backbone or spec.train.backbone
    tag = tag or f"{dataset}_k{k_eff if k_eff is not None else 'full'}_{backbone}"
    partitions = partitions or spec.partitions

    runs = {"image": [], "group": []}
    pairs = []
    for i in range(partitions):
        seed = spec.seed_base + i
        r_img = run_and_probe(dataset, df, "image", seed, spec, probe=False, tag=tag, backbone=backbone)
        r_grp = run_and_probe(dataset, df, "group", seed, spec, probe=probe, tag=tag, backbone=backbone)
        runs["image"].append(r_img); runs["group"].append(r_grp)
        pairs.append({"y_img": r_img["y_true"], "p_img": r_img["y_pred"], "y_grp": r_grp["y_true"], "p_grp": r_grp["y_pred"]})
        li = leakage_index(r_img["test_bal_acc"], r_grp["test_bal_acc"], n_classes)
        log.info("seed %d: image %.3f  group %.3f  LI %s", seed, r_img["test_bal_acc"], r_grp["test_bal_acc"],
                 "undefined" if li is None else f"{100*li:.1f}%")

    acc_img = summarise(runs["image"], "test_bal_acc")
    acc_grp = summarise(runs["group"], "test_bal_acc")
    boot = bootstrap_leakage(pairs, n_classes, n_boot=2000, seed=spec.seed_base)
    li_point = leakage_index(acc_img["mean"], acc_grp["mean"], n_classes)
    sis_runs = [r["sis"] for r in runs["group"] if r.get("sis") and not r["sis"].get("undefined")]
    summary = {
        "dataset": dataset, "k": k_eff, "backbone": backbone, "n_classes": n_classes, "classes": classes,
        "partitions": partitions,
        "image_level": acc_img, "group_disjoint": acc_grp,
        "cv_percent": {"image": 100 * acc_img["std"] / acc_img["mean"] if acc_img["mean"] else None,
                       "group": 100 * acc_grp["std"] / acc_grp["mean"] if acc_grp["mean"] else None},
        "LI_percent_of_means": None if li_point is None else 100 * li_point,
        "bootstrap": boot,
        "recall_image": np.nanmean([r["test_recall"] for r in runs["image"]], axis=0).tolist(),
        "recall_group": np.nanmean([r["test_recall"] for r in runs["group"]], axis=0).tolist(),
        "confusion_image": np.mean([r["test_confusion"] for r in runs["image"]], axis=0).tolist(),
        "confusion_group": np.mean([r["test_confusion"] for r in runs["group"]], axis=0).tolist(),
        "train_seconds": summarise(runs["group"], "train_seconds"),
        "sis": {key: summarise(sis_runs, key) for key in ("sis", "sis_control", "selectivity", "sis_adjusted",
                                                          "ratio_to_chance", "chance", "n_groups", "feature_seconds",
                                                          "probe_seconds", "feature_peak_mem_mb", "n_images")},
    }
    if summary["train_seconds"]["mean"] and sis_runs:
        audit_s = summary["sis"]["feature_seconds"]["mean"] + summary["sis"]["probe_seconds"]["mean"]
        summary["audit_cost"] = {"audit_seconds": audit_s, "train_seconds": summary["train_seconds"]["mean"],
                                 "audit_percent_of_training": 100 * audit_s / summary["train_seconds"]["mean"]}
    out = {"summary": summary, "runs": runs}
    write_results(out, spec, f"stage2_{tag}.json")
    print(f"\n[{tag}] image-level {acc_img['mean']:.3f} +- {acc_img['std']:.3f} | group-disjoint "
          f"{acc_grp['mean']:.3f} +- {acc_grp['std']:.3f} | drop {boot['drop_pp']:.2f} pp {boot['drop_ci95_pp']} | "
          f"LI {boot['LI_percent']} % CI {boot['LI_ci95_percent']}")
    if sis_runs:
        s = summary["sis"]
        print(f"[{tag}] SIS {s['sis']['mean']:.4f} +- {s['sis']['std']:.4f} | control {s['sis_control']['mean']:.4f} | "
              f"selectivity {s['selectivity']['mean']:.4f} | chance-adjusted {s['sis_adjusted']['mean']:.3f} | "
              f"|G_Xi| {s['n_groups']['mean']:.0f}")
    return out
