"""Stage 3 cross-backbone / cross-benchmark comparison (Table 8, Fig. 5; Section 4.8).

Runs the Stage 2 protocol with EfficientNet-B0 on OASIS and assembles Table 8
from the saved Stage 2 results (OASIS ResNet-18, TCGA-LGG ResNet-18) plus the
Stage 1 composition numbers used by Fig. 5(a).
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..utils import load_json
from .common import RunSpec, write_results
from .stage2 import run_stage2

log = logging.getLogger("sia")


def _row(path: Path) -> dict | None:
    if not path.exists():
        log.warning("missing %s -- run the corresponding stage first", path)
        return None
    s = load_json(path)["summary"]
    return {"dataset": s["dataset"], "backbone": s["backbone"], "d": None,
            "image_level": s["image_level"]["mean"], "group_disjoint": s["group_disjoint"]["mean"],
            "LI_percent": s["bootstrap"]["LI_percent"], "n_probe_groups": s["sis"]["n_groups"]["mean"],
            "sis": s["sis"]["sis"]["mean"], "sis_adjusted": s["sis"]["sis_adjusted"]["mean"]}


def run_stage3(spec: RunSpec, backbones=("efficientnet_b0",), skip_training: bool = False) -> dict:
    k_tag = spec.k if spec.k is not None else "full"
    if not skip_training:
        for bb in backbones:
            run_stage2(spec, "oasis", backbone=bb, tag=f"oasis_k{k_tag}_{bb}")
    rows = []
    for tag, d in ((f"oasis_k{k_tag}_resnet18", 512), *[(f"oasis_k{k_tag}_{bb}", 1280 if bb == "efficientnet_b0" else None) for bb in backbones],
                   (f"tcga_k{k_tag}_resnet18", 512)):
        r = _row(spec.out_dir / f"stage2_{tag}.json")
        if r:
            r["d"] = d
            rows.append(r)
    comp_path = spec.out_dir / "stage1_composition.json"
    comp = load_json(comp_path) if comp_path.exists() else {}
    out = {"table8": rows, "composition": comp}
    write_results(out, spec, "stage3_cross.json")
    fmt = lambda v, f: "undef." if v is None else (f % v)
    print("\n| Data | Backbone | d | Img-lvl | Grp-disj | LI | |G_Xi| | SIS~ |\n|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['backbone']} | {r['d']} | {fmt(r['image_level'], '%.3f')} | {fmt(r['group_disjoint'], '%.3f')} | "
              f"{fmt(r['LI_percent'], '%.1f%%')} | {fmt(r['n_probe_groups'], '%.0f')} | {fmt(r['sis_adjusted'], '%.3f')} |")
    return out
