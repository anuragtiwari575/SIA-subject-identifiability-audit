"""Print every table of the paper (markdown) from whatever JSON results exist."""
from __future__ import annotations

from pathlib import Path

from ..audit.composition import composition_table
from ..utils import load_json


def _ms(d, fmt="%.3f"):
    if not d or d.get("mean") is None:
        return "---"
    return (fmt % d["mean"]) + (" +- " + (fmt % d["std"]) if d.get("std") is not None else "")


def make_report(results_dir: str | Path) -> str:
    rd = Path(results_dir)
    out = []
    p = rd / "stage1_composition.json"
    if p.exists():
        out += ["## Table 1 -- Stage 1 composition", composition_table(load_json(p)), ""]
    rows = sorted(rd.glob("stage2_*.json"))
    if rows:
        out += ["## Tables 2-3 -- Stage 2 protocol ablation (+ Stage 3 on the standard encoder)",
                "| Run | Image-level | Group-disjoint | Drop (pp) [95% CI] | LI [95% CI] | SIS | Control | Selectivity | SIS~ | |G_Xi| | Audit cost |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in rows:
            s = load_json(p)["summary"]; b = s["bootstrap"]; si = s["sis"]
            li = "undefined" if b["LI_percent"] is None else "%.1f%% [%.1f, %.1f]" % (b["LI_percent"], *b["LI_ci95_percent"])
            cost = s.get("audit_cost", {}).get("audit_percent_of_training")
            out.append(f"| {p.stem[7:]} | {_ms(s['image_level'])} | {_ms(s['group_disjoint'])} | "
                       f"{b['drop_pp']:.2f} [{b['drop_ci95_pp'][0]:.2f}, {b['drop_ci95_pp'][1]:.2f}] | {li} | "
                       f"{_ms(si['sis'], '%.4f')} | {_ms(si['sis_control'], '%.4f')} | {_ms(si['selectivity'], '%.4f')} | "
                       f"{_ms(si['sis_adjusted'])} | {_ms(si['n_groups'], '%.0f')} | "
                       f"{'---' if cost is None else '%.2f%% of training' % cost} |")
        out.append("")
    p = rd / "budget_sweep.json"
    if p.exists():
        out += ["## Table 4 -- redundancy at fixed budget",
                "| M | k | N | Image-lvl | Group-disj. | LI | |G_Xi| | SIS | SIS/c (ceiling) | SIS~ |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in load_json(p)["sweep"]:
            s = r["sis"]
            out.append(f"| {r['M']} | {r['k']} | {r['N']:,} | {_ms(r['image_level'])} | {_ms(r['group_disjoint'])} | "
                       f"{'undef.' if r['LI_percent'] is None else '%.1f%%' % r['LI_percent']} | {_ms(s['n_groups'], '%.0f')} | "
                       f"{_ms(s['sis'])} | {_ms(s['ratio_to_chance'], '%.1f')} ({_ms(s['ratio_ceiling'], '%.1f')}) | {_ms(s['sis_adjusted'])} |")
        out.append("")
    for p in sorted(rd.glob("interventions_*.json")):
        res = load_json(p)
        out += [f"## Table 6 -- interventions ({res['dataset']}, group-disjoint)",
                "| Method | Bal. acc. | SIS | SIS~ | Selectivity | Partitions | Selected epochs |", "|---|---|---|---|---|---|---|"]
        for m, t in res["table6"].items():
            out.append(f"| {m} | {_ms(t['bal_acc'])} | {_ms(t['sis'])} | {_ms(t['sis_adjusted'])} | {_ms(t['selectivity'])} | "
                       f"{t['partitions']} | {t['selected_epochs']} |")
        out.append("")
    for p in sorted(rd.glob("stats_*.json")):
        s = load_json(p)
        out += [f"## Table 7 -- {s['treated']} vs {s['baseline']} over {len(s['shared_seeds'])} shared partitions",
                "| Quantity | Mean +- SD | t | p | d_z |", "|---|---|---|---|---|"]
        for name, d in (("Delta SIS", s["delta_sis"]), ("Delta Bal. acc.", s["delta_bal_acc"])):
            out.append(f"| {name} | {d['mean_diff']:+.4f} +- {d['sd_diff']:.4f} | {d.get('t', float('nan')):+.2f} | "
                       f"{d.get('p_ttest', float('nan')):.4f} (Wilcoxon {d.get('p_wilcoxon', float('nan')):.4f}) | {d.get('cohen_dz', float('nan')):+.2f} |")
        c = s["correlation_dSIS_dAcc"]
        out.append(f"| Pearson r | {c['pearson_r']:.3f} | --- | {c['p_pearson']:.4f} | --- |")
        out.append(f"| Spearman rho | {c['spearman_rho']:.3f} | --- | {c['p_spearman']:.4f} | --- |")
        out.append("\nPartitions required (alpha = 0.05, power = 0.8, paired SD = %.3f): " % s["paired_sd_bal_acc"]
                   + ", ".join(f"{k.replace('_points', '-point')} margin: {v:.0f}" for k, v in s["required_partitions"].items()) + "\n")
    p = rd / "stage3_cross.json"
    if p.exists():
        out += ["## Table 8 -- cross-backbone / cross-benchmark", "| Data | Backbone | d | Img-lvl | Grp-disj. | LI | |G_Xi| | SIS~ |",
                "|---|---|---|---|---|---|---|---|"]
        fmt = lambda v, f: "undef." if v is None else (f % v)
        for r in load_json(p)["table8"]:
            out.append(f"| {r['dataset']} | {r['backbone']} | {r['d']} | {fmt(r['image_level'], '%.3f')} | {fmt(r['group_disjoint'], '%.3f')} | "
                       f"{fmt(r['LI_percent'], '%.1f%%')} | {fmt(r['n_probe_groups'], '%.0f')} | {fmt(r['sis_adjusted'], '%.3f')} |")
        out.append("")
    return "\n".join(out) if out else "no results found in %s" % rd
