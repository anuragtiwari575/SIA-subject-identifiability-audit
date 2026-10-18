"""Auditing identity removal (Table 6, Fig. 4; Section 4.6).

A separate family of group-disjoint partitions (seeds ``seed_base + 100 + i``)
is used.  Every method is trained on the same partitions, selected on
validation balanced accuracy (except ``grl_convergence``, which keeps the final
checkpoint) and probed with Stage 3.  The validation curve and the annealed
lambda schedule of the first GRL run are what Fig. 4 shows.
"""
from __future__ import annotations

import logging

from .common import RunSpec, load_benchmark, run_and_probe, summarise, write_results

log = logging.getLogger("sia")

# number of partitions per method in the paper (Table 6)
DEFAULT_PARTITIONS = {"standard": 10, "grl": 3, "grl_convergence": 1, "mil": 5, "confusion": 10}


def run_interventions(spec: RunSpec, dataset: str = "oasis", partitions: dict | None = None, seed_offset: int = 100) -> dict:
    partitions = partitions or DEFAULT_PARTITIONS
    df, classes = load_benchmark(dataset, spec)
    runs = {m: [] for m in partitions}
    for method, n in partitions.items():
        for i in range(n):
            seed = spec.seed_base + seed_offset + i
            r = run_and_probe(dataset, df, "group", seed, spec, method=method, probe=True, tag=f"interv_{dataset}")
            runs[method].append(r)
            s = r.get("sis") or {}
            log.info("%s seed %d: bal_acc %.3f  SIS %s", method, seed, r["test_bal_acc"], s.get("sis"))
    table = {}
    for method, rs in runs.items():
        sis = [r["sis"] for r in rs if r.get("sis") and not r["sis"].get("undefined")]
        table[method] = {"bal_acc": summarise(rs, "test_bal_acc"), "sis": summarise(sis, "sis"),
                         "sis_adjusted": summarise(sis, "sis_adjusted"), "selectivity": summarise(sis, "selectivity"),
                         "partitions": len(rs), "selected_epochs": [r["selected_epoch"] for r in rs]}
    out = {"dataset": dataset, "table6": table, "runs": runs, "chance_bal_acc": 1 / len(classes)}
    write_results(out, spec, f"interventions_{dataset}.json")
    print("\n| Method | Bal. acc. | SIS | Partitions |\n|---|---|---|---|")
    for m, t in table.items():
        ba, s = t["bal_acc"], t["sis"]
        sis_txt = "---" if s["mean"] is None else "%.3f +- %.3f" % (s["mean"], s["std"])
        print("| %s | %.3f +- %.3f | %s | %d |" % (m, ba["mean"], ba["std"], sis_txt, t["partitions"]))
    return out
