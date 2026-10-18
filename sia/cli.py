"""Command-line entry point:  python -m sia.cli <command> [options]

Commands
  stage1         composition analysis (Table 1, Fig. 1)
  stage2         paired protocol ablation + SIS of the standard encoder (Tables 2-3, Fig. 2)
  budget         redundancy sweep at fixed budget (Table 4, Fig. 3)
  stage3         EfficientNet-B0 runs + Table 8 / Fig. 5 assembly
  interventions  identity-removal audit (Table 6, Fig. 4)
  stats          paired statistics + power analysis (Table 7)
  figures        regenerate all figures from results/
  report         print all tables from results/
  all            stage1 -> stage2 (oasis, tcga) -> budget -> stage3 -> interventions -> stats -> figures -> report

Example (Kaggle, T4):
  python -m sia.cli all --oasis-root /kaggle/input/imagesoasis/Data \\
                        --tcga-root /kaggle/input/lgg-mri-segmentation/kaggle_3m --out results
"""
from __future__ import annotations

import argparse
import logging
import sys

from .experiments.common import RunSpec
from .train.trainer import TrainConfig
from .utils import setup_logging

log = logging.getLogger("sia")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sia", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["stage1", "stage2", "budget", "stage3", "interventions", "stats", "figures", "report", "all"])
    p.add_argument("--oasis-root", default=None)
    p.add_argument("--tcga-root", default=None)
    p.add_argument("--out", default="results", help="results directory (JSON, checkpoints, figures/)")
    p.add_argument("--dataset", default="oasis", choices=["oasis", "tcga"], help="for stage2 / interventions / stats")
    p.add_argument("--k", type=int, default=20, help="slices retained per subject (0 = full benchmark)")
    p.add_argument("--partitions", type=int, default=3, help="paired partitions for stage2/budget/stage3")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-groups", type=int, default=None, help="subsample this many groups (quick runs)")
    p.add_argument("--tcga-channels", default="flair", choices=["flair", "all"])
    p.add_argument("--no-checkpoints", action="store_true")
    # training recipe (defaults = Section 4.1)
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--class-weights", default="none", choices=["none", "balanced"])
    p.add_argument("--bag-size", type=int, default=8)
    p.add_argument("--device", default="auto")
    # per-command extras
    p.add_argument("--full-oasis", action="store_true", help="stage2: also run the uncapped OASIS benchmark (rho=249)")
    p.add_argument("--sweep", default="347:10,174:20,87:40,44:80", help="budget: comma-separated M:k points")
    p.add_argument("--interv-partitions", default="standard:10,grl:3,grl_convergence:1,mil:5,confusion:10",
                   help="interventions: method:count list")
    p.add_argument("--skip-training", action="store_true", help="stage3: only assemble Table 8 from saved results")
    p.add_argument("--quick", action="store_true", help="smoke test: 1 epoch, 64 px, few subjects, tiny sweep")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def spec_from_args(a) -> RunSpec:
    train = TrainConfig(backbone=a.backbone, pretrained=not a.no_pretrained, epochs=a.epochs, batch_size=a.batch_size,
                        lr=a.lr, weight_decay=a.weight_decay, img_size=a.img_size, amp=not a.no_amp,
                        num_workers=a.num_workers, class_weights=a.class_weights, bag_size=a.bag_size, device=a.device)
    spec = RunSpec(oasis_root=a.oasis_root, tcga_root=a.tcga_root, out=a.out, k=(None if a.k == 0 else a.k),
                   partitions=a.partitions, seed_base=a.seed_base, max_groups=a.max_groups,
                   tcga_channels=a.tcga_channels, save_checkpoints=not a.no_checkpoints, train=train)
    if a.quick:
        spec.train.epochs = 1
        spec.train.img_size = 64
        spec.train.batch_size = 16
        spec.train.num_workers = 0
        spec.max_groups = spec.max_groups or 24
        spec.partitions = min(spec.partitions, 2)
        spec.k = min(spec.k or 8, 8)
    return spec


def _parse_sweep(s: str):
    return [(int(m), int(k)) for m, k in (t.split(":") for t in s.split(",") if t)]


def _parse_interv(s: str):
    return {m: int(n) for m, n in (t.split(":") for t in s.split(",") if t)}


def main(argv=None):
    a = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if a.verbose else logging.INFO)
    spec = spec_from_args(a)
    from .experiments import budget, figures, interventions, report, stage1, stage2, stage3, stats

    sweep = _parse_sweep(a.sweep)
    interv = _parse_interv(a.interv_partitions)
    if a.quick:
        sweep = [(24, 4), (12, 8)]
        interv = {m: min(n, 2) for m, n in interv.items()}

    cmd = a.command
    if cmd in ("stage1", "all"):
        stage1.run_stage1(spec)
    if cmd in ("stage2", "all"):
        datasets = ["oasis", "tcga"] if cmd == "all" else [a.dataset]
        for ds in datasets:
            if (ds == "oasis" and spec.oasis_root) or (ds == "tcga" and spec.tcga_root):
                stage2.run_stage2(spec, ds)
        if a.full_oasis and spec.oasis_root:
            stage2.run_stage2(spec, "oasis", k=None, tag="oasis_kfull_resnet18")
    if cmd in ("budget", "all") and spec.oasis_root:
        budget.run_budget(spec, sweep)
    if cmd in ("stage3", "all") and spec.oasis_root:
        stage3.run_stage3(spec, skip_training=a.skip_training)
    if cmd in ("interventions", "all"):
        interventions.run_interventions(spec, a.dataset if cmd != "all" else "oasis", interv)
    if cmd in ("stats", "all"):
        stats.run_stats(spec, a.dataset if cmd != "all" else "oasis")
    if cmd in ("figures", "all"):
        figures.make_all(spec.out_dir, spec.out_dir / "figures")
    if cmd in ("report", "all"):
        text = report.make_report(spec.out_dir)
        (spec.out_dir / "REPORT.md").write_text(text)
        print(text)


if __name__ == "__main__":
    sys.exit(main())
