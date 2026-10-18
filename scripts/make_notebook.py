"""Write scripts/sia_kaggle.ipynb (kept as a script so the notebook JSON stays reproducible)."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


md("# SIA — Subject Identifiability Auditing (ICVGIP 2026)\n\n"
   "Reproduces every table and figure of the paper on Kaggle (GPU: T4).\n\n"
   "**Before running:** add the two datasets to the notebook (`ninadaithal/imagesoasis` and "
   "`mateuszbuda/lgg-mri-segmentation`), turn on **Internet** (ImageNet weights) and select a **GPU** accelerator.")
code("# Option A: the code is attached as a Kaggle dataset named 'sia-audit'\n"
     "import os, shutil\n"
     "if os.path.exists('/kaggle/input/sia-audit'):\n"
     "    shutil.copytree('/kaggle/input/sia-audit', '/kaggle/working/sia-audit', dirs_exist_ok=True)\n"
     "# Option B: clone from GitHub\n"
     "# !git clone https://github.com/<user>/sia-audit.git /kaggle/working/sia-audit\n"
     "%cd /kaggle/working/sia-audit\n"
     "!pip install -q -e .")
code("OASIS = '/kaggle/input/imagesoasis/Data'\n"
     "TCGA  = '/kaggle/input/lgg-mri-segmentation/kaggle_3m'\n"
     "OUT   = '/kaggle/working/results'\n"
     "import os; print(os.listdir(OASIS)); print(len(os.listdir(TCGA)), 'TCGA entries')")
md("## Stage 1 — composition (Table 1, Fig. 1)")
code("!python -m sia.cli stage1 --oasis-root $OASIS --tcga-root $TCGA --out $OUT")
md("## Stage 2 — paired protocol ablation + SIS of the standard encoder (Tables 2–3, Fig. 2)\n\n"
   "≈ 30 min each on a T4 with the default 3 partitions.")
code("!python -m sia.cli stage2 --dataset oasis --oasis-root $OASIS --out $OUT")
code("!python -m sia.cli stage2 --dataset tcga --tcga-root $TCGA --out $OUT")
md("Optional — the uncapped benchmark (ρ = 249, Table 2 † row). ≈ 1 h per run; one partition per protocol is enough.")
code("# !python -m sia.cli stage2 --dataset oasis --oasis-root $OASIS --k 0 --partitions 1 --out $OUT")
md("## Redundancy sweep at fixed budget (Table 4, Fig. 3) — ≈ 1.5 h")
code("!python -m sia.cli budget --oasis-root $OASIS --out $OUT")
md("## Stage 3 — cross-backbone / cross-benchmark (Table 8, Fig. 5) — ≈ 30 min")
code("!python -m sia.cli stage3 --oasis-root $OASIS --tcga-root $TCGA --out $OUT")
md("## Interventions (Table 6, Fig. 4) — ≈ 3 h\n\n"
   "Reduce `--interv-partitions` (e.g. `standard:5,grl:2,grl_convergence:1,mil:2,confusion:5`) for a shorter session; "
   "`stats` needs the same seeds for `confusion` and `standard`.")
code("!python -m sia.cli interventions --oasis-root $OASIS --out $OUT")
md("## Statistics (Table 7), figures and report")
code("!python -m sia.cli stats --out $OUT\n"
     "!python -m sia.cli figures --out $OUT\n"
     "!python -m sia.cli report --out $OUT")
code("from IPython.display import Image, display, Markdown\n"
     "display(Markdown(open(f'{OUT}/REPORT.md').read()))\n"
     "for f in ['fig1_composition', 'fig3_confusion', 'fig2_budget', 'fig6_lambda', 'fig_dissociation']:\n"
     "    display(Image(f'{OUT}/figures/{f}.png', width=700))")
md("## Download everything (JSON results, figures, REPORT.md; checkpoints excluded)")
code("!cd /kaggle/working && zip -qr sia_results.zip results -x 'results/checkpoints/*' && ls -la sia_results.zip")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("sia_kaggle.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
