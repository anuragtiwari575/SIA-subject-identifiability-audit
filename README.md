[README.md](https://github.com/user-attachments/files/33629574/README.md)
# SIA-subject-identifiability-audit# SIA — Subject Identifiability Auditing

Code for **"SIA: A Subject Identifiability Auditing Framework for Detecting Patient-Level
Leakage in Medical Imaging Benchmarks"** (ICVGIP 2026).

SIA treats patient-level leakage as a *measurable* property of a benchmark and of a trained
representation, in three stages that are ordered by cost:

| Stage | What it needs | What it returns | Module |
|---|---|---|---|
| 1 — composition analysis | metadata only | ρ (group redundancy), η (within-group label homogeneity), M_c (class coverage), Proposition 1 check | `sia/audit/composition.py` |
| 2 — paired protocol ablation | paired image-level / group-disjoint training runs | Leakage Index LI = (A_img − A_grp)/(A_img − 1/C) with bootstrap CIs | `sia/audit/leakage.py` |
| 3 — Subject Identifiability Score | one trained encoder + group identifiers | SIS, permuted-label control, selectivity σ, chance-adjusted SIS (Eq. 5) | `sia/audit/sis.py` |

Everything else (the audited ResNet-18 / EfficientNet-B0 encoder with CBAM-style spatial
attention, gradient reversal, bounded confusion + subject-excluded SupCon, attention MIL, the
redundancy sweep, the paired statistics and power analysis, and all figures) lives under
`sia/models`, `sia/train` and `sia/experiments`.

## Installation

```bash
git clone <this repository>
cd sia-audit
pip install -e .            # or: pip install -r requirements.txt
```

Python ≥ 3.9, PyTorch ≥ 2.1, torchvision, scikit-learn, scipy, pandas, Pillow, matplotlib.
A single NVIDIA T4 (Kaggle) is enough; ImageNet weights are downloaded by torchvision, so
enable *Internet* in the Kaggle notebook settings (without it the encoder falls back to random
initialisation and says so).

## Data

| Benchmark | Kaggle dataset | Pass as |
|---|---|---|
| OASIS-1 2-D slices (86,437 images, 4 class folders, 347 subjects) | `ninadaithal/imagesoasis` | `--oasis-root /kaggle/input/imagesoasis/Data` |
| TCGA-LGG (3,929 FLAIR slices + masks, 110 patients) | `mateuszbuda/lgg-mri-segmentation` | `--tcga-root /kaggle/input/lgg-mri-segmentation/kaggle_3m` |

Subject identity is parsed from the OASIS file names (`OAS1_0001_MR1_mpr-1_100.jpg` → subject
`OAS1_0001`); the TCGA patient is the folder prefix (`TCGA_CS_4941`) and the slice label is
*tumour present* (mask non-empty).  OASIS is used as three classes (Moderate merged into Mild).

## Reproducing the paper

```bash
OASIS=/kaggle/input/imagesoasis/Data
TCGA=/kaggle/input/lgg-mri-segmentation/kaggle_3m

python -m sia.cli stage1        --oasis-root $OASIS --tcga-root $TCGA           # Table 1, Fig. 1
python -m sia.cli stage2        --oasis-root $OASIS --dataset oasis              # Tables 2-3, Fig. 2, SIS of the standard encoder
python -m sia.cli stage2        --tcga-root  $TCGA  --dataset tcga
python -m sia.cli stage2        --oasis-root $OASIS --dataset oasis --k 0 --partitions 1   # optional: full benchmark (rho = 249), Table 2 dagger row
python -m sia.cli budget        --oasis-root $OASIS                              # Table 4, Fig. 3
python -m sia.cli stage3        --oasis-root $OASIS --tcga-root $TCGA            # EfficientNet-B0 runs + Table 8 / Fig. 5
python -m sia.cli interventions --oasis-root $OASIS                              # Table 6, Fig. 4
python -m sia.cli stats                                                          # Table 7 + power analysis
python -m sia.cli figures && python -m sia.cli report                            # figures/ and REPORT.md
```

`python -m sia.cli all --oasis-root $OASIS --tcga-root $TCGA` runs the whole chain.  Every
command writes JSON into `results/` (plus encoder checkpoints in `results/checkpoints/`), so
stages can be re-run or extended independently; `report` and `figures` only read the JSON.

Defaults are the recipe of Section 4.1: ImageNet-pretrained ResNet-18 + spatial attention,
AdamW (lr 3e-4, wd 1e-4), 20 epochs, cosine decay, batch 64, AMP, 128×128 inputs, light affine
augmentation + horizontal flips, 60:20:20 splits, selection on validation balanced accuracy,
single test pass, `k = 20` evenly spaced slices per subject, 3 audit partitions, 10
intervention partitions (`--interv-partitions standard:10,grl:3,grl_convergence:1,mil:5,confusion:10`).

Approximate cost on a T4: one capped OASIS run ≈ 5 min, so `stage2` (OASIS + TCGA) ≈ 1 h,
`budget` ≈ 1.5 h, `stage3` ≈ 30 min, `interventions` ≈ 3 h; the uncapped OASIS run
(`--k 0`) ≈ 1 h per run.

### Smoke test (no data needed, CPU, ~2 min)

```bash
bash tests/smoke_test.sh
```

builds tiny synthetic OASIS / TCGA folder trees and runs every command with `--quick`.

## Layout

```
sia/data         indexing (oasis.py, tcga.py), slice capping & subject subsampling (index.py),
                 image-level / group-disjoint splits (splits.py), datasets & bags (dataset.py)
sia/models       encoder.py (ResNet-18 / EfficientNet-B0 + spatial attention + three heads),
                 grl.py (gradient reversal + lambda schedule), mil.py (attention MIL)
sia/audit        composition.py (Stage 1), leakage.py (Stage 2), sis.py (Stage 3), stats.py
sia/train        losses.py (subject-excluded SupCon, bounded confusion), trainer.py (all methods),
                 features.py (frozen-feature extraction with timing / memory)
sia/experiments  one driver per table/figure + figures.py + report.py
sia/cli.py       command-line entry point
tests/           synthetic-data generator and smoke test
scripts/         Kaggle notebook
```

## Implementation notes (choices the paper leaves open)

* **TCGA-LGG channels.** The Kaggle `.tif` slices store (pre-contrast, FLAIR, post-contrast).
  The paper describes them as FLAIR slices, so the default keeps the FLAIR channel only
  (`--tcga-channels flair`); `--tcga-channels all` feeds the three channels as stored.
* **Class weighting.** Plain cross-entropy by default (`--class-weights balanced` for
  inverse-frequency weights); balanced accuracy is the reported metric either way.
* **Control task.** The probe is refitted on *image-level permuted* group labels of the fit set
  and evaluated against the true labels, so SIS_ctrl sits at chance when the probe cannot
  memorise.
* **Bootstrap CIs.** Percentile bootstrap over resampled test-set predictions of each paired
  partition, averaged over partitions (2,000 draws).
* **Budget sweep.** Subjects are subsampled proportionally per class (at least three per class)
  with the same seed for every (M, k) point, so the k = 10 → 20 comparison at M = 347 is the
  confound-free one.
* **MIL metrics** are bag-level (bags of `--bag-size` slices per subject); the MIL encoder is
  probed like the others, so the SIS column of Table 6 can be filled if desired.

## Citation

```bibtex
@inproceedings{tiwari2026sia,
  title     = {SIA: A Subject Identifiability Auditing Framework for Detecting Patient-Level Leakage in Medical Imaging Benchmarks},
  author    = {Tiwari, Anurag and Tiwari, Shailendra},
  booktitle = {Proceedings of the 17th Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP)},
  year      = {2026},
  publisher = {ACM}
}
```
