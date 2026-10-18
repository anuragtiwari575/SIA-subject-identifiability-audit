"""Build tiny synthetic OASIS / TCGA-LGG folder trees so the whole pipeline can be smoke-tested without the real data.

    python tests/make_synthetic.py /tmp/sia_synth

creates
    /tmp/sia_synth/oasis/Data/{Non Demented,Very mild Dementia,Mild Dementia,Moderate Dementia}/OAS1_xxxx_MR1_mpr-1_<slice>.jpg
    /tmp/sia_synth/tcga/kaggle_3m/TCGA_XX_nnnn_19990101/TCGA_XX_nnnn_19990101_<slice>.tif (+ _mask.tif)

Each synthetic subject has its own "anatomy" (a random blob pattern) plus a class-dependent
intensity bias, so that identity is learnable from the images -- enough for the probes to
produce non-trivial numbers in a one-epoch run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]


def _subject_pattern(rng, size):
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size), float)
    for _ in range(6):
        cx, cy, r = rng.uniform(8, size - 8, 2).tolist() + [rng.uniform(4, 12)]
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    return img / img.max()


def make_oasis(root: Path, n_subjects=(14, 10, 8, 4), slices=12, size=64, seed=0):
    rng = np.random.default_rng(seed)
    sid = 1
    for c, n in enumerate(n_subjects):
        d = root / "Data" / CLASSES[c]
        d.mkdir(parents=True, exist_ok=True)
        for _ in range(n):
            pat = _subject_pattern(rng, size)
            for s in range(slices):
                img = 0.6 * pat + 0.1 * c + 0.02 * s + 0.1 * rng.standard_normal((size, size))
                arr = (255 * np.clip(img, 0, 1)).astype(np.uint8)
                Image.fromarray(arr).save(d / f"OAS1_{sid:04d}_MR1_mpr-1_{100 + s}.jpg")
            sid += 1


def make_tcga(root: Path, n_patients=20, slices=10, size=64, seed=1):
    rng = np.random.default_rng(seed)
    for p in range(n_patients):
        folder = root / "kaggle_3m" / f"TCGA_XX_{4000 + p}_19990101"
        folder.mkdir(parents=True, exist_ok=True)
        pat = _subject_pattern(rng, size)
        for s in range(slices):
            tumour = s >= slices // 2
            img = np.stack([0.5 * pat + 0.1 * rng.standard_normal((size, size)) for _ in range(3)], -1)
            mask = np.zeros((size, size), np.uint8)
            if tumour:
                cx, cy = rng.integers(16, size - 16, 2)
                yy, xx = np.mgrid[0:size, 0:size]
                blob = (xx - cx) ** 2 + (yy - cy) ** 2 < 36
                mask[blob] = 255
                img[..., 1][blob] += 0.4
            Image.fromarray((255 * np.clip(img, 0, 1)).astype(np.uint8)).save(folder / f"{folder.name}_{s + 1}.tif")
            Image.fromarray(mask).save(folder / f"{folder.name}_{s + 1}_mask.tif")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
    make_oasis(out / "oasis")
    make_tcga(out / "tcga")
    print("synthetic data written to", out)
