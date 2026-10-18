"""Index the Kaggle TCGA-LGG MRI benchmark (Buda et al., 2019; "lgg-mri-segmentation").

Expected layout (e.g. /kaggle/input/lgg-mri-segmentation/kaggle_3m):

    <root>/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1.tif
    <root>/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1_mask.tif
    ...

Each patient folder holds one FLAIR volume as 2-D .tif slices with a binary
tumour mask per slice.  The paper uses the slice-level *tumour presence*
label (1 if the mask contains any positive pixel) and the patient as the
acquisition group, giving 3,929 slices from 110 patients, eta = 0.552.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_SLICE_RE = re.compile(r"_(\d+)_mask\.tiff?$", re.IGNORECASE)


def _patient_from_folder(folder: str) -> str:
    # TCGA_CS_4941_19960909 -> TCGA_CS_4941 (the trailing token is the scan date)
    parts = folder.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else folder


def index_tcga(root: str | Path, mask_threshold: int = 0) -> pd.DataFrame:
    """Return one row per slice: path, mask_path, patient, slice_idx, label (tumour present), tumour_pixels."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"TCGA-LGG root not found: {root}")
    rows = []
    for mask in sorted(root.rglob("*_mask.tif")) + sorted(root.rglob("*_mask.tiff")):
        m = _SLICE_RE.search(mask.name)
        if m is None:
            continue
        img = mask.with_name(mask.name.replace("_mask", ""))
        if not img.exists():
            continue
        arr = np.asarray(Image.open(mask).convert("L"))
        n_pos = int((arr > mask_threshold).sum())
        rows.append({
            "path": str(img),
            "mask_path": str(mask),
            "patient": _patient_from_folder(mask.parent.name),
            "slice_idx": int(m.group(1)),
            "tumour_pixels": n_pos,
            "label": int(n_pos > 0),
        })
    if not rows:
        raise RuntimeError(f"No *_mask.tif files found under {root}")
    df = pd.DataFrame(rows)
    df["class_name"] = df["label"].map({0: "no tumour", 1: "tumour"})
    df["group"] = df["patient"]
    df["dataset"] = "tcga"
    return df.sort_values(["patient", "slice_idx"]).reset_index(drop=True)
