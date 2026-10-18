"""Index the Kaggle OASIS-1 2-D slice benchmark ("OASIS Alzheimer's Detection").

Expected layout (any top-level folder name, e.g. /kaggle/input/imagesoasis/Data):

    <root>/Non Demented/OAS1_0001_MR1_mpr-1_100.jpg
    <root>/Very mild Dementia/OAS1_0003_MR1_mpr-1_100.jpg
    <root>/Mild Dementia/...
    <root>/Moderate Dementia/...

Subject identity is recovered from the file name: ``OAS1_0001_MR1_mpr-1_100.jpg``
-> subject ``OAS1_0001``, session ``MR1``, slice index ``100``.  Every subject
contributes exactly one diagnosis, so the benchmark has eta = 1 (Section 4.2).

Labels
------
``label4``: 0 = Non Demented, 1 = Very mild, 2 = Mild, 3 = Moderate.
``label``  (3-class, used throughout the paper): Moderate is merged into Mild
because it contains only two subjects and cannot support group-disjoint
evaluation: 0 = ND, 1 = VM, 2 = Mild+Moderate.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

OASIS_CLASSES_4 = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
OASIS_CLASSES_3 = ["Non Demented", "Very mild Dementia", "Mild+Moderate Dementia"]

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_FNAME_RE = re.compile(r"^(?P<subject>OAS\d+_\d+)_(?P<session>MR\d+).*?(?P<slice>\d+)\.\w+$", re.IGNORECASE)


def _class_from_folder(name: str) -> int | None:
    n = name.lower().replace("_", " ").replace("-", " ")
    if "non" in n:
        return 0
    if "very" in n:
        return 1
    if "moderate" in n:
        return 3
    if "mild" in n:
        return 2
    return None


def index_oasis(root: str | Path) -> pd.DataFrame:
    """Return one row per image: path, subject, session, slice_idx, label4, label, class_name."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"OASIS root not found: {root}")

    # Locate the four class folders anywhere under root (handles an extra Data/ level).
    class_dirs: dict[int, Path] = {}
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        c = _class_from_folder(d.name)
        if c is not None and c not in class_dirs and any(f.suffix.lower() in _IMG_EXT for f in d.iterdir()):
            class_dirs[c] = d
    if not class_dirs:
        raise RuntimeError(f"No OASIS class folders found under {root}")

    rows = []
    unparsed = 0
    for c, d in class_dirs.items():
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in _IMG_EXT:
                continue
            m = _FNAME_RE.match(f.name)
            if m is None:
                unparsed += 1
                continue
            rows.append({
                "path": str(f),
                "subject": m.group("subject").upper(),
                "session": m.group("session").upper(),
                "slice_idx": int(m.group("slice")),
                "label4": c,
            })
    if unparsed:
        import logging
        logging.getLogger("sia").warning("OASIS: %d files did not match the OAS1_xxxx_MRx_..._<slice> pattern and were skipped", unparsed)

    df = pd.DataFrame(rows)
    df["label"] = df["label4"].map({0: 0, 1: 1, 2: 2, 3: 2})   # Moderate -> Mild
    df["class_name"] = df["label"].map(dict(enumerate(OASIS_CLASSES_3)))
    df["group"] = df["subject"]
    df["dataset"] = "oasis"
    # A subject with two sessions (MR1/MR2) is still one acquisition group.
    df = df.sort_values(["label", "subject", "session", "slice_idx"]).reset_index(drop=True)
    return df
