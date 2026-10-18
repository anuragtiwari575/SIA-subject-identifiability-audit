"""Small shared helpers: seeding, device selection, JSON I/O, logging."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("sia")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("matplotlib", "fontTools", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is only needed for training
        pass


def get_device(name: str = "auto"):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2)
    log.info("wrote %s", path)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


class Timer:
    """Context manager measuring wall-clock seconds (and CUDA-synchronised when a device is given)."""

    def __init__(self, device=None):
        self.device = device
        self.seconds = 0.0

    def __enter__(self):
        self._sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.seconds = time.perf_counter() - self._t0

    def _sync(self):
        if self.device is not None and str(self.device).startswith("cuda"):
            import torch
            torch.cuda.synchronize(self.device)


def mean_std(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0
