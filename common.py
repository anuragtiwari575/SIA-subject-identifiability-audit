"""Shared plumbing for the experiment drivers."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..audit.sis import subject_identifiability
from ..data import (OASIS_CLASSES_3, cap_slices_per_group, describe_index, group_disjoint_split,
                    image_level_split, index_oasis, index_tcga, subsample_groups)
from ..train.features import extract_features
from ..train.trainer import TrainConfig, train_run, strip_model
from ..utils import get_device, save_json

log = logging.getLogger("sia")

TCGA_CLASSES = ["no tumour", "tumour"]


@dataclass
class RunSpec:
    """Everything an experiment needs to know about where data lives and how to train."""
    oasis_root: str | None = None
    tcga_root: str | None = None
    out: str = "results"
    k: int | None = 20
    partitions: int = 3
    seed_base: int = 0
    max_groups: int | None = None          # subsample groups (quick runs)
    tcga_channels: str = "flair"
    save_checkpoints: bool = True
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def out_dir(self) -> Path:
        p = Path(self.out)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_benchmark(name: str, spec: RunSpec, k: int | None = "default", max_groups: int | None = "default") -> tuple[pd.DataFrame, list[str]]:
    """Index a benchmark, optionally subsample groups, and cap slices per group."""
    if name == "oasis":
        if not spec.oasis_root:
            raise SystemExit("--oasis-root is required")
        df, classes = index_oasis(spec.oasis_root), OASIS_CLASSES_3
    elif name == "tcga":
        if not spec.tcga_root:
            raise SystemExit("--tcga-root is required")
        df, classes = index_tcga(spec.tcga_root), TCGA_CLASSES
    else:
        raise ValueError(name)
    full = describe_index(df)
    if max_groups == "default":
        max_groups = spec.max_groups
    if max_groups:
        df = subsample_groups(df, max_groups, seed=spec.seed_base)
    if k == "default":
        k = spec.k
    df = cap_slices_per_group(df, k)
    log.info("%s: full index %s -> after subsample/cap(k=%s): %s", name, full, k, describe_index(df))
    return df, classes


def train_config_for(name: str, spec: RunSpec, seed: int, **overrides) -> TrainConfig:
    cfg = TrainConfig(**{**spec.train.__dict__, **overrides})
    cfg.seed = seed
    cfg.channels = spec.tcga_channels if name == "tcga" else "all"
    return cfg


def make_partition(df: pd.DataFrame, protocol: str, seed: int):
    return image_level_split(df, seed) if protocol == "image" else group_disjoint_split(df, seed)


def probe_model(result: dict, test_df: pd.DataFrame, cfg: TrainConfig, seed: int) -> dict:
    """Stage 3 on a trained run: extract frozen features of the held-out test groups and probe them."""
    model = result["model"]
    encoder = model[0] if isinstance(model, tuple) else model.encoder
    device = get_device(cfg.device)
    feats = extract_features(encoder, test_df, device, img_size=cfg.img_size, channels=cfg.channels,
                             batch_size=max(cfg.batch_size, 64), num_workers=cfg.num_workers)
    sis = subject_identifiability(feats["Z"], feats["groups"], feats["labels"], seed=seed)
    sis.update({"feature_seconds": feats["seconds"], "feature_peak_mem_mb": feats["peak_mem_mb"],
                "n_images": int(len(test_df)), "n_test_groups": int(test_df["group"].nunique()),
                "feature_dim": int(feats["Z"].shape[1])})
    return sis


def save_checkpoint(result: dict, path: Path) -> None:
    model = result["model"]
    state = {"encoder": (model[0] if isinstance(model, tuple) else model.encoder).state_dict(),
             "method": result["method"], "protocol": result["protocol"], "seed": result["seed"],
             "config": result["config"]}
    if isinstance(model, tuple):
        state["mil_head"] = model[1].state_dict()
    else:
        state["heads"] = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def run_and_probe(name: str, df: pd.DataFrame, protocol: str, seed: int, spec: RunSpec, method: str = "standard",
                  probe: bool = True, tag: str = "", **cfg_overrides) -> dict:
    """Train one (protocol, seed, method) run and, for group-disjoint runs, probe the encoder."""
    cfg = train_config_for(name, spec, seed, **cfg_overrides)
    part = make_partition(df, protocol, seed)
    log.info("=== %s | %s | %s | seed %d | %s", name, protocol, method, seed, part.summary())
    res = train_run(cfg, part, method)
    out = strip_model(res)
    out["dataset"] = name
    out["k"] = spec.k
    if probe and protocol == "group":
        out["sis"] = probe_model(res, part.test, cfg, seed)
        log.info("SIS %s", {k: v for k, v in out["sis"].items() if k in ("sis", "sis_control", "selectivity", "sis_adjusted", "n_groups")})
    if spec.save_checkpoints:
        ck = spec.out_dir / "checkpoints" / f"{tag or name}_{protocol}_{method}_seed{seed}.pt"
        save_checkpoint(res, ck)
        out["checkpoint"] = str(ck)
    del res["model"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarise(values, key=None):
    arr = np.array([v if key is None else v[key] for v in values if (v if key is None else v.get(key)) is not None], dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0, "n": int(arr.size)}


def write_results(obj: dict, spec: RunSpec, filename: str) -> Path:
    path = spec.out_dir / filename
    save_json(obj, path)
    return path
