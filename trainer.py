"""One training run = one (partition, method) pair, with the shared recipe of Section 4.1:

AdamW (lr 3e-4, wd 1e-4), 20 epochs, cosine decay, batch 64, mixed precision,
128 x 128 inputs, light affine augmentation + horizontal flips; model selection
on validation balanced accuracy with a single test pass.

Methods (Section 3.7 / Table 6)
    standard          L = L_cls
    grl               L = L_cls + L_adv through a gradient-reversal layer with the annealed
                      lambda_e schedule; validation-selected checkpoint
    grl_convergence   identical configuration, final-epoch checkpoint (no early stopping)
    confusion         encoder: L = L_cls + L_ctr + L_conf (subject-excluded SupCon + bounded
                      confusion); the adversary h_adv minimises L_adv independently on detached z
    mil               attention-based MIL over per-subject bags (subject-level aggregation)
"""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..audit.leakage import balanced_accuracy, per_class_recall, row_normalised_confusion
from ..data.dataset import SliceDataset, BagDataset, bag_collate
from ..data.splits import Partition
from ..models.encoder import AuditModel, AuditedEncoder
from ..models.grl import GradientReversal, grl_lambda
from ..models.mil import AttentionMIL
from ..utils import Timer, get_device, seed_everything
from .losses import SubjectExcludedSupCon, confusion_loss

log = logging.getLogger("sia")

METHODS = ["standard", "grl", "grl_convergence", "confusion", "mil"]


@dataclass
class TrainConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    epochs: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    img_size: int = 128
    channels: str = "all"          # "flair" for TCGA-LGG FLAIR-only
    amp: bool = True
    num_workers: int = 2
    class_weights: str = "none"    # "balanced" -> inverse-frequency CE weights
    supcon_temperature: float = 0.1
    bag_size: int = 8
    bags_per_batch: int = 8
    device: str = "auto"
    seed: int = 0


def _class_weight_tensor(labels: np.ndarray, n_classes: int, mode: str, device) -> torch.Tensor | None:
    if mode != "balanced":
        return None
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    w = len(labels) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def _predict(model: AuditModel, loader: DataLoader, device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y, _, _ in loader:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        ps.append(out["logits"].argmax(1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def _loader(df, cfg: TrainConfig, train: bool, group_to_idx, device):
    ds = SliceDataset(df, img_size=cfg.img_size, train=train, channels=cfg.channels, group_to_idx=group_to_idx)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=cfg.num_workers,
                      pin_memory=device.type == "cuda", drop_last=False, persistent_workers=cfg.num_workers > 0)


def train_run(cfg: TrainConfig, part: Partition, method: str = "standard", n_classes: int | None = None) -> dict:
    """Train on ``part.train``, select on ``part.val``, test once on ``part.test``.

    Returns a dict with the test metrics, the per-epoch history and the trained
    ``model`` (an AuditModel, or (AuditedEncoder, AttentionMIL) for MIL).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method}; choose from {METHODS}")
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    amp = cfg.amp and device.type == "cuda"
    n_classes = n_classes or int(max(part.train["label"].max(), part.val["label"].max(), part.test["label"].max()) + 1)
    if method == "mil":
        return _train_mil(cfg, part, n_classes, device, amp)

    group_to_idx = {g: i for i, g in enumerate(sorted(part.train["group"].unique()))}
    tr = _loader(part.train, cfg, True, group_to_idx, device)
    va = _loader(part.val, cfg, False, group_to_idx, device)
    te = _loader(part.test, cfg, False, group_to_idx, device)

    model = AuditModel(n_classes, len(group_to_idx), cfg.backbone, cfg.pretrained).to(device)
    cls_w = _class_weight_tensor(part.train["label"].to_numpy(), n_classes, cfg.class_weights, device)
    grl = GradientReversal().to(device)
    supcon = SubjectExcludedSupCon(cfg.supcon_temperature)

    if method == "confusion":
        main_params = list(model.encoder.parameters()) + list(model.h_cls.parameters()) + list(model.h_proj.parameters())
        opt = torch.optim.AdamW(main_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        opt_adv = torch.optim.AdamW(model.h_adv.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        opt_adv = None
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    select_last = method == "grl_convergence"
    history = []
    best = {"val_bal_acc": -1.0, "epoch": -1, "state": None}
    with Timer(device) as t_train:
        for epoch in range(cfg.epochs):
            model.train()
            grl.lam = grl_lambda(epoch, cfg.epochs) if method in ("grl", "grl_convergence") else 0.0
            sums = {"loss": 0.0, "cls": 0.0, "adv": 0.0, "ctr": 0.0, "conf": 0.0}
            n_batches = 0
            for x, y, g, _ in tr:
                x, y, g = x.to(device, non_blocking=True), y.to(device), g.to(device)
                with torch.autocast(device_type=device.type, enabled=amp):
                    out = model(x)
                    l_cls = F.cross_entropy(out["logits"], y, weight=cls_w)
                    loss = l_cls
                    parts = {"cls": l_cls.item()}
                    if method in ("grl", "grl_convergence"):
                        l_adv = F.cross_entropy(model.h_adv(grl(out["z"])), g)
                        loss = loss + l_adv
                        parts["adv"] = l_adv.item()
                    elif method == "confusion":
                        l_ctr = supcon(out["proj"].float(), y, g)
                        l_conf = confusion_loss(model.h_adv(out["z"]))
                        loss = loss + l_ctr + l_conf
                        parts.update({"ctr": l_ctr.item(), "conf": l_conf.item()})
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                if method == "confusion":  # adversary step on detached features (its own optimiser)
                    opt_adv.zero_grad(set_to_none=True)   # also drops the grads L_conf left on h_adv
                    with torch.autocast(device_type=device.type, enabled=amp):
                        l_adv = F.cross_entropy(model.h_adv(out["z"].detach()), g)
                    scaler.scale(l_adv).backward()
                    scaler.step(opt_adv)
                    parts["adv"] = l_adv.item()
                scaler.update()
                sums["loss"] += loss.item(); n_batches += 1
                for k, v in parts.items():
                    sums[k] += v
            sched.step()
            y_va, p_va = _predict(model, va, device, amp)
            val_bal = balanced_accuracy(y_va, p_va)
            rec = {"epoch": epoch, "lambda": grl.lam, "val_bal_acc": val_bal, "lr": sched.get_last_lr()[0]}
            rec.update({k: v / max(n_batches, 1) for k, v in sums.items()})
            history.append(rec)
            log.info("[%s] epoch %02d loss %.3f val_bal_acc %.3f lambda %.2f", method, epoch, rec["loss"], val_bal, grl.lam)
            if val_bal > best["val_bal_acc"] or select_last:
                best = {"val_bal_acc": val_bal, "epoch": epoch,
                        "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}

    model.load_state_dict(best["state"])
    y_te, p_te = _predict(model, te, device, amp)
    return {
        "method": method, "protocol": part.protocol, "seed": part.seed, "config": asdict(cfg),
        "n_classes": n_classes, "n_train_groups": len(group_to_idx),
        "selected_epoch": best["epoch"], "selection": "last" if select_last else "best_val",
        "val_bal_acc": best["val_bal_acc"],
        "test_bal_acc": balanced_accuracy(y_te, p_te),
        "test_recall": per_class_recall(y_te, p_te, n_classes),
        "test_confusion": row_normalised_confusion(y_te, p_te, n_classes),
        "y_true": y_te.tolist(), "y_pred": p_te.tolist(),
        "history": history, "train_seconds": t_train.seconds,
        "partition": part.summary(), "model": model,
    }


# ---------------------------------------------------------------------------------------------
# Attention-based MIL
# ---------------------------------------------------------------------------------------------

def _mil_forward(encoder, head, bags: list[torch.Tensor], device, amp):
    sizes = [b.shape[0] for b in bags]
    x = torch.cat(bags).to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, enabled=amp):
        h = encoder(x)
    logits = []
    for hb in torch.split(h.float(), sizes):
        lg, _ = head(hb)
        logits.append(lg)
    return torch.stack(logits)


@torch.no_grad()
def _mil_predict(encoder, head, loader, device, amp):
    encoder.eval(); head.eval()
    ys, ps = [], []
    for bags, y, _ in loader:
        ps.append(_mil_forward(encoder, head, bags, device, amp).argmax(1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def _train_mil(cfg: TrainConfig, part: Partition, n_classes: int, device, amp: bool) -> dict:
    mk = lambda df, train, seed: DataLoader(
        BagDataset(df, bag_size=cfg.bag_size, img_size=cfg.img_size, train=train, channels=cfg.channels, seed=seed),
        batch_size=cfg.bags_per_batch, shuffle=train, num_workers=cfg.num_workers, collate_fn=bag_collate,
        pin_memory=device.type == "cuda")
    tr, va, te = mk(part.train, True, cfg.seed), mk(part.val, False, cfg.seed), mk(part.test, False, cfg.seed)

    encoder = AuditedEncoder(cfg.backbone, cfg.pretrained).to(device)
    head = AttentionMIL(encoder.dim, n_classes).to(device)
    cls_w = _class_weight_tensor(part.train["label"].to_numpy(), n_classes, cfg.class_weights, device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history, best = [], {"val_bal_acc": -1.0, "epoch": -1, "state": None}
    with Timer(device) as t_train:
        for epoch in range(cfg.epochs):
            encoder.train(); head.train()
            tr.dataset.resample()
            tot, nb = 0.0, 0
            for bags, y, _ in tr:
                y = y.to(device)
                logits = _mil_forward(encoder, head, bags, device, amp)
                loss = F.cross_entropy(logits, y, weight=cls_w)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tot += loss.item(); nb += 1
            sched.step()
            y_va, p_va = _mil_predict(encoder, head, va, device, amp)
            val_bal = balanced_accuracy(y_va, p_va)
            history.append({"epoch": epoch, "loss": tot / max(nb, 1), "val_bal_acc": val_bal, "lambda": 0.0})
            log.info("[mil] epoch %02d loss %.3f val_bal_acc(bags) %.3f", epoch, tot / max(nb, 1), val_bal)
            if val_bal > best["val_bal_acc"]:
                best = {"val_bal_acc": val_bal, "epoch": epoch,
                        "state": (copy.deepcopy({k: v.detach().cpu() for k, v in encoder.state_dict().items()}),
                                  copy.deepcopy({k: v.detach().cpu() for k, v in head.state_dict().items()}))}
    encoder.load_state_dict(best["state"][0]); head.load_state_dict(best["state"][1])
    y_te, p_te = _mil_predict(encoder, head, te, device, amp)
    return {
        "method": "mil", "protocol": part.protocol, "seed": part.seed, "config": asdict(cfg),
        "n_classes": n_classes, "n_train_groups": int(part.train["group"].nunique()),
        "selected_epoch": best["epoch"], "selection": "best_val", "val_bal_acc": best["val_bal_acc"],
        "test_bal_acc": balanced_accuracy(y_te, p_te),
        "test_recall": per_class_recall(y_te, p_te, n_classes),
        "test_confusion": row_normalised_confusion(y_te, p_te, n_classes),
        "y_true": y_te.tolist(), "y_pred": p_te.tolist(),
        "history": history, "train_seconds": t_train.seconds, "partition": part.summary(),
        "model": (encoder, head), "note": "metrics are bag-level (bags of %d slices per subject)" % cfg.bag_size,
    }


def strip_model(result: dict) -> dict:
    """Return a JSON-serialisable copy of a train_run result (without the model object)."""
    return {k: v for k, v in result.items() if k != "model"}
