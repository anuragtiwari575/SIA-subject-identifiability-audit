"""Regenerate the paper's figures from the saved JSON results.

    fig1_composition   Stage 1 (stage1_composition.json)
    fig2_budget        redundancy sweep (budget_sweep.json)
    fig3_confusion     Stage 2 confusion matrices (stage2_oasis_k20_resnet18.json)
    fig6_lambda        GRL failure mode (interventions_oasis.json)
    fig_dissociation   rho/eta plane and LI vs SIS~ (stage3_cross.json + stage1)

Figures are written as PDF (Type 42 fonts, as ACM requires) and PNG.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.spines.top": False,
                            "axes.spines.right": False})
import matplotlib.pyplot as plt  # noqa: E402

from ..utils import load_json  # noqa: E402

log = logging.getLogger("sia")
RED, BLUE, ORANGE, GREY = "#c44e52", "#4c72b0", "#dd8452", "#7f7f7f"


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s/%s.{pdf,png}", out_dir, name)


def fig_composition(results_dir: Path, out_dir: Path):
    comp = load_json(results_dir / "stage1_composition.json")
    key = next(k for k in comp if k.startswith("OASIS (4-class"))
    r = comp[key]
    names = [pc["class_name"].replace(" Dementia", "").replace("Very mild", "Very Mild") for pc in r["per_class"]]
    imgs = [pc["n_images"] for pc in r["per_class"]]
    subs = [pc["n_groups"] for pc in r["per_class"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    y = np.arange(len(names))[::-1]
    for ax, vals, color, title, xlabel in ((axes[0], imgs, RED, "What is counted", "Images"),
                                          (axes[1], subs, BLUE, "What is independent", "Independent subjects")):
        ax.barh(y, vals, color=color)
        for yi, v in zip(y, vals):
            ax.text(v + 0.01 * max(vals), yi, f"{v:,}", va="center", fontsize=8)
        ax.set_yticks(y); ax.set_yticklabels(names); ax.set_title(title); ax.set_xlabel(xlabel)
        ax.set_xlim(0, max(vals) * 1.3)
    axes[1].set_yticklabels([])
    i_min = int(np.argmin(subs))
    axes[1].annotate(f"{subs[i_min]} patients", xy=(subs[i_min], y[i_min]), xytext=(max(subs) * 0.45, y[i_min]),
                     color=ORANGE, fontsize=8, arrowprops=dict(arrowstyle="->", color=ORANGE))
    fig.suptitle(f"{r['N']:,} images  =  {r['M']} subjects  (~{r['rho']:.0f} slices each)", y=1.02)
    _save(fig, out_dir, "fig1_composition")


def fig_budget(results_dir: Path, out_dir: Path):
    sw = load_json(results_dir / "budget_sweep.json")["sweep"]
    ks = [p["k"] for p in sw]; ms = [p["M"] for p in sw]
    a_i = [p["image_level"]["mean"] for p in sw]; s_i = [p["image_level"]["std"] or 0 for p in sw]
    a_g = [p["group_disjoint"]["mean"] for p in sw]; s_g = [p["group_disjoint"]["std"] or 0 for p in sw]
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.errorbar(ks, a_i, yerr=s_i, color=RED, marker="o", capsize=3, label="Apparent (image-level split)")
    ax.errorbar(ks, a_g, yerr=s_g, color=BLUE, marker="s", capsize=3, label="Real (subject-disjoint split)")
    ax.fill_between(ks, a_g, a_i, color=RED, alpha=0.08)
    chance = 1 / 3
    ax.axhline(chance, ls=":", color=GREY); ax.text(ks[-1], chance + 0.01, "chance", color=GREY, ha="right", fontsize=8)
    for k, g, p in zip(ks, a_g, sw):
        if p["LI_percent"] is not None:
            ax.annotate(f"LI {p['LI_percent']:.0f}%", (k, g), textcoords="offset points", xytext=(0, -14),
                        ha="center", color=BLUE, fontsize=8)
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off()
    ax.set_xlabel(f"Images per subject   (total N ≈ {int(round(np.mean([p['N'] for p in sw]), -2)):,})")
    ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0.25, 1.1); ax.legend(loc="lower left", fontsize=8, frameon=False)
    top = ax.secondary_xaxis("top"); top.set_xticks(ks); top.set_xticklabels([str(m) for m in ms]); top.set_xlabel("Subjects")
    top.minorticks_off()
    _save(fig, out_dir, "fig2_budget")


def fig_confusion(results_dir: Path, out_dir: Path, stage2_file: str = "stage2_oasis_k20_resnet18.json"):
    s = load_json(results_dir / stage2_file)["summary"]
    labels = ["ND", "VM", "M+M"] if s["n_classes"] == 3 else [str(i) for i in range(s["n_classes"])]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.9))
    for ax, cm, title, acc in ((axes[0], s["confusion_image"], "Random slice split", s["image_level"]["mean"]),
                               (axes[1], s["confusion_group"], "Subject-disjoint split", s["group_disjoint"]["mean"])):
        cm = np.array(cm)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_title(f"{title}\nbal. acc = {acc:.3f}")
        ax.spines[:].set_visible(True)
    axes[0].set_ylabel("True")
    _save(fig, out_dir, "fig3_confusion")


def fig_lambda(results_dir: Path, out_dir: Path, interventions_file: str = "interventions_oasis.json"):
    res = load_json(results_dir / interventions_file)
    runs = res["runs"].get("grl") or res["runs"].get("grl_convergence")
    if not runs:
        log.warning("no GRL run found; skipping fig6_lambda"); return
    r = runs[0]
    ep = [h["epoch"] for h in r["history"]]; va = [h["val_bal_acc"] for h in r["history"]]; lam = [h["lambda"] for h in r["history"]]
    sel = r["selected_epoch"]
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.plot(ep, va, color=RED, marker="o", ms=4); ax.set_xlabel("Epoch"); ax.set_ylabel("Balanced accuracy", color=RED)
    ax.tick_params(axis="y", colors=RED)
    ax.axhline(res.get("chance_bal_acc", 1 / 3), ls=":", color=GREY); ax.text(ep[-1], res.get("chance_bal_acc", 1 / 3) + 0.005, "chance", color=GREY, ha="right", fontsize=8)
    ax.scatter([ep[sel]], [va[sel]], s=120, facecolors="none", edgecolors="black", zorder=5)
    ax.annotate("checkpoint selected\nby validation", (ep[sel], va[sel]), textcoords="offset points", xytext=(40, 30),
                fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax2 = ax.twinx(); ax2.plot(ep, lam, color=BLUE, ls="--"); ax2.set_ylabel("reversal strength $\\lambda$", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE); ax2.set_ylim(0, 1.05); ax2.spines["right"].set_visible(True)
    _save(fig, out_dir, "fig6_lambda")


def fig_dissociation(results_dir: Path, out_dir: Path):
    cross = load_json(results_dir / "stage3_cross.json")
    comp = cross.get("composition") or load_json(results_dir / "stage1_composition.json")
    pts = []
    for r in cross["table8"]:
        if r["backbone"] != "resnet18":
            continue
        key = "OASIS (3-class, Moderate merged)" if r["dataset"] == "oasis" else "TCGA-LGG"
        c = comp.get(key)
        if c is None or r["LI_percent"] is None or r["sis_adjusted"] is None:
            continue
        pts.append({"name": "OASIS" if r["dataset"] == "oasis" else "TCGA-LGG", "rho": c["rho"], "excess": c["eta_minus_ref"],
                    "LI": r["LI_percent"], "sis": r["sis_adjusted"], "color": RED if r["dataset"] == "oasis" else BLUE})
    if not pts:
        log.warning("no points for fig_dissociation"); return
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.7))
    a.axhspan(0.35, 1.2, xmin=0, xmax=1, color=RED, alpha=0.07)
    a.text(0.97, 1.05, "both conditions present", color=RED, ha="right", fontsize=8, transform=a.get_yaxis_transform())
    for p in pts:
        a.scatter(p["rho"], p["excess"], s=140, color=p["color"], edgecolor="black", zorder=5)
        a.annotate(f"{p['name']}\nLI = {p['LI']:.1f}%", (p["rho"], p["excess"]), textcoords="offset points", xytext=(-12, -26), fontsize=8)
        b.scatter(p["sis"], p["LI"], s=140, color=p["color"], edgecolor="black", zorder=5)
        b.annotate(p["name"], (p["sis"], p["LI"]), textcoords="offset points", xytext=(-10, 12), fontsize=8)
    a.set_xscale("log"); a.set_xlabel("Group redundancy  $\\rho$"); a.set_ylabel("Excess homogeneity  $\\eta - 1/C$")
    a.minorticks_off(); a.set_ylim(-0.1, 1.2); a.set_title("(a) Leakage needs both conditions", fontsize=9)
    rhos = [p["rho"] for p in pts]; a.set_xlim(min(rhos) / 3, max(rhos) * 3)
    b.set_xlabel("Identity encoded  (chance-adjusted SIS)"); b.set_ylabel("Identity exploited  (LI, %)")
    sis_v = [p["sis"] for p in pts]; li_v = [p["LI"] for p in pts]
    b.set_xlim(max(0.0, min(sis_v) - 0.2), min(1.05, max(sis_v) + 0.15)); b.set_ylim(0, max(10, max(li_v) * 1.3))
    b.set_title("(b) Encoding is necessary, not sufficient", fontsize=9)
    _save(fig, out_dir, "fig_dissociation")


def make_all(results_dir: str | Path, out_dir: str | Path):
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    stage2_files = sorted(results_dir.glob("stage2_oasis_k*_resnet18.json"),
                          key=lambda p: (p.name != "stage2_oasis_k20_resnet18.json", p.name))
    jobs = [(fig_composition, "stage1_composition.json", {}), (fig_budget, "budget_sweep.json", {}),
            (fig_lambda, "interventions_oasis.json", {}), (fig_dissociation, "stage3_cross.json", {})]
    if stage2_files:
        jobs.append((fig_confusion, stage2_files[0].name, {"stage2_file": stage2_files[0].name}))
    else:
        log.warning("no stage2_oasis_k*_resnet18.json -> skipping fig3_confusion")
    for fn, needs, kw in jobs:
        if (results_dir / needs).exists():
            try:
                fn(results_dir, out_dir, **kw)
            except Exception as e:  # keep going: one missing figure should not block the others
                log.exception("%s failed: %s", fn.__name__, e)
        else:
            log.warning("%s missing -> skipping %s", needs, fn.__name__)
