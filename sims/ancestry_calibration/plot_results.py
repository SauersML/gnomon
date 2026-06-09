"""Render the study figures from the aggregated result tables.

Reads results/{accuracy,calibration}_{binary,survival}.csv and writes to figures/:
  1) discrimination.png   -- GLOBAL discrimination (binary AUC, liability R2, Brier;
                             survival Harrell C), 5 methods x (2 dem x 3 pheno).
  2) calibration_bss.png  -- ancestry-stratified calibration via Brier Skill Score,
                             held-out training-ancestry vs non-training populations.
  3) bss_vs_distance.png  -- BSS decay with genetic distance from the training deme.
  4) murphy.png           -- Murphy reliability (miscalibration) and resolution
                             (discrimination) in the non-training populations.
All mean +/- SD over seeds. Display order: gamfit, PGS+PCs, z-norm, CalPred, raw PGS.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("sims/results_hpc/ancestry_calibration")
RES = OUT / "results"
FIG = OUT / "figures"

WONG = {"gamfit": "#0072B2", "linpc": "#E69F00", "znorm": "#009E73",
        "calpred": "#CC79A7", "rawpgs": "#999999"}
MLAB = {"gamfit": "gamfit (marginal-slope)", "linpc": "PGS + PCs (linear)",
        "znorm": "z-norm (PC-adjusted)", "calpred": "CalPred", "rawpgs": "PGS (unadjusted)"}
MORD = ["gamfit", "linpc", "znorm", "calpred", "rawpgs"]
PHORD = ["phenoA", "phenoB", "phenoC"]
PHLAB = {"phenoA": "deme-varying baseline", "phenoB": "constant baseline",
         "phenoC": "drift-proof (equal prevalence)"}
DEMLAB = {"serial1d": "1-D chain", "grid2d": "2-D grid"}
CELLS = [(d, p) for d in ["serial1d", "grid2d"] for p in PHORD]


def _read(name):
    f = RES / name
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def _cellbars(ax, valfn, ylabel, title, hline=None, legend=False):
    x = np.arange(len(CELLS))
    w = 0.16
    for i, m in enumerate(MORD):
        mu = [valfn(d, p, m)[0] for d, p in CELLS]
        sd = [valfn(d, p, m)[1] for d, p in CELLS]
        ax.bar(x + (i - 2) * w, mu, w, yerr=sd, capsize=2, color=WONG[m],
               label=MLAB[m], edgecolor="white", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{DEMLAB[d]}\n{PHLAB[p]}" for d, p in CELLS], fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    if hline is not None:
        ax.axhline(hline, c="k", lw=0.8, ls="--")
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="best", ncol=2)


def _acc_valfn(acc, metric):
    g = acc[acc.metric == metric].groupby(["dem", "pheno", "method"]).value.agg(["mean", "std"]).reset_index()

    def fn(d, p, m):
        r = g[(g.dem == d) & (g.pheno == p) & (g.method == m)]
        return (float(r["mean"].iloc[0]), float(np.nan_to_num(r["std"].iloc[0]))) if len(r) else (np.nan, 0.0)
    return fn


def _cal_valfn(cal, metric, binval):
    sub = cal[(cal.metric == metric) & (cal.ancestry_bin_kind == "train_ancestry") & (cal.ancestry_bin == binval)]
    g = sub.groupby(["dem", "pheno", "method"]).value.agg(["mean", "std"]).reset_index()

    def fn(d, p, m):
        r = g[(g.dem == d) & (g.pheno == p) & (g.method == m)]
        return (float(r["mean"].iloc[0]), float(np.nan_to_num(r["std"].iloc[0]))) if len(r) else (np.nan, 0.0)
    return fn


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    bacc, bcal = _read("accuracy_binary.csv"), _read("calibration_binary.csv")
    sacc = _read("accuracy_survival.csv")

    # 1) discrimination
    fig, ax = plt.subplots(2, 2, figsize=(17, 10))
    if not bacc.empty:
        _cellbars(ax[0, 0], _acc_valfn(bacc, "auc"), "AUC (global)", "Binary discrimination — AUC",
                  hline=0.5, legend=True)
        _cellbars(ax[0, 1], _acc_valfn(bacc, "liability_r2"), "liability R2 (Lee 2011)",
                  "Binary discrimination — liability-scale R2")
        _cellbars(ax[1, 0], _acc_valfn(bacc, "brier"), "Brier (lower better)", "Binary — Brier score")
    if not sacc.empty:
        _cellbars(ax[1, 1], _acc_valfn(sacc, "cindex"), "Harrell's C (global)",
                  "Survival discrimination — C-index", hline=0.5)
    fig.suptitle("Real P+T PGS — GLOBAL discrimination (mean±SD over seeds)", fontsize=12, weight="bold", y=1.0)
    plt.tight_layout()
    fig.savefig(FIG / "discrimination.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote discrimination.png")

    if bcal.empty:
        print("PLOT_DONE (no binary calibration table)")
        return

    # 2) calibration BSS: training-ancestry vs non-training
    fig, ax = plt.subplots(1, 2, figsize=(17, 5.5), sharey=True)
    _cellbars(ax[0], _cal_valfn(bcal, "bss", "train_deme"), "Brier Skill Score (vs base rate)",
              "Calibration — training-ancestry test (held out)", hline=0.0, legend=True)
    _cellbars(ax[1], _cal_valfn(bcal, "bss", "other_deme"), "Brier Skill Score (vs base rate)",
              "Calibration — non-training populations", hline=0.0)
    fig.suptitle("Real P+T PGS — calibration skill (BSS>0 beats stratum base rate; emergent portability decay)",
                 fontsize=12, weight="bold", y=1.0)
    plt.tight_layout()
    fig.savefig(FIG / "calibration_bss.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote calibration_bss.png")

    # 3) BSS vs distance
    dsub = bcal[(bcal.metric == "bss") & (bcal.ancestry_bin_kind == "dist_bin")]
    if not dsub.empty:
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        for j, (d, p) in enumerate(CELLS):
            ax = axes[j // 3, j % 3]
            cell = dsub[(dsub.dem == d) & (dsub.pheno == p)]
            for m in MORD:
                cm = cell[cell.method == m]
                if cm.empty:
                    continue
                gg = cm.groupby("ancestry_bin").value.mean().reset_index()
                gg["xb"] = range(len(gg))
                ax.plot(gg["xb"], gg["value"], "o-", color=WONG[m], label=MLAB[m], ms=4, lw=1.5)
            ax.axhline(0, c="k", lw=0.8, ls="--")
            ax.set_title(f"{DEMLAB[d]} — {PHLAB[p]}", fontsize=10)
            ax.set_xlabel("distance-from-training quantile bin")
            ax.set_ylabel("BSS")
            ax.grid(alpha=0.25)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0, 0].legend(frameon=False, fontsize=7.5, loc="best")
        fig.suptitle("Real P+T PGS — calibration skill (BSS) vs genetic distance from the training population",
                     fontsize=12, weight="bold", y=1.0)
        plt.tight_layout()
        fig.savefig(FIG / "bss_vs_distance.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("wrote bss_vs_distance.png")

    # 4) Murphy reliability / resolution (non-training)
    fig, ax = plt.subplots(1, 2, figsize=(17, 5.5))
    _cellbars(ax[0], _cal_valfn(bcal, "reliability", "other_deme"),
              "reliability (miscalibration, lower better)", "Murphy reliability — non-training", legend=True)
    _cellbars(ax[1], _cal_valfn(bcal, "resolution", "other_deme"),
              "resolution (higher better)", "Murphy resolution — non-training")
    fig.suptitle("Real P+T PGS — Murphy decomposition of the Brier score (non-training ancestries)",
                 fontsize=12, weight="bold", y=1.0)
    plt.tight_layout()
    fig.savefig(FIG / "murphy.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote murphy.png")
    print("PLOT_DONE")


if __name__ == "__main__":
    main()
