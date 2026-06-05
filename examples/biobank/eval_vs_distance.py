#!/usr/bin/env python3
"""Individual-level evaluation vs. distance from a reference centroid, smoothed
with the gamfit GAM and its uncertainty band.

For each model (marginal-slope GAM, z-norm Cox, score+PC Cox) we take a
per-individual evaluation residual on the held-out test set, the Euclidean
distance of each individual to a reference centroid in PC space, and smooth
residual-vs-distance with gamfit (`value ~ matern(distance)`, Gaussian),
reading the mean and confidence band straight from `model.predict`.

Biobank (Figure 4b): reference = EUR-ancestry centroid; residual = per-individual
calibration error at a horizon. Simulation (Figure 2b): reference = training-deme
centroid; residual = |predicted - true risk|. Same engine, different centroid.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_STYLE = {
    "gamfit":         ("#c0392b", "marginal-slope"),
    "marginal-slope": ("#c0392b", "marginal-slope"),
    "linpc":          ("#2d4a8a", "linear-PC"),
    "linear-PC":      ("#2d4a8a", "linear-PC"),
    "znorm":          ("#7a7a7a", "z-norm"),
    "z-norm":         ("#7a7a7a", "z-norm"),
}


# ------------------------------------------------------- gamfit smoother + CI --
def gamfit_smooth(x, y, w=None, centers=20, ngrid=160, ci=0.95):
    """Smooth y ~ x with gamfit (Matern smooth, Gaussian family) and read the
    mean and confidence band straight from gamfit's own prediction
    (`predict(..., interval=ci)` -> mean / mean_lower / mean_upper). Optional
    per-point weights `w` (e.g. IPCW) are passed through to gamfit.fit.
    Returns (grid, mean, lo, hi)."""
    import gamfit
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if w is not None:
        w = np.asarray(w, float); ok &= np.isfinite(w)
    x, y = x[ok], y[ok]
    train = pd.DataFrame({"distance": x, "value": y})
    fit_kw = {}
    if w is not None:
        train["wt"] = w[ok]; fit_kw["weights"] = "wt"
    model = gamfit.fit(train, f"value ~ matern(distance, centers={centers})",
                       family="gaussian", **fit_kw)
    grid = pd.DataFrame({"distance": np.linspace(np.quantile(x, 0.01),
                                                 np.quantile(x, 0.99), ngrid)})
    pred = model.predict(grid, interval=ci)
    return (grid["distance"].to_numpy(float), pred["mean"].to_numpy(float),
            pred["mean_lower"].to_numpy(float), pred["mean_upper"].to_numpy(float))


# ------------------------------------------------------------- distances -------
def centroid_distance(P, mask_ref, k=None):
    """Euclidean distance of every row of P to the centroid of rows where
    mask_ref is True, using the first k PCs (all if k is None)."""
    P = np.asarray(P, float)
    P = P[:, :k] if k else P
    if mask_ref.sum() == 0:
        raise ValueError("reference group is empty (no individuals for centroid)")
    centroid = P[mask_ref].mean(0)
    return np.sqrt(((P - centroid) ** 2).sum(1))


# ----------------------------------------------------------- per-indiv eval ----
def binary_abs_error(p_hat, y):
    return np.abs(np.asarray(p_hat, float) - np.asarray(y, float))


def survival_horizon_error(F_hat, exit_age, entry_age, event, horizon):
    """Per-individual absolute calibration error at `horizon`:
    |1{T<=h, event} - F_hat(h)|, defined only for individuals whose status at the
    horizon is known (event by h, or still under follow-up past h). Returns
    (err, defined_mask)."""
    s = np.asarray(exit_age, float) - np.asarray(entry_age, float)
    ev = np.asarray(event, bool)
    F = np.asarray(F_hat, float)
    label = ((s <= horizon) & ev).astype(float)
    defined = (s > horizon) | ev
    return np.abs(label - F), defined


# ----------------------------------------------------------------- plotting ----
def plot_eval_vs_distance(df, dist_col, eval_col, method_col, ax=None,
                          xlabel="distance from EUR centroid in PC space",
                          ylabel="individual calibration error", centers=20):
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.2))
    for method, g in df.groupby(method_col):
        color, label = METHOD_STYLE.get(str(method), ("#444", str(method)))
        grid, mean, lo, hi = gamfit_smooth(g[dist_col].to_numpy(), g[eval_col].to_numpy(),
                                           centers=centers)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.18, lw=0)
        ax.plot(grid, mean, color=color, lw=1.8, label=label)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax


# --------------------------------------------------------------------- CLI -----
def run_sim_demo(individual_csv, out_png):
    df = pd.read_csv(individual_csv)
    dems = df["dem"].unique()
    fig, axes = plt.subplots(1, len(dems), figsize=(5.2 * len(dems), 4.2), squeeze=False)
    for ax, (dem, g) in zip(axes[0], df.groupby("dem")):
        plot_eval_vs_distance(g, "dist_from_train", "abs_error_true_risk", "method",
                              ax=ax, xlabel="distance from training deme",
                              ylabel="|predicted − true risk|")
        ax.set_title(dem, fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=170)
    print("WROTE", out_png)


def run_biobank(individual_csv, out_png, horizon, npc=8, eur_label="eur"):
    """Figure-4b: per-individual calibration error at `horizon` vs distance from
    the EUR centroid, gamfit-smoothed with bands. Expects a per-individual table
    (PC1..PCk, entry_age, exit_age, event, ancestry, Fhat_<method>) emitted by
    marginal_slope_diseases.py."""
    df = pd.read_csv(individual_csv)
    pcs = df[sorted([c for c in df.columns if c.startswith("PC")],
                    key=lambda c: int(c[2:]))].to_numpy()
    eur = df["ancestry"].astype(str).str.lower().eq(eur_label).to_numpy()
    dist = centroid_distance(pcs, eur, k=npc)
    fcols = {c[len("Fhat_"):]: c for c in df.columns if c.startswith("Fhat_")}
    if not fcols:
        raise SystemExit("no Fhat_<method> columns in the per-individual table")
    parts = []
    for method, fcol in fcols.items():
        err, defined = survival_horizon_error(df[fcol].to_numpy(), df["exit_age"],
                                              df["entry_age"], df["event"], horizon)
        parts.append(pd.DataFrame({"method": method, "dist": dist, "err": err})[defined])
    long = pd.concat(parts, ignore_index=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    plot_eval_vs_distance(long, "dist", "err", "method", ax=ax,
                          xlabel="distance from EUR centroid in PC space",
                          ylabel=f"individual calibration error at {horizon:g} (IPCW)")
    fig.tight_layout(); fig.savefig(out_png, dpi=170)
    print("WROTE", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "biobank"], default="sim")
    ap.add_argument("--individual-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=float, default=10.0)
    ap.add_argument("--npc", type=int, default=8)
    args = ap.parse_args()
    if args.mode == "sim":
        run_sim_demo(args.individual_csv, args.out)
    else:
        run_biobank(args.individual_csv, args.out, args.horizon, args.npc)


if __name__ == "__main__":
    main()
