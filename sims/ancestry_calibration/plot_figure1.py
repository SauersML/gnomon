#!/usr/bin/env python3
"""Figure 1 (study design).

(a) Workflow: msprime demography -> genotypes -> real P+T GWAS in the training
    ancestry -> standardized PGS -> calibration fit -> held-out evaluation on a
    held-out training-ancestry test set and other-ancestry test sets.
(b) the one-dimensional serial-founder chain: formal demography (demesdraw) and
    a color-matched PC plot.
(c) the two-dimensional stepping-stone grid: formal demography and a
    color-matched PC plot.

Deme colours are shared between each demography panel and its PC scatter:
the chain uses an ordered sequential ramp (training deme -> most diverged); the
grid uses a 2-D bilinear field so a deme's grid position *is* its colour, and
the PC scatter shows that PC space recovers the 2-D layout.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import demesdraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_real_pt as G  # noqa: E402

DATA = os.path.join(HERE, "..", "results_hpc", "ancestry_calibration", "data")
OUTDIR = os.path.join(HERE, "..", "results_hpc", "ancestry_calibration", "plots")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "savefig.dpi": 200,
})

# ------------------------------------------------------------------ colours ---
CHAIN_CMAP = plt.cm.cividis          # ordered, CVD-safe; training=dark end
GRID_CORNERS = {                     # bilinear anchors over the 6x6 lattice
    (0, 0): "#2d4a8a",   # training deme (top-left): indigo
    (0, 1): "#d1495b",   # rose
    (1, 0): "#edae49",   # amber
    (1, 1): "#5b8c5a",   # sage
}

def chain_colour(k, D=10):
    return CHAIN_CMAP(0.08 + 0.84 * k / (D - 1))

def grid_colour(r, c, side=6):
    u, v = r / (side - 1), c / (side - 1)
    cc = {k: np.array(mcolors.to_rgb(h)) for k, h in GRID_CORNERS.items()}
    col = ((1 - u) * (1 - v) * cc[(0, 0)] + (1 - u) * v * cc[(0, 1)]
           + u * (1 - v) * cc[(1, 0)] + u * v * cc[(1, 1)])
    return tuple(np.clip(col, 0, 1))

# -------------------------------------------------------------- data loader ---
def load_pcs(pattern, npc):
    f = sorted(glob.glob(os.path.join(DATA, pattern)))[0]
    cols = ["deme"] + [f"PC{i+1}" for i in range(npc)]
    d = pd.read_parquet(f, columns=cols + ["split"])
    d = d[d["split"] != "pgs_train"].copy()   # drop the huge GWAS block
    d["deme"] = d["deme"].astype(int)
    rng = np.random.default_rng(0)
    if len(d) > 4500:
        d = d.iloc[rng.choice(len(d), 4500, replace=False)]
    return d

# ----------------------------------------------------- demography (demesdraw) --
def draw_demography(ax, demo, colour_of_name, title):
    graph = demo.to_demes()
    colours = {d.name: colour_of_name(d.name) for d in graph.demes}
    colours["ANC"] = "#b9b9b9"
    try:
        demesdraw.tubes(graph, ax=ax, colours=colours, log_time=True,
                        labels=None, title=None, seed=1)
    except TypeError:
        demesdraw.tubes(graph, ax=ax, colours=colours, log_time=True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# --------------------------------------------------- grid spatial lattice -----
def draw_grid_lattice(ax, side=6, train=(0, 0)):
    """The 2-D stepping-stone model drawn as its spatial lattice: nodes coloured
    by the same 2-D field used in the PC scatter, nearest-neighbour migration
    as edges. Doubles as the colour key for panel (c)."""
    ax.set_aspect("equal"); ax.axis("off")
    pos = lambda r, c: (c, side - 1 - r)            # row 0 at top
    for r in range(side):                           # migration edges
        for c in range(side):
            for dr, dc in [(1, 0), (0, 1)]:
                r2, c2 = r + dr, c + dc
                if r2 < side and c2 < side:
                    (x1, y1), (x2, y2) = pos(r, c), pos(r2, c2)
                    ax.plot([x1, x2], [y1, y2], color="#cfcfcf", lw=1.0, zorder=1)
    for r in range(side):                           # deme nodes
        for c in range(side):
            x, y = pos(r, c)
            ax.scatter(x, y, s=210, c=[grid_colour(r, c)], edgecolor="white",
                       lw=1.0, zorder=2)
            if (r, c) == train:
                ax.scatter(x, y, s=330, facecolors="none", edgecolor="black",
                           lw=1.8, zorder=3)
    tx, ty = pos(*train)
    ax.annotate("training population", xy=(tx, ty), xytext=(tx + 0.35, ty + 0.85),
                fontsize=8, ha="left", va="bottom", color="#222",
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=0.9))
    ax.set_xlim(-0.7, side - 0.3); ax.set_ylim(-0.5, side + 0.4)

# ----------------------------------------------------------- PC scatter -------
def pc_scatter(ax, df, colour_of_deme, title, edges=(), train_deme=0):
    """Scatter of individuals (faint) with the demographic graph overlaid:
    deme centroids joined by the migration topology, so PC space is shown to
    recover the chain/grid structure. Colours match the demography panel."""
    cols = np.array([colour_of_deme(int(d)) for d in df["deme"]])
    ax.scatter(df["PC1"].values, df["PC2"].values, s=5, c=cols, lw=0,
               alpha=0.35, rasterized=True, zorder=1)
    cen = df.groupby("deme")[["PC1", "PC2"]].mean()
    for a, b in edges:                          # migration topology as a mesh
        if a in cen.index and b in cen.index:
            ax.plot([cen.loc[a, "PC1"], cen.loc[b, "PC1"]],
                    [cen.loc[a, "PC2"], cen.loc[b, "PC2"]],
                    color="#666666", lw=0.7, alpha=0.55, zorder=2)
    if train_deme in cen.index:                 # training deme (only marker kept)
        ax.scatter(cen.loc[train_deme, "PC1"], cen.loc[train_deme, "PC2"], s=120,
                   marker="*", c="white", edgecolor="black", lw=1.0, zorder=4)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# ----------------------------------------------------------------- assemble ----
def make_chain():
    """Panel b: 1-D serial-founder chain — demesdraw demography + PC plot."""
    fig = plt.figure(figsize=(8.6, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.24,
                          left=0.07, right=0.975, top=0.965, bottom=0.12)
    demo = G.dem_serial1d()[0]
    ax1 = fig.add_subplot(gs[0, 0])
    draw_demography(ax1, demo,
                    lambda nm: chain_colour(int(nm[1:])) if nm.startswith("d") and nm[1:].isdigit() else "#b9b9b9",
                    "1-D serial-founder chain")
    df = load_pcs("serial1d_phenoA_realpt_s1.parquet", 2)
    ax2 = fig.add_subplot(gs[0, 1])
    chain_edges = [(i, i + 1) for i in range(9)]
    pc_scatter(ax2, df, lambda d: chain_colour(int(d)), "PC space (chain)", edges=chain_edges)
    out = os.path.join(OUTDIR, "figure1b_chain.png")
    fig.savefig(out); print("WROTE", out)

def make_grid():
    """Panel c: 2-D stepping-stone grid — spatial lattice + PC plot."""
    fig = plt.figure(figsize=(8.6, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.24,
                          left=0.05, right=0.975, top=0.965, bottom=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    draw_grid_lattice(ax1, side=6, train=(0, 0))
    df = load_pcs("grid2d_phenoA_realpt_s1.parquet", 3)
    ax2 = fig.add_subplot(gs[0, 1])
    side = 6; grid_edges = []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if c + 1 < side: grid_edges.append((i, i + 1))
            if r + 1 < side: grid_edges.append((i, i + side))
    pc_scatter(ax2, df, lambda d: grid_colour(int(d) // 6, int(d) % 6),
               "PC space (grid)", edges=grid_edges)
    out = os.path.join(OUTDIR, "figure1c_grid.png")
    fig.savefig(out); print("WROTE", out)

if __name__ == "__main__":
    make_chain()
    make_grid()
