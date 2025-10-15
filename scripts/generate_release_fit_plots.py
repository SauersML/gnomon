#!/usr/bin/env python3
"""Generate PCA projection plots for the release-fit workflow."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


OUTPUT_DIR = Path("artifacts")
HWE_PATH = Path("phased_haplotypes_v2.hwe.json")
SAMPLES_PATH = Path("phased_haplotypes_v2.samples.tsv")
IGSR_URL = "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/igsr_samples.tsv"
IGSR_FILENAME = "igsr_samples.tsv"


def _load_model_scores(model_path: Path) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(f"Expected model at {model_path} from gnomon fit run")

    with model_path.open("r", encoding="utf-8") as fh:
        model = json.load(fh)

    scores_meta = model["sample_scores"]
    rows = scores_meta["nrows"]
    cols = scores_meta["ncols"]
    values = scores_meta["data"]

    if len(values) != rows * cols:
        raise ValueError("sample_scores data length mismatch")

    scores = np.array(values, dtype=float).reshape((cols, rows)).T
    return pd.DataFrame(scores, columns=[f"PC{i + 1}" for i in range(cols)])


def _load_samples(samples_path: Path) -> pd.Series:
    if not samples_path.exists():
        raise FileNotFoundError("Sample manifest from fit run was not found")

    samples_df = pd.read_csv(samples_path, sep="\t", dtype=str)
    return samples_df["IID"].astype(str).str.strip()


def _download_igsr(target_path: Path) -> None:
    if target_path.exists():
        return

    req = urllib.request.Request(IGSR_URL, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req) as response, target_path.open("wb") as out:
        total = response.getheader("Content-Length")
        total_bytes = int(total) if total is not None else None
        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {target_path.name}",
            leave=False,
            dynamic_ncols=True,
        ) as progress:
            while True:
                chunk = response.read(1024 * 64)
                if not chunk:
                    break
                out.write(chunk)
                progress.update(len(chunk))


def _load_igsr(target_path: Path) -> pd.DataFrame:
    igsr_df = pd.read_csv(target_path, sep="\t", dtype=str)
    igsr_df["SampleID"] = igsr_df["Sample name"].astype(str).str.strip()
    igsr_df["Population"] = igsr_df["Population code"].astype(str).str.strip()
    igsr_df["Superpopulation"] = igsr_df["Superpopulation code"].astype(str).str.strip()
    igsr_df["PopulationNameLong"] = igsr_df.get("Population name", igsr_df["Population"])
    return igsr_df[["SampleID", "Population", "PopulationNameLong", "Superpopulation"]]


def _build_color_map(annotated: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    superpopulations = set(annotated["Superpopulation"])
    color_map: dict[tuple[str, str], np.ndarray] = {}
    base_palette = plt.get_cmap("tab10")

    for idx, superpop in enumerate(sorted(superpopulations)):
        base_color = np.array(mcolors.to_rgb(base_palette(idx % base_palette.N)))
        subpopulations = sorted(
            set(annotated.loc[annotated["Superpopulation"] == superpop, "Population"])
        )
        if not subpopulations:
            continue

        shades = np.linspace(0.45, 1.0, num=len(subpopulations))
        for shade, subpop in zip(shades, subpopulations):
            adjusted = np.clip(base_color * shade, 0, 1)
            color_map[(superpop, subpop)] = adjusted

    return color_map


def _plot_projection(
    annotated: pd.DataFrame,
    color_map: dict[tuple[str, str], np.ndarray],
    x: str,
    y: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    seen_labels: set[str] = set()

    for (superpop, subpop), group in annotated.groupby(["Superpopulation", "Population"]):
        color = color_map.get((superpop, subpop), (0.4, 0.4, 0.4))
        label = f"{superpop} – {subpop}"
        ax.scatter(
            group[x],
            group[y],
            s=14,
            c=[color],
            edgecolors="none",
            alpha=0.8,
            label=label,
        )
        seen_labels.add(label)

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y} (annotated)")

    handles, labels = [], []
    for (superpop, subpop) in sorted(color_map.keys()):
        label = f"{superpop} – {subpop}"
        if label in seen_labels:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=color_map[(superpop, subpop)],
                    label=label,
                )
            )
            labels.append(label)

    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(1.35, 1),
            fontsize="small",
            ncol=1,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pcs_df = _load_model_scores(HWE_PATH)
    sample_ids = _load_samples(SAMPLES_PATH)
    pcs_df.insert(0, "SampleID", sample_ids)

    igsr_path = OUTPUT_DIR / IGSR_FILENAME
    _download_igsr(igsr_path)
    igsr_df = _load_igsr(igsr_path)

    ann_pca = pcs_df.merge(igsr_df, on="SampleID", how="left")
    ann_pca["Population"] = ann_pca["Population"].fillna("UNK")
    ann_pca["Superpopulation"] = ann_pca["Superpopulation"].fillna("OTH")
    ann_pca["PopulationNameLong"] = ann_pca["PopulationNameLong"].fillna(ann_pca["Population"])

    color_map = _build_color_map(ann_pca)

    _plot_projection(ann_pca, color_map, "PC1", "PC2", OUTPUT_DIR / "pc1_pc2.png")
    _plot_projection(ann_pca, color_map, "PC3", "PC4", OUTPUT_DIR / "pc3_pc4.png")


if __name__ == "__main__":
    main()
