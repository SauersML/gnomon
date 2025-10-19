#!/usr/bin/env python3
"""Generate PCA projection plots for the release-fit workflow."""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from collections import Counter
from collections.abc import Sequence
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate release PCA projection plots with optional downsampling."
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=1,
        help="Downsampling factor applied per population for visualization (>=1).",
    )
    return parser.parse_args(argv)


def _normalize_value(value: object) -> str | None:
    if pd.isna(value):
        return None

    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()

    if not text or text.lower() == "nan":
        return None

    return text


def _resolve_consensus(values: pd.Series) -> str | None:
    cleaned = [_normalize_value(value) for value in values]
    cleaned = [value for value in cleaned if value is not None]

    if not cleaned:
        return None

    counts = Counter(cleaned)
    most_common = counts.most_common()
    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        return most_common[0][0]

    return None


def _consolidate_sample_annotations(igsr_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, str | None]] = []

    for sample_id, group in igsr_df.groupby("SampleID", sort=False):
        population = _resolve_consensus(group["Population"])

        normalized_population = group["Population"].map(_normalize_value)
        if population is not None:
            matching_rows = group.loc[normalized_population == population]
            if matching_rows.empty:
                matching_rows = group
        else:
            matching_rows = group

        superpopulation = _resolve_consensus(matching_rows["Superpopulation"])
        population_name = _resolve_consensus(matching_rows["PopulationNameLong"])
        if population_name is None:
            population_name = population

        records.append(
            {
                "SampleID": sample_id,
                "Population": population,
                "PopulationNameLong": population_name,
                "Superpopulation": superpopulation,
            }
        )

    consolidated = pd.DataFrame.from_records(
        records,
        columns=["SampleID", "Population", "PopulationNameLong", "Superpopulation"],
    )

    if not consolidated.empty:
        consolidated["SampleID"] = consolidated["SampleID"].astype("string")
        for column in ("Population", "PopulationNameLong", "Superpopulation"):
            consolidated[column] = consolidated[column].astype("string")

    return consolidated


def _downsample_dataframe(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    if factor <= 1 or df.empty:
        return df

    factor = max(1, factor)
    group_columns = [
        column
        for column in ("Superpopulation", "Population")
        if column in df.columns
    ]

    if not group_columns:
        size = len(df)
        target = max(1, math.ceil(size / factor))
        if target >= size:
            return df
        indices = np.linspace(0, size - 1, num=target, dtype=int)
        return df.iloc[indices].reset_index(drop=True)

    groups: list[pd.DataFrame] = []
    for _, group in df.groupby(group_columns, sort=False, dropna=False):
        size = len(group)
        target = max(1, math.ceil(size / factor))
        if target >= size:
            groups.append(group)
            continue

        indices = np.linspace(0, size - 1, num=target, dtype=int)
        groups.append(group.iloc[indices])

    downsampled = pd.concat(groups, axis=0)
    return downsampled.sort_index().reset_index(drop=True)


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


def _load_samples(samples_path: Path) -> pd.DataFrame:
    if not samples_path.exists():
        raise FileNotFoundError("Sample manifest from fit run was not found")

    samples_df = pd.read_csv(samples_path, sep="\t", dtype=str).applymap(
        lambda value: value.strip() if isinstance(value, str) else value
    )

    lower_to_actual = {column.lower(): column for column in samples_df.columns}

    sample_col: str | None = None
    for candidate in ("iid", "sampleid", "sample id", "sample", "id"):
        if candidate in lower_to_actual:
            sample_col = lower_to_actual[candidate]
            break

    if sample_col is None:
        raise ValueError("Sample manifest is missing a column containing sample IDs")

    filename_col: str | None = None
    for candidate in ("filename", "file", "filepath", "path", "source"):
        if candidate in lower_to_actual:
            filename_col = lower_to_actual[candidate]
            break

    if filename_col is None and "fid" in lower_to_actual:
        fid_col = lower_to_actual["fid"]
        fid_values = samples_df[fid_col].astype(str)
        if fid_values.str.contains(r"[./]").any():
            filename_col = fid_col

    if filename_col is None:
        # Fall back to using the sample identifier itself as the filename so
        # downstream logic can still operate deterministically.
        filename_col = sample_col

    sample_ids = samples_df[sample_col].astype(str)
    filenames = samples_df[filename_col].astype(str)

    return pd.DataFrame({"Filename": filenames, "SampleID": sample_ids})


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

    if "Population name" in igsr_df.columns:
        igsr_df["PopulationNameLong"] = igsr_df["Population name"].astype(str).str.strip()
    else:
        igsr_df["PopulationNameLong"] = igsr_df["Population"]

    subset = igsr_df[["SampleID", "Population", "PopulationNameLong", "Superpopulation"]]
    subset = subset[subset["SampleID"].map(lambda value: isinstance(value, str) and value != "")]
    return _consolidate_sample_annotations(subset)


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


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    downsample_factor = max(1, args.downsample_factor)
    if args.downsample_factor < 1:
        print(
            f"Downsample factor {args.downsample_factor} is less than 1; defaulting to 1.",
            flush=True,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pcs_df = _load_model_scores(HWE_PATH)
    samples_df = _load_samples(SAMPLES_PATH)

    if len(samples_df) != len(pcs_df):
        raise ValueError(
            "Sample manifest row count does not match the number of score rows"
        )

    pcs_df.insert(0, "Filename", samples_df["Filename"].tolist())
    filename_to_sample = samples_df.drop_duplicates("Filename").set_index("Filename")[
        "SampleID"
    ]
    pcs_df.insert(1, "SampleID", pcs_df["Filename"].map(filename_to_sample))
    pcs_df["SampleID"] = pcs_df["SampleID"].fillna(pcs_df["Filename"])

    igsr_path = OUTPUT_DIR / IGSR_FILENAME
    _download_igsr(igsr_path)
    igsr_df = _load_igsr(igsr_path)

    ann_pca = pcs_df.merge(igsr_df, on="SampleID", how="left")
    ann_pca["Population"] = ann_pca["Population"].fillna("UNK")
    ann_pca["Superpopulation"] = ann_pca["Superpopulation"].fillna("OTH")
    ann_pca["PopulationNameLong"] = ann_pca["PopulationNameLong"].fillna(ann_pca["Population"])

    plot_pca = _downsample_dataframe(ann_pca, downsample_factor)
    if downsample_factor > 1:
        print(
            "Downsampled annotated PCA data from "
            f"{len(ann_pca)} to {len(plot_pca)} rows using factor {downsample_factor}.",
            flush=True,
        )

    color_map = _build_color_map(plot_pca)

    score_columns = [column for column in pcs_df.columns if column.startswith("PC")]
    export_columns = [
        "SampleID",
        "Population",
        "Superpopulation",
        *score_columns,
    ]

    available_columns = [
        column for column in export_columns if column in ann_pca.columns
    ]

    projection_path = OUTPUT_DIR / "pca_projection_scores.tsv"
    ann_pca.loc[:, available_columns].rename(
        columns={"SampleID": "SampleName", "Population": "Subpopulation"}
    ).to_csv(projection_path, sep="\t", index=False)

    _plot_projection(plot_pca, color_map, "PC1", "PC2", OUTPUT_DIR / "pc1_pc2.png")
    _plot_projection(plot_pca, color_map, "PC3", "PC4", OUTPUT_DIR / "pc3_pc4.png")


if __name__ == "__main__":
    main()
