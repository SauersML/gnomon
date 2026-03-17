#!/usr/bin/env python3
"""Plot projected PCA scores with superpopulation fill and subpopulation outlines."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECTION_MATRIX_MAGIC = b"GNPRJ001"
PROJECTION_MATRIX_VERSION = 3
PROJECTION_MATRIX_HEADER_LEN = 32
ROW_IDS_MAGIC = b"GNPSID01"
ROW_IDS_VERSION = 1
ROW_IDS_HEADER_LEN = 32
SAMPLE_MAPPING_URL = (
    "https://raw.githubusercontent.com/SauersML/genomic_pca/refs/heads/main/data/"
    "sample_population_mapping.tsv"
)
SUPERPOP_FILL_COLORS = {
    "AFR": "#d1495b",
    "AMR": "#edae49",
    "CSA": "#4f6d7a",
    "EAS": "#00798c",
    "EUR": "#30638e",
    "MID": "#8f5d5d",
    "OCE": "#6a994e",
    "SAS": "#7b2cbf",
    "OTH": "#7a7a7a",
    "UNK": "#a6a6a6",
}
SUPERPOP_ORDER = ["AFR", "AMR", "CSA", "EAS", "EUR", "MID", "OCE", "SAS", "OTH", "UNK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read gnomon projection_scores.bin and render a PCA plot using "
            "superpopulation fill colors and subpopulation outline colors."
        )
    )
    parser.add_argument("--scores-bin", required=True, help="Path to projection_scores.bin")
    parser.add_argument(
        "--metadata-json",
        required=True,
        help="Path to projection_scores.metadata.json",
    )
    parser.add_argument("--output-png", required=True, help="Output PNG path")
    parser.add_argument(
        "--output-tsv",
        help="Optional annotated TSV output path",
    )
    parser.add_argument(
        "--mapping-tsv",
        help="Optional local sample population mapping TSV. If omitted, it is downloaded.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name used in the plot title",
    )
    parser.add_argument(
        "--pc-x",
        default="PC1",
        help="Principal component for the x-axis",
    )
    parser.add_argument(
        "--pc-y",
        default="PC2",
        help="Principal component for the y-axis",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=1,
        help="Downsample equally within each superpopulation/subpopulation group",
    )
    return parser.parse_args()


def _normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    stripped = df.copy()
    for column in stripped.columns:
        series = stripped[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            stripped[column] = series.map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
    return stripped


def _download_if_missing(path: Path) -> None:
    if path.exists():
        return
    request = urllib.request.Request(
        SAMPLE_MAPPING_URL,
        headers={"User-Agent": "python-urllib"},
    )
    with urllib.request.urlopen(request) as response, path.open("wb") as handle:
        handle.write(response.read())


def _load_population_mapping(path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(path, sep="\t", dtype=str)
    mapping = _strip_string_columns(mapping)

    lower_to_actual = {column.lower(): column for column in mapping.columns}
    subpop_col = lower_to_actual.get("meta_subpopulation")
    superpop_col = lower_to_actual.get("meta_superpopulation")
    if subpop_col is None or superpop_col is None:
        raise ValueError(
            "Population mapping must contain meta_subpopulation and meta_superpopulation columns."
        )

    def is_sample_identifier(column: str) -> bool:
        normalized = "".join(char for char in column.lower() if char.isalnum())
        return "sampleid" in normalized or "samplename" in normalized

    sample_cols = [column for column in mapping.columns if is_sample_identifier(column)]
    if not sample_cols:
        raise ValueError("Population mapping did not contain any sample identifier columns.")

    records: list[dict[str, str | None]] = []
    for _, row in mapping.iterrows():
        subpop = _normalize_text(row[subpop_col])
        superpop = _normalize_text(row[superpop_col])
        for sample_col in sample_cols:
            sample_id = _normalize_text(row[sample_col])
            if sample_id is None:
                continue
            records.append(
                {
                    "SampleID": sample_id,
                    "Subpopulation": subpop,
                    "Superpopulation": superpop,
                }
            )

    if not records:
        raise ValueError("Population mapping did not yield any sample identifiers.")

    expanded = pd.DataFrame.from_records(records)
    expanded = expanded.drop_duplicates(subset="SampleID", keep="first")
    expanded["SampleID"] = expanded["SampleID"].astype("string")
    expanded["Subpopulation"] = expanded["Subpopulation"].astype("string")
    expanded["Superpopulation"] = expanded["Superpopulation"].astype("string")
    return expanded


def _read_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    version = metadata.get("version")
    if version != PROJECTION_MATRIX_VERSION:
        raise ValueError(
            f"Unsupported projection metadata version {version}; expected {PROJECTION_MATRIX_VERSION}."
        )
    if metadata.get("kind") != "scores":
        raise ValueError(f"Expected scores metadata, found {metadata.get('kind')!r}.")
    if metadata.get("layout") != "column_major":
        raise ValueError(f"Unsupported layout {metadata.get('layout')!r}.")
    if metadata.get("dtype") != "f64_le":
        raise ValueError(f"Unsupported dtype {metadata.get('dtype')!r}.")
    if metadata.get("row_ids_embedded") is not True:
        raise ValueError("Projection metadata must declare embedded row IDs.")
    if metadata.get("row_id_field") != "IID":
        raise ValueError(f"Unsupported row ID field {metadata.get('row_id_field')!r}.")
    return metadata


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _read_projection_binary(path: Path, rows: int, cols: int) -> tuple[np.ndarray, list[str]]:
    raw = path.read_bytes()
    if len(raw) < PROJECTION_MATRIX_HEADER_LEN:
        raise ValueError("Projection binary is shorter than the fixed header.")

    if raw[:8] != PROJECTION_MATRIX_MAGIC:
        raise ValueError("Projection binary magic did not match expected value.")
    version = int.from_bytes(raw[8:12], "little")
    if version != PROJECTION_MATRIX_VERSION:
        raise ValueError(
            f"Projection binary version {version} does not match expected version {PROJECTION_MATRIX_VERSION}."
        )
    header_rows = int.from_bytes(raw[12:20], "little")
    header_cols = int.from_bytes(raw[20:28], "little")
    if header_rows != rows or header_cols != cols:
        raise ValueError(
            f"Projection header shape ({header_rows}, {header_cols}) did not match metadata ({rows}, {cols})."
        )

    matrix_bytes = rows * cols * 8
    matrix_end = PROJECTION_MATRIX_HEADER_LEN + matrix_bytes
    if len(raw) < matrix_end + ROW_IDS_HEADER_LEN:
        raise ValueError("Projection binary is truncated before the row-ID section.")

    scores = np.frombuffer(
        raw[PROJECTION_MATRIX_HEADER_LEN:matrix_end],
        dtype="<f8",
        count=rows * cols,
    ).reshape((cols, rows)).T

    row_section = raw[matrix_end:]
    if row_section[:8] != ROW_IDS_MAGIC:
        raise ValueError("Projection row-ID section magic did not match expected value.")
    row_version = int.from_bytes(row_section[8:12], "little")
    if row_version != ROW_IDS_VERSION:
        raise ValueError(
            f"Projection row-ID section version {row_version} does not match expected version {ROW_IDS_VERSION}."
        )
    row_count = int.from_bytes(row_section[16:24], "little")
    string_bytes = int.from_bytes(row_section[24:32], "little")
    if row_count != rows:
        raise ValueError(
            f"Projection row-ID section has {row_count} rows but score matrix has {rows} rows."
        )

    offsets_start = ROW_IDS_HEADER_LEN
    offsets_end = offsets_start + (row_count + 1) * 8
    if len(row_section) != offsets_end + string_bytes:
        raise ValueError("Projection row-ID section length does not match its header.")

    offsets = [
        int.from_bytes(row_section[offsets_start + idx * 8 : offsets_start + (idx + 1) * 8], "little")
        for idx in range(row_count + 1)
    ]
    if offsets[0] != 0 or offsets[-1] != string_bytes:
        raise ValueError("Projection row-ID string offsets are inconsistent with the payload.")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("Projection row-ID offsets are not monotone.")

    string_table = row_section[offsets_end:]
    row_ids = [
        string_table[offsets[idx] : offsets[idx + 1]].decode("utf-8")
        for idx in range(row_count)
    ]
    return scores, row_ids


def _downsample(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    if factor <= 1 or df.empty:
        return df

    groups: list[pd.DataFrame] = []
    for _, group in df.groupby(["Superpopulation", "Subpopulation"], sort=False, dropna=False):
        size = len(group)
        target = max(1, math.ceil(size / factor))
        if target >= size:
            groups.append(group)
            continue
        indices = np.linspace(0, size - 1, num=target, dtype=int)
        groups.append(group.iloc[indices])
    return pd.concat(groups, axis=0).sort_index().reset_index(drop=True)


def _lighten(color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color))
    white = np.ones(3)
    mixed = rgb * (1.0 - amount) + white * amount
    return tuple(np.clip(mixed, 0.0, 1.0))


def _superpop_fill_color(superpop: str) -> tuple[float, float, float]:
    base = SUPERPOP_FILL_COLORS.get(superpop, SUPERPOP_FILL_COLORS["OTH"])
    return _lighten(base, 0.18)


def _subpop_edge_colors(subpops: list[str]) -> dict[str, tuple[float, float, float]]:
    if not subpops:
        return {}
    color_map: dict[str, tuple[float, float, float]] = {}
    total = max(1, len(subpops))
    for idx, subpop in enumerate(sorted(subpops)):
        hue = idx / total
        color_map[subpop] = colorsys.hls_to_rgb(hue, 0.38, 0.85)
    return color_map


def _legend_columns(size: int) -> int:
    if size > 36:
        return 3
    if size > 18:
        return 2
    return 1


def _ordered_superpops(values: list[str]) -> list[str]:
    remaining = sorted(value for value in values if value not in SUPERPOP_ORDER)
    return [value for value in SUPERPOP_ORDER if value in values] + remaining


def _build_dataframe(scores: np.ndarray, row_ids: list[str]) -> pd.DataFrame:
    cols = [f"PC{idx + 1}" for idx in range(scores.shape[1])]
    df = pd.DataFrame(scores, columns=cols)
    df.insert(0, "IID", row_ids)
    return df


def _plot_projection(
    annotated: pd.DataFrame,
    x: str,
    y: str,
    model_name: str,
    output_path: Path,
) -> None:
    superpops = _ordered_superpops(
        annotated["Superpopulation"].dropna().astype(str).unique().tolist()
    )
    subpop_pairs = (
        annotated.loc[:, ["Superpopulation", "Subpopulation"]]
        .dropna()
        .drop_duplicates()
        .assign(
            super_rank=lambda df: df["Superpopulation"].map(
                lambda value: SUPERPOP_ORDER.index(value)
                if value in SUPERPOP_ORDER
                else len(SUPERPOP_ORDER)
            )
        )
        .sort_values(["super_rank", "Superpopulation", "Subpopulation"], kind="stable")
    )
    subpops = subpop_pairs["Subpopulation"].astype(str).tolist()
    subpop_edges = _subpop_edge_colors(subpops)
    annotated_count = len(annotated)
    matched_count = int(
        (
            (annotated["Superpopulation"] != "OTH")
            | (annotated["Subpopulation"] != "UNK")
        ).sum()
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    fig.patch.set_facecolor("#f4f1ea")
    ax.set_facecolor("#fcfbf8")

    for (superpop, subpop), group in annotated.groupby(
        ["Superpopulation", "Subpopulation"], sort=True, dropna=False
    ):
        fill = _superpop_fill_color(str(superpop))
        edge = subpop_edges.get(str(subpop), (0.2, 0.2, 0.2))
        ax.scatter(
            group[x],
            group[y],
            s=22,
            c=[fill],
            edgecolors=[edge],
            linewidths=0.9,
            alpha=0.88,
            rasterized=annotated_count > 5000,
        )

    ax.axhline(0.0, color="#4a4a4a", linewidth=0.7, alpha=0.35)
    ax.axvline(0.0, color="#4a4a4a", linewidth=0.7, alpha=0.35)
    ax.set_xlabel(f"{x} score", fontsize=12)
    ax.set_ylabel(f"{y} score", fontsize=12)
    ax.set_title(
        f"{model_name}: {x} vs {y}",
        fontsize=15,
        loc="left",
        pad=12,
    )
    ax.text(
        0.0,
        1.01,
        f"{annotated_count:,} projected samples, {matched_count:,} with population labels",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#d8d8d8", linewidth=0.6, alpha=0.45)

    super_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=_superpop_fill_color(superpop),
            markeredgecolor="#2f2f2f",
            markeredgewidth=0.8,
            label=superpop,
        )
        for superpop in superpops
    ]
    super_legend = ax.legend(
        handles=super_handles,
        title="Superpopulation (fill)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=9,
        title_fontsize=10,
    )
    ax.add_artist(super_legend)

    sub_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor="#ffffff",
            markeredgecolor=subpop_edges[subpop],
            markeredgewidth=1.3,
            label=f"{subpop} ({superpop})",
        )
        for superpop, subpop in zip(
            subpop_pairs["Superpopulation"].astype(str).tolist(),
            subpops,
        )
    ]
    ax.legend(
        handles=sub_handles,
        title="Subpopulation (outline)",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.62),
        frameon=True,
        fontsize=8,
        title_fontsize=10,
        ncol=_legend_columns(len(sub_handles)),
    )

    fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    scores_bin = Path(args.scores_bin)
    metadata_json = Path(args.metadata_json)
    output_png = Path(args.output_png)
    output_tsv = Path(args.output_tsv) if args.output_tsv else None

    _require_file(scores_bin, "Projection scores binary")
    _require_file(metadata_json, "Projection metadata JSON")
    metadata = _read_metadata(metadata_json)
    rows = int(metadata["rows"])
    cols = int(metadata["cols"])
    scores, row_ids = _read_projection_binary(scores_bin, rows, cols)

    plot_df = _build_dataframe(scores, row_ids)
    if args.pc_x not in plot_df.columns or args.pc_y not in plot_df.columns:
        raise ValueError(f"Requested axes {args.pc_x}/{args.pc_y} do not exist in the score matrix.")

    if args.mapping_tsv:
        mapping_path = Path(args.mapping_tsv)
    else:
        mapping_path = output_png.parent / "sample_population_mapping.tsv"
        _download_if_missing(mapping_path)

    mapping = _load_population_mapping(mapping_path)
    annotated = plot_df.merge(mapping, how="left", left_on="IID", right_on="SampleID")
    annotated = annotated.drop(columns=["SampleID"])
    unmatched = annotated["Subpopulation"].isna() & annotated["Superpopulation"].isna()
    if unmatched.any():
        sample_preview = ", ".join(
            annotated.loc[unmatched, "IID"].dropna().astype(str).head(10).tolist()
        )
        print(
            "Warning: {} projected samples were not found in the population mapping.{}".format(
                int(unmatched.sum()),
                f" Examples: {sample_preview}" if sample_preview else "",
            ),
            flush=True,
        )
    annotated["Subpopulation"] = annotated["Subpopulation"].fillna("UNK")
    annotated["Superpopulation"] = annotated["Superpopulation"].fillna("OTH")
    plot_annotated = _downsample(annotated, max(1, args.downsample_factor))

    if output_tsv is not None:
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        ordered_columns = ["IID", "Subpopulation", "Superpopulation"] + [
            f"PC{idx + 1}" for idx in range(cols)
        ]
        annotated.loc[:, ordered_columns].to_csv(output_tsv, sep="\t", index=False)

    if len(plot_annotated) != len(annotated):
        print(
            "Downsampled plot input from {} to {} rows using factor {}.".format(
                len(annotated),
                len(plot_annotated),
                max(1, args.downsample_factor),
            ),
            flush=True,
        )

    _plot_projection(plot_annotated, args.pc_x, args.pc_y, args.model_name, output_png)


if __name__ == "__main__":
    main()
