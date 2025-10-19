#!/usr/bin/env python3
"""Compare release-fit projection TSVs using cross-validation classifiers."""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class DatasetSummary:
    name: str
    num_samples: int
    num_features: int
    class_counts: Counter

    def describe(self) -> str:
        header = f"Dataset '{self.name}': {self.num_samples} samples, {self.num_features} features"
        parts = [header, "  Class distribution:"]
        for label, count in self.class_counts.most_common():
            parts.append(f"    - {label}: {count}")
        return "\n".join(parts)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PCA projection TSV files using stratified cross-validated classifiers "
            "and paired statistical testing across any number of projection sources."
        )
    )
    parser.add_argument(
        "--projection",
        action="append",
        metavar="LABEL=PATH",
        help=(
            "Projection TSV to include in the comparison, provided as LABEL=PATH. "
            "Supply at least two instances to compare multiple datasets."
        ),
        required=True,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=50,
        help="Requested number of stratified CV folds (defaults to 50).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when shuffling splits.",
    )
    args = parser.parse_args(argv)

    parsed: list[tuple[str, Path]] = []
    for raw in args.projection:
        if "=" not in raw:
            parser.error(
                f"Invalid --projection value '{raw}'. Expected format LABEL=PATH."
            )
        label, path_str = raw.split("=", 1)
        label = label.strip()
        if not label:
            parser.error("Projection label must be non-empty.")
        path = Path(path_str.strip())
        parsed.append((label, path))

    if len(parsed) < 2:
        parser.error("Provide at least two --projection arguments to perform a comparison.")

    seen_labels: set[str] = set()
    for label, _ in parsed:
        if label in seen_labels:
            parser.error(f"Duplicate projection label specified: '{label}'")
        seen_labels.add(label)

    args.projections = parsed
    return args


def _read_projection(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Projection TSV for {label} not found: {path}")

    print(f"Loading projection data for {label} from {path}…", flush=True)
    df = pd.read_csv(path, sep="\t")

    expected_cols = {"SampleName", "Subpopulation"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {label} projection TSV: {sorted(missing)}"
        )

    feature_columns = [col for col in df.columns if col.startswith("PC")]
    if not feature_columns:
        raise ValueError(f"No principal component columns (PC*) found in {label} TSV")

    df = df[["SampleName", "Subpopulation", *feature_columns]].copy()
    return df


def _normalize_subpop(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    disallowed = ("nan", "unk", "unknown", "other", "others", "na", "none")
    if any(token in lowered for token in disallowed):
        return None

    return text


def _filter_projection(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["Subpopulation"] = df["Subpopulation"].map(_normalize_subpop)
    before = len(df)
    df = df.dropna(subset=["Subpopulation"])
    removed = before - len(df)
    if removed:
        print(
            f"Removed {removed} rows with undefined or disallowed subpopulation labels from {label}.",
            flush=True,
        )

    duplicates = df.duplicated(subset="SampleName")
    if duplicates.any():
        dup_count = duplicates.sum()
        print(
            f"Warning: detected {dup_count} duplicated SampleName entries in {label}; keeping first occurrence.",
            flush=True,
        )
        df = df[~duplicates]

    df = df.set_index("SampleName", drop=False)
    return df


def _align_datasets(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    sample_sets = [set(df.index) for df in datasets.values()]
    common_samples = sorted(set.intersection(*sample_sets)) if sample_sets else []
    if not common_samples:
        raise ValueError("No overlapping samples found across the provided projections")

    aligned: dict[str, pd.DataFrame] = {}
    for label, df in datasets.items():
        missing = len(df) - len(common_samples)
        if missing:
            print(
                f"Omitted {missing} samples unique to {label} data to align datasets.",
                flush=True,
            )
        aligned[label] = df.loc[common_samples].copy()

    ref_label = next(iter(aligned))
    ref_subpops = aligned[ref_label]["Subpopulation"]
    for label, df in aligned.items():
        mismatched = df["Subpopulation"] != ref_subpops
        if mismatched.any():
            mismatching_samples = mismatched[mismatched].index.tolist()
            sample_preview = ", ".join(mismatching_samples[:10])
            raise ValueError(
                "Subpopulation labels disagree between projections '{}' and '{}' for samples: {}".format(
                    ref_label, label, sample_preview
                )
            )

    return aligned


def _determine_feature_columns(datasets: dict[str, pd.DataFrame]) -> list[str]:
    feature_sets = {
        label: {col for col in df.columns if col.startswith("PC")}
        for label, df in datasets.items()
    }
    common = set.intersection(*feature_sets.values()) if feature_sets else set()
    if not common:
        raise ValueError("No shared principal component columns across all datasets")

    def sort_key(name: str) -> tuple[int, str]:
        suffix = name[2:]
        if suffix.isdigit():
            return (0, int(suffix))
        return (1, name)

    selected = sorted(common, key=sort_key)

    for label, features in feature_sets.items():
        removed = sorted(features - set(selected), key=sort_key)
        if removed:
            print(
                f"Projection '{label}' lacks {len(removed)} PCs present in others;"
                f" they will be excluded: {', '.join(removed[:10])}",
                flush=True,
            )

    print(
        f"Using {len(selected)} shared principal components: {', '.join(selected[:20])}"
        + (" …" if len(selected) > 20 else ""),
        flush=True,
    )

    return selected


def _filter_small_classes(df: pd.DataFrame, min_size: int) -> pd.DataFrame:
    counts = df["Subpopulation"].value_counts()
    allowed = counts[counts >= min_size].index
    removed_labels = sorted(set(counts.index) - set(allowed))
    if removed_labels:
        print(
            "Removing subpopulations with insufficient support (< {} samples): {}".format(
                min_size, ", ".join(removed_labels)
            ),
            flush=True,
        )
    return df[df["Subpopulation"].isin(allowed)].copy()


def _summarize_dataset(df: pd.DataFrame, name: str) -> DatasetSummary:
    feature_columns = [col for col in df.columns if col.startswith("PC")]
    counts = Counter(df["Subpopulation"].tolist())
    return DatasetSummary(
        name=name,
        num_samples=len(df),
        num_features=len(feature_columns),
        class_counts=counts,
    )


def _build_models() -> list[tuple[str, Callable[[], BaseEstimator]]]:
    return [
        (
            "LogisticRegression",
            lambda: LogisticRegression(
                penalty="none",
                solver="lbfgs",
                max_iter=1000,
                multi_class="auto",
            ),
        ),
        ("KNeighborsClassifier", lambda: KNeighborsClassifier(n_neighbors=10)),
        (
            "RandomForestClassifier",
            lambda: RandomForestClassifier(random_state=42),
        ),
    ]


def _evaluate_models(
    models: list[tuple[str, Callable[[], BaseEstimator]]],
    datasets: dict[str, pd.DataFrame],
    feature_columns: list[str],
    folds: int,
    random_state: int,
) -> list[dict[str, object]]:
    labels = list(datasets.keys())
    y = next(iter(datasets.values()))["Subpopulation"].to_numpy()

    matrices = {
        label: df[feature_columns].to_numpy(dtype=float)
        for label, df in datasets.items()
    }

    min_class_count = next(iter(datasets.values()))["Subpopulation"].value_counts().min()
    if min_class_count < 2:
        raise ValueError("Insufficient samples per class after filtering (need >= 2)")

    n_splits = min(folds, min_class_count)
    if n_splits < folds:
        print(
            f"Requested {folds} folds but the smallest class has only {min_class_count} samples;"
            f" using {n_splits} folds instead.",
            flush=True,
        )
    if n_splits < 2:
        raise ValueError("Unable to perform stratified CV with fewer than 2 folds")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(splitter.split(next(iter(matrices.values())), y))

    results: list[dict[str, object]] = []
    for model_name, factory in models:
        print(f"\nEvaluating {model_name} with {n_splits}-fold stratified CV…", flush=True)
        score_tracker: dict[str, list[float]] = {label: [] for label in labels}

        for fold_index, (train_idx, test_idx) in enumerate(splits, start=1):
            print(f"  Fold {fold_index:02d} results:", flush=True)
            for label in labels:
                model = factory()
                X = matrices[label]
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[test_idx])
                acc = accuracy_score(y[test_idx], preds)
                score_tracker[label].append(acc)
                print(
                    f"    {label:<20} accuracy = {acc:.4f}",
                    flush=True,
                )

        dataset_summaries: dict[str, dict[str, object]] = {}
        for label, scores in score_tracker.items():
            arr = np.array(scores)
            dataset_summaries[label] = {
                "scores": arr,
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            }
            print(
                "  Summary for {}: mean = {:.4f}, std = {:.4f}".format(
                    label, dataset_summaries[label]["mean"], dataset_summaries[label]["std"]
                ),
                flush=True,
            )

        comparisons: list[dict[str, object]] = []
        for a_label, b_label in itertools.combinations(labels, 2):
            scores_a = dataset_summaries[a_label]["scores"]
            scores_b = dataset_summaries[b_label]["scores"]
            diff = scores_b - scores_a
            t_stat, p_value = stats.ttest_rel(scores_b, scores_a)
            mean_diff = float(diff.mean())
            comparisons.append(
                {
                    "a": a_label,
                    "b": b_label,
                    "mean_diff": mean_diff,
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                }
            )
            print(
                "  Paired t-test ({} - {}): diff = {:.4f}, t = {:.4f}, p = {:.6f}".format(
                    b_label, a_label, mean_diff, t_stat, p_value
                ),
                flush=True,
            )

        results.append(
            {
                "model": model_name,
                "folds": n_splits,
                "datasets": dataset_summaries,
                "comparisons": comparisons,
            }
        )

    return results


def _print_summary(results: list[dict[str, object]]) -> None:
    print("\n===== Final Summary =====", flush=True)
    for entry in results:
        print(
            f"Model: {entry['model']} (folds={entry['folds']})",
            flush=True,
        )
        print("  Dataset accuracies:", flush=True)
        for label, stats_dict in entry["datasets"].items():
            print(
                f"    {label:<20} mean = {stats_dict['mean']:.4f}, std = {stats_dict['std']:.4f}",
                flush=True,
            )
        print("  Pairwise comparisons (B - A):", flush=True)
        for comp in entry["comparisons"]:
            print(
                f"    {comp['b']} - {comp['a']:<16} diff = {comp['mean_diff']:.4f},"
                f" t = {comp['t_stat']:.4f}, p = {comp['p_value']:.6f}",
                flush=True,
            )
        print("", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    datasets: dict[str, pd.DataFrame] = {}
    for label, path in args.projections:
        datasets[label] = _filter_projection(_read_projection(path, label), label)

    aligned = _align_datasets(datasets)
    feature_columns = _determine_feature_columns(aligned)

    reference_label = next(iter(aligned))
    filtered_reference = _filter_small_classes(aligned[reference_label], min_size=2)
    retained_samples = filtered_reference.index.tolist()

    trimmed_datasets: dict[str, pd.DataFrame] = {}
    for label, df in aligned.items():
        trimmed = df.loc[retained_samples, ["SampleName", "Subpopulation", *feature_columns]].copy()
        trimmed_datasets[label] = trimmed

    if not trimmed_datasets or next(iter(trimmed_datasets.values())).empty:
        raise ValueError("No samples remain after filtering invalid labels and small classes")

    for label, df in trimmed_datasets.items():
        print(_summarize_dataset(df, label).describe(), flush=True)

    models = _build_models()
    results = _evaluate_models(models, trimmed_datasets, feature_columns, args.folds, args.random_state)
    _print_summary(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
