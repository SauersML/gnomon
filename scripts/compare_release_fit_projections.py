#!/usr/bin/env python3
"""Compare release-fit projection TSVs using cross-validation classifiers."""

from __future__ import annotations

import argparse
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
            "Compare PCA projection TSV files (LD vs. no LD) using 50-fold cross-validated"
            " classifiers and paired statistical testing."
        )
    )
    parser.add_argument(
        "--no-ld",
        required=True,
        type=Path,
        help="Path to the TSV exported from the no-LD release-fit run.",
    )
    parser.add_argument(
        "--ld",
        required=True,
        type=Path,
        help="Path to the TSV exported from the LD release-fit run.",
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
    return parser.parse_args(argv)


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


def _align_datasets(no_ld: pd.DataFrame, ld: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_samples = sorted(set(no_ld.index) & set(ld.index))
    if not common_samples:
        raise ValueError("No overlapping samples between the LD and no-LD projections")

    missing_no_ld = len(no_ld) - len(common_samples)
    missing_ld = len(ld) - len(common_samples)
    if missing_no_ld:
        print(
            f"Omitted {missing_no_ld} samples unique to no-LD data to align datasets.",
            flush=True,
        )
    if missing_ld:
        print(
            f"Omitted {missing_ld} samples unique to LD data to align datasets.",
            flush=True,
        )

    aligned_no_ld = no_ld.loc[common_samples].copy()
    aligned_ld = ld.loc[common_samples].copy()

    if not (aligned_no_ld["Subpopulation"] == aligned_ld["Subpopulation"]).all():
        mismatched = (
            aligned_no_ld["Subpopulation"] != aligned_ld["Subpopulation"]
        )
        mismatching_samples = mismatched[mismatched].index.tolist()
        raise ValueError(
            "Subpopulation labels disagree between LD and no-LD data for samples: "
            + ", ".join(mismatching_samples[:10])
        )

    no_ld_features = [col for col in aligned_no_ld.columns if col.startswith("PC")]
    ld_features = [col for col in aligned_ld.columns if col.startswith("PC")]
    if no_ld_features != ld_features:
        raise ValueError(
            "Principal component columns differ between LD and no-LD datasets"
        )

    return aligned_no_ld, aligned_ld


def _filter_small_classes(df: pd.DataFrame, min_size: int) -> pd.DataFrame:
    counts = df["Subpopulation"].value_counts()
    allowed = counts[counts >= min_size].index
    removed_labels = sorted(set(counts.index) - set(allowed))
    if removed_labels:
        print(
            "Removing subpopulations with insufficient support (" "< {} samples): {}".format(
                min_size, ", ".join(removed_labels)
            ),
            flush=True,
        )
    return df[df["Subpopulation"].isin(allowed)].copy()


def _summarize_dataset(df: pd.DataFrame, name: str) -> DatasetSummary:
    feature_columns = [col for col in df.columns if col.startswith("PC")]
    counts = Counter(df["Subpopulation"].tolist())
    return DatasetSummary(name=name, num_samples=len(df), num_features=len(feature_columns), class_counts=counts)


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
    no_ld_df: pd.DataFrame,
    ld_df: pd.DataFrame,
    folds: int,
    random_state: int,
) -> list[dict[str, object]]:
    feature_columns = [col for col in no_ld_df.columns if col.startswith("PC")]
    X_no_ld = no_ld_df[feature_columns].to_numpy(dtype=float)
    X_ld = ld_df[feature_columns].to_numpy(dtype=float)
    y = no_ld_df["Subpopulation"].to_numpy()

    min_class_count = no_ld_df["Subpopulation"].value_counts().min()
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
    splits = list(splitter.split(X_no_ld, y))

    results: list[dict[str, object]] = []
    for model_name, factory in models:
        print(f"\nEvaluating {model_name} with {n_splits}-fold stratified CV…", flush=True)
        no_ld_scores: list[float] = []
        ld_scores: list[float] = []

        for fold_index, (train_idx, test_idx) in enumerate(splits, start=1):
            model_no_ld = factory()
            model_no_ld.fit(X_no_ld[train_idx], y[train_idx])
            preds_no_ld = model_no_ld.predict(X_no_ld[test_idx])
            acc_no_ld = accuracy_score(y[test_idx], preds_no_ld)
            no_ld_scores.append(acc_no_ld)

            model_ld = factory()
            model_ld.fit(X_ld[train_idx], y[train_idx])
            preds_ld = model_ld.predict(X_ld[test_idx])
            acc_ld = accuracy_score(y[test_idx], preds_ld)
            ld_scores.append(acc_ld)

            print(
                f"  Fold {fold_index:02d}: no-LD accuracy = {acc_no_ld:.4f}, LD accuracy = {acc_ld:.4f}",
                flush=True,
            )

        no_ld_scores_arr = np.array(no_ld_scores)
        ld_scores_arr = np.array(ld_scores)
        diff = ld_scores_arr - no_ld_scores_arr
        mean_no_ld = float(no_ld_scores_arr.mean())
        mean_ld = float(ld_scores_arr.mean())
        mean_diff = float(diff.mean())
        t_stat, p_value = stats.ttest_rel(ld_scores_arr, no_ld_scores_arr)

        print(
            "  Summary: no-LD mean = {:.4f}, LD mean = {:.4f}, diff = {:.4f}".format(
                mean_no_ld, mean_ld, mean_diff
            ),
            flush=True,
        )
        print(
            "  Paired t-test (LD - no-LD): t = {:.4f}, p = {:.6f}".format(t_stat, p_value),
            flush=True,
        )

        results.append(
            {
                "model": model_name,
                "folds": n_splits,
                "no_ld_mean": mean_no_ld,
                "ld_mean": mean_ld,
                "mean_diff": mean_diff,
                "t_stat": float(t_stat),
                "p_value": float(p_value),
            }
        )

    return results


def _print_summary(results: list[dict[str, object]]) -> None:
    print("\n===== Final Summary =====", flush=True)
    header = (
        f"{'Model':<25} {'Folds':<5} {'no-LD Acc':<12} {'LD Acc':<12} "
        f"{'LD - no-LD':<12} {'t-stat':<10} {'p-value':<12}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for entry in results:
        print(
            f"{entry['model']:<25} {entry['folds']:<5d} "
            f"{entry['no_ld_mean']:<12.4f} {entry['ld_mean']:<12.4f} "
            f"{entry['mean_diff']:<12.4f} {entry['t_stat']:<10.4f} {entry['p_value']:<12.6f}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    no_ld_df = _filter_projection(_read_projection(args.no_ld, "no-LD"), "no-LD")
    ld_df = _filter_projection(_read_projection(args.ld, "LD"), "LD")
    no_ld_df, ld_df = _align_datasets(no_ld_df, ld_df)

    combined_df = no_ld_df.copy()
    combined_df = _filter_small_classes(combined_df, min_size=2)

    retained_samples = combined_df.index.tolist()
    no_ld_df = no_ld_df.loc[retained_samples]
    ld_df = ld_df.loc[retained_samples]

    if no_ld_df.empty:
        raise ValueError("No samples remain after filtering invalid labels and small classes")

    for summary in (
        _summarize_dataset(no_ld_df, "no-LD"),
        _summarize_dataset(ld_df, "LD"),
    ):
        print(summary.describe(), flush=True)

    models = _build_models()
    results = _evaluate_models(models, no_ld_df, ld_df, args.folds, args.random_state)
    _print_summary(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
