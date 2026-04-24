#!/usr/bin/env python3
"""Export the dashboard scalarized portability plots as PNG files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


@dataclass(frozen=True)
class Config:
    max_generations: int = 400
    plot_points: int = 240
    ne: float = 10_000
    mu: float = 0.00001
    mig: float = 0.0
    recomb: float = 0.02
    tag_distance: float = 8.0
    tag_causal_distance: float = 12.0
    source_direct_score: float = 0.18
    source_proxy_score: float = 0.16
    source_context_score: float = -0.01
    source_outcome_variance: float = 0.64
    source_tag_af: float = 0.32
    source_causal_af: float = 0.28
    source_prevalence: float = 0.09
    standing_tag_shift: float = 0.35
    mutation_tag_shift: float = 0.08
    standing_causal_shift: float = 0.28
    mutation_causal_shift: float = 0.06
    standing_effect_fst_shift: float = 0.0
    context_fst_shift: float = -0.18
    outcome_variance_fst_shift: float = 0.20
    prevalence_fst_shift: float = 0.15
    novel_direct_score: float = 0.07
    novel_proxy_score: float = 0.04
    broken_tagging_scale: float = 0.06
    ancestry_ld_scale: float = 0.05
    source_overfit: float = 0.5
    novel_untaggable_scale: float = 0.035

    @property
    def source_score_variance(self) -> float:
        return (
            self.source_direct_score
            + self.source_proxy_score
            + self.source_context_score
        )


BG = "#ffffff"
GRID = "#e8ecf3"
AXIS = "#cfd6df"
TEXT = "#17202a"
SUBTLE = "#5b6572"
TEAL = "#0f6d67"
SOFT_TEAL = "#1a8f86"
RUST = "#9c4f29"
CLAY = "#c06b42"
AMBER = "#8d6500"
BRASS = "#b88711"
SLATE = "#32454c"


def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def apply_base_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": AXIS,
            "axes.labelcolor": TEXT,
            "xtick.color": SUBTLE,
            "ytick.color": SUBTLE,
            "text.color": TEXT,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=1.0)
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(length=0, labelsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))


def annotate_last(ax: plt.Axes, x: list[float], y: list[float], label: str, color: str) -> None:
    ax.annotate(
        label,
        xy=(x[-1], y[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        color=color,
        fontsize=9.5,
        va="center",
        fontweight="bold",
    )


def style_panel_title(ax: plt.Axes, title: str, color: str) -> None:
    ax.set_title(
        title,
        loc="left",
        pad=10,
        fontsize=11.5,
        fontweight="bold",
        color=color,
    )


def add_figure_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(
        0.07,
        0.972,
        title,
        fontsize=18.5,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="top",
    )
    fig.text(
        0.07,
        0.942,
        subtitle,
        fontsize=10.4,
        color=SUBTLE,
        ha="left",
        va="top",
    )


def tighten_ylim(ax: plt.Axes, values: list[float], center: float | None = None) -> None:
    data_min = min(values)
    data_max = max(values)
    if center is not None:
        data_min = min(data_min, center)
        data_max = max(data_max, center)
    span = data_max - data_min
    pad = 0.0008 if span == 0 else span * 0.22
    ax.set_ylim(data_min - pad, data_max + pad)


def add_endpoint(ax: plt.Axes, x: list[float], y: list[float], color: str) -> None:
    ax.scatter([x[-1]], [y[-1]], s=24, color=color, zorder=5)


def pad_x_limits(ax: plt.Axes, x: list[float]) -> None:
    left = x[0]
    right = x[-1]
    span = right - left
    pad = 8 if span == 0 else span * 0.05
    ax.set_xlim(left - pad, right + pad)


def sample_generations(max_generations: int, count: int) -> list[int]:
    generations: list[int] = []
    step = max_generations / max(1, count - 1)
    seen: set[int] = set()
    for index in range(count):
        raw = max_generations if index == count - 1 else round(index * step)
        if raw not in seen:
            generations.append(raw)
            seen.add(raw)
    if 0 not in seen:
        generations.insert(0, 0)
    if max_generations not in seen:
        generations.append(max_generations)
    return sorted(generations)


def state_at(cfg: Config, t: int) -> dict[str, float]:
    theta = 4 * cfg.ne * cfg.mu
    big_m = 4 * cfg.ne * cfg.mig
    tau = t / (2 * cfg.ne)
    het_decay = (1 - 1 / (2 * cfg.ne)) * (1 - theta / (2 * cfg.ne))
    fst = (1 / (1 + theta + big_m)) * (1 - het_decay**t)
    mutation_shared_retention = math.exp(-theta * tau)
    migration_shared_boost = 1 + (big_m * tau) / (1 + big_m)
    novel_innovation = 1 - mutation_shared_retention
    ld_tag_decay = math.exp(-(cfg.recomb * fst * cfg.tag_distance))
    ld_tag_causal_decay = math.exp(-(cfg.recomb * fst * cfg.tag_causal_distance))

    source_predictive_covariance = cfg.source_score_variance
    source_signal = source_predictive_covariance**2 / cfg.source_score_variance
    source_residual = cfg.source_outcome_variance - source_signal

    target_tag_af = (
        cfg.source_tag_af
        + cfg.standing_tag_shift * fst
        + cfg.mutation_tag_shift * novel_innovation
    )
    target_causal_af = (
        cfg.source_causal_af
        + cfg.standing_causal_shift * fst
        + cfg.mutation_causal_shift * novel_innovation
    )
    target_prevalence = cfg.source_prevalence + cfg.prevalence_fst_shift * fst

    tag_retention = math.exp(-abs(target_tag_af - cfg.source_tag_af))
    causal_retention = math.exp(-abs(target_causal_af - cfg.source_causal_af))
    standing_effect_multiplier = 1 + cfg.standing_effect_fst_shift * fst

    shared_direct_kernel = (
        mutation_shared_retention
        * migration_shared_boost
        * tag_retention
        * causal_retention
    )
    shared_proxy_kernel = (
        ld_tag_causal_decay
        * mutation_shared_retention
        * migration_shared_boost
        * tag_retention
        * causal_retention
    )
    novel_direct_kernel = (
        novel_innovation
        * (1 / migration_shared_boost)
        * tag_retention
        * causal_retention
    )
    novel_proxy_kernel = (
        ld_tag_causal_decay
        * novel_innovation
        * (1 / migration_shared_boost)
        * tag_retention
        * causal_retention
    )

    target_direct_channel = (
        cfg.source_direct_score * shared_direct_kernel * standing_effect_multiplier
        + cfg.novel_direct_score * novel_direct_kernel
    )
    target_proxy_channel = (
        cfg.source_proxy_score * shared_proxy_kernel * standing_effect_multiplier
        + cfg.novel_proxy_score * novel_proxy_kernel
    )
    target_context_channel = cfg.source_context_score + cfg.context_fst_shift * fst
    target_predictive_covariance = (
        target_direct_channel + target_proxy_channel + target_context_channel
    )

    target_score_variance = (
        cfg.source_score_variance
        * ld_tag_decay
        * mutation_shared_retention
        * migration_shared_boost
        * tag_retention
        * tag_retention
    )
    target_outcome_variance = (
        cfg.source_outcome_variance + cfg.outcome_variance_fst_shift * fst
    )
    broken_tagging_residual = cfg.broken_tagging_scale * (1 - shared_proxy_kernel) ** 2
    ancestry_ld_residual = cfg.ancestry_ld_scale * (1 - ld_tag_decay) ** 2
    novel_untaggable_residual = cfg.novel_untaggable_scale * novel_innovation
    source_overfit_residual = (
        cfg.source_overfit * (target_context_channel - cfg.source_context_score) ** 2
    )
    effective_target_outcome_variance = (
        target_outcome_variance
        + broken_tagging_residual
        + ancestry_ld_residual
        + source_overfit_residual
        + novel_untaggable_residual
    )

    target_signal = target_predictive_covariance**2 / target_score_variance
    target_residual = effective_target_outcome_variance - target_signal

    source_r2 = source_signal / cfg.source_outcome_variance
    source_auc = phi(math.sqrt(source_signal / (2 * source_residual)))
    source_comparable_brier = target_prevalence * (1 - target_prevalence) * (1 - source_r2)
    source_raw_brier = (
        cfg.source_prevalence * (1 - cfg.source_prevalence) * (1 - source_r2)
    )

    target_r2 = target_signal / effective_target_outcome_variance
    target_auc = phi(math.sqrt(target_signal / (2 * target_residual)))
    target_brier = target_prevalence * (1 - target_prevalence) * (1 - target_r2)

    return {
        "generation": float(t),
        "source_r2": source_r2,
        "source_auc": source_auc,
        "source_comparable_brier": source_comparable_brier,
        "source_raw_brier": source_raw_brier,
        "target_r2": target_r2,
        "target_auc": target_auc,
        "target_brier": target_brier,
        "ratio_r2": target_r2 / source_r2,
        "ratio_auc": target_auc / source_auc,
        "ratio_brier": target_brier / source_comparable_brier,
    }


def export_ratio_plot(states: list[dict[str, float]], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 9.2), dpi=220, sharex=True)
    x = [state["generation"] for state in states]
    ratio_r2 = [state["ratio_r2"] for state in states]
    ratio_auc = [state["ratio_auc"] for state in states]
    ratio_brier = [state["ratio_brier"] for state in states]
    panel_specs = [
        (axes[0], ratio_r2, TEAL, "R² Ratio"),
        (axes[1], ratio_auc, RUST, "AUC Ratio"),
        (axes[2], ratio_brier, AMBER, "Brier Ratio"),
    ]
    add_figure_header(
        fig,
        "Scalarized Target-to-Source Metric Ratios",
        "Dashboard defaults after enforcing target = source at generation 0. Brier uses the source profile on the target prevalence scale.",
    )
    for ax, values, color, label in panel_specs:
        ax.plot(x, values, color=color, linewidth=3.0)
        ax.axhline(1.0, color=SLATE, linewidth=1.2, linestyle=(0, (5, 4)), alpha=0.9)
        ax.fill_between(x, values, 1.0, color=color, alpha=0.08)
        ax.set_ylabel("T / S")
        style_axes(ax)
        style_panel_title(ax, label, color)
        tighten_ylim(ax, values, center=1.0)
        pad_x_limits(ax, x)
        add_endpoint(ax, x, values, color)
        annotate_last(ax, x, values, f"{values[-1]:.3f}", color)
    axes[-1].set_xlabel("Generations since divergence")
    axes[0].text(x[0], 1.0, " parity", color=SLATE, fontsize=9.5, va="bottom", ha="left")
    fig.subplots_adjust(top=0.865, left=0.07, right=0.94, bottom=0.09, hspace=0.31)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def export_raw_plot(states: list[dict[str, float]], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 9.6), dpi=220, sharex=True)
    x = [state["generation"] for state in states]
    target_r2 = [state["target_r2"] for state in states]
    source_r2 = [state["source_r2"] for state in states]
    target_auc = [state["target_auc"] for state in states]
    source_auc = [state["source_auc"] for state in states]
    target_brier = [state["target_brier"] for state in states]
    source_brier = [state["source_raw_brier"] for state in states]
    add_figure_header(
        fig,
        "Scalarized Source and Target Metric Curves",
        "Solid lines are target metrics. Dashed lines are source baselines. Source Brier stays on the source prevalence scale.",
    )
    panel_specs = [
        (axes[0], target_r2, source_r2, TEAL, SOFT_TEAL, "R²"),
        (axes[1], target_auc, source_auc, RUST, CLAY, "AUC"),
        (axes[2], target_brier, source_brier, AMBER, BRASS, "Brier"),
    ]
    for ax, target_values, source_values, target_color, source_color, label in panel_specs:
        ax.plot(x, target_values, color=target_color, linewidth=3.0, label=f"target {label}")
        ax.plot(
            x,
            source_values,
            color=source_color,
            linewidth=2.2,
            linestyle=(0, (7, 4)),
            label=f"source {label}",
        )
        ax.set_ylabel(label)
        style_axes(ax)
        style_panel_title(ax, label, target_color)
        tighten_ylim(ax, target_values + source_values)
        pad_x_limits(ax, x)
        add_endpoint(ax, x, target_values, target_color)
        add_endpoint(ax, x, source_values, source_color)
        annotate_last(ax, x, target_values, "target", target_color)
        annotate_last(ax, x, source_values, "source", source_color)
    axes[-1].set_xlabel("Generations since divergence")
    fig.subplots_adjust(top=0.868, left=0.07, right=0.94, bottom=0.085, hspace=0.31)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def export_source_normalized_r2_plot(
    states: list[dict[str, float]],
    output_path: Path,
    source_baseline: float = 0.09,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=220)
    x = [state["generation"] for state in states]
    source_r2 = [source_baseline for _ in states]
    target_r2 = [source_baseline * state["ratio_r2"] for state in states]

    add_figure_header(
        fig,
        "Source-Normalized Target R²",
        "Display target R² = source baseline × exact target/source R² ratio. This preserves the mechanistic portability curve without rescaling the biological state.",
    )
    style_axes(ax)
    ax.plot(x, target_r2, color=TEAL, linewidth=3.2)
    ax.plot(x, source_r2, color=SOFT_TEAL, linewidth=2.2, linestyle=(0, (7, 4)))
    style_panel_title(ax, "R²", TEAL)
    pad_x_limits(ax, x)
    ax.set_ylim(0.0, 0.1)
    ax.set_xlabel("Generations since divergence")
    ax.set_ylabel("R²")
    add_endpoint(ax, x, target_r2, TEAL)
    add_endpoint(ax, x, source_r2, SOFT_TEAL)
    annotate_last(ax, x, target_r2, f"target {target_r2[-1]:.3f}", TEAL)
    annotate_last(ax, x, source_r2, f"source {source_baseline:.3f}", SOFT_TEAL)
    fig.subplots_adjust(top=0.83, left=0.08, right=0.93, bottom=0.13)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main() -> None:
    apply_base_style()
    cfg = Config()
    generations = sample_generations(cfg.max_generations, cfg.plot_points)
    states = [state_at(cfg, generation) for generation in generations]

    dashboard_dir = Path("/Users/user/gnomon/proofs/dashboard")
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    export_ratio_plot(states, dashboard_dir / "scalarized_metric_ratios.png")
    export_raw_plot(states, dashboard_dir / "scalarized_source_target_metric_curves.png")
    export_source_normalized_r2_plot(states, dashboard_dir / "source_normalized_r2_curve.png")


if __name__ == "__main__":
    main()
