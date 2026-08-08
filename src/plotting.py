"""
Comparison plots for the Horn experiment.

Generates publication-quality dark-themed plots showing entropy trajectories,
repetition rates, temperature curves, and self-similarity across conditions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .experiment import RunResult

# Color palette — matches the Horn Logic Engine viz
PALETTE = {
    "horn_1/x": "#6366f1",
    "horn_1/√x": "#8b5cf6",
    "horn_1/log(x)": "#a78bfa",
    "const_τ=0.7": "#f59e0b",
    "const_τ=1.0": "#ef4444",
    "linear_decay": "#22d3ee",
    "rep_penalty": "#34d399",
}

BG_COLOR = "#0a0a1a"
PANEL_COLOR = "#0f0f2a"
GRID_COLOR = "#1e1e3a"
TEXT_COLOR = "#ccc"
MUTED_COLOR = "#666"


def _style_axis(ax):
    """Apply consistent dark styling to a matplotlib axis."""
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=9)
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.08, color="#fff")


def _get_color(condition: str) -> str:
    return PALETTE.get(condition, "#888888")


def generate_plots(results: list[RunResult], out_dir: Path):
    """Generate all comparison plots and save to out_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = sorted(set(r.condition for r in results))

    # ── 4-Panel Comparison ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        "Gabriel's Horn Experiment — Condition Comparison",
        color="#fff",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for ax in axes.flat:
        _style_axis(ax)

    # Panel 1: Raw entropy trajectories
    ax = axes[0, 0]
    for cond in conditions:
        cond_results = [r for r in results if r.condition == cond]
        min_len = min(len(r.raw_entropies) for r in cond_results)
        if min_len == 0:
            continue
        avg = np.mean(
            [r.raw_entropies[:min_len] for r in cond_results], axis=0
        )
        ax.plot(avg, color=_get_color(cond), alpha=0.8, label=cond, linewidth=1.5)
    ax.set_title("Raw Entropy (before adjustment)", color=TEXT_COLOR, fontsize=11)
    ax.set_xlabel("Step", color=MUTED_COLOR)
    ax.set_ylabel("H (bits)", color=MUTED_COLOR)
    ax.legend(fontsize=8, facecolor="#1a1a3a", edgecolor="#333", labelcolor=TEXT_COLOR)

    # Panel 2: Adjusted entropy + target overlays
    ax = axes[0, 1]
    for cond in conditions:
        cond_results = [r for r in results if r.condition == cond]
        min_len = min(len(r.adjusted_entropies) for r in cond_results)
        if min_len == 0:
            continue
        avg = np.mean(
            [r.adjusted_entropies[:min_len] for r in cond_results], axis=0
        )
        ax.plot(avg, color=_get_color(cond), alpha=0.8, label=cond, linewidth=1.5)

    # Overlay target curves for horn conditions
    for cond in ["horn_1/x", "horn_1/√x", "horn_1/log(x)"]:
        cond_results = [r for r in results if r.condition == cond]
        if not cond_results:
            continue
        min_len = min(len(r.target_entropies) for r in cond_results)
        targets = []
        for r in cond_results:
            t = r.target_entropies[:min_len]
            t = [v if v is not None and not np.isnan(v) else 0 for v in t]
            targets.append(t)
        avg_target = np.mean(targets, axis=0)
        ax.plot(
            avg_target,
            color=_get_color(cond),
            alpha=0.3,
            linestyle="--",
            linewidth=1,
        )

    ax.set_title("Adjusted Entropy (after constraint)", color=TEXT_COLOR, fontsize=11)
    ax.set_xlabel("Step", color=MUTED_COLOR)
    ax.set_ylabel("H (bits)", color=MUTED_COLOR)
    ax.legend(fontsize=8, facecolor="#1a1a3a", edgecolor="#333", labelcolor=TEXT_COLOR)

    # Panel 3: Repetition rates (bar chart)
    ax = axes[1, 0]
    x_pos = np.arange(len(conditions))
    rep4_means = []
    rep8_means = []
    for cond in conditions:
        cond_results = [r for r in results if r.condition == cond]
        rep4_means.append(np.mean([r.rep_rate_4gram for r in cond_results]))
        rep8_means.append(np.mean([r.rep_rate_8gram for r in cond_results]))

    w = 0.35
    ax.bar(
        x_pos - w / 2,
        rep4_means,
        w,
        label="4-gram",
        color=[_get_color(c) for c in conditions],
        alpha=0.7,
    )
    ax.bar(
        x_pos + w / 2,
        rep8_means,
        w,
        label="8-gram",
        color=[_get_color(c) for c in conditions],
        alpha=0.4,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=8, color=MUTED_COLOR)
    ax.set_title("N-gram Repetition Rate (lower = better)", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Repetition rate", color=MUTED_COLOR)
    ax.legend(fontsize=8, facecolor="#1a1a3a", edgecolor="#333", labelcolor=TEXT_COLOR)

    # Panel 4: Temperature trajectories
    ax = axes[1, 1]
    for cond in conditions:
        cond_results = [r for r in results if r.condition == cond]
        min_len = min(len(r.temperatures) for r in cond_results)
        if min_len == 0:
            continue
        avg = np.mean(
            [r.temperatures[:min_len] for r in cond_results], axis=0
        )
        ax.plot(avg, color=_get_color(cond), alpha=0.8, label=cond, linewidth=1.5)
    ax.set_title("Temperature τ over time", color=TEXT_COLOR, fontsize=11)
    ax.set_xlabel("Step", color=MUTED_COLOR)
    ax.set_ylabel("τ", color=MUTED_COLOR)
    ax.legend(fontsize=8, facecolor="#1a1a3a", edgecolor="#333", labelcolor=TEXT_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = out_dir / "horn_comparison.png"
    fig.savefig(plot_path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {plot_path}")

    # ── Self-Similarity Plot ──
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    fig2.patch.set_facecolor(BG_COLOR)
    _style_axis(ax2)

    max_windows = 50
    for cond in conditions:
        cond_results = [r for r in results if r.condition == cond]
        padded = [
            r.self_similarity[:max_windows]
            + [0] * max(0, max_windows - len(r.self_similarity))
            for r in cond_results
        ]
        avg_sim = np.mean(padded, axis=0)
        ax2.plot(
            avg_sim,
            color=_get_color(cond),
            alpha=0.8,
            label=cond,
            linewidth=1.5,
        )

    ax2.set_title(
        "Sliding Self-Similarity (lower = more novel)",
        color=TEXT_COLOR,
        fontsize=11,
    )
    ax2.set_xlabel("Window index", color=MUTED_COLOR)
    ax2.set_ylabel("Jaccard similarity", color=MUTED_COLOR)
    ax2.legend(fontsize=8, facecolor="#1a1a3a", edgecolor="#333", labelcolor=TEXT_COLOR)

    sim_path = out_dir / "horn_self_similarity.png"
    fig2.savefig(sim_path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig2)
    print(f"  -> {sim_path}")
