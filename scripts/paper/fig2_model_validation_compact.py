"""Fig 2 — Compact model validation summary for the main paper.

This is the main-paper counterpart to the full leaderboard. It restricts the
display to the curated 9 models with full coverage across H0, unconditional,
and conditional artifacts, while preserving rank context from the full model
sets in each panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from scripts.paper._data import (
    RUN_NONH0,
    load_run,
    mean_brier_by_model,
    mean_normalized_crps_by_model,
    rank_models,
    score_binary,
    score_continuous_with_opus,
)
from scripts.paper._style import (
    CURATED_LINE_COLORS,
    CURATED_MODELS,
    apply_style,
    display_name,
    save_figure,
)


def _rank_lookup(score_by_model: dict[str, float]) -> tuple[dict[str, int], int]:
    ranked = rank_models(score_by_model)
    return {model: rank for rank, model, _score in ranked}, len(ranked)


def _draw_panel(ax, labels, values, colors, title, ylabel, ranks):
    xs = list(range(len(labels)))
    bars = ax.bar(xs, values, color=colors, edgecolor="white", linewidth=0.5, width=0.74)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ymax = max(values) * 1.18
    ax.set_ylim(0, ymax)
    for bar, value, rank in zip(bars, values, ranks):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.018,
            f"#{rank}",
            va="bottom",
            ha="center",
            fontsize=8.5,
            color="#475569",
        )


def main() -> None:
    apply_style()

    main_run = load_run(RUN_NONH0)
    binary_scores = mean_brier_by_model(score_binary(main_run))
    continuous_scores = mean_normalized_crps_by_model(score_continuous_with_opus())
    binary_rank, binary_n = _rank_lookup(binary_scores)
    continuous_rank, continuous_n = _rank_lookup(continuous_scores)

    rows = []
    for model in CURATED_MODELS:
        rows.append({
            "model": model,
            "label": display_name(model),
            "binary": binary_scores[model],
            "continuous": continuous_scores[model],
            "binary_rank": binary_rank[model],
            "continuous_rank": continuous_rank[model],
        })
    # Matplotlib draws larger y values higher on the page. Sort worst-to-best
    # so the best binary model appears at the top without fighting shared
    # y-axis inversion across panels.
    rows.sort(key=lambda row: row["binary"], reverse=True)

    labels = [row["label"] for row in rows]
    colors = [CURATED_LINE_COLORS[row["model"]] for row in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.35, 5.85), sharex=True)
    _draw_panel(
        ax1,
        labels,
        [row["binary"] for row in rows],
        colors,
        f"Binary events ({binary_n} models)",
        "Mean Brier",
        [row["binary_rank"] for row in rows],
    )
    _draw_panel(
        ax2,
        labels,
        [row["continuous"] for row in rows],
        colors,
        f"Continuous quantities ({continuous_n} models)",
        "Mean normalized CRPS",
        [row["continuous_rank"] for row in rows],
    )

    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=41, ha="right", rotation_mode="anchor")
    ax2.tick_params(axis="x", labelsize=7.4)

    fig.tight_layout(h_pad=0.85)
    out = save_figure(fig, "fig2_model_validation_compact")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
