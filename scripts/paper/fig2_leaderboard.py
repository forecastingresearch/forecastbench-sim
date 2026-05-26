"""Fig 2 — Headline model leaderboard across the full benchmark.

Two stacked horizontal-bar panels, sorted independently within each panel.

  - Top:    Binary Brier per model (lower is better). 30-model main run.
  - Bottom: Continuous normalized CRPS per model (lower is better).
            31 models = 30 main + Claude Opus 4.5 spliced from the
            archived continuous unconditional run on the same questions.

Curated 9 are colored per their lab/per-model line color and bolded.
Non-curated models are drawn in light grey. Outlier models on the
continuous panel (>= 1.0 normalized CRPS) are shown with a clipped axis
and an annotation listing their off-chart values.
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


CONT_CLIP = 1.0  # x-axis clip for the continuous panel.


def _draw_panel(ax, ranked, title, xlabel, clip=None):
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    annot_off: list[tuple[int, str, float]] = []  # (y_index, label, value)

    for i, (rank, model_id, score) in enumerate(ranked):
        labels.append(f"{display_name(model_id)}  ({rank}/{len(ranked)})")
        plotted = score
        if clip is not None and score > clip:
            plotted = clip
            annot_off.append((i, display_name(model_id), score))
        values.append(plotted)
        if model_id in CURATED_MODELS:
            colors.append(CURATED_LINE_COLORS[model_id])
        else:
            colors.append("#CBD5E1")  # light slate grey

    ys = list(range(len(labels)))
    ax.barh(ys, values, color=colors, edgecolor="white", linewidth=0.4, height=0.78)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # best at top
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")

    # Bold the curated labels.
    for tick, (rank, model_id, _score) in zip(ax.get_yticklabels(), ranked):
        if model_id in CURATED_MODELS:
            tick.set_fontweight("bold")

    if clip is not None:
        ax.set_xlim(0, clip)
        if annot_off:
            # Mark each off-chart bar at the right edge so readers see why it
            # ran out of axis, plus a single legend-line annotation up top.
            for y_idx, name, v in annot_off:
                ax.text(
                    clip * 0.998, y_idx,
                    f"  {v:.2f} >>", va="center", ha="right",
                    fontsize=6.5, color="#1F2937", fontweight="bold",
                )
            txt = "off-chart: " + "; ".join(f"{n} = {v:.2f}" for _, n, v in annot_off)
            ax.text(
                clip * 0.5, -0.6, txt,
                ha="center", va="bottom", fontsize=7, style="italic",
                color="#475569",
            )


def main() -> None:
    apply_style()

    main_run = load_run(RUN_NONH0)
    bin_scores = mean_brier_by_model(score_binary(main_run))
    cont_scores = mean_normalized_crps_by_model(score_continuous_with_opus())

    ranked_bin = rank_models(bin_scores)
    ranked_cont = rank_models(cont_scores)

    # Height proportional to total bars (30 + 31 = 61) so labels don't crowd.
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 11.0),
        gridspec_kw={"height_ratios": [len(ranked_bin), len(ranked_cont)]},
    )

    _draw_panel(
        ax_top, ranked_bin,
        title=f"Binary forecasting — Brier score (n={len(ranked_bin)} models)",
        xlabel="Mean Brier (lower is better)",
    )
    _draw_panel(
        ax_bot, ranked_cont,
        title=f"Continuous forecasting — normalized CRPS (n={len(ranked_cont)} models)",
        xlabel="Mean normalized CRPS (lower is better)",
        clip=CONT_CLIP,
    )

    fig.suptitle(
        "Model performance across CivBench (curated 9 highlighted)",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = save_figure(fig, "fig2_leaderboard")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
