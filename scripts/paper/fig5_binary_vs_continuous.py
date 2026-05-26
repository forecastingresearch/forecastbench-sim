"""Fig 5 — Binary vs continuous performance scatter.

One dot per model, x = binary Brier rank (out of 30), y = continuous
normalized-CRPS rank (out of 31). Top-left = strong on both. Off-diagonal
dots = models that are good at one task and bad at the other.

The headline finding this surfaces: GPT-5.1 is rank 1/30 on binary Brier
but rank 23/31 on continuous CRPS. Calibrated quantile forecasting and
calibrated binary classification are demonstrably different skills on
this benchmark.

Inputs:
  data/evaluations/runs/final_2026-03-07_newq_uncond_nonh0_bin_plus_cont_all30.json
  archive/20260302_121948/evaluations/runs/cont_uncond_all.anthropic_retry.json
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


def main() -> None:
    apply_style()

    bin_means = mean_brier_by_model(score_binary(load_run(RUN_NONH0)))
    cont_means = mean_normalized_crps_by_model(score_continuous_with_opus())

    bin_rank = {m: r for r, m, _s in rank_models(bin_means)}
    cont_rank = {m: r for r, m, _s in rank_models(cont_means)}
    n_bin = len(bin_means)
    n_cont = len(cont_means)

    shared = sorted(set(bin_rank) & set(cont_rank))

    fig, ax = plt.subplots(figsize=(6.6, 6.4))

    # Diagonal reference: rank-equivalent (after normalizing the two axes
    # to a 0..1 scale because n_bin != n_cont).
    ax.plot(
        [0, 1], [0, 1],
        transform=ax.transAxes,
        color="#94A3B8", linestyle="--", linewidth=0.8, label="rank-equivalent",
    )

    # Scatter — non-curated grey, curated colored, all labelled when curated.
    for m in shared:
        x = bin_rank[m]
        y = cont_rank[m]
        if m in CURATED_MODELS:
            ax.scatter(
                x, y, s=110, color=CURATED_LINE_COLORS[m],
                edgecolor="white", linewidth=0.8, zorder=3,
            )
            # Place label slightly off the marker.
            ax.annotate(
                display_name(m), (x, y),
                xytext=(6, 4), textcoords="offset points",
                fontsize=8, fontweight="bold",
                color=CURATED_LINE_COLORS[m],
            )
        else:
            ax.scatter(
                x, y, s=24, color="#CBD5E1",
                edgecolor="white", linewidth=0.4, zorder=2,
            )

    ax.set_xlabel(f"Binary Brier rank  (1 = best of {n_bin})")
    ax.set_ylabel(f"Continuous normalized-CRPS rank  (1 = best of {n_cont})")
    ax.set_xlim(0, n_bin + 1)
    ax.set_ylim(0, n_cont + 1)
    ax.invert_yaxis()
    ax.invert_xaxis()
    # Now top-right corner = best on both. Put quadrant guides at midpoints.
    ax.axvline(n_bin / 2, color="#E5E7EB", linewidth=0.8, zorder=1)
    ax.axhline(n_cont / 2, color="#E5E7EB", linewidth=0.8, zorder=1)

    ax.set_title(
        "Binary skill vs continuous skill (one dot per model)",
        loc="left", fontweight="bold",
    )
    ax.text(
        0.02, 0.02,
        "top-right = best on both;\noff-diagonal = task-specialized",
        transform=ax.transAxes, fontsize=7.5, color="#475569", style="italic",
        ha="left", va="bottom",
    )

    fig.tight_layout()
    out = save_figure(fig, "fig5_binary_vs_continuous")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
