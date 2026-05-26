"""Fig 3 — Performance vs forecast horizon (H1..H7).

Two side-by-side panels, one line per curated 9 model.
  - Left:   Binary Brier vs horizon.
  - Right:  Continuous normalized CRPS vs horizon.

H0 (the comprehension floor) is excluded from the main paper. The
appendix reports H0 numbers separately and the caption notes that H0
performance was strong across the board, so the H1+ curves represent
forecasting difficulty rather than report-comprehension difficulty.

Inputs:
  data/evaluations/runs/final_2026-03-07_newq_uncond_nonh0_bin_plus_cont_all30.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from scripts.paper._data import (
    RUN_NONH0,
    load_run,
    mean_brier_by_model_horizon,
    mean_normalized_crps_by_model_horizon,
    score_binary,
    score_continuous,
)
from scripts.paper._style import (
    CURATED_LINE_COLORS,
    CURATED_MODELS,
    apply_style,
    display_name,
    save_figure,
)


HORIZONS = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]


def _draw_panel(ax, by_model_horizon, ylabel, title):
    for model in CURATED_MODELS:
        ys = []
        xs = []
        for h_idx, h in enumerate(HORIZONS):
            value = by_model_horizon.get((model, h))
            if value is None:
                continue
            xs.append(h_idx)
            ys.append(value)
        if not xs:
            continue
        ax.plot(
            xs, ys,
            color=CURATED_LINE_COLORS[model],
            marker="o", linewidth=2.3, markersize=6,
            label=display_name(model),
        )
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels(HORIZONS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylim(bottom=0)


def main() -> None:
    apply_style()

    run = load_run(RUN_NONH0)
    binary = mean_brier_by_model_horizon(score_binary(run))
    continuous = mean_normalized_crps_by_model_horizon(score_continuous(run))

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(4.35, 5.75), sharex=True)

    _draw_panel(
        ax_top, binary,
        ylabel="Mean Brier",
        title="Binary forecasting",
    )
    _draw_panel(
        ax_bottom, continuous,
        ylabel="Mean normalized CRPS",
        title="Continuous forecasting",
    )
    ax_bottom.set_xlabel("Forecast horizon")

    handles, labels = ax_bottom.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3, fontsize=8.5,
        bbox_to_anchor=(0.5, -0.01), frameon=False,
    )
    fig.tight_layout(rect=[0, 0.115, 1, 1])
    out = save_figure(fig, "fig3_horizon_curve")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
