"""Fig A2 — Per-template performance heatmap (appendix).

Two side-by-side heatmaps:
  - Left:  binary Brier per (curated model, binary template).
  - Right: continuous normalized CRPS per (curated model, family).

Rows are the curated 9 models in the same order in both panels. Columns
are the templates that appear in the run.

Inputs:
  data/evaluations/runs/final_2026-03-07_newq_uncond_nonh0_bin_plus_cont_all30.json
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper._data import (
    RUN_NONH0,
    load_run,
    score_binary,
    score_continuous,
)
from scripts.paper._style import (
    CURATED_MODELS,
    apply_style,
    display_name,
    save_figure,
)


def _build_grid(scores, value_attr, model_keyer, col_keyer):
    """Generic mean(values) per (model, column) builder."""
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    cols: set[str] = set()
    for s in scores:
        m = model_keyer(s)
        if m is None:
            continue
        c = col_keyer(s)
        cells[(m, c)].append(getattr(s, value_attr))
        cols.add(c)
    return {k: mean(v) for k, v in cells.items()}, sorted(cols)


def _draw_heatmap(ax, models, cols, cell_lookup, title, vmin, vmax, fmt):
    matrix = np.full((len(models), len(cols)), np.nan)
    for i, m in enumerate(models):
        for j, c in enumerate(cols):
            v = cell_lookup.get((m, c))
            if v is not None:
                matrix[i, j] = v

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([display_name(m) for m in models], fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Annotate every cell.
    for i in range(len(models)):
        for j in range(len(cols)):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            text_color = "white" if v >= (vmin + vmax) / 2 else "#1F2937"
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9, color=text_color)
    return im


def main() -> None:
    apply_style()

    run = load_run(RUN_NONH0)

    # Binary: model -> template_id -> mean Brier.
    binary_scores = score_binary(run)
    binary_cells, binary_cols = _build_grid(
        [s for s in binary_scores if s.model in CURATED_MODELS],
        value_attr="brier",
        model_keyer=lambda s: s.model,
        col_keyer=lambda s: s.template_id,
    )

    # Continuous: model -> family -> mean normalized CRPS.
    cont_scores = score_continuous(run)
    cont_cells, cont_cols = _build_grid(
        [s for s in cont_scores if s.model in CURATED_MODELS],
        value_attr="normalized_crps",
        model_keyer=lambda s: s.model,
        col_keyer=lambda s: s.family,
    )

    fig = plt.figure(figsize=(14.5, 5.4))
    ratio = max(1, len(binary_cols)) / max(1, len(cont_cols)) * 0.6 + 0.4
    gs = fig.add_gridspec(1, 2, width_ratios=[len(binary_cols), len(cont_cols) * 1.3])
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    # Pick clip ranges for color scales: 5th-95th percentile-ish.
    bin_values = [v for v in binary_cells.values()]
    cont_values = [v for v in cont_cells.values()]

    im_l = _draw_heatmap(
        ax_l, CURATED_MODELS, binary_cols, binary_cells,
        title="Binary — mean Brier per template",
        vmin=0, vmax=max(0.4, max(bin_values) if bin_values else 0.4),
        fmt="{:.2f}",
    )
    im_r = _draw_heatmap(
        ax_r, CURATED_MODELS, cont_cols, cont_cells,
        title="Continuous — normalized CRPS per template family",
        vmin=0, vmax=max(0.7, max(cont_values) if cont_values else 0.7),
        fmt="{:.2f}",
    )

    fig.colorbar(im_l, ax=ax_l, fraction=0.04, pad=0.02).set_label("Brier", fontsize=11)
    fig.colorbar(im_r, ax=ax_r, fraction=0.04, pad=0.02).set_label("Norm. CRPS", fontsize=11)

    fig.suptitle(
        "Per-template performance for the curated 9 models",
        fontsize=15, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = save_figure(fig, "figA2_template_heatmap")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
