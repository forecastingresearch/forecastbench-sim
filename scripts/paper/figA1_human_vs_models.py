"""Fig A1 — Human pilot vs models (appendix).

Horizontal bars showing normalized CRPS on the primary aggregate-proxy
setting (same templates and horizons as the human pilot: cities, techs,
treasury at H1/H3/H4/H6). All 31 models are shown; the curated 9 are
highlighted with their per-model line color. Three reference markers are
drawn as vertical dashed lines: uniform-bins baseline, individual human
mean, and human crowd mean.

Caveat to surface in the paper caption: humans saw 24 questions on 2
worlds; models scored on 660 primary-proxy questions across 11 worlds x
5 civilizations x 3 templates x 4 horizons. This is an aggregate proxy,
not a head-to-head matched-question comparison.

Inputs:
  data/human_baseline/aggregate_model_comparison/human_reference_summary.csv
  Continuous score function with Opus splice (covers 31 models).

Output:
  data/evaluations/plots/paper/figA1_human_vs_models.png
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from scripts.paper._data import HB_HUMAN_REF, score_continuous_with_opus
from scripts.paper._style import (
    CURATED_LINE_COLORS,
    CURATED_MODELS,
    apply_style,
    display_name,
    save_figure,
)


PRIMARY_HORIZONS = {"H1", "H3", "H4", "H6"}
PRIMARY_FAMILIES = {"cities", "techs", "treasury"}

REF_STYLE = {
    "uniform_bins":     {"color": "#6B7280", "label": "uniform baseline"},
    "human_individual": {"color": "#0F172A", "label": "human (individual mean)"},
    "human_crowd_mean": {"color": "#DC2626", "label": "human (crowd mean)"},
}


def _model_primary_proxy_scores() -> dict[str, float]:
    """Mean normalized CRPS per model on the human-pilot-matched subset."""
    by_model: dict[str, list[float]] = defaultdict(list)
    for s in score_continuous_with_opus():
        if s.horizon not in PRIMARY_HORIZONS or s.family not in PRIMARY_FAMILIES:
            continue
        by_model[s.model].append(s.normalized_crps)
    return {m: mean(v) for m, v in by_model.items() if v}


def _read_human_refs() -> dict[str, float]:
    out: dict[str, float] = {}
    with HB_HUMAN_REF.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["source"]] = float(row["normalized_crps"])
    return out


def main() -> None:
    apply_style()

    model_scores = _model_primary_proxy_scores()
    refs = _read_human_refs()

    sorted_models = sorted(model_scores.keys(), key=lambda m: model_scores[m])
    labels = [f"{display_name(m)}  ({i + 1}/{len(sorted_models)})" for i, m in enumerate(sorted_models)]
    values = [model_scores[m] for m in sorted_models]
    colors = [
        CURATED_LINE_COLORS[m] if m in CURATED_MODELS else "#CBD5E1"
        for m in sorted_models
    ]

    # Clip x-axis: anything past 0.6 is annotated rather than drawn.
    # Tight clip gives the three reference lines (uniform / individual /
    # crowd, all near 0.15) enough horizontal room to be distinguishable.
    clip = 0.6
    plotted_values = [min(v, clip) for v in values]
    off_chart = [(display_name(m), v) for m, v in zip(sorted_models, values) if v > clip]

    fig, ax = plt.subplots(figsize=(7.5, 9.5))
    ys = list(range(len(labels)))
    ax.barh(ys, plotted_values, color=colors, edgecolor="white", linewidth=0.4, height=0.78)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    for tick, m in zip(ax.get_yticklabels(), sorted_models):
        if m in CURATED_MODELS:
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, clip)
    ax.set_xlabel("Mean normalized CRPS on human-matched subset (lower is better)")
    ax.set_title(
        "Human pilot vs models on the primary aggregate-proxy setting",
        loc="left", fontweight="bold",
    )

    for ref_key, ref_value in refs.items():
        style = REF_STYLE[ref_key]
        ax.axvline(
            ref_value, color=style["color"], linewidth=1.2,
            linestyle="--", label=f"{style['label']}: {ref_value:.3f}",
        )

    ax.legend(loc="upper right", fontsize=8)

    if off_chart:
        txt = "off-chart: " + "; ".join(f"{n} = {v:.2f}" for n, v in off_chart)
        ax.text(
            clip * 0.99, len(labels) - 0.5, txt,
            ha="right", va="bottom", fontsize=7, style="italic", color="#475569",
        )

    fig.tight_layout()
    out = save_figure(fig, "figA1_human_vs_models")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
