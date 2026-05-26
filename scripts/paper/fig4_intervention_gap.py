"""Fig 4 — Conditional / intervention reasoning (republic + gold500).

Two panels (republic on the left, gold500 on the right). For each
intervention we compute, per matched question pair, how much the model
*benefited* from updating its forecast under the conditional framing
relative to using its baseline forecast on the post-intervention world:

    intervention_gain_q = brier(p_baseline, gt_intervention)
                       -  brier(p_conditional, gt_intervention)

A positive bar means the model's conditional forecast was closer to the
intervention-world ground truth than its baseline forecast would have
been. A negative bar means the model updated in the wrong direction or
overcorrected. Bars are sorted within each panel; curated 9 only since
that is the coverage we have.

Inputs (all from archive/20260302_121948/evaluations/runs/):
  republic_baseline_binary_all.anthropic_retry.json
  republic_conditional_binary_all.anthropic_retry.json
  gold500_baseline_binary_all.anthropic_retry.json
  gold500_conditional_binary_all.anthropic_retry.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from scripts.paper._data import (
    ARCHIVE_GOLD500_BASELINE_BIN,
    ARCHIVE_GOLD500_CONDITIONAL_BIN,
    ARCHIVE_REPUBLIC_BASELINE_BIN,
    ARCHIVE_REPUBLIC_CONDITIONAL_BIN,
    load_run,
)
from scripts.paper._style import (
    CURATED_LINE_COLORS,
    CURATED_MODELS,
    apply_style,
    display_name,
    save_figure,
)


def _intervention_gain_per_model(baseline_run: dict, conditional_run: dict) -> dict[str, float]:
    """For each model, mean intervention gain across matched question pairs."""
    base_by_qid = {q["question_id"]: q for q in baseline_run["questions"]}
    per_model: dict[str, list[float]] = {m: [] for m in CURATED_MODELS}

    for q_cond in conditional_run["questions"]:
        bid = q_cond["question_id"].removesuffix("_intervention")
        q_base = base_by_qid.get(bid)
        if q_base is None:
            continue
        gt_intervention = int(bool(q_cond["ground_truth"]))
        for model in CURATED_MODELS:
            cp = q_cond["predictions"].get(model, {})
            bp = q_base["predictions"].get(model, {})
            if cp.get("error") or bp.get("error"):
                continue
            p_cond = cp.get("probability")
            p_base = bp.get("probability")
            if p_cond is None or p_base is None:
                continue
            try:
                p_cond = float(p_cond)
                p_base = float(p_base)
            except (TypeError, ValueError):
                continue
            naive = (p_base - gt_intervention) ** 2
            updated = (p_cond - gt_intervention) ** 2
            per_model[model].append(naive - updated)

    return {m: mean(v) for m, v in per_model.items() if v}


def _draw_panel(ax, gains: dict[str, float], title: str):
    sorted_models = sorted(gains.keys(), key=lambda m: gains[m], reverse=True)
    labels = [display_name(m) for m in sorted_models]
    values = [gains[m] for m in sorted_models]
    colors = [CURATED_LINE_COLORS[m] for m in sorted_models]

    ys = list(range(len(labels)))
    ax.barh(ys, values, color=colors, edgecolor="white", linewidth=0.4, height=0.78)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(0, color="#1F2937", linewidth=0.8)
    ax.set_xlabel("Intervention gain  (baseline-Brier − conditional-Brier)")
    ax.set_title(title, loc="left", fontweight="bold")

    # Annotate value on each bar.
    xmax = max(abs(v) for v in values) if values else 1.0
    for y, v in zip(ys, values):
        offset = xmax * 0.02 if v >= 0 else -xmax * 0.02
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, y, f"{v:+.3f}", va="center", ha=ha, fontsize=7.5)

    # Padding so annotations don't clip.
    ax.set_xlim(-xmax * 1.25, xmax * 1.25)


def main() -> None:
    apply_style()

    rep_gain = _intervention_gain_per_model(
        load_run(ARCHIVE_REPUBLIC_BASELINE_BIN),
        load_run(ARCHIVE_REPUBLIC_CONDITIONAL_BIN),
    )
    gold_gain = _intervention_gain_per_model(
        load_run(ARCHIVE_GOLD500_BASELINE_BIN),
        load_run(ARCHIVE_GOLD500_CONDITIONAL_BIN),
    )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    _draw_panel(ax_l, rep_gain, "Republic intervention (government switch)")
    _draw_panel(ax_r, gold_gain, "Gold500 intervention (+500 treasury)")

    fig.suptitle(
        "Conditional reasoning: did models benefit from updating on the intervention?",
        fontsize=11, y=0.995,
    )
    fig.text(
        0.5, 0.005,
        "Positive = conditional forecast closer to intervention-world truth than the baseline forecast would have been.",
        ha="center", fontsize=7.5, style="italic", color="#475569",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = save_figure(fig, "fig4_intervention_gap")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
