"""Fig A3 — Binary calibration diagnostics (appendix).

ECE = sum_b (|B_b| / N) * |conf(B_b) - acc(B_b)|

where bins B_b are 10 equal-width bins of predicted probability,
conf(B_b) is the mean predicted probability in the bin, and acc(B_b) is
the empirical outcome rate. Lower is better; 0 means perfect calibration.

Two stacked panels:
  - ECE bar per curated model.
  - Signed calibration bias, mean(predicted probability - outcome).

This avoids the visually tangled reliability-curve plot while preserving the
two checks a reader needs: magnitude of calibration error and direction of
average over/under-confidence.

Inputs:
  data/evaluations/runs/final_2026-03-07_newq_uncond_nonh0_bin_plus_cont_all30.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from scripts.paper._data import RUN_NONH0, load_run
from scripts.paper._style import (
    CURATED_LINE_COLORS,
    CURATED_MODELS,
    apply_style,
    display_name,
    save_figure,
)


N_BINS = 10


def _calibration_stats(run, model_id):
    """Return ECE, signed mean bias, and prediction count for one model."""
    bins = [[0.0, 0.0, 0] for _ in range(N_BINS)]  # sum_p, sum_outcome, count
    total = 0
    bias_sum = 0.0
    for q in run["questions"]:
        if q.get("question_type") != "binary":
            continue
        gt = q.get("ground_truth")
        if gt is None:
            continue
        outcome = int(bool(gt))
        pred = q["predictions"].get(model_id, {})
        if not pred or pred.get("error"):
            continue
        p = pred.get("probability")
        if p is None:
            continue
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        p = min(max(p, 0.0), 1.0)
        idx = min(int(p * N_BINS), N_BINS - 1)
        bins[idx][0] += p
        bins[idx][1] += outcome
        bins[idx][2] += 1
        total += 1
        bias_sum += p - outcome

    ece = 0.0
    for sum_p, sum_o, count in bins:
        if count == 0:
            continue
        conf = sum_p / count
        acc = sum_o / count
        ece += (count / total) * abs(conf - acc)
    return {
        "ece": ece,
        "bias": bias_sum / total,
        "n": total,
    }


def main() -> None:
    apply_style()
    run = load_run(RUN_NONH0)

    stats = {m: _calibration_stats(run, m) for m in CURATED_MODELS}
    ece_by_model = {m: stats[m]["ece"] for m in CURATED_MODELS}

    sorted_models = sorted(CURATED_MODELS, key=lambda m: ece_by_model[m])
    labels = [display_name(m) for m in sorted_models]
    eces = [ece_by_model[m] for m in sorted_models]
    biases = [stats[m]["bias"] for m in sorted_models]
    colors = [CURATED_LINE_COLORS[m] for m in sorted_models]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.5, 7.0),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    ys = list(range(len(labels)))
    ax_top.barh(ys, eces, color=colors, edgecolor="white", linewidth=0.4, height=0.78)
    ax_top.set_yticks(ys)
    ax_top.set_yticklabels(labels, fontweight="bold")
    ax_top.invert_yaxis()
    ax_top.set_xlabel("Expected Calibration Error  (lower is better)")
    ax_top.set_title(
        "Calibration error per model on binary forecasts",
        loc="left", fontweight="bold",
    )
    for y, v in zip(ys, eces):
        ax_top.text(v + max(eces) * 0.01, y, f"{v:.3f}", va="center", ha="left", fontsize=7.5)
    ax_top.set_xlim(0, max(eces) * 1.18)

    ax_bot.barh(ys, biases, color=colors, edgecolor="white", linewidth=0.4, height=0.78)
    ax_bot.set_yticks(ys)
    ax_bot.set_yticklabels(labels, fontweight="bold")
    ax_bot.invert_yaxis()
    ax_bot.axvline(0, color="#1F2937", linewidth=0.8)
    ax_bot.set_xlabel("Mean signed calibration bias  (prediction − outcome)")
    ax_bot.set_title(
        "Average direction of calibration error",
        loc="left", fontweight="bold",
    )
    xmax = max(abs(v) for v in biases) if biases else 0.1
    ax_bot.set_xlim(-xmax * 1.25, xmax * 1.25)
    for y, v in zip(ys, biases):
        offset = xmax * 0.02 if v >= 0 else -xmax * 0.02
        ha = "left" if v >= 0 else "right"
        ax_bot.text(v + offset, y, f"{v:+.3f}", va="center", ha=ha, fontsize=7.5)

    fig.suptitle("Binary calibration diagnostics for curated models", fontsize=11, y=0.995)
    fig.text(
        0.5, 0.01,
        "Positive bias means forecasts are high on average relative to realized outcomes; negative means low.",
        ha="center", fontsize=7.5, color="#475569", style="italic",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = save_figure(fig, "figA3_reliability")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
