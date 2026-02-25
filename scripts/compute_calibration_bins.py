"""
Compute per-bin calibration data for baseline vs Republic conditional,
averaged across all 10 models, then show per-model detail for top 3 bins.
"""

import json
import sys
from collections import defaultdict

BASELINE_PATH = "data/results/republic_baseline_binary_all.json"
CONDITIONAL_PATH = "data/results/republic_conditional_binary_all.json"

BIN_EDGES = [(i / 10, (i + 1) / 10) for i in range(10)]
BIN_LABELS = [f"[{lo:.1f}-{hi:.1f})" if hi < 1.0 else f"[{lo:.1f}-{hi:.1f}]"
              for lo, hi in BIN_EDGES]


def load_data(path):
    with open(path) as f:
        return json.load(f)


def bin_index(p):
    """Return bin index 0-9 for a probability in [0, 1]."""
    idx = int(p * 10)
    if idx == 10:  # p == 1.0 goes in last bin
        idx = 9
    return idx


def compute_model_bins(questions, model_id):
    """For a single model, compute calibration bins.

    Returns dict: bin_idx -> {"sum_pred": float, "sum_truth": float, "n": int}
    """
    bins = {i: {"sum_pred": 0.0, "sum_truth": 0.0, "n": 0} for i in range(10)}
    for q in questions:
        if q["question_type"] != "binary":
            continue
        pred_info = q["predictions"].get(model_id)
        if pred_info is None or pred_info["probability"] is None:
            continue
        p = pred_info["probability"]
        truth = 1.0 if q["ground_truth"] else 0.0
        idx = bin_index(p)
        bins[idx]["sum_pred"] += p
        bins[idx]["sum_truth"] += truth
        bins[idx]["n"] += 1
    return bins


def bins_to_stats(bins):
    """Convert raw bin sums to pred_mean, obs_freq, n."""
    stats = {}
    for i in range(10):
        b = bins[i]
        if b["n"] > 0:
            stats[i] = {
                "pred_mean": b["sum_pred"] / b["n"],
                "obs_freq": b["sum_truth"] / b["n"],
                "n": b["n"],
            }
        else:
            stats[i] = {"pred_mean": None, "obs_freq": None, "n": 0}
    return stats


def main():
    baseline_data = load_data(BASELINE_PATH)
    conditional_data = load_data(CONDITIONAL_PATH)

    baseline_questions = baseline_data["questions"]
    conditional_questions = conditional_data["questions"]

    models = sorted(baseline_data["model_results"].keys())
    print(f"Models ({len(models)}):")
    for m in models:
        print(f"  {m}")
    print()

    # Compute per-model bins for both conditions
    baseline_model_stats = {}
    conditional_model_stats = {}
    for model in models:
        b_bins = compute_model_bins(baseline_questions, model)
        c_bins = compute_model_bins(conditional_questions, model)
        baseline_model_stats[model] = bins_to_stats(b_bins)
        conditional_model_stats[model] = bins_to_stats(c_bins)

    # ---- Average across models ----
    print("=" * 100)
    print("AVERAGE CALIBRATION ACROSS ALL 10 MODELS")
    print("=" * 100)
    header = f"{'Bin':<14} {'Avg Pred (B)':>12} {'Avg Pred (C)':>12} {'Obs Freq (B)':>13} {'Obs Freq (C)':>13} {'n (B)':>8} {'n (C)':>8}"
    print(header)
    print("-" * 100)

    for i in range(10):
        # Collect stats from all models that have data in this bin
        b_pred_means = []
        b_obs_freqs = []
        b_ns = []
        c_pred_means = []
        c_obs_freqs = []
        c_ns = []
        for model in models:
            bs = baseline_model_stats[model][i]
            cs = conditional_model_stats[model][i]
            if bs["n"] > 0:
                b_pred_means.append(bs["pred_mean"])
                b_obs_freqs.append(bs["obs_freq"])
                b_ns.append(bs["n"])
            if cs["n"] > 0:
                c_pred_means.append(cs["pred_mean"])
                c_obs_freqs.append(cs["obs_freq"])
                c_ns.append(cs["n"])

        def fmt(vals):
            if vals:
                return f"{sum(vals)/len(vals):.4f}"
            return "   -   "

        def fmt_n(vals):
            if vals:
                return f"{sum(vals)/len(vals):.1f}"
            return "  -  "

        print(f"{BIN_LABELS[i]:<14} {fmt(b_pred_means):>12} {fmt(c_pred_means):>12} "
              f"{fmt(b_obs_freqs):>13} {fmt(c_obs_freqs):>13} "
              f"{fmt_n(b_ns):>8} {fmt_n(c_ns):>8}")

    # ---- Per-model detail for top 3 bins ----
    print()
    print("=" * 100)
    print("PER-MODEL DETAIL FOR HIGH-CONFIDENCE BINS (0.7-0.8, 0.8-0.9, 0.9-1.0)")
    print("=" * 100)

    top_bins = [7, 8, 9]
    for bin_i in top_bins:
        print(f"\n--- Bin {BIN_LABELS[bin_i]} ---")
        header2 = f"{'Model':<45} {'Pred (B)':>9} {'Obs (B)':>9} {'n (B)':>6}  {'Pred (C)':>9} {'Obs (C)':>9} {'n (C)':>6}  {'Gap':>7}"
        print(header2)
        print("-" * 110)
        for model in models:
            bs = baseline_model_stats[model][bin_i]
            cs = conditional_model_stats[model][bin_i]
            short_name = model.split("/")[-1]

            def f(v):
                return f"{v:.4f}" if v is not None else "  -  "

            gap = ""
            if bs["obs_freq"] is not None and cs["obs_freq"] is not None:
                g = bs["obs_freq"] - cs["obs_freq"]
                gap = f"{g:+.4f}"

            print(f"{short_name:<45} {f(bs['pred_mean']):>9} {f(bs['obs_freq']):>9} {bs['n']:>6}  "
                  f"{f(cs['pred_mean']):>9} {f(cs['obs_freq']):>9} {cs['n']:>6}  {gap:>7}")

    # Summary stats
    print()
    print("=" * 100)
    print("SUMMARY: Average obs_freq gap (baseline - conditional) in top 3 bins across all models")
    print("=" * 100)
    for bin_i in top_bins:
        gaps = []
        for model in models:
            bs = baseline_model_stats[model][bin_i]
            cs = conditional_model_stats[model][bin_i]
            if bs["obs_freq"] is not None and cs["obs_freq"] is not None:
                gaps.append(bs["obs_freq"] - cs["obs_freq"])
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            print(f"  {BIN_LABELS[bin_i]}: avg gap = {avg_gap:+.4f}  (across {len(gaps)} models)")


if __name__ == "__main__":
    main()
