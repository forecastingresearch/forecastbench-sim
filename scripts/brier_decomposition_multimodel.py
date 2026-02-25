"""Brier decomposition (REL, RES, UNC) across 10 models.

Decomposes the conditional Brier gap into calibration failure (REL),
resolution loss (RES), and difficulty (UNC). This is an exact decomposition
that sidesteps the Simas objection about bounds on optimal Brier.

Usage:
    uv run python scripts/brier_decomposition_multimodel.py
"""

import json
import numpy as np
from collections import defaultdict


MODEL_NAMES = {
    "openai/o3-2025-04-16": "o3",
    "openai/gpt-5.1-2025-11-13": "GPT-5.1",
    "openai/gpt-5-2025-08-07": "GPT-5",
    "openai/gpt-5-mini-2025-08-07": "GPT-5 mini",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "anthropic/claude-opus-4-5-20251101": "Opus 4.5",
    "anthropic/claude-sonnet-4-5-20250929": "Sonnet 4.5",
}

# Display order
MODEL_ORDER = [
    "openai/o3-2025-04-16",
    "openai/gpt-5.1-2025-11-13",
    "openai/gpt-5-2025-08-07",
    "openai/gpt-5-mini-2025-08-07",
    "openai/gpt-4.1-2025-04-14",
    "google/gemini-3-pro-preview",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "anthropic/claude-opus-4-5-20251101",
    "anthropic/claude-sonnet-4-5-20250929",
]

CONFIGS = {
    "republic": {
        "baseline": "data/results/republic_baseline_binary_all.json",
        "conditional": "data/results/republic_conditional_binary_all.json",
    },
    "gold500": {
        "baseline": "data/results/gold500_baseline_binary_all.json",
        "conditional": "data/results/gold500_conditional_binary_all.json",
    },
}

N_BINS = 10


def load_predictions(eval_path, model_id):
    """Load (prediction, ground_truth) pairs for one model from eval file."""
    data = json.load(open(eval_path))
    preds = []
    truths = []
    for q in data["questions"]:
        pred = q["predictions"].get(model_id)
        if pred and pred.get("error") is None:
            preds.append(pred["probability"])
            truths.append(1.0 if q["ground_truth"] else 0.0)
    return np.array(preds), np.array(truths)


def murphy_decomposition(preds, truths, n_bins=N_BINS):
    """Compute Murphy (1973) Brier decomposition: Brier = REL - RES + UNC."""
    n = len(preds)
    base_rate = truths.mean()
    unc = base_rate * (1 - base_rate)

    # Bin predictions into equal-width bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rel = 0.0
    res = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)

        n_b = mask.sum()
        if n_b == 0:
            continue

        f_b = truths[mask].mean()  # observed frequency in bin
        p_b = preds[mask].mean()   # mean prediction in bin

        rel += n_b * (f_b - p_b) ** 2
        res += n_b * (f_b - base_rate) ** 2

    rel /= n
    res /= n
    brier = ((preds - truths) ** 2).mean()

    return {
        "brier": brier,
        "rel": rel,
        "res": res,
        "unc": unc,
        "base_rate": base_rate,
        "n": n,
        "check": rel - res + unc,  # should ≈ brier
    }


def ece(preds, truths, n_bins=N_BINS):
    """Expected Calibration Error."""
    n = len(preds)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)

        n_b = mask.sum()
        if n_b == 0:
            continue

        f_b = truths[mask].mean()
        p_b = preds[mask].mean()
        ece_val += n_b * abs(f_b - p_b)

    return ece_val / n


def analyze_intervention(name, config):
    """Run Brier decomposition for one intervention."""
    print(f"\n{'='*100}")
    print(f"  {name.upper()}")
    print(f"{'='*100}")

    # Header
    print(f"\n  {'Model':<16} {'Cond':<6} {'Brier':>7} {'REL':>7} {'RES':>7} "
          f"{'UNC':>7} {'ECE':>7} {'BR':>6} {'n':>6}")
    print(f"  {'-'*80}")

    all_results = {}

    for model_id in MODEL_ORDER:
        short = MODEL_NAMES[model_id]
        model_results = {}

        for cond_name, path in [("BL", config["baseline"]), ("CD", config["conditional"])]:
            preds, truths = load_predictions(path, model_id)
            decomp = murphy_decomposition(preds, truths)
            ece_val = ece(preds, truths)
            decomp["ece"] = ece_val
            model_results[cond_name] = decomp

            print(f"  {short:<16} {cond_name:<6} {decomp['brier']:>7.4f} "
                  f"{decomp['rel']:>7.4f} {decomp['res']:>7.4f} "
                  f"{decomp['unc']:>7.4f} {ece_val:>7.4f} "
                  f"{decomp['base_rate']:>6.2f} {decomp['n']:>6}")

        all_results[model_id] = model_results

    # Gap decomposition
    print(f"\n  --- Brier gap decomposition: ΔBRIER = ΔREL + (-ΔRES) + ΔUNC ---")
    print(f"  {'Model':<16} {'ΔBRIER':>8} {'ΔREL':>8} {'-ΔRES':>8} {'ΔUNC':>8} "
          f"{'%REL':>7} {'%RES':>7} {'%UNC':>7}")
    print(f"  {'-'*80}")

    summary_rels = []
    summary_ress = []
    summary_uncs = []

    for model_id in MODEL_ORDER:
        short = MODEL_NAMES[model_id]
        bl = all_results[model_id]["BL"]
        cd = all_results[model_id]["CD"]

        d_brier = cd["brier"] - bl["brier"]
        d_rel = cd["rel"] - bl["rel"]
        d_res = -(cd["res"] - bl["res"])  # negative because RES is subtracted in Brier
        d_unc = cd["unc"] - bl["unc"]

        # Percentages (of total gap)
        if abs(d_brier) > 0.001:
            pct_rel = d_rel / d_brier * 100
            pct_res = d_res / d_brier * 100
            pct_unc = d_unc / d_brier * 100
        else:
            pct_rel = pct_res = pct_unc = float("nan")

        summary_rels.append(pct_rel)
        summary_ress.append(pct_res)
        summary_uncs.append(pct_unc)

        print(f"  {short:<16} {d_brier:>+8.4f} {d_rel:>+8.4f} {d_res:>+8.4f} "
              f"{d_unc:>+8.4f} {pct_rel:>6.1f}% {pct_res:>6.1f}% {pct_unc:>6.1f}%")

    # Averages
    print(f"  {'-'*80}")
    print(f"  {'AVERAGE':<16} {'':>8} {'':>8} {'':>8} {'':>8} "
          f"{np.mean(summary_rels):>6.1f}% {np.mean(summary_ress):>6.1f}% "
          f"{np.mean(summary_uncs):>6.1f}%")

    return all_results


def main():
    for name, config in CONFIGS.items():
        analyze_intervention(name, config)


if __name__ == "__main__":
    main()
