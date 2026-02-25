"""Brier decomposition (REL, RES, UNC) for prompting experiment results.

Computes Murphy decomposition per condition x model to see whether
prompting variants reduce REL (calibration failure) specifically.

Usage:
    uv run python scripts/prompting_brier_decomposition.py
"""

import json
import numpy as np
from pathlib import Path

CONDITIONS = [
    "baseline",
    "cot",
    "explicit_updating",
    "structured_reasoning",
    "base_rate_reanchoring",
    "calibration_nudge",
]

MODEL_NAMES = {
    "openai/o3-2025-04-16": "o3",
    "anthropic/claude-opus-4-5-20251101": "Opus 4.5",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
}

MODEL_ORDER = [
    "openai/o3-2025-04-16",
    "anthropic/claude-opus-4-5-20251101",
    "openai/gpt-4.1-2025-04-14",
]

DATA_DIR = Path("data/results/prompting_experiment")
N_BINS = 10


def load_predictions(condition, model_id):
    """Load (prediction, ground_truth) pairs for one model from one condition."""
    path = DATA_DIR / f"prompting_{condition}.json"
    data = json.load(open(path))
    preds = []
    truths = []
    for q in data["questions"]:
        pred = q["predictions"].get(model_id)
        if pred and pred.get("error") is None and pred.get("probability") is not None:
            preds.append(pred["probability"])
            truths.append(1.0 if q["ground_truth"] else 0.0)
    return np.array(preds), np.array(truths)


def murphy_decomposition(preds, truths, n_bins=N_BINS):
    """Compute Murphy (1973) Brier decomposition: Brier = REL - RES + UNC."""
    n = len(preds)
    base_rate = truths.mean()
    unc = base_rate * (1 - base_rate)

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

        f_b = truths[mask].mean()
        p_b = preds[mask].mean()

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
    }


def main():
    # Compute decomposition for all conditions x models
    results = {}
    for condition in CONDITIONS:
        results[condition] = {}
        for model_id in MODEL_ORDER:
            preds, truths = load_predictions(condition, model_id)
            if len(preds) == 0:
                continue
            decomp = murphy_decomposition(preds, truths)
            results[condition][model_id] = decomp

    # Print main results table
    print("=" * 100)
    print("BRIER DECOMPOSITION BY PROMPTING CONDITION")
    print("=" * 100)
    print(f"{'Condition':<25} {'Model':<12} {'N':>6} {'Brier':>8} {'REL':>8} {'RES':>8} {'UNC':>8} {'BaseRate':>8}")
    print("-" * 100)

    for condition in CONDITIONS:
        for model_id in MODEL_ORDER:
            d = results[condition].get(model_id)
            if d is None:
                continue
            name = MODEL_NAMES.get(model_id, model_id.split("/")[-1][:12])
            print(
                f"{condition:<25} {name:<12} {d['n']:>6} "
                f"{d['brier']:>8.4f} {d['rel']:>8.4f} {d['res']:>8.4f} "
                f"{d['unc']:>8.4f} {d['base_rate']:>8.3f}"
            )
        print()

    # Print delta from baseline table
    print("\n" + "=" * 100)
    print("DELTA FROM BASELINE (negative = improvement)")
    print("=" * 100)
    print(f"{'Condition':<25} {'Model':<12} {'ΔBrier':>8} {'ΔREL':>8} {'ΔRES':>8} {'ΔUNC':>8}")
    print("-" * 100)

    for condition in CONDITIONS:
        if condition == "baseline":
            continue
        for model_id in MODEL_ORDER:
            d = results[condition].get(model_id)
            bl = results["baseline"].get(model_id)
            if d is None or bl is None:
                continue
            name = MODEL_NAMES.get(model_id, model_id.split("/")[-1][:12])
            d_brier = d["brier"] - bl["brier"]
            d_rel = d["rel"] - bl["rel"]
            d_res = d["res"] - bl["res"]
            d_unc = d["unc"] - bl["unc"]
            print(
                f"{condition:<25} {name:<12} "
                f"{d_brier:>+8.4f} {d_rel:>+8.4f} {d_res:>+8.4f} {d_unc:>+8.4f}"
            )
        print()

    # Summary: best condition per model
    print("\n" + "=" * 100)
    print("BEST CONDITION PER MODEL (by Brier)")
    print("=" * 100)
    for model_id in MODEL_ORDER:
        name = MODEL_NAMES.get(model_id, model_id.split("/")[-1])
        best_cond = None
        best_brier = float("inf")
        bl_brier = results["baseline"][model_id]["brier"]
        for condition in CONDITIONS:
            d = results[condition].get(model_id)
            if d and d["brier"] < best_brier:
                best_brier = d["brier"]
                best_cond = condition
        improvement = bl_brier - best_brier
        pct = improvement / bl_brier * 100 if bl_brier > 0 else 0
        print(f"  {name:<12}: {best_cond:<25} Brier={best_brier:.4f} (Δ={-improvement:+.4f}, {pct:+.1f}%)")

    # Save full results as JSON
    output = {}
    for condition in CONDITIONS:
        output[condition] = {}
        for model_id in MODEL_ORDER:
            d = results[condition].get(model_id)
            if d:
                output[condition][model_id] = {k: float(v) for k, v in d.items()}

    out_path = DATA_DIR / "prompting_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
