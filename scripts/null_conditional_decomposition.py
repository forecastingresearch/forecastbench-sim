"""Brier decomposition for null conditional vs real conditional (Opus 4.5).

Compares calibration (REL) across baseline, null conditional, and real
conditional to show that calibration failure is specifically about
intervention content, not conditional framing.

Usage:
    uv run python scripts/null_conditional_decomposition.py
"""

import json
import numpy as np


MODEL = "anthropic/claude-opus-4-5-20251101"
N_BINS = 10

CONDITIONS = {
    "Baseline": "data/evaluations/results/baseline_opus45_republic_full_eval.json",
    "Null Conditional": "data/evaluations/results/conditional_no_opus45_republic_full_eval.json",
    "Real Conditional": "data/evaluations/results/conditional_opus45_republic_full_eval.json",
}


def load_predictions(eval_path, model_id):
    """Load (prediction, ground_truth) pairs."""
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
    """Compute Murphy (1973) Brier decomposition."""
    n = len(preds)
    base_rate = truths.mean()
    unc = base_rate * (1 - base_rate)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    rel = 0.0
    res = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)

        n_b = mask.sum()
        if n_b == 0:
            bin_data.append(None)
            continue

        f_b = truths[mask].mean()
        p_b = preds[mask].mean()

        rel += n_b * (f_b - p_b) ** 2
        res += n_b * (f_b - base_rate) ** 2

        bin_data.append({
            "lo": lo, "hi": hi, "n": int(n_b),
            "pred_mean": p_b, "obs_freq": f_b, "gap": f_b - p_b,
        })

    rel /= n
    res /= n
    brier = ((preds - truths) ** 2).mean()
    ece_val = sum(
        b["n"] * abs(b["gap"]) for b in bin_data if b is not None
    ) / n

    return {
        "brier": brier,
        "rel": rel,
        "res": res,
        "unc": unc,
        "ece": ece_val,
        "base_rate": base_rate,
        "n": n,
        "bins": bin_data,
    }


def main():
    print("Brier decomposition: Baseline vs Null Conditional vs Real Conditional")
    print("Model: Opus 4.5 (Republic intervention)")
    print(f"{'='*90}")

    results = {}
    for cond_name, path in CONDITIONS.items():
        preds, truths = load_predictions(path, MODEL)
        decomp = murphy_decomposition(preds, truths)
        results[cond_name] = decomp

    # Summary table
    print(f"\n  {'Condition':<20} {'Brier':>7} {'REL':>7} {'RES':>7} {'UNC':>7} "
          f"{'ECE':>7} {'BR':>6} {'n':>5}")
    print(f"  {'-'*70}")
    for cond_name, d in results.items():
        print(f"  {cond_name:<20} {d['brier']:>7.4f} {d['rel']:>7.4f} "
              f"{d['res']:>7.4f} {d['unc']:>7.4f} {d['ece']:>7.4f} "
              f"{d['base_rate']:>6.2f} {d['n']:>5}")

    # Gap decomposition
    bl = results["Baseline"]
    null = results["Null Conditional"]
    real = results["Real Conditional"]

    print(f"\n  --- Gap decomposition ---")
    for label, ref, cmp in [
        ("Null vs Baseline", bl, null),
        ("Real vs Baseline", bl, real),
        ("Real vs Null", null, real),
    ]:
        d_brier = cmp["brier"] - ref["brier"]
        d_rel = cmp["rel"] - ref["rel"]
        d_res = -(cmp["res"] - ref["res"])
        d_unc = cmp["unc"] - ref["unc"]

        if abs(d_brier) > 0.001:
            pct_rel = d_rel / d_brier * 100
            pct_res = d_res / d_brier * 100
            pct_unc = d_unc / d_brier * 100
        else:
            pct_rel = pct_res = pct_unc = float("nan")

        print(f"\n  {label}:")
        print(f"    ΔBRIER = {d_brier:+.4f}")
        print(f"    ΔREL   = {d_rel:+.4f} ({pct_rel:+.1f}%) — calibration change")
        print(f"    -ΔRES  = {d_res:+.4f} ({pct_res:+.1f}%) — resolution change")
        print(f"    ΔUNC   = {d_unc:+.4f} ({pct_unc:+.1f}%) — difficulty change")

    # Per-bin calibration detail
    print(f"\n  --- Per-bin calibration detail ---")
    print(f"  {'Bin':<12} {'BL freq':>8} {'Null freq':>10} {'Real freq':>10} "
          f"{'BL n':>5} {'Null n':>6} {'Real n':>6}")
    print(f"  {'-'*65}")

    for i in range(N_BINS):
        bl_b = bl["bins"][i]
        null_b = null["bins"][i]
        real_b = real["bins"][i]

        lo = i / N_BINS
        hi = (i + 1) / N_BINS
        label = f"[{lo:.1f}-{hi:.1f})"

        bl_freq = f"{bl_b['obs_freq']:.3f}" if bl_b else "—"
        null_freq = f"{null_b['obs_freq']:.3f}" if null_b else "—"
        real_freq = f"{real_b['obs_freq']:.3f}" if real_b else "—"
        bl_n = str(bl_b["n"]) if bl_b else "0"
        null_n = str(null_b["n"]) if null_b else "0"
        real_n = str(real_b["n"]) if real_b else "0"

        print(f"  {label:<12} {bl_freq:>8} {null_freq:>10} {real_freq:>10} "
              f"{bl_n:>5} {null_n:>6} {real_n:>6}")


if __name__ == "__main__":
    main()
