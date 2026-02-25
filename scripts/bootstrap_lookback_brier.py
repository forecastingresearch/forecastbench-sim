#!/usr/bin/env python3
"""
Bootstrap test for lookback conditional Brier score differences.

For each model, computes:
- Observed mean Brier gap (lookback - baseline) per question
- Bootstrap 95% CI on the mean gap
- p-value (two-sided): proportion of bootstrap samples where mean gap >= 0
  (or <= 0, depending on direction)

Usage:
    uv run python scripts/bootstrap_lookback_brier.py \
        --baseline data/results/lookback_baseline_eval.json \
        --conditional data/results/lookback_conditional_eval.json \
        [--n-boot 10000] [--seed 42]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


MODELS = [
    "openai/o3-2025-04-16",
    "anthropic/claude-opus-4-5-20251101",
    "openai/gpt-4.1-2025-04-14",
]

MODEL_SHORT = {
    "openai/o3-2025-04-16": "o3",
    "anthropic/claude-opus-4-5-20251101": "Opus 4.5",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
}


def brier_score(prob: float, outcome: bool) -> float:
    """Brier score for a single binary prediction."""
    return (prob - float(outcome)) ** 2


def load_eval(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_binary_brier_pairs(baseline_data, conditional_data, model: str):
    """Extract paired per-question Brier scores for a model.

    Returns (baseline_briers, conditional_briers) as aligned numpy arrays.
    """
    # Index baseline questions by (game_id, question_id) — IDs repeat across seeds
    base_by_key = {}
    for q in baseline_data["questions"]:
        if q["question_type"] == "binary":
            key = (q["game_id"], q["question_id"])
            base_by_key[key] = q

    baseline_briers = []
    conditional_briers = []

    for cq in conditional_data["questions"]:
        if cq["question_type"] != "binary":
            continue

        # Get paired baseline question
        base_id = cq["question_id"].replace("_lookback", "")
        key = (cq["game_id"], base_id)
        bq = base_by_key.get(key)
        if bq is None:
            continue

        # Get predictions
        base_pred = bq.get("predictions", {}).get(model)
        cond_pred = cq.get("predictions", {}).get(model)
        if base_pred is None or cond_pred is None:
            continue
        if base_pred.get("error") or cond_pred.get("error"):
            continue

        base_prob = base_pred["probability"]
        cond_prob = cond_pred["probability"]

        # Ground truth (same for both)
        gt = bq["ground_truth"]

        baseline_briers.append(brier_score(base_prob, gt))
        conditional_briers.append(brier_score(cond_prob, gt))

    return np.array(baseline_briers), np.array(conditional_briers)


def bootstrap_mean_diff(gaps: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Bootstrap the mean of per-question Brier gaps."""
    n = len(gaps)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = gaps[idx].mean()
    return boot_means


def main():
    parser = argparse.ArgumentParser(description="Bootstrap test for lookback Brier gaps")
    parser.add_argument("--baseline", required=True, help="Path to baseline eval JSON")
    parser.add_argument("--conditional", required=True, help="Path to conditional eval JSON")
    parser.add_argument("--n-boot", type=int, default=10000, help="Number of bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    baseline_data = load_eval(Path(args.baseline))
    conditional_data = load_eval(Path(args.conditional))
    rng = np.random.default_rng(args.seed)

    # Identify framing from first conditional question
    first_cq = next(
        (q for q in conditional_data["questions"] if q["question_type"] == "binary"), None
    )
    if first_cq:
        text = first_cq["question_text"]
        if text.startswith("Given that"):
            framing = '"Given that" (presuppositional)'
        elif text.startswith("If "):
            framing = '"If" (hypothetical)'
        else:
            framing = "unknown"
    else:
        framing = "unknown"

    print(f"Framing: {framing}")
    print(f"Bootstrap iterations: {args.n_boot}")
    print(f"Random seed: {args.seed}")
    print()

    # Also collect all gaps across models for a pooled test
    all_gaps = []

    for model in MODELS:
        name = MODEL_SHORT.get(model, model)
        base_b, cond_b = extract_binary_brier_pairs(baseline_data, conditional_data, model)

        if len(base_b) == 0:
            print(f"{name}: No paired questions found")
            continue

        gaps = cond_b - base_b  # positive = conditional is worse
        observed_mean = gaps.mean()
        all_gaps.append(gaps)

        boot_means = bootstrap_mean_diff(gaps, args.n_boot, rng)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        # Two-sided p-value: proportion of bootstrap samples on opposite side of 0
        if observed_mean < 0:
            # Improvement: p = 2 * P(boot_mean >= 0)
            p_value = 2 * (boot_means >= 0).mean()
        else:
            # Degradation or null: p = 2 * P(boot_mean <= 0)
            p_value = 2 * (boot_means <= 0).mean()
        p_value = min(p_value, 1.0)

        print(f"--- {name} (n={len(gaps)}) ---")
        print(f"  Baseline mean Brier:     {base_b.mean():.4f}")
        print(f"  Conditional mean Brier:  {cond_b.mean():.4f}")
        print(f"  Mean gap (cond - base):  {observed_mean:+.4f}")
        print(f"  95% CI:                  [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  p-value (two-sided):     {p_value:.4f}")
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        print(f"  Significance:            {sig}")
        print()

    # Pooled test across all models
    if all_gaps:
        pooled = np.concatenate(all_gaps)
        observed_pooled = pooled.mean()
        boot_pooled = bootstrap_mean_diff(pooled, args.n_boot, rng)
        ci_lo, ci_hi = np.percentile(boot_pooled, [2.5, 97.5])
        if observed_pooled < 0:
            p_value = 2 * (boot_pooled >= 0).mean()
        else:
            p_value = 2 * (boot_pooled <= 0).mean()
        p_value = min(p_value, 1.0)

        print(f"--- Pooled across models (n={len(pooled)}) ---")
        print(f"  Mean gap (cond - base):  {observed_pooled:+.4f}")
        print(f"  95% CI:                  [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  p-value (two-sided):     {p_value:.4f}")
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        print(f"  Significance:            {sig}")


if __name__ == "__main__":
    main()
