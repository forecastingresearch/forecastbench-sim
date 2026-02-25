#!/usr/bin/env python3
"""
Counterfactual Brier score analysis.

Question: If we take LLMs' predictions on conditional (Republic) questions
and score them against baseline ground truth instead of fork ground truth,
what Brier score do we get?

If models are just predicting baseline probabilities on conditional questions,
the "counterfactual Brier" should be close to the actual baseline Brier.

Four Brier scores per model (computed over matched question pairs only):
  - Baseline Brier:        baseline_prediction  vs  baseline_truth
  - Conditional Brier:     conditional_prediction vs fork_truth
  - Counterfactual Brier:  conditional_prediction vs baseline_truth  (KEY)
  - Cross Brier:           baseline_prediction  vs  fork_truth
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "results"
BASELINE_FILE = DATA_DIR / "republic_baseline_binary_all.json"
CONDITIONAL_FILE = DATA_DIR / "republic_conditional_binary_all.json"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def brier_score(probability: float, outcome: bool) -> float:
    """Brier score for a single prediction: (p - outcome)^2."""
    return (probability - float(outcome)) ** 2


def main():
    baseline_data = load_json(BASELINE_FILE)
    conditional_data = load_json(CONDITIONAL_FILE)

    # Index baseline questions by question_id
    baseline_by_id = {q["question_id"]: q for q in baseline_data["questions"]}

    # Collect per-model scores across matched pairs
    # Each entry: {model_id: [list of (baseline_brier, conditional_brier, counterfactual_brier, cross_brier)]}
    model_scores = defaultdict(lambda: {
        "baseline": [],
        "conditional": [],
        "counterfactual": [],
        "cross": [],
    })

    matched = 0
    unmatched = 0

    for cond_q in conditional_data["questions"]:
        # Only binary questions
        if cond_q["question_type"] != "binary":
            continue

        # Match to baseline
        base_id = cond_q["question_id"].replace("_intervention", "")
        base_q = baseline_by_id.get(base_id)
        if base_q is None:
            unmatched += 1
            continue

        matched += 1

        baseline_truth = base_q["ground_truth"]
        fork_truth = cond_q["ground_truth"]

        for model_id, cond_pred in cond_q["predictions"].items():
            base_pred = base_q["predictions"].get(model_id)
            if base_pred is None:
                continue

            p_base = base_pred["probability"]
            p_cond = cond_pred["probability"]

            if p_base is None or p_cond is None:
                continue

            model_scores[model_id]["baseline"].append(brier_score(p_base, baseline_truth))
            model_scores[model_id]["conditional"].append(brier_score(p_cond, fork_truth))
            model_scores[model_id]["counterfactual"].append(brier_score(p_cond, baseline_truth))
            model_scores[model_id]["cross"].append(brier_score(p_base, fork_truth))

    # Compute means
    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    models = sorted(model_scores.keys())

    # Print summary
    print(f"Matched question pairs: {matched}")
    print(f"Unmatched conditional questions: {unmatched}")
    print(f"Baseline base rate:    {sum(1 for q in baseline_data['questions'] if q['ground_truth']) / len(baseline_data['questions']):.3f}")
    print(f"Conditional base rate: {sum(1 for q in conditional_data['questions'] if q['ground_truth']) / len(conditional_data['questions']):.3f}")
    print()

    # Compute ground truth agreement on matched pairs
    agree = 0
    for cond_q in conditional_data["questions"]:
        if cond_q["question_type"] != "binary":
            continue
        base_id = cond_q["question_id"].replace("_intervention", "")
        base_q = baseline_by_id.get(base_id)
        if base_q is None:
            continue
        if base_q["ground_truth"] == cond_q["ground_truth"]:
            agree += 1
    print(f"Ground truth agreement (matched pairs): {agree}/{matched} ({agree/matched:.1%})")
    print()

    # Short model names for display
    def short_name(model_id: str) -> str:
        parts = model_id.split("/")
        name = parts[-1] if len(parts) > 1 else model_id
        # Trim date suffixes like -20251101
        for suffix in ["-20251101", "-20250929", "-2025-04-16", "-2025-04-14",
                       "-2025-08-07", "-2025-11-13"]:
            name = name.replace(suffix, "")
        return name

    # Table header
    name_width = max(len(short_name(m)) for m in models)
    header = f"{'Model':<{name_width}}  {'Baseline':>10}  {'Cond.':>10}  {'Countfact.':>10}  {'Cross':>10}  {'CF-Base':>8}"
    print(header)
    print("-" * len(header))

    for model_id in models:
        s = model_scores[model_id]
        b = mean(s["baseline"])
        c = mean(s["conditional"])
        cf = mean(s["counterfactual"])
        x = mean(s["cross"])
        diff = cf - b
        name = short_name(model_id)
        print(f"{name:<{name_width}}  {b:>10.4f}  {c:>10.4f}  {cf:>10.4f}  {x:>10.4f}  {diff:>+8.4f}")

    # Averages across models
    print("-" * len(header))
    avg_b = mean([mean(model_scores[m]["baseline"]) for m in models])
    avg_c = mean([mean(model_scores[m]["conditional"]) for m in models])
    avg_cf = mean([mean(model_scores[m]["counterfactual"]) for m in models])
    avg_x = mean([mean(model_scores[m]["cross"]) for m in models])
    avg_diff = avg_cf - avg_b
    print(f"{'AVERAGE':<{name_width}}  {avg_b:>10.4f}  {avg_c:>10.4f}  {avg_cf:>10.4f}  {avg_x:>10.4f}  {avg_diff:>+8.4f}")

    print()
    print("Legend:")
    print("  Baseline:     baseline_prediction vs baseline_truth  (normal baseline Brier)")
    print("  Cond.:        conditional_prediction vs fork_truth    (normal conditional Brier — the gap)")
    print("  Countfact.:   conditional_prediction vs baseline_truth (KEY: scoring cond. preds against baseline truth)")
    print("  Cross:        baseline_prediction vs fork_truth       (for reference)")
    print("  CF-Base:      Counterfactual minus Baseline           (if ~0, models predict baseline probs on cond. Qs)")
    print()
    print("Interpretation:")
    print("  If CF-Base ≈ 0, models are predicting baseline probabilities on conditional questions.")
    print("  If CF-Base >> 0, models are doing something different on conditional questions")
    print("  (but not necessarily tracking the fork — check Cond. Brier for that).")


if __name__ == "__main__":
    main()
