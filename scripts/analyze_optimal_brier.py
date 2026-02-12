"""Compute per-(template, horizon) base rates and optimal Brier scores.

Compares the model's Brier to a "template-horizon optimal forecaster" that
predicts the per-group base rate for every question. This is a tighter bound
than the aggregate base-rate forecaster used in the initial bounding argument.

Usage:
    uv run python scripts/analyze_optimal_brier.py
"""

import json
import glob
from collections import defaultdict
from pathlib import Path


MODEL = "anthropic/claude-opus-4-5-20251101"

CONDITIONS = {
    "baseline": {
        "eval_path": "data/evaluations/results/baseline_opus45_republic_full_eval.json",
        "source_glob": "data/conditional/republic/baseline/seed*/questions.json",
    },
    "conditional": {
        "eval_path": "data/evaluations/results/conditional_opus45_republic_full_eval.json",
        "source_glob": "data/conditional/republic/conditional/seed*/conditional_questions.json",
    },
}


def load_eval_and_source(eval_path, source_glob):
    """Load eval results and join with source questions to get horizons."""
    eval_data = json.load(open(eval_path))

    # Build source lookup: (game_id, question_id) -> question
    source_lookup = {}
    for qfile in glob.glob(source_glob):
        data = json.load(open(qfile))
        game_id = data["game_id"]
        for q in data["questions"]:
            source_lookup[(game_id, q["question_id"])] = q

    # Join eval questions with source
    questions = []
    missed = 0
    for eq in eval_data["questions"]:
        key = (eq["game_id"], eq["question_id"])
        if key in source_lookup:
            sq = source_lookup[key]
            pred = eq["predictions"].get(MODEL)
            if pred and pred["error"] is None:
                questions.append({
                    "question_id": eq["question_id"],
                    "game_id": eq["game_id"],
                    "template_id": sq["template_id"]
                        .replace("conditional_", "")
                        .replace("null_conditional_", ""),
                    "horizon": sq["horizon"],
                    "resolution_turn": sq["resolution_turn"],
                    "ground_truth": 1 if eq["ground_truth"] else 0,
                    "model_prob": pred["probability"],
                })
        else:
            missed += 1

    if missed:
        print(f"  Warning: {missed} eval questions not found in source files")

    return questions


def compute_analysis(questions):
    """Compute per-(template, horizon) base rates and optimal/model Brier."""
    # Group by (template, horizon)
    groups = defaultdict(list)
    for q in questions:
        groups[(q["template_id"], q["horizon"])].append(q)

    results = []
    for (template, horizon), qs in sorted(groups.items()):
        n = len(qs)
        base_rate = sum(q["ground_truth"] for q in qs) / n

        # Optimal Brier: predict base_rate for every question in this group
        optimal_brier = sum(
            (base_rate - q["ground_truth"]) ** 2 for q in qs
        ) / n

        # Model Brier
        model_brier = sum(
            (q["model_prob"] - q["ground_truth"]) ** 2 for q in qs
        ) / n

        results.append({
            "template": template,
            "horizon": horizon,
            "n": n,
            "base_rate": base_rate,
            "optimal_brier": optimal_brier,
            "model_brier": model_brier,
            "skill": model_brier - optimal_brier,
        })

    return results


def print_table(condition_name, results):
    """Print results table."""
    print(f"\n{'='*90}")
    print(f"  {condition_name.upper()}")
    print(f"{'='*90}")
    print(f"{'Template':<25} {'Horizon':<8} {'n':>4} {'Base Rate':>10} "
          f"{'Optimal':>8} {'Model':>8} {'Skill':>8}")
    print("-" * 90)

    for r in results:
        print(f"{r['template']:<25} {r['horizon']:<8} {r['n']:>4} "
              f"{r['base_rate']:>10.3f} {r['optimal_brier']:>8.3f} "
              f"{r['model_brier']:>8.3f} {r['skill']:>+8.3f}")


def print_summary(condition_name, results):
    """Print weighted summary stats."""
    total_n = sum(r["n"] for r in results)
    avg_optimal = sum(r["optimal_brier"] * r["n"] for r in results) / total_n
    avg_model = sum(r["model_brier"] * r["n"] for r in results) / total_n
    avg_skill = avg_model - avg_optimal

    print(f"\n  Weighted average (n={total_n}):")
    print(f"    Optimal Brier (template-horizon):  {avg_optimal:.4f}")
    print(f"    Model Brier:                       {avg_model:.4f}")
    print(f"    Skill (model - optimal):           {avg_skill:+.4f}")

    return avg_optimal, avg_model, avg_skill


def main():
    all_summaries = {}

    for condition_name, config in CONDITIONS.items():
        print(f"\nLoading {condition_name}...")
        questions = load_eval_and_source(config["eval_path"], config["source_glob"])
        print(f"  Joined {len(questions)} questions")

        results = compute_analysis(questions)
        print_table(condition_name, results)
        avg_optimal, avg_model, avg_skill = print_summary(condition_name, results)
        all_summaries[condition_name] = {
            "optimal": avg_optimal,
            "model": avg_model,
            "skill": avg_skill,
            "results": results,
        }

    # Decomposition
    print(f"\n{'='*90}")
    print("  DIFFICULTY vs CAPABILITY DECOMPOSITION")
    print(f"{'='*90}")

    b_opt = all_summaries["baseline"]["optimal"]
    c_opt = all_summaries["conditional"]["optimal"]
    b_mod = all_summaries["baseline"]["model"]
    c_mod = all_summaries["conditional"]["model"]

    observed_gap = c_mod - b_mod
    difficulty_gap = c_opt - b_opt
    capability_gap = observed_gap - difficulty_gap

    print(f"\n  Observed gap (conditional - baseline model Brier): {observed_gap:+.4f}")
    print(f"  Difficulty gap (conditional - baseline optimal):   {difficulty_gap:+.4f}")
    print(f"  Capability gap (observed - difficulty):            {capability_gap:+.4f}")
    print(f"\n  Difficulty explains: {abs(difficulty_gap / observed_gap) * 100:.1f}% of the gap")
    print(f"  Capability explains: {abs(capability_gap / observed_gap) * 100:.1f}% of the gap")

    print(f"\n  Baseline skill:     {all_summaries['baseline']['skill']:+.4f} "
          f"({'positive' if all_summaries['baseline']['skill'] < 0 else 'NEGATIVE'})")
    print(f"  Conditional skill:  {all_summaries['conditional']['skill']:+.4f} "
          f"({'positive' if all_summaries['conditional']['skill'] < 0 else 'NEGATIVE'})")

    # Comparison to aggregate bound
    print(f"\n  --- Comparison to aggregate bounding argument ---")
    print(f"  Aggregate base-rate bound:      difficulty ≤ 9.5%")
    print(f"  Per-template-horizon bound:     difficulty = {abs(difficulty_gap / observed_gap) * 100:.1f}%")
    if abs(difficulty_gap / observed_gap) < 0.095:
        print(f"  → Tighter bound STRENGTHENS the argument")
    else:
        print(f"  → Tighter bound is less tight than aggregate (unexpected)")


if __name__ == "__main__":
    main()
