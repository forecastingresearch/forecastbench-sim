"""Per-(template, horizon) optimal Brier analysis across 10 models.

Extends analyze_optimal_brier.py (single-model Opus 4.5) to all 10 models
from the multi-model conditional gap evaluation. Computes neg_skill
(model Brier - optimal Brier) for both baseline and conditional, which
factors out irreducible uncertainty and enables direct comparison.

Usage:
    uv run python scripts/analyze_optimal_brier_multimodel.py
"""

import json
import glob
from collections import defaultdict
from pathlib import Path


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

INTERVENTIONS = {
    "republic": {
        "eval_baseline": "data/results/republic_baseline_binary_all.json",
        "eval_conditional": "data/results/republic_conditional_binary_all.json",
        "source_baseline": "data/conditional/republic/baseline/seed*/questions.json",
        "source_conditional": "data/conditional/republic/conditional/seed*/conditional_questions.json",
    },
    "gold500": {
        "eval_baseline": "data/results/gold500_baseline_binary_all.json",
        "eval_conditional": "data/results/gold500_conditional_binary_all.json",
        "source_baseline": "data/conditional/gold500/baseline/seed*/questions.json",
        "source_conditional": "data/conditional/gold500/conditional/seed*/conditional_questions.json",
    },
}


def build_source_lookup(source_glob):
    """Build (game_id, question_id) -> {horizon, template_id} from source files."""
    lookup = {}
    for qfile in glob.glob(source_glob):
        data = json.load(open(qfile))
        game_id = data["game_id"]
        for q in data["questions"]:
            if q["question_type"] == "binary":
                lookup[(game_id, q["question_id"])] = {
                    "horizon": q["horizon"],
                    "template_id": q["template_id"],
                }
    return lookup


def join_eval_with_source(eval_path, source_lookup):
    """Join eval questions with source to get horizon. Returns list of enriched questions."""
    eval_data = json.load(open(eval_path))
    questions = []
    missed = 0

    for eq in eval_data["questions"]:
        key = (eq["game_id"], eq["question_id"])
        if key in source_lookup:
            src = source_lookup[key]
            template = (
                src["template_id"]
                .replace("conditional_", "")
                .replace("null_conditional_", "")
            )
            questions.append({
                "question_id": eq["question_id"],
                "game_id": eq["game_id"],
                "template_id": template,
                "horizon": src["horizon"],
                "ground_truth": 1 if eq["ground_truth"] else 0,
                "predictions": eq["predictions"],
            })
        else:
            missed += 1

    if missed:
        print(f"  Warning: {missed} eval questions not found in source files")

    return questions


def compute_optimal_brier(questions):
    """Compute per-(template, horizon) optimal Brier (base-rate forecaster)."""
    groups = defaultdict(list)
    for q in questions:
        groups[(q["template_id"], q["horizon"])].append(q)

    group_stats = {}
    for (template, horizon), qs in sorted(groups.items()):
        n = len(qs)
        base_rate = sum(q["ground_truth"] for q in qs) / n
        optimal_brier = base_rate * (1 - base_rate)
        group_stats[(template, horizon)] = {
            "n": n,
            "base_rate": base_rate,
            "optimal_brier": optimal_brier,
        }

    # Weighted average
    total_n = sum(g["n"] for g in group_stats.values())
    avg_optimal = sum(g["optimal_brier"] * g["n"] for g in group_stats.values()) / total_n

    return group_stats, avg_optimal


def compute_model_brier(questions, model_id):
    """Compute per-(template, horizon) model Brier for one model."""
    groups = defaultdict(list)
    for q in questions:
        pred = q["predictions"].get(model_id)
        if pred and pred.get("error") is None:
            groups[(q["template_id"], q["horizon"])].append({
                "prob": pred["probability"],
                "ground_truth": q["ground_truth"],
            })

    group_briers = {}
    for (template, horizon), qs in sorted(groups.items()):
        n = len(qs)
        brier = sum((q["prob"] - q["ground_truth"]) ** 2 for q in qs) / n
        group_briers[(template, horizon)] = {"n": n, "brier": brier}

    total_n = sum(g["n"] for g in group_briers.values())
    if total_n == 0:
        return group_briers, float("nan")
    avg_brier = sum(g["brier"] * g["n"] for g in group_briers.values()) / total_n

    return group_briers, avg_brier


def match_baseline_to_conditional(bl_questions, cd_questions):
    """Filter baseline to match conditional seeds and horizons."""
    cd_seeds = set(q["game_id"] for q in cd_questions)
    cd_horizons = set(q["horizon"] for q in cd_questions)
    matched = [q for q in bl_questions
                if q["game_id"] in cd_seeds and q["horizon"] in cd_horizons]
    return matched


def analyze_intervention(name, config):
    """Run full analysis for one intervention (Republic or Gold)."""
    print(f"\n{'='*90}")
    print(f"  {name.upper()}")
    print(f"{'='*90}")

    # Build source lookups
    bl_lookup = build_source_lookup(config["source_baseline"])
    cd_lookup = build_source_lookup(config["source_conditional"])
    print(f"  Source lookup: {len(bl_lookup)} baseline, {len(cd_lookup)} conditional questions")

    # Join eval with source
    bl_questions_full = join_eval_with_source(config["eval_baseline"], bl_lookup)
    cd_questions = join_eval_with_source(config["eval_conditional"], cd_lookup)

    # Match baseline to conditional seeds/horizons
    bl_questions = match_baseline_to_conditional(bl_questions_full, cd_questions)
    print(f"  Joined: {len(bl_questions_full)} baseline (full), "
          f"{len(bl_questions)} baseline (matched), {len(cd_questions)} conditional")

    cd_seeds = sorted(set(q["game_id"] for q in cd_questions))
    cd_horizons = sorted(set(q["horizon"] for q in cd_questions))
    print(f"  Matched seeds: {cd_seeds}")
    print(f"  Matched horizons: {cd_horizons}")

    # Compute optimal Brier
    bl_groups, bl_optimal = compute_optimal_brier(bl_questions)
    cd_groups, cd_optimal = compute_optimal_brier(cd_questions)

    print(f"\n  --- Optimal Brier (per-template-horizon base-rate forecaster) ---")
    print(f"  Baseline optimal (matched):  {bl_optimal:.4f}")
    print(f"  Conditional optimal:         {cd_optimal:.4f}")
    print(f"  Difficulty gap (CD - BL):    {cd_optimal - bl_optimal:+.4f}")
    if cd_optimal <= bl_optimal:
        print(f"  Conditional questions are NOT harder for an optimal forecaster")
    else:
        print(f"  Conditional questions are harder (+{(cd_optimal - bl_optimal):.4f} irreducible Brier)")

    # Per-model analysis
    models = list(bl_questions[0]["predictions"].keys())

    print(f"\n  --- Per-model neg_skill (model Brier - optimal Brier) ---")
    print(f"  neg_skill factors out irreducible uncertainty → direct comparison is valid")
    print()
    print(f"  {'Model':<16} {'BL Brier':>9} {'BL neg_sk':>10} {'CD Brier':>9} "
          f"{'CD neg_sk':>10} {'Ratio':>7} {'CD > BL?':>9}")
    print(f"  {'-'*75}")

    results = []
    for model_id in models:
        short = MODEL_NAMES.get(model_id, model_id.split("/")[-1])
        _, bl_brier = compute_model_brier(bl_questions, model_id)
        _, cd_brier = compute_model_brier(cd_questions, model_id)

        bl_neg_skill = bl_brier - bl_optimal
        cd_neg_skill = cd_brier - cd_optimal
        ratio = cd_neg_skill / bl_neg_skill if bl_neg_skill > 0 else float("inf")

        # neg_skill comparison: does the model have more excess error on conditionals?
        proved = "YES" if cd_neg_skill > bl_neg_skill else "no"

        results.append({
            "model": short,
            "bl_brier": bl_brier,
            "cd_brier": cd_brier,
            "bl_neg_skill": bl_neg_skill,
            "cd_neg_skill": cd_neg_skill,
            "ratio": ratio,
            "proved": proved,
        })

        print(f"  {short:<16} {bl_brier:>9.4f} {bl_neg_skill:>+10.4f} "
              f"{cd_brier:>9.4f} {cd_neg_skill:>+10.4f} {ratio:>6.1f}x  {proved:>7}")

    # Decomposition
    all_proved = all(r["proved"] == "YES" for r in results)
    n_proved = sum(1 for r in results if r["proved"] == "YES")
    print(f"\n  {n_proved}/{len(results)} models: CD neg_skill > BL neg_skill")

    # Difficulty vs capability decomposition (using mean across models)
    avg_bl_brier = sum(r["bl_brier"] for r in results) / len(results)
    avg_cd_brier = sum(r["cd_brier"] for r in results) / len(results)
    observed_gap = avg_cd_brier - avg_bl_brier
    difficulty_gap = cd_optimal - bl_optimal
    capability_gap = observed_gap - difficulty_gap

    print(f"\n  --- Difficulty vs capability decomposition (avg across models) ---")
    print(f"  Observed Brier gap (CD - BL):      {observed_gap:+.4f} (100%)")
    print(f"  Difficulty gap (optimal CD - BL):   {difficulty_gap:+.4f} "
          f"({abs(difficulty_gap / observed_gap) * 100:.1f}%)")
    print(f"  Capability gap (remainder):         {capability_gap:+.4f} "
          f"({abs(capability_gap / observed_gap) * 100:.1f}%)")

    if all_proved:
        print(f"\n  ✓ ALL {len(results)} models have higher excess error on conditionals")
        print(f"    → conditional gap is a real capability deficit for every model tested")

    return {
        "bl_optimal": bl_optimal,
        "cd_optimal": cd_optimal,
        "difficulty_gap": difficulty_gap,
        "capability_pct": abs(capability_gap / observed_gap) * 100,
        "results": results,
        "all_proved": all_proved,
    }


def print_group_detail(name, config):
    """Print per-(template, horizon) breakdown for inspection."""
    bl_lookup = build_source_lookup(config["source_baseline"])
    cd_lookup = build_source_lookup(config["source_conditional"])
    bl_questions_full = join_eval_with_source(config["eval_baseline"], bl_lookup)
    cd_questions = join_eval_with_source(config["eval_conditional"], cd_lookup)
    bl_questions = match_baseline_to_conditional(bl_questions_full, cd_questions)

    bl_groups, _ = compute_optimal_brier(bl_questions)
    cd_groups, _ = compute_optimal_brier(cd_questions)

    print(f"\n  --- Per-group optimal Brier detail ({name}) ---")
    print(f"  {'Template':<25} {'H':>3} {'BL n':>5} {'BL BR':>6} {'BL opt':>7} "
          f"{'CD n':>5} {'CD BR':>6} {'CD opt':>7} {'Δ opt':>7}")
    print(f"  {'-'*80}")

    # Get all templates (normalize conditional prefix)
    all_templates = sorted(set(t for t, h in bl_groups) | set(t for t, h in cd_groups))
    all_horizons = sorted(set(h for t, h in bl_groups) | set(h for t, h in cd_groups))

    for template in all_templates:
        for horizon in all_horizons:
            bl = bl_groups.get((template, horizon))
            cd = cd_groups.get((template, horizon))
            if bl or cd:
                bl_n = bl["n"] if bl else 0
                bl_br = bl["base_rate"] if bl else float("nan")
                bl_opt = bl["optimal_brier"] if bl else float("nan")
                cd_n = cd["n"] if cd else 0
                cd_br = cd["base_rate"] if cd else float("nan")
                cd_opt = cd["optimal_brier"] if cd else float("nan")
                delta = (cd_opt - bl_opt) if (bl and cd) else float("nan")

                print(f"  {template:<25} {horizon:>3} {bl_n:>5} {bl_br:>6.3f} {bl_opt:>7.4f} "
                      f"{cd_n:>5} {cd_br:>6.3f} {cd_opt:>7.4f} {delta:>+7.4f}")


def main():
    summaries = {}
    for name, config in INTERVENTIONS.items():
        summaries[name] = analyze_intervention(name, config)
        print_group_detail(name, config)

    # Final summary
    print(f"\n{'='*90}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*90}")
    for name, s in summaries.items():
        status = "ALL PROVED" if s["all_proved"] else "INCOMPLETE"
        print(f"  {name}: difficulty={s['difficulty_gap']:+.4f}, "
              f"capability={s['capability_pct']:.1f}% of gap, "
              f"neg_skill proves gap for {'all' if s['all_proved'] else 'some'} models → {status}")


if __name__ == "__main__":
    main()
