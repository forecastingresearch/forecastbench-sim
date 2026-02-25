"""FSG Mechanism Analysis: Gap × Horizon and Gap × Template.

Computes per-horizon and per-template binary Brier gaps for all 10 models
across both interventions. Supports the theoretical argument in the
FSG mechanism behavioral experiment note.

A1: Gap × Causal Distance (per-template binary Brier gaps)
A2: Gap × Temporal Horizon (per-horizon binary Brier gaps)

Usage:
    uv run python scripts/fsg_mechanism_analysis.py
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

# Display order: best to worst gap (roughly)
MODEL_ORDER = [
    "openai/o3-2025-04-16",
    "openai/gpt-5-mini-2025-08-07",
    "anthropic/claude-opus-4-5-20251101",
    "openai/gpt-5-2025-08-07",
    "anthropic/claude-sonnet-4-5-20250929",
    "openai/gpt-5.1-2025-11-13",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-3-pro-preview",
    "openai/gpt-4.1-2025-04-14",
]

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
    """Join eval questions with source to get horizon and template."""
    eval_data = json.load(open(eval_path))
    questions = []
    missed = 0

    for eq in eval_data["questions"]:
        if eq["question_type"] != "binary":
            continue
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


def compute_brier_by_group(questions, model_id, group_key):
    """Compute Brier score grouped by a key ('horizon' or 'template_id').

    Returns dict of group_value -> {brier, n, base_rate}.
    """
    groups = defaultdict(list)
    for q in questions:
        pred = q["predictions"].get(model_id)
        if pred and pred.get("error") is None:
            groups[q[group_key]].append({
                "prob": pred["probability"],
                "ground_truth": q["ground_truth"],
            })

    results = {}
    for group_val, qs in sorted(groups.items()):
        n = len(qs)
        brier = sum((q["prob"] - q["ground_truth"]) ** 2 for q in qs) / n
        base_rate = sum(q["ground_truth"] for q in qs) / n
        results[group_val] = {"brier": brier, "n": n, "base_rate": base_rate}

    return results


def analyze_by_horizon(bl_questions, cd_questions, models):
    """A2: Compute per-horizon Brier gaps across all models."""
    print("\n" + "=" * 90)
    print("  A2: BINARY BRIER GAP × TEMPORAL HORIZON")
    print("=" * 90)

    horizons = sorted(set(q["horizon"] for q in cd_questions))
    print(f"  Horizons: {horizons}")
    print(f"  n baseline: {len(bl_questions)}, n conditional: {len(cd_questions)}")

    # Per-model, per-horizon
    all_model_data = {}
    for model_id in models:
        short = MODEL_NAMES.get(model_id, model_id.split("/")[-1])
        bl_by_h = compute_brier_by_group(bl_questions, model_id, "horizon")
        cd_by_h = compute_brier_by_group(cd_questions, model_id, "horizon")
        all_model_data[model_id] = {"bl": bl_by_h, "cd": cd_by_h}

    # Print header
    print(f"\n  {'Model':<16}", end="")
    for h in horizons:
        print(f" {h:>8}", end="")
    print(f" {'Trend':>8}")
    print(f"  {'-' * (16 + 9 * len(horizons) + 9)}")

    # Print per-model rows (gap = CD - BL Brier)
    avg_gaps = defaultdict(list)
    for model_id in MODEL_ORDER:
        if model_id not in all_model_data:
            continue
        short = MODEL_NAMES.get(model_id, model_id.split("/")[-1])
        data = all_model_data[model_id]
        print(f"  {short:<16}", end="")
        gaps = []
        for h in horizons:
            bl = data["bl"].get(h, {}).get("brier", float("nan"))
            cd = data["cd"].get(h, {}).get("brier", float("nan"))
            gap = cd - bl
            gaps.append(gap)
            avg_gaps[h].append(gap)
            print(f" {gap:>+8.3f}", end="")

        # Trend: is gap increasing with horizon?
        if len(gaps) >= 2:
            # Simple: compare first half avg to second half avg
            mid = len(gaps) // 2
            first_half = sum(gaps[:mid]) / mid
            second_half = sum(gaps[mid:]) / (len(gaps) - mid)
            trend = "UP" if second_half > first_half * 1.05 else (
                "DOWN" if second_half < first_half * 0.95 else "FLAT")
        else:
            trend = "?"
        print(f" {trend:>8}")

    # Average row
    print(f"  {'-' * (16 + 9 * len(horizons) + 9)}")
    print(f"  {'AVERAGE':<16}", end="")
    avg_gap_list = []
    for h in horizons:
        avg = sum(avg_gaps[h]) / len(avg_gaps[h])
        avg_gap_list.append(avg)
        print(f" {avg:>+8.3f}", end="")

    if len(avg_gap_list) >= 2:
        mid = len(avg_gap_list) // 2
        first_half = sum(avg_gap_list[:mid]) / mid
        second_half = sum(avg_gap_list[mid:]) / (len(avg_gap_list) - mid)
        trend = "UP" if second_half > first_half * 1.05 else (
            "DOWN" if second_half < first_half * 0.95 else "FLAT")
    else:
        trend = "?"
    print(f" {trend:>8}")

    # Also print n per horizon and base rates
    print(f"\n  --- Question counts and base rates per horizon ---")
    print(f"  {'Horizon':<8} {'BL n':>6} {'BL BR':>7} {'CD n':>6} {'CD BR':>7} {'ΔBR':>7}")
    for h in horizons:
        # Use first model's data for counts (all models see same questions)
        first_model = MODEL_ORDER[0]
        bl = all_model_data[first_model]["bl"].get(h, {"n": 0, "base_rate": 0})
        cd = all_model_data[first_model]["cd"].get(h, {"n": 0, "base_rate": 0})
        delta_br = cd["base_rate"] - bl["base_rate"]
        print(f"  {h:<8} {bl['n']:>6} {bl['base_rate']:>7.3f} {cd['n']:>6} {cd['base_rate']:>7.3f} {delta_br:>+7.3f}")

    return avg_gap_list, horizons


def analyze_by_template(bl_questions, cd_questions, models):
    """A1: Compute per-template binary Brier gaps across all models."""
    print("\n" + "=" * 90)
    print("  A1: BINARY BRIER GAP × QUESTION TEMPLATE")
    print("=" * 90)

    templates = sorted(set(q["template_id"] for q in cd_questions))
    print(f"  Templates: {templates}")

    # Per-model, per-template
    all_model_data = {}
    for model_id in models:
        bl_by_t = compute_brier_by_group(bl_questions, model_id, "template_id")
        cd_by_t = compute_brier_by_group(cd_questions, model_id, "template_id")
        all_model_data[model_id] = {"bl": bl_by_t, "cd": cd_by_t}

    # Compute average gap per template across models
    avg_gaps = {}
    for t in templates:
        gaps = []
        for model_id in MODEL_ORDER:
            if model_id not in all_model_data:
                continue
            bl = all_model_data[model_id]["bl"].get(t, {}).get("brier", float("nan"))
            cd = all_model_data[model_id]["cd"].get(t, {}).get("brier", float("nan"))
            if bl == bl and cd == cd:  # not nan
                gaps.append(cd - bl)
        avg_gaps[t] = sum(gaps) / len(gaps) if gaps else float("nan")

    # Sort templates by average gap (largest first)
    templates_sorted = sorted(templates, key=lambda t: avg_gaps.get(t, 0), reverse=True)

    # Short template names for display
    short_templates = {
        "treasury_comparative": "treasury",
        "territory_comparative": "territory",
        "score_comparative": "score",
        "score_rank_1": "score_rank",
        "population_comparative": "population",
        "city_count_comparative": "city_count",
        "tech_comparative": "tech",
        "tech_discovered": "tech_disc",
        "government_at": "govt",
        "wonder_completed": "wonder",
    }

    # Print table: models as rows, templates as columns
    print(f"\n  Brier gap (CD - BL) per template, sorted by avg gap (largest first)")
    print(f"\n  {'Model':<16}", end="")
    for t in templates_sorted:
        st = short_templates.get(t, t[:10])
        print(f" {st:>10}", end="")
    print()
    print(f"  {'-' * (16 + 11 * len(templates_sorted))}")

    for model_id in MODEL_ORDER:
        if model_id not in all_model_data:
            continue
        short = MODEL_NAMES.get(model_id, model_id.split("/")[-1])
        data = all_model_data[model_id]
        print(f"  {short:<16}", end="")
        for t in templates_sorted:
            bl = data["bl"].get(t, {}).get("brier", float("nan"))
            cd = data["cd"].get(t, {}).get("brier", float("nan"))
            if bl == bl and cd == cd:
                gap = cd - bl
                print(f" {gap:>+10.3f}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print()

    print(f"  {'-' * (16 + 11 * len(templates_sorted))}")
    print(f"  {'AVERAGE':<16}", end="")
    for t in templates_sorted:
        print(f" {avg_gaps[t]:>+10.3f}", end="")
    print()

    # Base rates per template
    print(f"\n  --- Base rates per template ---")
    print(f"  {'Template':<25} {'BL n':>6} {'BL BR':>7} {'CD n':>6} {'CD BR':>7} {'ΔBR':>7} {'Flip%':>7}")
    first_model = MODEL_ORDER[0]
    for t in templates_sorted:
        bl = all_model_data[first_model]["bl"].get(t, {"n": 0, "base_rate": 0})
        cd = all_model_data[first_model]["cd"].get(t, {"n": 0, "base_rate": 0})
        delta_br = cd["base_rate"] - bl["base_rate"]
        # Flip rate: how much did the base rate change?
        flip_pct = abs(delta_br) / max(bl["base_rate"], 1 - bl["base_rate"], 0.01) * 100
        print(f"  {short_templates.get(t, t):<25} {bl['n']:>6} {bl['base_rate']:>7.3f} "
              f"{cd['n']:>6} {cd['base_rate']:>7.3f} {delta_br:>+7.3f} {flip_pct:>6.1f}%")

    return templates_sorted, avg_gaps


def main():
    for intervention_name, config in INTERVENTIONS.items():
        print(f"\n{'#' * 90}")
        print(f"  INTERVENTION: {intervention_name.upper()}")
        print(f"{'#' * 90}")

        # Build lookups and join
        bl_lookup = build_source_lookup(config["source_baseline"])
        cd_lookup = build_source_lookup(config["source_conditional"])
        bl_questions = join_eval_with_source(config["eval_baseline"], bl_lookup)
        cd_questions = join_eval_with_source(config["eval_conditional"], cd_lookup)

        # Match baseline to conditional seeds
        cd_seeds = set(q["game_id"] for q in cd_questions)
        cd_horizons = set(q["horizon"] for q in cd_questions)
        bl_matched = [q for q in bl_questions
                      if q["game_id"] in cd_seeds and q["horizon"] in cd_horizons]

        models = [m for m in MODEL_ORDER if m in bl_matched[0]["predictions"]]
        print(f"  Models: {len(models)}")
        print(f"  Baseline (matched): {len(bl_matched)}, Conditional: {len(cd_questions)}")

        # A1: Per-template
        analyze_by_template(bl_matched, cd_questions, models)

        # A2: Per-horizon
        analyze_by_horizon(bl_matched, cd_questions, models)


if __name__ == "__main__":
    main()
