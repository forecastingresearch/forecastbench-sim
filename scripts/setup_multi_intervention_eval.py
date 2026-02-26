#!/usr/bin/env python3
"""
Set up evaluation directories for Multi-Intervention Scaling experiment.

Creates evaluation data for two compound-intervention conditions:
1. N=2: Republic + Gold+500
2. N=3: Republic + Gold+500 + Map Making

For each condition, creates:
- baseline/   — Unconditional framing, control (baseline) answer
- conditional/ — Compound conditional framing, intervention answer

Also sets up N=1 forks for seeds 0-10 (republic, gold500, mapmaking) if those
fork results exist but haven't been set up as eval directories yet.

Usage:
    python scripts/setup_multi_intervention_eval.py
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ALL_SEEDS = [f"seed{i}" for i in range(21)]

# === Fork directories ===

N2_FORKS = {
    f"seed{i}": f"logs/recordings/seed{i}forkgovRepublic_goldadd500_p0"
    for i in range(21)
}

N3_FORKS = {
    f"seed{i}": f"logs/recordings/seed{i}forkgovRepublic_goldadd500_tech45_p0"
    for i in range(21)
}

# === Question templates ===

# Baseline (unconditional) — shared across all conditions
BASELINE_TEXT_TEMPLATES = {
    "treasury_comparative": "Will {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?",
    "score_comparative": "Will {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?",
    "tech_comparative": "Will {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?",
    "population_comparative": "Will {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?",
    "city_count_comparative": "Will {civ_a} have more cities than {civ_b} at turn {resolution_turn}?",
    "territory_comparative": "Will {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?",
    "score_rank_1": "Will {civ} be ranked #1 at turn {resolution_turn}?",
    "tech_discovered": "Will {civ} have discovered {tech_name} by turn {resolution_turn}?",
    "wonder_completed": "Will {wonder_name} be completed by any civilization by turn {resolution_turn}?",
    "government_at": "Will {civ} be in {government_type} at turn {resolution_turn}?",
}

BASELINE_CONTINUOUS_TEXT_TEMPLATES = {
    "techs_continuous": "How many technologies will {civ} have discovered by turn {resolution_turn}?",
    "treasury_continuous": "How much gold will {civ} have at turn {resolution_turn}?",
    "population_continuous": "What will {civ}'s population be at turn {resolution_turn}?",
    "cities_count_continuous": "How many cities will {civ} have at turn {resolution_turn}?",
    "territory_continuous": "How many tiles will {civ} control at turn {resolution_turn}?",
    "scores_continuous": "What will {civ}'s score be at turn {resolution_turn}?",
}

# N=2: Republic + Gold+500
N2_CONDITIONAL_TEXT_TEMPLATES = {
    "treasury_comparative": "If {civ_a} adopts Republic AND receives +500 gold next turn, would {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?",
    "score_comparative": "If {civ_a} adopts Republic AND receives +500 gold next turn, would {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?",
    "tech_comparative": "If {civ_a} adopts Republic AND receives +500 gold next turn, would {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?",
    "population_comparative": "If {civ_a} adopts Republic AND receives +500 gold next turn, would {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?",
    "city_count_comparative": "If {civ_a} adopts Republic AND receives +500 gold next turn, would {civ_a} have more cities than {civ_b} at turn {resolution_turn}?",
    "territory_comparative": "If {civ_a} adopts Republic AND receives +500 gold next turn, would {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?",
    "score_rank_1": "If {civ} adopts Republic AND receives +500 gold next turn, would {civ} be ranked #1 at turn {resolution_turn}?",
    "tech_discovered": "If {civ} adopts Republic AND receives +500 gold next turn, would {civ} have discovered {tech_name} by turn {resolution_turn}?",
    "wonder_completed": "If the civilization adopts Republic AND receives +500 gold next turn, would {wonder_name} be completed by any civilization by turn {resolution_turn}?",
    "government_at": "If {civ} adopts Republic AND receives +500 gold next turn, would {civ} be in {government_type} at turn {resolution_turn}?",
}

N2_CONDITIONAL_CONTINUOUS_TEXT_TEMPLATES = {
    "techs_continuous": "If {civ} adopts Republic AND receives +500 gold next turn, how many technologies will {civ} have discovered by turn {resolution_turn}?",
    "treasury_continuous": "If {civ} adopts Republic AND receives +500 gold next turn, how much gold will {civ} have at turn {resolution_turn}?",
    "population_continuous": "If {civ} adopts Republic AND receives +500 gold next turn, what will {civ}'s population be at turn {resolution_turn}?",
    "cities_count_continuous": "If {civ} adopts Republic AND receives +500 gold next turn, how many cities will {civ} have at turn {resolution_turn}?",
    "territory_continuous": "If {civ} adopts Republic AND receives +500 gold next turn, how many tiles will {civ} control at turn {resolution_turn}?",
    "scores_continuous": "If {civ} adopts Republic AND receives +500 gold next turn, what will {civ}'s score be at turn {resolution_turn}?",
}

# N=3: Republic + Gold+500 + Map Making
N3_CONDITIONAL_TEXT_TEMPLATES = {
    "treasury_comparative": "If {civ_a} adopts Republic, receives +500 gold, AND discovers Map Making next turn, would {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?",
    "score_comparative": "If {civ_a} adopts Republic, receives +500 gold, AND discovers Map Making next turn, would {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?",
    "tech_comparative": "If {civ_a} adopts Republic, receives +500 gold, AND discovers Map Making next turn, would {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?",
    "population_comparative": "If {civ_a} adopts Republic, receives +500 gold, AND discovers Map Making next turn, would {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?",
    "city_count_comparative": "If {civ_a} adopts Republic, receives +500 gold, AND discovers Map Making next turn, would {civ_a} have more cities than {civ_b} at turn {resolution_turn}?",
    "territory_comparative": "If {civ_a} adopts Republic, receives +500 gold, AND discovers Map Making next turn, would {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?",
    "score_rank_1": "If {civ} adopts Republic, receives +500 gold, AND discovers Map Making next turn, would {civ} be ranked #1 at turn {resolution_turn}?",
    "tech_discovered": "If {civ} adopts Republic, receives +500 gold, AND discovers Map Making next turn, would {civ} have discovered {tech_name} by turn {resolution_turn}?",
    "wonder_completed": "If the civilization adopts Republic, receives +500 gold, AND discovers Map Making next turn, would {wonder_name} be completed by any civilization by turn {resolution_turn}?",
    "government_at": "If {civ} adopts Republic, receives +500 gold, AND discovers Map Making next turn, would {civ} be in {government_type} at turn {resolution_turn}?",
}

N3_CONDITIONAL_CONTINUOUS_TEXT_TEMPLATES = {
    "techs_continuous": "If {civ} adopts Republic, receives +500 gold, AND discovers Map Making next turn, how many technologies will {civ} have discovered by turn {resolution_turn}?",
    "treasury_continuous": "If {civ} adopts Republic, receives +500 gold, AND discovers Map Making next turn, how much gold will {civ} have at turn {resolution_turn}?",
    "population_continuous": "If {civ} adopts Republic, receives +500 gold, AND discovers Map Making next turn, what will {civ}'s population be at turn {resolution_turn}?",
    "cities_count_continuous": "If {civ} adopts Republic, receives +500 gold, AND discovers Map Making next turn, how many cities will {civ} have at turn {resolution_turn}?",
    "territory_continuous": "If {civ} adopts Republic, receives +500 gold, AND discovers Map Making next turn, how many tiles will {civ} control at turn {resolution_turn}?",
    "scores_continuous": "If {civ} adopts Republic, receives +500 gold, AND discovers Map Making next turn, what will {civ}'s score be at turn {resolution_turn}?",
}


def get_horizon(checkpoint_turn: int, resolution_turn: int) -> str:
    """Calculate horizon based on turn difference."""
    diff = resolution_turn - checkpoint_turn
    if diff <= 30:
        return "H1"
    elif diff <= 60:
        return "H2"
    else:
        return "H3"


def format_question_text(template_id: str, params: dict, text_templates: dict) -> str:
    """Format question text using the appropriate template."""
    text_template = text_templates.get(template_id)
    if text_template:
        try:
            params_with_civ = dict(params)
            if "civ" not in params_with_civ:
                params_with_civ["civ"] = params.get("civ_a", "the civilization")
            return text_template.format(**params_with_civ)
        except KeyError as e:
            return f"Question about {template_id} (missing param: {e})"
    return f"Question about {template_id}"


def generate_questions_from_conditional_results(
    conditional_results_path: Path,
    game_id: str,
    civilizations: dict,
    condition_type: str,  # "baseline" or "conditional"
    conditional_text_templates: dict,
    conditional_continuous_text_templates: dict,
    condition_metadata: dict,
) -> dict:
    """
    Generate questions from conditional_results.json.

    Args:
        conditional_results_path: Path to conditional_results.json
        game_id: Game identifier
        civilizations: Civilization info
        condition_type: "baseline" for unconditional framing + control answer,
                       "conditional" for compound conditional framing + intervention answer
        conditional_text_templates: Templates for conditional binary questions
        conditional_continuous_text_templates: Templates for conditional continuous questions
        condition_metadata: Metadata about the condition for the output

    Returns:
        Question bank dict ready for evaluation
    """
    with open(conditional_results_path) as f:
        cond_data = json.load(f)

    results = cond_data.get("results", {})
    checkpoint_turn = cond_data.get("checkpoint_turn", 60)

    if condition_type == "baseline":
        text_templates = BASELINE_TEXT_TEMPLATES
        continuous_text_templates = BASELINE_CONTINUOUS_TEXT_TEMPLATES
        answer_key = "answer_control"
        template_prefix = ""
        question_id_suffix = "_control"
    else:
        text_templates = conditional_text_templates
        continuous_text_templates = conditional_continuous_text_templates
        answer_key = "answer_intervention"
        template_prefix = "conditional_"
        question_id_suffix = "_intervention"

    questions = []
    skipped = 0

    for q in cond_data.get("questions", []):
        cond_id = q["conditional_id"]
        result = results.get(cond_id, {})

        answer = result.get(answer_key)
        if answer is None:
            skipped += 1
            continue

        template_id = q["target_template_id"]
        params = q["target_parameters"]
        resolution_turn = q["resolution_turn"]

        is_continuous = template_id.endswith("_continuous")

        templates_to_use = continuous_text_templates if is_continuous else text_templates
        question_text = format_question_text(template_id, params, templates_to_use)
        horizon = get_horizon(checkpoint_turn, resolution_turn)

        question_entry = {
            "question_id": f"{cond_id}{question_id_suffix}",
            "template_id": f"{template_prefix}{template_id}",
            "resolution_turn": resolution_turn,
            "horizon": horizon,
            "question_type": "continuous" if is_continuous else "binary",
            "parameters": {
                **params,
                "checkpoint_turn": checkpoint_turn,
            },
            "question_text": question_text,
            "resolution": {"value_at_resolution": answer} if is_continuous else {"answer": answer},
        }
        questions.append(question_entry)

    if skipped > 0:
        print(f"      Skipped {skipped} questions with missing {answer_key}")

    return {
        "game_id": game_id,
        "snapshot_turn": checkpoint_turn,
        "game_max_turn": cond_data.get("end_turn", 270),
        "generated_at": datetime.now().isoformat() + "Z",
        "civilizations": civilizations,
        "questions": questions,
        "condition_metadata": condition_metadata,
    }


def get_civilizations(seed: str, base_dir: Path) -> dict | None:
    """Get civilizations info from questions.json or game_data.json."""
    # Try questions.json first (seeds 11-20)
    questions_path = base_dir / "data" / "questions" / seed / "questions.json"
    if questions_path.exists():
        with open(questions_path) as f:
            return json.load(f).get("civilizations", {})

    # Fall back to game_data.json (seeds 0-10)
    seed_num = seed.replace("seed", "")
    game_data_path = base_dir / "data" / "games" / f"{seed}_data.json"
    if game_data_path.exists():
        with open(game_data_path) as f:
            return json.load(f).get("civilizations", {})

    return None


def get_world_report_dir(seed: str, base_dir: Path) -> Path | None:
    """Find world_report directory for a seed."""
    # Try standard location (seeds 11-20)
    wr = base_dir / "data" / "questions" / seed / "world_report"
    if wr.exists():
        return wr

    # Try nested location (seeds 0-10)
    wr = base_dir / "data" / "questions" / "data" / "questions" / seed / "world_report"
    if wr.exists():
        return wr

    return None


def setup_condition(
    condition_name: str,
    fork_dirs: dict,
    conditional_text_templates: dict,
    conditional_continuous_text_templates: dict,
    condition_metadata: dict,
    base_dir: Path,
) -> tuple[int, int]:
    """Set up evaluation directories for one compound condition.

    Returns:
        (total_baseline, total_conditional) question counts
    """
    conditional_dir = base_dir / "data" / "conditional" / condition_name / "conditional"
    baseline_dir = base_dir / "data" / "conditional" / condition_name / "baseline"

    total_conditional = 0
    total_baseline = 0

    for seed, fork_rel in fork_dirs.items():
        fork_path = base_dir / fork_rel
        conditional_results_path = fork_path / "conditional_results.json"

        if not conditional_results_path.exists():
            print(f"  {seed}: Skipping - no conditional_results.json")
            continue

        civilizations = get_civilizations(seed, base_dir)
        if civilizations is None:
            print(f"  {seed}: Skipping - no civilizations data")
            continue

        world_report_dir = get_world_report_dir(seed, base_dir)
        if world_report_dir is None:
            print(f"  {seed}: Skipping - no world_report")
            continue

        print(f"  Processing {seed}...")

        # 1. Baseline questions (unconditional framing, control answer)
        baseline_seed_dir = baseline_dir / seed
        baseline_seed_dir.mkdir(parents=True, exist_ok=True)
        baseline_questions = generate_questions_from_conditional_results(
            conditional_results_path, seed, civilizations, "baseline",
            conditional_text_templates, conditional_continuous_text_templates,
            condition_metadata,
        )
        with open(baseline_seed_dir / "questions.json", "w") as f:
            json.dump(baseline_questions, f, indent=2)
        if (baseline_seed_dir / "world_report").exists():
            shutil.rmtree(baseline_seed_dir / "world_report")
        shutil.copytree(world_report_dir, baseline_seed_dir / "world_report")
        n_baseline = len(baseline_questions.get("questions", []))
        total_baseline += n_baseline

        # 2. Conditional questions (compound intervention framing, intervention answer)
        cond_seed_dir = conditional_dir / seed
        cond_seed_dir.mkdir(parents=True, exist_ok=True)
        cond_questions = generate_questions_from_conditional_results(
            conditional_results_path, seed, civilizations, "conditional",
            conditional_text_templates, conditional_continuous_text_templates,
            condition_metadata,
        )
        with open(cond_seed_dir / "conditional_questions.json", "w") as f:
            json.dump(cond_questions, f, indent=2)
        if (cond_seed_dir / "world_report").exists():
            shutil.rmtree(cond_seed_dir / "world_report")
        shutil.copytree(world_report_dir, cond_seed_dir / "world_report")
        n_conditional = len(cond_questions.get("questions", []))
        total_conditional += n_conditional

        print(f"    Baseline: {n_baseline}, Conditional: {n_conditional}")

    return total_baseline, total_conditional


def main():
    base_dir = Path(__file__).parent.parent

    # === N=2: Republic + Gold+500 ===
    print("=" * 60)
    print("N=2: Republic + Gold+500")
    print("=" * 60)
    print()
    n2_baseline, n2_conditional = setup_condition(
        condition_name="republic_gold500",
        fork_dirs=N2_FORKS,
        conditional_text_templates=N2_CONDITIONAL_TEXT_TEMPLATES,
        conditional_continuous_text_templates=N2_CONDITIONAL_CONTINUOUS_TEXT_TEMPLATES,
        condition_metadata={
            "condition_type": "compound",
            "n_interventions": 2,
            "interventions": [
                {"type": "government", "value": "Republic"},
                {"type": "gold_add", "value": 500},
            ],
            "description": "Republic + 500 gold",
            "source": "conditional_results.json",
        },
        base_dir=base_dir,
    )
    print()
    print(f"  N=2 Baseline: {n2_baseline} questions")
    print(f"  N=2 Conditional: {n2_conditional} questions")
    print()

    # === N=3: Republic + Gold+500 + Map Making ===
    print("=" * 60)
    print("N=3: Republic + Gold+500 + Map Making")
    print("=" * 60)
    print()
    n3_baseline, n3_conditional = setup_condition(
        condition_name="republic_gold500_mapmaking",
        fork_dirs=N3_FORKS,
        conditional_text_templates=N3_CONDITIONAL_TEXT_TEMPLATES,
        conditional_continuous_text_templates=N3_CONDITIONAL_CONTINUOUS_TEXT_TEMPLATES,
        condition_metadata={
            "condition_type": "compound",
            "n_interventions": 3,
            "interventions": [
                {"type": "government", "value": "Republic"},
                {"type": "gold_add", "value": 500},
                {"type": "tech", "value": 45, "name": "Map Making"},
            ],
            "description": "Republic + 500 gold + Map Making",
            "source": "conditional_results.json",
        },
        base_dir=base_dir,
    )
    print()
    print(f"  N=3 Baseline: {n3_baseline} questions")
    print(f"  N=3 Conditional: {n3_conditional} questions")
    print()

    # === Summary ===
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  N=2 (Republic + Gold):              {n2_baseline} baseline, {n2_conditional} conditional")
    print(f"  N=3 (Republic + Gold + Map Making):  {n3_baseline} baseline, {n3_conditional} conditional")
    print(f"  Total:                               {n2_baseline + n3_baseline} baseline, {n2_conditional + n3_conditional} conditional")
    print()
    print("Evaluation directories:")
    print(f"  data/conditional/republic_gold500/baseline/")
    print(f"  data/conditional/republic_gold500/conditional/")
    print(f"  data/conditional/republic_gold500_mapmaking/baseline/")
    print(f"  data/conditional/republic_gold500_mapmaking/conditional/")
    print()
    print("To run evaluations:")
    print("  python scripts/evaluate_llm_forecasts_parallel.py --data-dir data/conditional/republic_gold500/conditional --models ...")
    print("  python scripts/evaluate_llm_forecasts_parallel.py --data-dir data/conditional/republic_gold500_mapmaking/conditional --models ...")


if __name__ == "__main__":
    main()
