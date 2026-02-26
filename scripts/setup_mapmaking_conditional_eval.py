#!/usr/bin/env python3
"""
Set up evaluation directories for Map Making conditional experiment.

Creates two evaluation directories from the same underlying questions:
1. conditional/mapmaking/baseline/ - Unconditional framing, baseline (control) answer
2. conditional/mapmaking/conditional/ - "If discovers Map Making" framing, fork (intervention) answer

This is the "plausible next-turn conditional" experiment — Map Making is a tech
the player could naturally research, producing a fully in-distribution game state.

Usage:
    python scripts/setup_mapmaking_conditional_eval.py
"""

import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from civrealm.world_reports import generate_txt_report

# Map Making fork directories for each seed
MAPMAKING_FORKS = {
    "seed0": "logs/recordings/seed0forktech45p0",
    "seed1": "logs/recordings/seed1forktech45p0",
    "seed2": "logs/recordings/seed2forktech45p0",
    # seed3 skipped — Map Making not researchable at turn 60
    "seed4": "logs/recordings/seed4forktech45p0",
    "seed5": "logs/recordings/seed5forktech45p0",
    "seed6": "logs/recordings/seed6forktech45p0",
    "seed7": "logs/recordings/seed7forktech45p0",
    "seed8": "logs/recordings/seed8forktech45p0",
    "seed9": "logs/recordings/seed9forktech45p0",
    "seed10": "logs/recordings/seed10forktech45p0",
    "seed11": "logs/recordings/seed11forktech45p0",
    "seed12": "logs/recordings/seed12forktech45p0",
    "seed13": "logs/recordings/seed13forktech45p0",
    "seed14": "logs/recordings/seed14forktech45p0",
    "seed15": "logs/recordings/seed15forktech45p0",
    "seed16": "logs/recordings/seed16forktech45p0",
    "seed17": "logs/recordings/seed17forktech45p0",
    "seed18": "logs/recordings/seed18forktech45p0",
    "seed19": "logs/recordings/seed19forktech45p0",
    "seed20": "logs/recordings/seed20forktech45p0",
}

# Template text for baseline (unconditional) questions
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

# Template text for conditional (intervention) questions
CONDITIONAL_TEXT_TEMPLATES = {
    "treasury_comparative": "If {civ_a} discovers Map Making next turn, would {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?",
    "score_comparative": "If {civ_a} discovers Map Making next turn, would {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?",
    "tech_comparative": "If {civ_a} discovers Map Making next turn, would {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?",
    "population_comparative": "If {civ_a} discovers Map Making next turn, would {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?",
    "city_count_comparative": "If {civ_a} discovers Map Making next turn, would {civ_a} have more cities than {civ_b} at turn {resolution_turn}?",
    "territory_comparative": "If {civ_a} discovers Map Making next turn, would {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?",
    "score_rank_1": "If {civ} discovers Map Making next turn, would {civ} be ranked #1 at turn {resolution_turn}?",
    "tech_discovered": "If {civ} discovers Map Making next turn, would {civ} have discovered {tech_name} by turn {resolution_turn}?",
    "wonder_completed": "If the civilization discovers Map Making next turn, would {wonder_name} be completed by any civilization by turn {resolution_turn}?",
    "government_at": "If {civ} discovers Map Making next turn, would {civ} be in {government_type} at turn {resolution_turn}?",
}

# Continuous template text
BASELINE_CONTINUOUS_TEXT_TEMPLATES = {
    "techs_continuous": "How many technologies will {civ} have discovered by turn {resolution_turn}?",
    "treasury_continuous": "How much gold will {civ} have at turn {resolution_turn}?",
    "population_continuous": "What will {civ}'s population be at turn {resolution_turn}?",
    "cities_count_continuous": "How many cities will {civ} have at turn {resolution_turn}?",
    "territory_continuous": "How many tiles will {civ} control at turn {resolution_turn}?",
    "scores_continuous": "What will {civ}'s score be at turn {resolution_turn}?",
}

CONDITIONAL_CONTINUOUS_TEXT_TEMPLATES = {
    "techs_continuous": "If {civ} discovers Map Making next turn, how many technologies will {civ} have discovered by turn {resolution_turn}?",
    "treasury_continuous": "If {civ} discovers Map Making next turn, how much gold will {civ} have at turn {resolution_turn}?",
    "population_continuous": "If {civ} discovers Map Making next turn, what will {civ}'s population be at turn {resolution_turn}?",
    "cities_count_continuous": "If {civ} discovers Map Making next turn, how many cities will {civ} have at turn {resolution_turn}?",
    "territory_continuous": "If {civ} discovers Map Making next turn, how many tiles will {civ} control at turn {resolution_turn}?",
    "scores_continuous": "If {civ} discovers Map Making next turn, what will {civ}'s score be at turn {resolution_turn}?",
}


def get_horizon(checkpoint_turn: int, resolution_turn: int) -> str:
    diff = resolution_turn - checkpoint_turn
    if diff <= 30:
        return "H1"
    elif diff <= 60:
        return "H2"
    else:
        return "H3"


def format_question_text(template_id: str, params: dict, text_templates: dict) -> str:
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
    condition_type: str,
) -> dict:
    with open(conditional_results_path) as f:
        cond_data = json.load(f)

    results = cond_data.get("results", {})
    checkpoint_turn = cond_data.get("checkpoint_turn", 60)

    if condition_type == "baseline":
        text_templates = BASELINE_TEXT_TEMPLATES
        continuous_text_templates = BASELINE_CONTINUOUS_TEXT_TEMPLATES
        answer_key = "answer_control"
        template_prefix = ""
        question_id_suffix = ""
    elif condition_type == "conditional":
        text_templates = CONDITIONAL_TEXT_TEMPLATES
        continuous_text_templates = CONDITIONAL_CONTINUOUS_TEXT_TEMPLATES
        answer_key = "answer_intervention"
        template_prefix = "conditional_"
        question_id_suffix = "_intervention"
    else:
        raise ValueError(f"Unknown condition_type: {condition_type}")

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
        "condition_metadata": {
            "condition_type": condition_type,
            "source": "conditional_results.json",
            "intervention": "Map Making tech discovery",
        },
    }


def setup_evaluation_directories():
    base_dir = Path(__file__).parent.parent

    baseline_eval_dir = base_dir / "data" / "conditional" / "mapmaking" / "baseline"
    conditional_eval_dir = base_dir / "data" / "conditional" / "mapmaking" / "conditional"

    questions_dir = base_dir / "data" / "questions"
    games_dir = base_dir / "data" / "games"

    total_baseline = 0
    total_conditional = 0

    print("Setting up evaluation directories for Map Making conditional experiment...")
    print()

    for seed, fork_dir in MAPMAKING_FORKS.items():
        fork_path = base_dir / fork_dir
        conditional_results_path = fork_path / "conditional_results.json"

        if not conditional_results_path.exists():
            print(f"  {seed}: Skipping - no conditional_results.json")
            continue

        # Get civilizations from game_data file
        game_data_path = games_dir / f"{seed}_data.json"
        if not game_data_path.exists():
            print(f"  {seed}: Skipping - no game_data file")
            continue

        world_report_dir = questions_dir / seed / "world_report"
        if not world_report_dir.exists():
            print(f"  {seed}: Skipping - no world_report")
            continue

        print(f"  Processing {seed}...")

        with open(game_data_path) as f:
            game_data = json.load(f)
        civilizations = game_data.get("civilizations", {})

        # 1. Baseline questions (unconditional framing, control answer)
        baseline_seed_dir = baseline_eval_dir / seed
        baseline_seed_dir.mkdir(parents=True, exist_ok=True)
        baseline_questions = generate_questions_from_conditional_results(
            conditional_results_path, seed, civilizations, "baseline"
        )
        with open(baseline_seed_dir / "questions.json", "w") as f:
            json.dump(baseline_questions, f, indent=2)
        if (baseline_seed_dir / "world_report").exists():
            shutil.rmtree(baseline_seed_dir / "world_report")
        shutil.copytree(world_report_dir, baseline_seed_dir / "world_report")
        n_baseline = len(baseline_questions.get("questions", []))
        total_baseline += n_baseline

        # 2. Conditional questions ("If discovers Map Making" framing, intervention answer)
        conditional_seed_dir = conditional_eval_dir / seed
        conditional_seed_dir.mkdir(parents=True, exist_ok=True)
        conditional_questions = generate_questions_from_conditional_results(
            conditional_results_path, seed, civilizations, "conditional"
        )
        with open(conditional_seed_dir / "conditional_questions.json", "w") as f:
            json.dump(conditional_questions, f, indent=2)
        if (conditional_seed_dir / "world_report").exists():
            shutil.rmtree(conditional_seed_dir / "world_report")
        shutil.copytree(world_report_dir, conditional_seed_dir / "world_report")
        n_conditional = len(conditional_questions.get("questions", []))
        total_conditional += n_conditional

        print(f"    Baseline: {n_baseline}, Conditional: {n_conditional}")

    print()
    print("=" * 60)
    print(f"  Baseline questions:    {total_baseline}")
    print(f"  Conditional questions: {total_conditional}")
    print()
    print("Evaluation directories:")
    print(f"  {baseline_eval_dir}")
    print(f"  {conditional_eval_dir}")
    print()
    print("To run evaluations:")
    print("  uv run python scripts/evaluate_llm_forecasts_parallel.py \\")
    print("    --data-dir data/conditional/mapmaking/conditional \\")
    print("    --models openai/o3-2025-04-16 anthropic/claude-opus-4-5-20251101 openai/gpt-4.1-2025-04-14")


if __name__ == "__main__":
    setup_evaluation_directories()
