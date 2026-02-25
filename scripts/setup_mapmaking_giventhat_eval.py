#!/usr/bin/env python3
"""
Set up evaluation directories for Map Making conditional experiment with "Given that" framing.

Creates conditional evaluation directory with "Given that" presuppositional framing
(vs the original "If" hypothetical framing) to test whether framing affects the
fork conditional gap. Reuses existing baseline from mapmaking/baseline/.

Output: conditional/mapmaking/conditional_giventhat/

Usage:
    python scripts/setup_mapmaking_giventhat_eval.py
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

# Template text for conditional (intervention) questions — "Given that" framing
CONDITIONAL_TEXT_TEMPLATES = {
    "treasury_comparative": "Given that {civ_a} discovers Map Making next turn, would {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?",
    "score_comparative": "Given that {civ_a} discovers Map Making next turn, would {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?",
    "tech_comparative": "Given that {civ_a} discovers Map Making next turn, would {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?",
    "population_comparative": "Given that {civ_a} discovers Map Making next turn, would {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?",
    "city_count_comparative": "Given that {civ_a} discovers Map Making next turn, would {civ_a} have more cities than {civ_b} at turn {resolution_turn}?",
    "territory_comparative": "Given that {civ_a} discovers Map Making next turn, would {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?",
    "score_rank_1": "Given that {civ} discovers Map Making next turn, would {civ} be ranked #1 at turn {resolution_turn}?",
    "tech_discovered": "Given that {civ} discovers Map Making next turn, would {civ} have discovered {tech_name} by turn {resolution_turn}?",
    "wonder_completed": "Given that the civilization discovers Map Making next turn, would {wonder_name} be completed by any civilization by turn {resolution_turn}?",
    "government_at": "Given that {civ} discovers Map Making next turn, would {civ} be in {government_type} at turn {resolution_turn}?",
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
    "techs_continuous": "Given that {civ} discovers Map Making next turn, how many technologies will {civ} have discovered by turn {resolution_turn}?",
    "treasury_continuous": "Given that {civ} discovers Map Making next turn, how much gold will {civ} have at turn {resolution_turn}?",
    "population_continuous": "Given that {civ} discovers Map Making next turn, what will {civ}'s population be at turn {resolution_turn}?",
    "cities_count_continuous": "Given that {civ} discovers Map Making next turn, how many cities will {civ} have at turn {resolution_turn}?",
    "territory_continuous": "Given that {civ} discovers Map Making next turn, how many tiles will {civ} control at turn {resolution_turn}?",
    "scores_continuous": "Given that {civ} discovers Map Making next turn, what will {civ}'s score be at turn {resolution_turn}?",
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
            "framing": "given_that",
        },
    }


def setup_evaluation_directories():
    base_dir = Path(__file__).parent.parent

    conditional_eval_dir = base_dir / "data" / "conditional" / "mapmaking" / "conditional_giventhat"

    questions_dir = base_dir / "data" / "questions"
    games_dir = base_dir / "data" / "games"

    total_conditional = 0

    print("Setting up 'Given that' evaluation directories for Map Making conditional...")
    print("(Baseline reused from data/conditional/mapmaking/baseline/)")
    print()

    for seed, fork_dir in MAPMAKING_FORKS.items():
        fork_path = base_dir / fork_dir
        conditional_results_path = fork_path / "conditional_results.json"

        if not conditional_results_path.exists():
            print(f"  {seed}: Skipping - no conditional_results.json")
            continue

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

        # Conditional questions ("Given that discovers Map Making" framing, intervention answer)
        conditional_seed_dir = conditional_eval_dir / seed
        conditional_seed_dir.mkdir(parents=True, exist_ok=True)
        conditional_questions = generate_questions_from_conditional_results(
            conditional_results_path, seed, civilizations, "conditional"
        )
        with open(conditional_seed_dir / "questions.json", "w") as f:
            json.dump(conditional_questions, f, indent=2)
        if (conditional_seed_dir / "world_report").exists():
            shutil.rmtree(conditional_seed_dir / "world_report")
        shutil.copytree(world_report_dir, conditional_seed_dir / "world_report")
        n_conditional = len(conditional_questions.get("questions", []))
        total_conditional += n_conditional

        print(f"    Conditional: {n_conditional}")

    print()
    print("=" * 60)
    print(f"  Conditional questions: {total_conditional}")
    print()
    print("Evaluation directory:")
    print(f"  {conditional_eval_dir}")
    print()
    print("To run evaluation:")
    print(f"  uv run python scripts/evaluate_llm_forecasts_parallel.py \\")
    print(f"    --data-dir data/conditional/mapmaking/conditional_giventhat --all \\")
    print(f"    --models openai/o3-2025-04-16 anthropic/claude-opus-4-5-20251101 openai/gpt-4.1-2025-04-14 \\")
    print(f"    -o data/results/mapmaking_giventhat_eval.json")


if __name__ == "__main__":
    setup_evaluation_directories()
