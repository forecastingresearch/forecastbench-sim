#!/usr/bin/env python3
"""
Set up the evaluation directories for Gold +500 conditional experiment.

Uses conditional_results.json from gold +500 forks to create:
1. questions_gold500_eval/ - "If receives +500 gold" framing, intervention answer

Usage:
    python scripts/setup_gold500_conditional_eval.py
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Gold +500 fork directories for each seed
GOLD500_FORKS = {
    "seed0": "logs/recordings/seed0forkgoldadd500p0",
    "seed2": "logs/recordings/seed2forkgoldadd500p0",
    "seed3": "logs/recordings/seed3forkgoldadd500p0",
    "seed4": "logs/recordings/seed4forkgoldadd500p0",
    "seed5": "logs/recordings/seed5forkgoldadd500p0",
    "seed6": "logs/recordings/seed6forkgoldadd500p0",
    "seed7": "logs/recordings/seed7forkgoldadd500p0",
    "seed8": "logs/recordings/seed8forkgoldadd500p0",
}

# Template text for conditional (intervention) questions - Gold +500
CONDITIONAL_TEXT_TEMPLATES = {
    "treasury_comparative": "If {civ_a} received +500 gold next turn, would {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?",
    "score_comparative": "If {civ_a} received +500 gold next turn, would {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?",
    "tech_comparative": "If {civ_a} received +500 gold next turn, would {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?",
    "population_comparative": "If {civ_a} received +500 gold next turn, would {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?",
    "city_count_comparative": "If {civ_a} received +500 gold next turn, would {civ_a} have more cities than {civ_b} at turn {resolution_turn}?",
    "territory_comparative": "If {civ_a} received +500 gold next turn, would {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?",
    "score_rank_1": "If {civ} received +500 gold next turn, would {civ} be ranked #1 at turn {resolution_turn}?",
    "tech_discovered": "If {civ} received +500 gold next turn, would {civ} have discovered {tech_name} by turn {resolution_turn}?",
    "wonder_completed": "If the civilization received +500 gold next turn, would {wonder_name} be completed by any civilization by turn {resolution_turn}?",
    "government_at": "If {civ} received +500 gold next turn, would {civ} be in {government_type} at turn {resolution_turn}?",
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
) -> dict:
    """
    Generate conditional questions from conditional_results.json.

    Args:
        conditional_results_path: Path to conditional_results.json
        game_id: Game identifier
        civilizations: Civilization info from baseline questions

    Returns:
        Question bank dict ready for evaluation
    """
    with open(conditional_results_path) as f:
        cond_data = json.load(f)

    results = cond_data.get("results", {})
    checkpoint_turn = cond_data.get("checkpoint_turn", 60)

    text_templates = CONDITIONAL_TEXT_TEMPLATES
    answer_key = "answer_intervention"
    template_prefix = "conditional_"
    question_id_suffix = "_intervention"

    questions = []
    skipped = 0

    for q in cond_data.get("questions", []):
        cond_id = q["conditional_id"]
        result = results.get(cond_id, {})

        # Get the appropriate answer
        answer = result.get(answer_key)
        if answer is None:
            skipped += 1
            continue

        template_id = q["target_template_id"]
        params = q["target_parameters"]
        resolution_turn = q["resolution_turn"]

        # Generate question text with appropriate framing
        question_text = format_question_text(template_id, params, text_templates)
        horizon = get_horizon(checkpoint_turn, resolution_turn)

        questions.append({
            "question_id": f"{cond_id}{question_id_suffix}",
            "template_id": f"{template_prefix}{template_id}",
            "resolution_turn": resolution_turn,
            "horizon": horizon,
            "parameters": {
                **params,
                "checkpoint_turn": checkpoint_turn,
            },
            "question_text": question_text,
            "resolution": {"answer": answer},
        })

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
            "condition_type": "gold_add",
            "condition_value": 500,
            "source": "conditional_results.json",
            "intervention": "+500 gold bonus",
        },
    }


def setup_evaluation_directories():
    """Set up Gold +500 evaluation directory."""
    base_dir = Path(__file__).parent.parent

    # Output directory
    gold500_eval_dir = base_dir / "data" / "questions_gold500_eval"

    # Source directory for baseline questions (for civilizations info and world_report)
    questions_dir = base_dir / "data" / "questions"

    total_questions = 0

    print("Setting up evaluation directories for Gold +500 conditional experiment...")
    print()

    for seed, fork_dir in GOLD500_FORKS.items():
        fork_path = base_dir / fork_dir
        conditional_results_path = fork_path / "conditional_results.json"

        if not conditional_results_path.exists():
            print(f"  {seed}: Skipping - no conditional_results.json")
            continue

        baseline_questions_path = questions_dir / seed / "questions.json"
        if not baseline_questions_path.exists():
            print(f"  {seed}: Skipping - no baseline questions.json")
            continue

        world_report_dir = questions_dir / seed / "world_report"
        if not world_report_dir.exists():
            print(f"  {seed}: Skipping - no world_report")
            continue

        print(f"  Processing {seed}...")

        # Load baseline data for civilizations info
        with open(baseline_questions_path) as f:
            baseline_data = json.load(f)
        civilizations = baseline_data.get("civilizations", {})

        # Generate conditional questions (gold +500 intervention framing, intervention answer)
        gold500_seed_dir = gold500_eval_dir / seed
        gold500_seed_dir.mkdir(parents=True, exist_ok=True)
        gold500_questions = generate_questions_from_conditional_results(
            conditional_results_path, seed, civilizations
        )
        with open(gold500_seed_dir / "conditional_questions.json", "w") as f:
            json.dump(gold500_questions, f, indent=2)
        if (gold500_seed_dir / "world_report").exists():
            shutil.rmtree(gold500_seed_dir / "world_report")
        shutil.copytree(world_report_dir, gold500_seed_dir / "world_report")
        n_questions = len(gold500_questions.get("questions", []))
        total_questions += n_questions

        print(f"    Questions: {n_questions}")

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Total Gold +500 questions: {total_questions}")
    print()
    print("Evaluation directory created:")
    print(f"  {gold500_eval_dir}")
    print()
    print("To run evaluation:")
    print(f"  python scripts/evaluate_llm_forecasts_parallel.py --data-dir data/questions_gold500_eval --models anthropic/claude-opus-4-5-20251101 -n 20 -o data/evaluations/gold500_opus45_eval.json")


if __name__ == "__main__":
    setup_evaluation_directories()
