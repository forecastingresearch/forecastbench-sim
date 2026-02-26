#!/usr/bin/env python3
"""
Set up the five evaluation directories for Republic conditional experiment.

Uses the same underlying questions (from conditional_results.json) for all five conditions,
with different framing and appropriate ground truth:

1. conditional/republic/baseline/ - Unconditional framing, baseline (control) answer
2. conditional/republic/conditional/ - "If switches to Republic" framing, fork (intervention) answer
3. conditional/republic/conditional_no/ - "If does NOT switch to Republic" framing, baseline (control) answer
4. conditional/republic/given_that/ - "Given that will switch to Republic" framing, fork (intervention) answer
5. conditional/republic/post_intervention/ - Unconditional framing, fork (intervention) answer, post-intervention world report

Conditions 1-4 share the same pre-intervention world_report per seed.
Condition 5 uses a merged recording dir (baseline turns 1-59 + fork turn 60 state)
to generate a post-intervention world report showing Republic already applied.

Usage:
    python scripts/setup_republic_conditional_eval.py
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

# Republic fork directories for each seed
REPUBLIC_FORKS = {
    "seed0": "logs/recordings/seed0forkgovRepp0v1770146193",
    "seed1": "logs/recordings/seed1forkgovRepublicp0",
    "seed2": "logs/recordings/seed2forkgovRepublicp0",
    "seed3": "logs/recordings/seed3forkgovRepublicp0",
    "seed4": "logs/recordings/seed4forkgovRepublicp0",
    "seed5": "logs/recordings/seed5forkgovRepublicp0",
    "seed6": "logs/recordings/seed6forkgovRepublicp0",
    "seed7": "logs/recordings/seed7forkgovRepublicp0",
    "seed8": "logs/recordings/seed8forkgovRepublicp0",
    "seed9": "logs/recordings/seed9forkgovRepublicp0",
    "seed10": "logs/recordings/seed10forkgovRepublicp0",
    "seed11": "logs/recordings/seed11forkgovRepublicp0",
    "seed12": "logs/recordings/seed12forkgovRepublicp0",
    "seed13": "logs/recordings/seed13forkgovRepublicp0",
    "seed14": "logs/recordings/seed14forkgovRepublicp0",
    "seed15": "logs/recordings/seed15forkgovRepublicp0",
    "seed16": "logs/recordings/seed16forkgovRepublicp0",
    "seed17": "logs/recordings/seed17forkgovRepublicp0",
    "seed18": "logs/recordings/seed18forkgovRepublicp0",
    "seed19": "logs/recordings/seed19forkgovRepublicp0",
    "seed20": "logs/recordings/seed20forkgovRepublicp0",
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
    "treasury_comparative": "If {civ_a} switches to Republic next turn, would {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?",
    "score_comparative": "If {civ_a} switches to Republic next turn, would {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?",
    "tech_comparative": "If {civ_a} switches to Republic next turn, would {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?",
    "population_comparative": "If {civ_a} switches to Republic next turn, would {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?",
    "city_count_comparative": "If {civ_a} switches to Republic next turn, would {civ_a} have more cities than {civ_b} at turn {resolution_turn}?",
    "territory_comparative": "If {civ_a} switches to Republic next turn, would {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?",
    "score_rank_1": "If {civ} switches to Republic next turn, would {civ} be ranked #1 at turn {resolution_turn}?",
    "tech_discovered": "If {civ} switches to Republic next turn, would {civ} have discovered {tech_name} by turn {resolution_turn}?",
    "wonder_completed": "If the civilization switches to Republic next turn, would {wonder_name} be completed by any civilization by turn {resolution_turn}?",
    "government_at": "If {civ} switches to Republic next turn, would {civ} be in {government_type} at turn {resolution_turn}?",
}

# Template text for null conditional (negated intervention) questions
NULL_CONDITIONAL_TEXT_TEMPLATES = {
    "treasury_comparative": "If {civ_a} does NOT switch to Republic next turn, would {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?",
    "score_comparative": "If {civ_a} does NOT switch to Republic next turn, would {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?",
    "tech_comparative": "If {civ_a} does NOT switch to Republic next turn, would {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?",
    "population_comparative": "If {civ_a} does NOT switch to Republic next turn, would {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?",
    "city_count_comparative": "If {civ_a} does NOT switch to Republic next turn, would {civ_a} have more cities than {civ_b} at turn {resolution_turn}?",
    "territory_comparative": "If {civ_a} does NOT switch to Republic next turn, would {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?",
    "score_rank_1": "If {civ} does NOT switch to Republic next turn, would {civ} be ranked #1 at turn {resolution_turn}?",
    "tech_discovered": "If {civ} does NOT switch to Republic next turn, would {civ} have discovered {tech_name} by turn {resolution_turn}?",
    "wonder_completed": "If no civilization switches to Republic next turn, would {wonder_name} be completed by any civilization by turn {resolution_turn}?",
    "government_at": "If {civ} does NOT switch to Republic next turn, would {civ} be in {government_type} at turn {resolution_turn}?",
}

# Continuous template text for baseline (unconditional) questions
BASELINE_CONTINUOUS_TEXT_TEMPLATES = {
    "techs_continuous": "How many technologies will {civ} have discovered by turn {resolution_turn}?",
    "treasury_continuous": "How much gold will {civ} have at turn {resolution_turn}?",
    "population_continuous": "What will {civ}'s population be at turn {resolution_turn}?",
    "cities_count_continuous": "How many cities will {civ} have at turn {resolution_turn}?",
    "territory_continuous": "How many tiles will {civ} control at turn {resolution_turn}?",
    "scores_continuous": "What will {civ}'s score be at turn {resolution_turn}?",
}

# Continuous template text for conditional (intervention) questions
CONDITIONAL_CONTINUOUS_TEXT_TEMPLATES = {
    "techs_continuous": "If {civ} switches to Republic next turn, how many technologies will {civ} have discovered by turn {resolution_turn}?",
    "treasury_continuous": "If {civ} switches to Republic next turn, how much gold will {civ} have at turn {resolution_turn}?",
    "population_continuous": "If {civ} switches to Republic next turn, what will {civ}'s population be at turn {resolution_turn}?",
    "cities_count_continuous": "If {civ} switches to Republic next turn, how many cities will {civ} have at turn {resolution_turn}?",
    "territory_continuous": "If {civ} switches to Republic next turn, how many tiles will {civ} control at turn {resolution_turn}?",
    "scores_continuous": "If {civ} switches to Republic next turn, what will {civ}'s score be at turn {resolution_turn}?",
}

# Continuous template text for null conditional (negated intervention) questions
NULL_CONDITIONAL_CONTINUOUS_TEXT_TEMPLATES = {
    "techs_continuous": "If {civ} does NOT switch to Republic next turn, how many technologies will {civ} have discovered by turn {resolution_turn}?",
    "treasury_continuous": "If {civ} does NOT switch to Republic next turn, how much gold will {civ} have at turn {resolution_turn}?",
    "population_continuous": "If {civ} does NOT switch to Republic next turn, what will {civ}'s population be at turn {resolution_turn}?",
    "cities_count_continuous": "If {civ} does NOT switch to Republic next turn, how many cities will {civ} have at turn {resolution_turn}?",
    "territory_continuous": "If {civ} does NOT switch to Republic next turn, how many tiles will {civ} control at turn {resolution_turn}?",
    "scores_continuous": "If {civ} does NOT switch to Republic next turn, what will {civ}'s score be at turn {resolution_turn}?",
}

# Continuous template text for "given that" conditional (presuppositional framing) questions
GIVEN_THAT_CONTINUOUS_TEXT_TEMPLATES = {
    "techs_continuous": "Given that {civ} will switch to Republic next turn, how many technologies will {civ} have discovered by turn {resolution_turn}?",
    "treasury_continuous": "Given that {civ} will switch to Republic next turn, how much gold will {civ} have at turn {resolution_turn}?",
    "population_continuous": "Given that {civ} will switch to Republic next turn, what will {civ}'s population be at turn {resolution_turn}?",
    "cities_count_continuous": "Given that {civ} will switch to Republic next turn, how many cities will {civ} have at turn {resolution_turn}?",
    "territory_continuous": "Given that {civ} will switch to Republic next turn, how many tiles will {civ} control at turn {resolution_turn}?",
    "scores_continuous": "Given that {civ} will switch to Republic next turn, what will {civ}'s score be at turn {resolution_turn}?",
}

# Template text for "given that" conditional (presuppositional framing) questions
GIVEN_THAT_TEXT_TEMPLATES = {
    "treasury_comparative": "Given that {civ_a} will switch to Republic next turn, would {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?",
    "score_comparative": "Given that {civ_a} will switch to Republic next turn, would {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?",
    "tech_comparative": "Given that {civ_a} will switch to Republic next turn, would {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?",
    "population_comparative": "Given that {civ_a} will switch to Republic next turn, would {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?",
    "city_count_comparative": "Given that {civ_a} will switch to Republic next turn, would {civ_a} have more cities than {civ_b} at turn {resolution_turn}?",
    "territory_comparative": "Given that {civ_a} will switch to Republic next turn, would {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?",
    "score_rank_1": "Given that {civ} will switch to Republic next turn, would {civ} be ranked #1 at turn {resolution_turn}?",
    "tech_discovered": "Given that {civ} will switch to Republic next turn, would {civ} have discovered {tech_name} by turn {resolution_turn}?",
    "wonder_completed": "Given that the civilization will switch to Republic next turn, would {wonder_name} be completed by any civilization by turn {resolution_turn}?",
    "government_at": "Given that {civ} will switch to Republic next turn, would {civ} be in {government_type} at turn {resolution_turn}?",
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
    condition_type: str,  # "baseline", "conditional", "null_conditional", "given_that", or "post_intervention"
) -> dict:
    """
    Generate questions from conditional_results.json with appropriate framing and answers.

    Args:
        conditional_results_path: Path to conditional_results.json
        game_id: Game identifier
        civilizations: Civilization info from baseline questions
        condition_type: Which condition to generate ("baseline", "conditional", "null_conditional", "given_that", "post_intervention")

    Returns:
        Question bank dict ready for evaluation
    """
    with open(conditional_results_path) as f:
        cond_data = json.load(f)

    results = cond_data.get("results", {})
    checkpoint_turn = cond_data.get("checkpoint_turn", 60)

    # Select text templates and answer key based on condition type
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
    elif condition_type == "given_that":
        text_templates = GIVEN_THAT_TEXT_TEMPLATES
        continuous_text_templates = GIVEN_THAT_CONTINUOUS_TEXT_TEMPLATES
        answer_key = "answer_intervention"
        template_prefix = "given_that_"
        question_id_suffix = "_given"
    elif condition_type == "post_intervention":
        text_templates = BASELINE_TEXT_TEMPLATES  # unconditional framing
        continuous_text_templates = BASELINE_CONTINUOUS_TEXT_TEMPLATES
        answer_key = "answer_intervention"        # fork ground truth
        template_prefix = "post_intervention_"
        question_id_suffix = "_postintervention"
    else:  # null_conditional
        text_templates = NULL_CONDITIONAL_TEXT_TEMPLATES
        continuous_text_templates = NULL_CONDITIONAL_CONTINUOUS_TEXT_TEMPLATES
        answer_key = "answer_control"
        template_prefix = "null_conditional_"
        question_id_suffix = "_null"

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

        # Determine if continuous
        is_continuous = template_id.endswith("_continuous")

        # Generate question text with appropriate framing
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
            "intervention": "Republic government change",
        },
    }


def create_merged_recording_dir(
    baseline_recording_dir: Path,
    fork_recording_dir: Path,
    output_dir: Path,
    fork_turn: int = 60,
) -> Path:
    """
    Create a merged recording directory with baseline states + fork's turn-60 state.

    Symlinks all files from baseline recording dir, then adds fork's turn_060 state file(s).
    Since the fork uses step_0000 and the baseline uses a higher step number for turn 60,
    the DataLoader picks the fork's state (lowest step number per turn) without collision.

    Args:
        baseline_recording_dir: Path to baseline recording (turns 1-60+)
        fork_recording_dir: Path to fork recording (turn 60+)
        output_dir: Path where merged dir will be created
        fork_turn: Turn number where the fork diverges (default: 60)

    Returns:
        output_dir path
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Symlink all files from baseline recording dir
    for f in baseline_recording_dir.iterdir():
        if f.is_file():
            os.symlink(f.resolve(), output_dir / f.name)
        elif f.is_dir():
            os.symlink(f.resolve(), output_dir / f.name)

    # Symlink fork's turn_060 state files (step_0000 beats baseline's higher step)
    fork_turn_pattern = re.compile(rf"turn_{fork_turn:03d}_step_\d+_state\.json")
    for f in fork_recording_dir.iterdir():
        if f.is_file() and fork_turn_pattern.match(f.name):
            target = output_dir / f.name
            if not target.exists():
                os.symlink(f.resolve(), target)

    return output_dir


def setup_evaluation_directories():
    """Set up all five evaluation directories."""
    base_dir = Path(__file__).parent.parent

    # Output directories
    baseline_eval_dir = base_dir / "data" / "conditional" / "republic" / "baseline"
    conditional_eval_dir = base_dir / "data" / "conditional" / "republic" / "conditional"
    conditional_no_eval_dir = base_dir / "data" / "conditional" / "republic" / "conditional_no"
    given_that_eval_dir = base_dir / "data" / "conditional" / "republic" / "given_that"
    post_intervention_eval_dir = base_dir / "data" / "conditional" / "republic" / "post_intervention"

    # Source directory for baseline questions (for civilizations info and world_report)
    questions_dir = base_dir / "data" / "questions"

    total_baseline = 0
    total_conditional = 0
    total_null_conditional = 0
    total_given_that = 0
    total_post_intervention = 0

    print("Setting up evaluation directories for Republic conditional experiment...")
    print("Using same underlying questions across all five conditions.")
    print()

    for seed, fork_dir in REPUBLIC_FORKS.items():
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

        # 1. Generate baseline questions (unconditional framing, control answer)
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

        # 2. Generate conditional questions (intervention framing, intervention answer)
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

        # 3. Generate null conditional questions (negated intervention framing, control answer)
        null_cond_seed_dir = conditional_no_eval_dir / seed
        null_cond_seed_dir.mkdir(parents=True, exist_ok=True)
        null_cond_questions = generate_questions_from_conditional_results(
            conditional_results_path, seed, civilizations, "null_conditional"
        )
        with open(null_cond_seed_dir / "questions.json", "w") as f:
            json.dump(null_cond_questions, f, indent=2)
        if (null_cond_seed_dir / "world_report").exists():
            shutil.rmtree(null_cond_seed_dir / "world_report")
        shutil.copytree(world_report_dir, null_cond_seed_dir / "world_report")
        n_null_conditional = len(null_cond_questions.get("questions", []))
        total_null_conditional += n_null_conditional

        # 4. Generate "given that" conditional questions (presuppositional framing, intervention answer)
        given_that_seed_dir = given_that_eval_dir / seed
        given_that_seed_dir.mkdir(parents=True, exist_ok=True)
        given_that_questions = generate_questions_from_conditional_results(
            conditional_results_path, seed, civilizations, "given_that"
        )
        with open(given_that_seed_dir / "conditional_questions.json", "w") as f:
            json.dump(given_that_questions, f, indent=2)
        if (given_that_seed_dir / "world_report").exists():
            shutil.rmtree(given_that_seed_dir / "world_report")
        shutil.copytree(world_report_dir, given_that_seed_dir / "world_report")
        n_given_that = len(given_that_questions.get("questions", []))
        total_given_that += n_given_that

        # 5. Generate post-intervention questions (unconditional framing, intervention answer, post-intervention world report)
        post_intervention_seed_dir = post_intervention_eval_dir / seed
        post_intervention_seed_dir.mkdir(parents=True, exist_ok=True)
        post_intervention_questions = generate_questions_from_conditional_results(
            conditional_results_path, seed, civilizations, "post_intervention"
        )
        with open(post_intervention_seed_dir / "questions.json", "w") as f:
            json.dump(post_intervention_questions, f, indent=2)

        # Build merged recording dir (baseline turns 1-59 + fork turn 60 state)
        baseline_recording_dir = base_dir / "logs" / "recordings" / seed
        merged_dir = create_merged_recording_dir(
            baseline_recording_dir=baseline_recording_dir,
            fork_recording_dir=fork_path,
            output_dir=base_dir / "logs" / "recordings" / f"_merged_{seed}_republic",
            fork_turn=60,
        )

        # Generate post-intervention world report from merged recording
        game_data_path = base_dir / "data" / "games" / f"{seed}_data.json"
        generate_txt_report(
            game_data_path=game_data_path,
            recording_dir=merged_dir,
            output_dir=post_intervention_seed_dir / "world_report",
            turn=60,
            sample_interval=5,
            map_interval=10,
        )

        n_post_intervention = len(post_intervention_questions.get("questions", []))
        total_post_intervention += n_post_intervention

        print(f"    Baseline: {n_baseline}, Conditional: {n_conditional}, Null-conditional: {n_null_conditional}, Given-that: {n_given_that}, Post-intervention: {n_post_intervention}")

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Baseline questions:          {total_baseline}")
    print(f"  Conditional questions:       {total_conditional}")
    print(f"  Null-conditional questions:   {total_null_conditional}")
    print(f"  Given-that questions:        {total_given_that}")
    print(f"  Post-intervention questions: {total_post_intervention}")
    print()
    print("All five conditions use the SAME underlying questions with:")
    print("  - Baseline: unconditional framing, control (baseline) answer, pre-intervention world report")
    print("  - Conditional: 'If switches to Republic' framing, intervention (fork) answer, pre-intervention world report")
    print("  - Null-conditional: 'If does NOT switch' framing, control (baseline) answer, pre-intervention world report")
    print("  - Given-that: 'Given that will switch to Republic' framing, intervention (fork) answer, pre-intervention world report")
    print("  - Post-intervention: unconditional framing, intervention (fork) answer, POST-intervention world report")
    print()
    print("Evaluation directories created:")
    print(f"  {baseline_eval_dir}")
    print(f"  {conditional_eval_dir}")
    print(f"  {conditional_no_eval_dir}")
    print(f"  {given_that_eval_dir}")
    print(f"  {post_intervention_eval_dir}")
    print()
    print("To run evaluations:")
    print("  # Baseline")
    print(f"  python scripts/evaluate_llm_forecasts_parallel.py --data-dir data/conditional/republic/baseline --models anthropic/claude-opus-4-5-20251101")
    print()
    print("  # Conditional (Republic intervention)")
    print(f"  python scripts/evaluate_llm_forecasts_parallel.py --data-dir data/conditional/republic/conditional --models anthropic/claude-opus-4-5-20251101")
    print()
    print("  # Null conditional (NOT Republic)")
    print(f"  python scripts/evaluate_llm_forecasts_parallel.py --data-dir data/conditional/republic/conditional_no --models anthropic/claude-opus-4-5-20251101")
    print()
    print("  # Given-that conditional (presuppositional framing)")
    print(f"  python scripts/evaluate_llm_forecasts_parallel.py --data-dir data/conditional/republic/given_that --models anthropic/claude-opus-4-5-20251101")
    print()
    print("  # Post-intervention baseline (unconditional + post-intervention world state)")
    print(f"  python scripts/evaluate_llm_forecasts_parallel.py --data-dir data/conditional/republic/post_intervention --models anthropic/claude-opus-4-5-20251101")


if __name__ == "__main__":
    setup_evaluation_directories()
