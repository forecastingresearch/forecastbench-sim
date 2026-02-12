#!/usr/bin/env python3
"""
Set up post-intervention evaluation directory for Gold +500 experiment.

Unconditional question framing + post-intervention world report (gold already added).
Uses merged recording dirs (baseline turns 1-59 + gold fork turn 60 state).

Usage:
    python scripts/setup_gold500_post_intervention_eval.py
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

# Unconditional question templates (same as baseline)
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


def generate_post_intervention_questions(
    conditional_results_path: Path,
    game_id: str,
    civilizations: dict,
) -> dict:
    with open(conditional_results_path) as f:
        cond_data = json.load(f)

    results = cond_data.get("results", {})
    checkpoint_turn = cond_data.get("checkpoint_turn", 60)

    questions = []
    skipped = 0

    for q in cond_data.get("questions", []):
        cond_id = q["conditional_id"]
        result = results.get(cond_id, {})

        answer = result.get("answer_intervention")
        if answer is None:
            skipped += 1
            continue

        template_id = q["target_template_id"]
        params = q["target_parameters"]
        resolution_turn = q["resolution_turn"]

        question_text = format_question_text(template_id, params, BASELINE_TEXT_TEMPLATES)
        horizon = get_horizon(checkpoint_turn, resolution_turn)

        questions.append({
            "question_id": f"{cond_id}_postintervention",
            "template_id": f"post_intervention_{template_id}",
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
        print(f"      Skipped {skipped} questions with missing answer_intervention")

    return {
        "game_id": game_id,
        "snapshot_turn": checkpoint_turn,
        "game_max_turn": cond_data.get("end_turn", 270),
        "generated_at": datetime.now().isoformat() + "Z",
        "civilizations": civilizations,
        "questions": questions,
        "condition_metadata": {
            "condition_type": "post_intervention",
            "source": "conditional_results.json",
            "intervention": "+500 gold bonus",
        },
    }


def create_merged_recording_dir(
    baseline_recording_dir: Path,
    fork_recording_dir: Path,
    output_dir: Path,
    fork_turn: int = 60,
) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for f in baseline_recording_dir.iterdir():
        if f.is_file():
            os.symlink(f.resolve(), output_dir / f.name)
        elif f.is_dir():
            os.symlink(f.resolve(), output_dir / f.name)

    fork_turn_pattern = re.compile(rf"turn_{fork_turn:03d}_step_\d+_state\.json")
    for f in fork_recording_dir.iterdir():
        if f.is_file() and fork_turn_pattern.match(f.name):
            target = output_dir / f.name
            if not target.exists():
                os.symlink(f.resolve(), target)

    return output_dir


def setup_evaluation_directories():
    base_dir = Path(__file__).parent.parent

    post_intervention_eval_dir = base_dir / "data" / "questions_gold500_post_intervention_eval"
    questions_dir = base_dir / "data" / "questions"

    total = 0

    print("Setting up post-intervention evaluation for Gold +500...")
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

        print(f"  Processing {seed}...")

        with open(baseline_questions_path) as f:
            baseline_data = json.load(f)
        civilizations = baseline_data.get("civilizations", {})

        # Generate questions (unconditional framing, fork ground truth)
        seed_dir = post_intervention_eval_dir / seed
        seed_dir.mkdir(parents=True, exist_ok=True)
        questions = generate_post_intervention_questions(
            conditional_results_path, seed, civilizations
        )
        with open(seed_dir / "questions.json", "w") as f:
            json.dump(questions, f, indent=2)

        # Build merged recording dir
        baseline_recording_dir = base_dir / "logs" / "recordings" / seed
        merged_dir = create_merged_recording_dir(
            baseline_recording_dir=baseline_recording_dir,
            fork_recording_dir=fork_path,
            output_dir=base_dir / "logs" / "recordings" / f"_merged_{seed}_gold500",
            fork_turn=60,
        )

        # Generate post-intervention world report with patched treasury
        # The report reads treasury from game_data JSON (not recordings),
        # so we need to patch the turn-60 treasury value to reflect +500 gold.
        game_data_path = base_dir / "data" / "games" / f"{seed}_data.json"
        patched_game_data_path = seed_dir / f"{seed}_data_patched.json"
        with open(game_data_path) as f:
            game_data = json.load(f)
        # Read the fork's turn-60 state to get the actual gold value
        fork_state_files = list(fork_path.glob("turn_060_step_*_state.json"))
        if fork_state_files:
            with open(sorted(fork_state_files)[0]) as f:
                fork_state = json.load(f)
            # Patch treasury at turn 60 for player 0
            ts = game_data.get("time_series", {})
            if "treasury" in ts and "60" in ts["treasury"]:
                p0_gold = fork_state.get("player", {}).get("0", {}).get("gold", 0)
                ts["treasury"]["60"]["0"] = float(p0_gold)
        with open(patched_game_data_path, "w") as f:
            json.dump(game_data, f)

        generate_txt_report(
            game_data_path=patched_game_data_path,
            recording_dir=merged_dir,
            output_dir=seed_dir / "world_report",
            turn=60,
            sample_interval=5,
            map_interval=10,
        )

        n = len(questions.get("questions", []))
        total += n
        print(f"    Questions: {n}")

    print()
    print("=" * 60)
    print(f"Summary: {total} post-intervention Gold +500 questions")
    print(f"Output: {post_intervention_eval_dir}")
    print()
    print("To run evaluation:")
    print(f"  uv run python scripts/evaluate_llm_forecasts_parallel.py \\")
    print(f"    --data-dir data/questions_gold500_post_intervention_eval \\")
    print(f"    --models anthropic/claude-opus-4-5-20251101 \\")
    print(f"    --output data/evaluations/gold500_post_intervention_opus45_eval.json")


if __name__ == "__main__":
    setup_evaluation_directories()
