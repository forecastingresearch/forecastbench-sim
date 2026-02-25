#!/usr/bin/env python3
"""
Set up evaluation directories for the diverged-state experiment.

Tests whether unconditional forecasting works better when the model sees
a world state where the intervention's consequences have been baked in
(turn 90, 30 turns after the fork) rather than immediately after the change (turn 61).

Creates two conditions:
1. diverged_baseline/ - Unconditional framing, baseline (control) answer, turn-90 baseline world report
2. diverged_fork/ - Unconditional framing, fork (intervention) answer, turn-90 fork world report

Only includes questions with resolution_turn >= 120 (H2+ horizons), since turn 90
is now the snapshot and H1 questions would be "predict the present."

Usage:
    python scripts/setup_diverged_state_eval.py
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from civrealm.world_reports import generate_txt_report

# Reuse from existing setup script
from setup_republic_conditional_eval import (
    BASELINE_CONTINUOUS_TEXT_TEMPLATES,
    BASELINE_TEXT_TEMPLATES,
    REPUBLIC_FORKS,
    format_question_text,
)

SNAPSHOT_TURN = 90
MIN_RESOLUTION_TURN = 120  # Only H2+ horizons


def get_horizon(checkpoint_turn: int, resolution_turn: int) -> str:
    """Calculate horizon based on turn difference from snapshot."""
    diff = resolution_turn - checkpoint_turn
    if diff <= 30:
        return "H1"
    elif diff <= 60:
        return "H2"
    else:
        return "H3"


def generate_diverged_questions(
    conditional_results_path: Path,
    game_id: str,
    civilizations: dict,
    condition_type: str,  # "diverged_baseline" or "diverged_fork"
) -> dict:
    """
    Generate questions for diverged-state experiment.

    Filters to resolution_turn >= 120 and uses unconditional framing for both conditions.
    Ground truth comes from answer_control (baseline) or answer_intervention (fork).
    """
    with open(conditional_results_path) as f:
        cond_data = json.load(f)

    results = cond_data.get("results", {})

    if condition_type == "diverged_baseline":
        answer_key = "answer_control"
        question_id_suffix = "_divbaseline"
    else:  # diverged_fork
        answer_key = "answer_intervention"
        question_id_suffix = "_divfork"

    questions = []
    skipped = 0
    filtered = 0

    for q in cond_data.get("questions", []):
        resolution_turn = q["resolution_turn"]

        # Filter: only H2+ horizons (resolution_turn >= 120)
        if resolution_turn < MIN_RESOLUTION_TURN:
            filtered += 1
            continue

        cond_id = q["conditional_id"]
        result = results.get(cond_id, {})

        answer = result.get(answer_key)
        if answer is None:
            skipped += 1
            continue

        template_id = q["target_template_id"]
        params = q["target_parameters"]

        is_continuous = template_id.endswith("_continuous")
        templates = BASELINE_CONTINUOUS_TEXT_TEMPLATES if is_continuous else BASELINE_TEXT_TEMPLATES
        question_text = format_question_text(template_id, params, templates)
        horizon = get_horizon(SNAPSHOT_TURN, resolution_turn)

        question_entry = {
            "question_id": f"{cond_id}{question_id_suffix}",
            "template_id": template_id,
            "resolution_turn": resolution_turn,
            "horizon": horizon,
            "question_type": "continuous" if is_continuous else "binary",
            "parameters": {
                **params,
                "checkpoint_turn": SNAPSHOT_TURN,
                "snapshot_turn": SNAPSHOT_TURN,
            },
            "question_text": question_text,
            "resolution": {"value_at_resolution": answer} if is_continuous else {"answer": answer},
        }
        questions.append(question_entry)

    if skipped > 0:
        print(f"      Skipped {skipped} questions with missing {answer_key}")
    if filtered > 0:
        print(f"      Filtered {filtered} questions with resolution_turn < {MIN_RESOLUTION_TURN}")

    return {
        "game_id": game_id,
        "snapshot_turn": SNAPSHOT_TURN,
        "game_max_turn": cond_data.get("end_turn", 270),
        "generated_at": datetime.now().isoformat() + "Z",
        "civilizations": civilizations,
        "questions": questions,
        "condition_metadata": {
            "condition_type": condition_type,
            "source": "conditional_results.json",
            "intervention": "Republic government change",
            "snapshot_turn": SNAPSHOT_TURN,
            "min_resolution_turn": MIN_RESOLUTION_TURN,
            "description": "Diverged-state experiment: unconditional framing with turn-90 world report",
        },
    }


def create_merged_turn90_recording(
    baseline_recording_dir: Path,
    fork_recording_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Create a merged recording directory combining baseline and fork recordings.

    Symlinks ALL files from both baseline and fork recordings. Since fork step
    numbers are offset by -59 (fork turn 60 = step 0, baseline turn 60 = step 58),
    fork states always have lower step numbers for the same turn. The DataLoader
    picks fork states for turns 60+ automatically (lowest step number per turn).
    No filename collisions since different step numbers produce different filenames.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Symlink all files from baseline recording
    for f in baseline_recording_dir.iterdir():
        if f.is_file() or f.is_dir():
            os.symlink(f.resolve(), output_dir / f.name)

    # Symlink all files from fork recording (no collisions due to different step numbers)
    for f in fork_recording_dir.iterdir():
        target = output_dir / f.name
        if f.is_file() or f.is_dir():
            if not target.exists():
                os.symlink(f.resolve(), target)

    return output_dir


def setup_diverged_state_directories():
    """Set up diverged-state evaluation directories."""
    base_dir = Path(__file__).parent.parent

    diverged_baseline_dir = base_dir / "data" / "conditional" / "republic" / "diverged_baseline"
    diverged_fork_dir = base_dir / "data" / "conditional" / "republic" / "diverged_fork"

    # Civilizations info comes from existing baseline condition directory
    existing_baseline_dir = base_dir / "data" / "conditional" / "republic" / "baseline"

    total_baseline = 0
    total_fork = 0

    print("Setting up diverged-state experiment directories...")
    print(f"  Snapshot turn: {SNAPSHOT_TURN}")
    print(f"  Min resolution turn: {MIN_RESOLUTION_TURN} (H2+ only)")
    print()

    for seed, fork_dir in REPUBLIC_FORKS.items():
        fork_path = base_dir / fork_dir
        conditional_results_path = fork_path / "conditional_results.json"

        if not conditional_results_path.exists():
            print(f"  {seed}: Skipping - no conditional_results.json")
            continue

        baseline_questions_path = existing_baseline_dir / seed / "questions.json"
        if not baseline_questions_path.exists():
            print(f"  {seed}: Skipping - no baseline questions.json at {baseline_questions_path}")
            continue

        print(f"  Processing {seed}...")

        # Load baseline data for civilizations info
        with open(baseline_questions_path) as f:
            baseline_data = json.load(f)
        civilizations = baseline_data.get("civilizations", {})

        # --- Create merged turn-90 recording ---
        baseline_recording_dir = base_dir / "logs" / "recordings" / seed
        merged_dir = create_merged_turn90_recording(
            baseline_recording_dir=baseline_recording_dir,
            fork_recording_dir=fork_path,
            output_dir=base_dir / "logs" / "recordings" / f"_merged_turn90_{seed}_republic",
        )
        print(f"    Created merged recording: {merged_dir.name}")

        # --- Generate turn-90 world reports ---
        game_data_path = base_dir / "data" / "games" / f"{seed}_data.json"

        # Baseline turn-90 world report
        baseline_seed_dir = diverged_baseline_dir / seed
        baseline_seed_dir.mkdir(parents=True, exist_ok=True)

        generate_txt_report(
            game_data_path=game_data_path,
            recording_dir=baseline_recording_dir,
            output_dir=baseline_seed_dir / "world_report",
            turn=SNAPSHOT_TURN,
            sample_interval=5,
            map_interval=10,
        )
        print(f"    Generated baseline turn-{SNAPSHOT_TURN} world report")

        # Fork turn-90 world report (using merged recording)
        fork_seed_dir = diverged_fork_dir / seed
        fork_seed_dir.mkdir(parents=True, exist_ok=True)

        generate_txt_report(
            game_data_path=game_data_path,
            recording_dir=merged_dir,
            output_dir=fork_seed_dir / "world_report",
            turn=SNAPSHOT_TURN,
            sample_interval=5,
            map_interval=10,
        )
        print(f"    Generated fork turn-{SNAPSHOT_TURN} world report")

        # --- Generate question files ---

        # Diverged baseline: unconditional framing + answer_control
        baseline_questions = generate_diverged_questions(
            conditional_results_path, seed, civilizations, "diverged_baseline"
        )
        with open(baseline_seed_dir / "questions.json", "w") as f:
            json.dump(baseline_questions, f, indent=2)
        n_baseline = len(baseline_questions.get("questions", []))
        total_baseline += n_baseline

        # Diverged fork: unconditional framing + answer_intervention
        fork_questions = generate_diverged_questions(
            conditional_results_path, seed, civilizations, "diverged_fork"
        )
        with open(fork_seed_dir / "questions.json", "w") as f:
            json.dump(fork_questions, f, indent=2)
        n_fork = len(fork_questions.get("questions", []))
        total_fork += n_fork

        print(f"    Questions: baseline={n_baseline}, fork={n_fork}")

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Diverged-baseline questions: {total_baseline}")
    print(f"  Diverged-fork questions:     {total_fork}")
    print()
    print("Both conditions use:")
    print("  - Unconditional framing (no 'If switches to Republic')")
    print(f"  - Turn-{SNAPSHOT_TURN} world report (30 turns after intervention)")
    print(f"  - Only resolution_turn >= {MIN_RESOLUTION_TURN} questions (H2+ horizons)")
    print("  - Diverged-baseline: baseline world state + control ground truth")
    print("  - Diverged-fork: fork world state + intervention ground truth")
    print()
    print("Evaluation directories created:")
    print(f"  {diverged_baseline_dir}")
    print(f"  {diverged_fork_dir}")
    print()
    print("To run evaluations:")
    print("  # Diverged baseline (X=0)")
    print(f"  uv run python scripts/evaluate_llm_forecasts_parallel.py \\")
    print(f"      --data-dir data/conditional/republic/diverged_baseline \\")
    print(f"      --models anthropic/claude-opus-4-5-20251101 openai/o3-2025-04-16 --all")
    print()
    print("  # Diverged fork (X=1)")
    print(f"  uv run python scripts/evaluate_llm_forecasts_parallel.py \\")
    print(f"      --data-dir data/conditional/republic/diverged_fork \\")
    print(f"      --models anthropic/claude-opus-4-5-20251101 openai/o3-2025-04-16 --all")


if __name__ == "__main__":
    setup_diverged_state_directories()
