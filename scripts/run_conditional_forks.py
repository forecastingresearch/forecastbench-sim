#!/usr/bin/env python3
"""
Run conditional forks to generate P(B|A) forecasting questions.

This script:
1. Loads game data from a completed recording
2. Generates conditional questions (pairing conditions with target questions)
3. Runs paired forks (control vs intervention) for each question
4. Saves results in both:
   - conditional_results.json (full conditional data)
   - conditional_questions.json (evaluation-compatible format)

Usage:
    # Auto-generate conditions for all players
    uv run python scripts/run_conditional_forks.py \\
        --seed 100 --checkpoint-turn 50 --end-turn 100 \\
        --output data/questions/s100/

    # Specific condition
    uv run python scripts/run_conditional_forks.py \\
        --seed 100 --checkpoint-turn 50 --end-turn 100 \\
        --condition "gold:0:5000" \\
        --target-template "treasury_comparative"

    # Dry run (generate questions without running forks)
    uv run python scripts/run_conditional_forks.py \\
        --seed 100 --checkpoint-turn 50 --end-turn 100 \\
        --dry-run

Prerequisites:
    - A completed game recording at logs/recordings/s{seed}/
    - Savegames downloaded from Docker to savegames/ subdirectory
    - Game data at data/games/s{seed}_data.json
"""

import argparse
import json
import sys
from pathlib import Path

# Parse arguments before importing civrealm (which has its own argparse)
_parser = argparse.ArgumentParser(
    description="Run conditional forks for P(B|A) forecasting questions"
)
_parser.add_argument("--seed", type=int, required=True, help="Game seed")
_parser.add_argument("--checkpoint-turn", type=int, required=True, help="Turn to fork from")
_parser.add_argument("--end-turn", type=int, required=True, help="Turn to run forks until")
_parser.add_argument("--recording-dir", type=str, default=None, help="Override recording directory")
_parser.add_argument("--game-data", type=str, default=None, help="Override game data path")
_parser.add_argument("--output", type=str, default=None, help="Output directory for results")
_parser.add_argument("--condition", type=str, action="append", default=None,
                    help="Specific condition in format 'type:player_id:value' (e.g., 'gold:0:5000')")
_parser.add_argument("--target-template", type=str, action="append", default=None,
                    help="Specific target templates to use")
_parser.add_argument("--parallel", action="store_true", default=True,
                    help="Run control/intervention forks in parallel (default: True)")
_parser.add_argument("--sequential", action="store_true", default=False,
                    help="Run forks sequentially (for debugging)")
_parser.add_argument("--dry-run", action="store_true", default=False,
                    help="Generate questions without running forks")
_parser.add_argument("--verbose", "-v", action="store_true", default=False,
                    help="Print detailed progress")
_args = _parser.parse_args()


def parse_condition(cond_str: str) -> dict:
    """Parse condition string 'type:player_id:value' into dict."""
    parts = cond_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid condition format: {cond_str}. Expected 'type:player_id:value'")

    cond_type = parts[0]
    player_id = int(parts[1])

    if cond_type == "gold":
        value = int(parts[2])
    elif cond_type == "gold_add":
        value = int(parts[2])
    elif cond_type == "government":
        value = parts[2]
    elif cond_type == "tech":
        value = int(parts[2])
    else:
        raise ValueError(f"Unknown condition type: {cond_type}")

    return {
        "type": cond_type,
        "player_id": player_id,
        "value": value,
    }


def main():
    args = _args

    # Determine paths
    civbench_dir = Path(__file__).parent.parent
    recording_dir = args.recording_dir or f"logs/recordings/seed{args.seed}"
    recording_path = civbench_dir / recording_dir

    if not recording_path.exists():
        print(f"Error: Recording directory not found: {recording_path}")
        print(f"Run `python scripts/run_world.py --seed {args.seed}` first.")
        return 1

    savegames_path = recording_path / "savegames"
    if not savegames_path.exists():
        print(f"Error: Savegames not found: {savegames_path}")
        print("Download savegames from Docker first.")
        return 1

    game_data_path = args.game_data or civbench_dir / "data" / "games" / f"seed{args.seed}_data.json"
    game_data_path = Path(game_data_path)

    if not game_data_path.exists():
        print(f"Error: Game data not found: {game_data_path}")
        print(f"Run `python scripts/generate_questions.py --seed {args.seed}` first.")
        return 1

    output_dir = Path(args.output) if args.output else civbench_dir / "data" / "questions" / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load game data
    print(f"Loading game data from {game_data_path}")
    with open(game_data_path) as f:
        game_data = json.load(f)

    # Import after argparse
    sys.path.insert(0, str(civbench_dir / "src"))
    from civrealm.world_reports.questions import (
        ConditionalQuestionGenerator,
        ConditionalQuestionRunner,
        create_condition,
        save_conditional_bank,
        save_as_question_bank,
    )
    from civrealm.world_reports.questions.schema import CivilizationInfo

    # Parse specific conditions if provided
    conditions = None
    if args.condition:
        conditions = []
        for cond_str in args.condition:
            cond_dict = parse_condition(cond_str)

            # Get civ name from game data
            civ_data = game_data.get("civilizations", {}).get(str(cond_dict["player_id"]), {})
            civ_name = civ_data.get("name", f"Player {cond_dict['player_id']}")

            conditions.append(create_condition(
                condition_type=cond_dict["type"],
                player_id=cond_dict["player_id"],
                value=cond_dict["value"],
                civ_name=civ_name,
            ))

    # Generate conditional questions
    print(f"\nGenerating conditional questions...")
    print(f"  Seed: {args.seed}")
    print(f"  Checkpoint turn: {args.checkpoint_turn}")
    print(f"  End turn: {args.end_turn}")

    generator = ConditionalQuestionGenerator()
    cond_bank = generator.generate_conditional_bank(
        game_id=f"seed{args.seed}",
        game_data=game_data,
        checkpoint_turn=args.checkpoint_turn,
        end_turn=args.end_turn,
        conditions=conditions,
        target_templates=args.target_template,
    )

    print(f"\nGenerated {len(cond_bank.questions)} conditional questions")
    print(f"  Conditions: {len(cond_bank.conditions)}")

    if args.verbose:
        print("\nConditions:")
        for c in cond_bank.conditions:
            print(f"  - {c.condition_id}: {c.description}")

        print("\nQuestions:")
        for q in cond_bank.questions[:5]:  # Show first 5
            print(f"  - {q.conditional_id}: {q.target_template_id}")
        if len(cond_bank.questions) > 5:
            print(f"  ... and {len(cond_bank.questions) - 5} more")

    if args.dry_run:
        print("\nDry run - not running forks")

        # Save the unresolved bank
        results_path = output_dir / "conditional_results.json"
        save_conditional_bank(cond_bank, results_path)
        print(f"\nSaved conditional questions to {results_path}")
        return 0

    # Run forks
    print(f"\nRunning conditional forks...")
    parallel = not args.sequential

    runner = ConditionalQuestionRunner(
        recording_dir=str(recording_path),
        base_seed=args.seed,
        parallel=parallel,
        verbose=args.verbose,
    )

    cond_bank = runner.run_batch(cond_bank)

    # Summarize results
    successful = sum(1 for r in cond_bank.results.values()
                    if r.control_outcome.success and r.intervention_outcome.success)
    effects = [r.conditional_effect for r in cond_bank.results.values()
               if r.conditional_effect is not None]

    print(f"\nResults summary:")
    print(f"  Total questions: {len(cond_bank.questions)}")
    print(f"  Successful runs: {successful}")
    if effects:
        avg_effect = sum(effects) / len(effects)
        differing = sum(1 for e in effects if e > 0)
        print(f"  Questions with conditional effect: {differing}/{len(effects)} ({100*differing/len(effects):.1f}%)")
        print(f"  Average conditional effect: {avg_effect:.3f}")

    # Extract civilizations for saving
    civs = {}
    for pid_str, civ_data in game_data.get("civilizations", {}).items():
        civs[int(pid_str)] = CivilizationInfo(
            name=civ_data.get("name", f"Player {pid_str}"),
            nation_id=civ_data.get("nation_id", 0),
        )

    # Save results
    results_path = output_dir / "conditional_results.json"
    save_conditional_bank(cond_bank, results_path)
    print(f"\nSaved full conditional results to {results_path}")

    # Save evaluation-compatible format
    eval_path = output_dir / "conditional_questions.json"
    save_as_question_bank(cond_bank, eval_path, civs)
    print(f"Saved evaluation-compatible questions to {eval_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
