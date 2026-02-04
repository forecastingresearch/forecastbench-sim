#!/usr/bin/env python3
"""Run game forks from a checkpoint with modifications.

This script runs forks from an existing game recording. Use
generate_conditional_results.py afterward to generate conditional questions.

Usage:
    # Run baseline (no modifications) from turn 60 to 100
    uv run python scripts/run_fork.py --base-seed 100 --checkpoint-turn 60 --end-turn 100

    # Run with gold modification
    uv run python scripts/run_fork.py --base-seed 100 --checkpoint-turn 60 \
        --modification "gold:0:5000" --end-turn 100

    # Run multiple modifications
    uv run python scripts/run_fork.py --base-seed 100 --checkpoint-turn 60 \
        --modification "gold:0:5000" --modification "tech:0:23" --end-turn 100

    # Compare baseline vs modified (runs both)
    uv run python scripts/run_fork.py --base-seed 100 --checkpoint-turn 60 \
        --modification "gold:0:5000" --compare --end-turn 100

Modification format:
    gold:player_id:amount        - Set player's gold
    gold_add:player_id:amount    - Add gold to player's treasury
    government:player_id:name    - Set player's government (e.g., Republic)
    tech:player_id:tech_id       - Grant technology by ID

Output:
    logs/recordings/{seed}fork{name}/savegames/ — Fork savegames

Prerequisites:
    - A completed game recording at logs/recordings/seed{seed}/
    - Savegames downloaded from Docker to savegames/ subdirectory
    - Run `uv run python scripts/run_world.py --seed {seed} --max_turns {turns}` first
"""

import sys
import argparse
from pathlib import Path

# Parse our arguments first, before importing civrealm (which has its own argparse)
# We use parse_known_args to ignore civrealm's arguments
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--base-seed", type=int)
_parser.add_argument("--checkpoint-turn", type=int)
_parser.add_argument("--end-turn", type=int)
_parser.add_argument("--modification", "-m", action="append", dest="modifications", default=[])
_parser.add_argument("--compare", action="store_true")
_parser.add_argument("--fork-name", type=str, default=None)
_parser.add_argument("--recording-dir", type=str, default=None)
_pre_args, _ = _parser.parse_known_args()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Now import civrealm (this will parse its own config args)
from civrealm.forking import ForkManager


def parse_modification(mod_str: str) -> dict:
    """Parse a modification string into a modification dict.

    Args:
        mod_str: Modification in format "type:player_id:value"

    Returns:
        Modification dict suitable for ForkManager
    """
    parts = mod_str.split(":")

    if len(parts) < 3:
        raise ValueError(
            f"Invalid modification format: {mod_str}. "
            "Expected format: type:player_id:value"
        )

    mod_type = parts[0].lower()
    player_id = int(parts[1])

    if mod_type == "gold":
        return {
            "type": "gold",
            "player_id": player_id,
            "value": int(parts[2])
        }
    elif mod_type == "gold_add":
        return {
            "type": "gold_add",
            "player_id": player_id,
            "value": int(parts[2])
        }
    elif mod_type == "government":
        return {
            "type": "government",
            "player_id": player_id,
            "value": parts[2]
        }
    elif mod_type == "tech":
        return {
            "type": "tech",
            "player_id": player_id,
            "tech_id": int(parts[2])
        }
    else:
        raise ValueError(f"Unknown modification type: {mod_type}")


def modification_to_name(mod: dict) -> str:
    """Convert a modification dict to a name-friendly string."""
    mod_type = mod["type"]
    player_id = mod["player_id"]

    if mod_type == "gold":
        return f"gold{mod['value']}p{player_id}"
    elif mod_type == "gold_add":
        return f"goldadd{mod['value']}p{player_id}"
    elif mod_type == "government":
        return f"gov{mod['value']}p{player_id}"
    elif mod_type == "tech":
        return f"tech{mod['tech_id']}p{player_id}"
    else:
        return f"{mod_type}p{player_id}"


def main():
    # Use pre-parsed args from module-level parsing
    args = _pre_args

    # Validate required arguments
    if args.base_seed is None:
        print("Error: --base-seed is required")
        print("Usage: uv run python scripts/run_fork.py --base-seed 100 --checkpoint-turn 50 --end-turn 100")
        return 1
    if args.checkpoint_turn is None:
        print("Error: --checkpoint-turn is required")
        print("Usage: uv run python scripts/run_fork.py --base-seed 100 --checkpoint-turn 50 --end-turn 100")
        return 1
    if args.end_turn is None:
        print("Error: --end-turn is required")
        print("Usage: uv run python scripts/run_fork.py --base-seed 100 --checkpoint-turn 50 --end-turn 100")
        return 1

    # Determine recording directory
    if args.recording_dir:
        recording_dir = args.recording_dir
    else:
        recording_dir = f"logs/recordings/s{args.base_seed}"

    # Verify recording exists
    if not Path(recording_dir).exists():
        print(f"Error: Recording directory not found: {recording_dir}")
        print(f"Run `python scripts/run_world.py --seed {args.base_seed}` first.")
        return 1

    # Parse modifications
    modifications = []
    for mod_str in args.modifications:
        try:
            mod = parse_modification(mod_str)
            modifications.append(mod)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    # Generate fork name
    if args.fork_name:
        fork_name = args.fork_name
    elif modifications:
        fork_name = "_".join(modification_to_name(m) for m in modifications)
    else:
        fork_name = "baseline"

    print(f"Fork Manager: Running forks from seed {args.base_seed}")
    print(f"  Recording directory: {recording_dir}")
    print(f"  Checkpoint turn: {args.checkpoint_turn}")
    print(f"  End turn: {args.end_turn}")
    print(f"  Modifications: {modifications if modifications else 'none (baseline)'}")
    print()

    # Create fork manager
    try:
        manager = ForkManager(recording_dir, args.base_seed)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Find the checkpoint savegame
    savegame = manager.find_savegame(args.checkpoint_turn)
    if savegame is None:
        print(f"Error: No savegame found for turn {args.checkpoint_turn}")
        return 1

    print(f"  Using savegame: {savegame.name}")
    print()

    # Create forks
    forks_to_run = []

    if args.compare:
        # Create baseline fork first
        print("Creating baseline fork...")
        baseline = manager.create_fork(
            checkpoint_turn=args.checkpoint_turn,
            modifications=[],
            fork_name="baseline"
        )
        forks_to_run.append(baseline)
        print(f"  Output: {baseline.output_dir}")

    if modifications:
        print(f"Creating modified fork '{fork_name}'...")
        modified = manager.create_fork(
            checkpoint_turn=args.checkpoint_turn,
            modifications=modifications,
            fork_name=fork_name
        )
        forks_to_run.append(modified)
        print(f"  Output: {modified.output_dir}")
    elif not args.compare:
        # No modifications and no --compare, just run baseline
        print("Creating baseline fork...")
        baseline = manager.create_fork(
            checkpoint_turn=args.checkpoint_turn,
            modifications=[],
            fork_name="baseline"
        )
        forks_to_run.append(baseline)
        print(f"  Output: {baseline.output_dir}")

    print()

    # Run forks
    print("Running forks...")
    results = manager.run_all_forks(args.end_turn)

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    for fork_name, result in results.items():
        print(f"\nFork: {fork_name}")
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Final turn: {result.final_turn}")
            print(f"  Players:")
            for pid, state in sorted(result.player_states.items()):
                print(f"    Player {pid}: {state.get('name', 'Unknown')}")
                print(f"      Score: {state.get('score', 0)}")
                print(f"      Gold: {state.get('gold', 0)}")
                print(f"      Alive: {state.get('is_alive', False)}")
        else:
            print(f"  Error: {result.error}")

    # Compare if we ran multiple forks
    if len(results) > 1:
        print()
        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)

        for metric in ["player_score", "player_gold"]:
            print(f"\n{metric}:")
            comparison = manager.compare_outcomes(metric)
            for fork_name, values in comparison.items():
                if values is None:
                    print(f"  {fork_name}: FAILED")
                else:
                    for pid, val in sorted(values.items()):
                        print(f"  {fork_name} / Player {pid}: {val}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
