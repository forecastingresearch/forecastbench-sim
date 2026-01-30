#!/usr/bin/env python3
"""Test that loading a savegame produces deterministic results.

This script verifies fork determinism by:
1. Loading the same savegame twice via ForkManager (in separate subprocesses)
2. Running both forward N turns
3. Comparing player scores, gold, alive status
4. Reporting whether results are deterministic

Critical for conditional forecasting - if forks aren't deterministic,
we can't reliably compare P(B|A=yes) vs P(B|A=no).

Usage:
    uv run python scripts/test_determinism.py --seed 100 --checkpoint-turn 50 --run-turns 10

Prerequisites:
    - A completed game recording at logs/recordings/s{seed}/
    - Savegames downloaded from Docker to savegames/ subdirectory
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


# Parse our arguments first, before importing civrealm (which has its own argparse)
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--seed", type=int, help="Base seed of the recording")
_parser.add_argument("--checkpoint-turn", type=int, help="Turn to fork from")
_parser.add_argument("--run-turns", type=int, help="Number of turns to run forward")
_parser.add_argument("--recording-dir", type=str, default=None, help="Override recording directory")
_parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed state info")
_pre_args, _ = _parser.parse_known_args()


def run_fork_subprocess(
    recording_dir: str,
    seed: int,
    checkpoint_turn: int,
    end_turn: int,
    fork_name: str
) -> Dict:
    """Run a fork in a subprocess to avoid global state issues.

    Returns:
        Dict with success, final_turn, player_states, error
    """
    script = f'''
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(".") / "src"))
from civrealm.forking import ForkManager

manager = ForkManager("{recording_dir}", {seed})
fork = manager.create_fork(
    checkpoint_turn={checkpoint_turn},
    modifications=[],
    fork_name="{fork_name}"
)
result = manager.run_fork(fork, {end_turn})

# Output result as JSON on the last line
output = {{
    "success": result.success,
    "final_turn": result.final_turn,
    "player_states": result.player_states,
    "error": result.error
}}
print("RESULT_JSON:" + json.dumps(output))
'''

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent)
    )

    # Find the JSON result line
    for line in result.stdout.split('\n'):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])

    # If no result found, return error
    return {
        "success": False,
        "final_turn": 0,
        "player_states": {},
        "error": f"Subprocess failed: {result.stderr[-500:] if result.stderr else 'no output'}"
    }


def compare_results(result_a: Dict, result_b: Dict) -> List[str]:
    """Compare two fork results and return list of differences.

    Args:
        result_a: First fork result dict
        result_b: Second fork result dict

    Returns:
        List of difference descriptions (empty if deterministic)
    """
    differences = []

    # Check if both succeeded
    if not result_a["success"]:
        differences.append(f"Fork A failed: {result_a.get('error', 'unknown')}")
    if not result_b["success"]:
        differences.append(f"Fork B failed: {result_b.get('error', 'unknown')}")

    if not result_a["success"] or not result_b["success"]:
        return differences

    # Compare final turn
    if result_a["final_turn"] != result_b["final_turn"]:
        differences.append(
            f"Final turn: A={result_a['final_turn']}, B={result_b['final_turn']}"
        )

    # Compare player states
    states_a = result_a.get("player_states", {})
    states_b = result_b.get("player_states", {})

    # Convert string keys to int for comparison (JSON serializes int keys as strings)
    states_a = {int(k): v for k, v in states_a.items()}
    states_b = {int(k): v for k, v in states_b.items()}

    all_player_ids = set(states_a.keys()) | set(states_b.keys())

    for pid in sorted(all_player_ids):
        state_a = states_a.get(pid, {})
        state_b = states_b.get(pid, {})

        # Fields to compare
        fields = ['score', 'gold', 'is_alive', 'researching']

        for field in fields:
            val_a = state_a.get(field)
            val_b = state_b.get(field)

            if val_a != val_b:
                name = state_a.get('name', state_b.get('name', f'Player {pid}'))
                differences.append(
                    f"{name} (id={pid}) {field}: A={val_a}, B={val_b}"
                )

    return differences


def print_fork_result(result: Dict, label: str):
    """Print detailed fork result."""
    print(f"\n{label}:")
    print(f"  Success: {result['success']}")
    if not result["success"]:
        print(f"  Error: {result.get('error', 'unknown')}")
        return

    print(f"  Final turn: {result['final_turn']}")
    print(f"  Players:")

    states = result.get("player_states", {})
    # Convert string keys to int (JSON serializes int keys as strings)
    states = {int(k): v for k, v in states.items()}

    for pid, state in sorted(states.items()):
        print(f"    {state.get('name', 'Unknown')} (id={pid}):")
        print(f"      Score: {state.get('score', 0)}")
        print(f"      Gold: {state.get('gold', 0)}")
        print(f"      Alive: {state.get('is_alive', False)}")
        print(f"      Researching: {state.get('researching', None)}")


def main():
    args = _pre_args

    # Validate required arguments
    if args.seed is None:
        print("Error: --seed is required")
        print("Usage: uv run python scripts/test_determinism.py --seed 100 --checkpoint-turn 50 --run-turns 10")
        return 1
    if args.checkpoint_turn is None:
        print("Error: --checkpoint-turn is required")
        return 1
    if args.run_turns is None:
        print("Error: --run-turns is required")
        return 1

    # Determine recording directory
    if args.recording_dir:
        recording_dir = args.recording_dir
    else:
        recording_dir = f"logs/recordings/s{args.seed}"

    # Verify recording exists
    civbench_dir = Path(__file__).parent.parent
    if not (civbench_dir / recording_dir).exists():
        print(f"Error: Recording directory not found: {recording_dir}")
        print(f"Run `python scripts/run_world.py --seed {args.seed}` first.")
        return 1

    end_turn = args.checkpoint_turn + args.run_turns

    print("=" * 60)
    print("DETERMINISM TEST")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Recording: {recording_dir}")
    print(f"Checkpoint turn: {args.checkpoint_turn}")
    print(f"Run turns: {args.run_turns}")
    print(f"End turn: {end_turn}")
    print()
    print("Note: Running forks in separate subprocesses to avoid global state issues")
    print()

    # Run forks in separate subprocesses
    print("Running fork A (deta)...")
    result_a = run_fork_subprocess(
        recording_dir=recording_dir,
        seed=args.seed,
        checkpoint_turn=args.checkpoint_turn,
        end_turn=end_turn,
        fork_name="deta"
    )
    print(f"  {'Success' if result_a['success'] else 'Failed'}")

    print()
    print("Running fork B (detb)...")
    result_b = run_fork_subprocess(
        recording_dir=recording_dir,
        seed=args.seed,
        checkpoint_turn=args.checkpoint_turn,
        end_turn=end_turn,
        fork_name="detb"
    )
    print(f"  {'Success' if result_b['success'] else 'Failed'}")

    # Print detailed results if verbose
    if args.verbose:
        print()
        print("=" * 60)
        print("DETAILED RESULTS")
        print("=" * 60)
        print_fork_result(result_a, "Fork A (deta)")
        print_fork_result(result_b, "Fork B (detb)")

    # Compare results
    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)

    differences = compare_results(result_a, result_b)

    if not differences:
        print()
        print("✅ DETERMINISTIC")
        print()
        print("Both forks produced identical results.")
        print(f"Final turn: {result_a['final_turn']}")
        states = result_a.get("player_states", {})
        print(f"Players compared: {len(states)}")
        return 0
    else:
        print()
        print("❌ NON-DETERMINISTIC")
        print()
        print(f"Found {len(differences)} difference(s):")
        for diff in differences:
            print(f"  - {diff}")
        print()
        print("This means fork results are NOT reliable for conditional forecasting.")
        print("Consider:")
        print("  1. Force-seeding on load (if Freeciv supports it)")
        print("  2. Running multiple forks and averaging outcomes")
        print("  3. Using statistical comparison with noise tolerance")
        return 1


if __name__ == "__main__":
    sys.exit(main())
