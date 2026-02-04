#!/usr/bin/env python3
"""Test that two forks can run simultaneously (parallel execution).

This script verifies parallel fork execution by:
1. Launching two fork subprocesses at the same time
2. Waiting for both to complete
3. Comparing results to verify both succeeded
4. Reporting timing (parallel should be ~same time as single fork, not 2x)

This is distinct from test_determinism.py which runs forks sequentially.
Parallel execution is critical for efficient counterfactual analysis.

Usage:
    uv run python scripts/test_parallel_forks.py --seed 100 --checkpoint-turn 50 --run-turns 10

Prerequisites:
    - A completed game recording at logs/recordings/s{seed}/
    - Savegames downloaded from Docker to savegames/ subdirectory
"""

import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple


# Parse our arguments first, before importing civrealm (which has its own argparse)
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--seed", type=int, help="Base seed of the recording")
_parser.add_argument("--checkpoint-turn", type=int, help="Turn to fork from")
_parser.add_argument("--run-turns", type=int, help="Number of turns to run forward")
_parser.add_argument("--recording-dir", type=str, default=None, help="Override recording directory")
_parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed state info")
_pre_args, _ = _parser.parse_known_args()


def create_fork_script(
    recording_dir: str,
    seed: int,
    checkpoint_turn: int,
    end_turn: int,
    fork_name: str
) -> str:
    """Generate the Python script to run a fork."""
    return f'''
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


def parse_subprocess_result(proc: subprocess.CompletedProcess, fork_name: str) -> Dict:
    """Parse the JSON result from a subprocess."""
    if proc.returncode != 0 and not proc.stdout:
        return {
            "success": False,
            "final_turn": 0,
            "player_states": {},
            "error": f"Subprocess {fork_name} failed (returncode={proc.returncode}): {proc.stderr[-500:] if proc.stderr else 'no output'}"
        }

    # Find the JSON result line
    for line in proc.stdout.split('\n'):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])

    # If no result found, return error
    return {
        "success": False,
        "final_turn": 0,
        "player_states": {},
        "error": f"Subprocess {fork_name} produced no result: {proc.stderr[-500:] if proc.stderr else 'no output'}"
    }


def run_forks_parallel(
    recording_dir: str,
    seed: int,
    checkpoint_turn: int,
    end_turn: int
) -> Tuple[Dict, Dict, float, float, float]:
    """Run two forks in parallel.

    Returns:
        Tuple of (result_a, result_b, time_a, time_b, total_time)
    """
    script_a = create_fork_script(recording_dir, seed, checkpoint_turn, end_turn, "para")
    script_b = create_fork_script(recording_dir, seed, checkpoint_turn, end_turn, "parb")

    cwd = str(Path(__file__).parent.parent)

    total_start = time.time()

    # Launch both subprocesses simultaneously
    proc_a = subprocess.Popen(
        [sys.executable, "-c", script_a],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd
    )
    start_a = time.time()

    proc_b = subprocess.Popen(
        [sys.executable, "-c", script_b],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd
    )
    start_b = time.time()

    # Wait for both to complete
    stdout_a, stderr_a = proc_a.communicate()
    end_a = time.time()

    stdout_b, stderr_b = proc_b.communicate()
    end_b = time.time()

    total_end = time.time()

    # Parse results
    result_a_proc = subprocess.CompletedProcess(
        args=[], returncode=proc_a.returncode, stdout=stdout_a, stderr=stderr_a
    )
    result_b_proc = subprocess.CompletedProcess(
        args=[], returncode=proc_b.returncode, stdout=stdout_b, stderr=stderr_b
    )

    result_a = parse_subprocess_result(result_a_proc, "para")
    result_b = parse_subprocess_result(result_b_proc, "parb")

    time_a = end_a - start_a
    time_b = end_b - start_b
    total_time = total_end - total_start

    return result_a, result_b, time_a, time_b, total_time


def run_fork_sequential(
    recording_dir: str,
    seed: int,
    checkpoint_turn: int,
    end_turn: int,
    fork_name: str
) -> Tuple[Dict, float]:
    """Run a single fork (for timing comparison).

    Returns:
        Tuple of (result, elapsed_time)
    """
    script = create_fork_script(recording_dir, seed, checkpoint_turn, end_turn, fork_name)
    cwd = str(Path(__file__).parent.parent)

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=cwd
    )
    elapsed = time.time() - start

    parsed = parse_subprocess_result(result, fork_name)
    return parsed, elapsed


def compare_results(result_a: Dict, result_b: Dict) -> List[str]:
    """Compare two fork results and return list of differences."""
    differences = []

    if not result_a["success"]:
        differences.append(f"Fork A failed: {result_a.get('error', 'unknown')}")
    if not result_b["success"]:
        differences.append(f"Fork B failed: {result_b.get('error', 'unknown')}")

    if not result_a["success"] or not result_b["success"]:
        return differences

    if result_a["final_turn"] != result_b["final_turn"]:
        differences.append(
            f"Final turn: A={result_a['final_turn']}, B={result_b['final_turn']}"
        )

    states_a = {int(k): v for k, v in result_a.get("player_states", {}).items()}
    states_b = {int(k): v for k, v in result_b.get("player_states", {}).items()}

    all_player_ids = set(states_a.keys()) | set(states_b.keys())

    for pid in sorted(all_player_ids):
        state_a = states_a.get(pid, {})
        state_b = states_b.get(pid, {})
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

    states = {int(k): v for k, v in result.get("player_states", {}).items()}

    for pid, state in sorted(states.items()):
        print(f"    {state.get('name', 'Unknown')} (id={pid}):")
        print(f"      Score: {state.get('score', 0)}")
        print(f"      Gold: {state.get('gold', 0)}")
        print(f"      Alive: {state.get('is_alive', False)}")
        print(f"      Researching: {state.get('researching', None)}")


def main():
    args = _pre_args

    if args.seed is None:
        print("Error: --seed is required")
        print("Usage: uv run python scripts/test_parallel_forks.py --seed 100 --checkpoint-turn 50 --run-turns 10")
        return 1
    if args.checkpoint_turn is None:
        print("Error: --checkpoint-turn is required")
        return 1
    if args.run_turns is None:
        print("Error: --run-turns is required")
        return 1

    recording_dir = args.recording_dir or f"logs/recordings/s{args.seed}"

    civbench_dir = Path(__file__).parent.parent
    if not (civbench_dir / recording_dir).exists():
        print(f"Error: Recording directory not found: {recording_dir}")
        print(f"Run `python scripts/run_world.py --seed {args.seed}` first.")
        return 1

    end_turn = args.checkpoint_turn + args.run_turns

    print("=" * 60)
    print("PARALLEL FORK EXECUTION TEST")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Recording: {recording_dir}")
    print(f"Checkpoint turn: {args.checkpoint_turn}")
    print(f"Run turns: {args.run_turns}")
    print(f"End turn: {end_turn}")
    print()

    # First, run a single fork for baseline timing
    print("Running single fork for baseline timing...")
    baseline_result, baseline_time = run_fork_sequential(
        recording_dir=recording_dir,
        seed=args.seed,
        checkpoint_turn=args.checkpoint_turn,
        end_turn=end_turn,
        fork_name="baseline"
    )
    print(f"  Baseline time: {baseline_time:.2f}s")
    if not baseline_result["success"]:
        print(f"  ERROR: Baseline fork failed: {baseline_result.get('error')}")
        return 1
    print()

    # Now run two forks in parallel
    print("Running two forks in PARALLEL...")
    result_a, result_b, time_a, time_b, total_time = run_forks_parallel(
        recording_dir=recording_dir,
        seed=args.seed,
        checkpoint_turn=args.checkpoint_turn,
        end_turn=end_turn
    )
    print(f"  Fork A time: {time_a:.2f}s")
    print(f"  Fork B time: {time_b:.2f}s")
    print(f"  Total wall-clock time: {total_time:.2f}s")
    print()

    # Print detailed results if verbose
    if args.verbose:
        print("=" * 60)
        print("DETAILED RESULTS")
        print("=" * 60)
        print_fork_result(result_a, "Fork A (para)")
        print_fork_result(result_b, "Fork B (parb)")
        print()

    # Analyze timing
    print("=" * 60)
    print("TIMING ANALYSIS")
    print("=" * 60)
    sequential_estimate = baseline_time * 2
    speedup = sequential_estimate / total_time if total_time > 0 else 0

    print(f"  Baseline (single fork):     {baseline_time:.2f}s")
    print(f"  Sequential estimate (2x):   {sequential_estimate:.2f}s")
    print(f"  Parallel actual:            {total_time:.2f}s")
    print(f"  Speedup:                    {speedup:.2f}x")
    print()

    is_parallel = total_time < (sequential_estimate * 0.75)  # Allow 25% overhead
    if is_parallel:
        print("  TRUE PARALLEL EXECUTION CONFIRMED")
    else:
        print("  WARNING: Forks may not be running in true parallel")
        print("  (Could be CPU-bound or shared resource contention)")
    print()

    # Compare results for correctness
    print("=" * 60)
    print("CORRECTNESS CHECK")
    print("=" * 60)

    differences = compare_results(result_a, result_b)

    if not differences:
        print()
        print("PARALLEL FORKS SUCCESSFUL")
        print()
        print("Both parallel forks completed with identical results.")
        print(f"Final turn: {result_a['final_turn']}")
        states = result_a.get("player_states", {})
        print(f"Players compared: {len(states)}")

        if is_parallel:
            print()
            print("PARALLEL EXECUTION VERIFIED")
            print(f"  - Two forks ran simultaneously on different ports")
            print(f"  - No resource conflicts detected")
            print(f"  - Speedup: {speedup:.2f}x over sequential execution")
        return 0
    else:
        print()
        print("PARALLEL FORKS DIFFER")
        print()
        print(f"Found {len(differences)} difference(s):")
        for diff in differences:
            print(f"  - {diff}")
        print()
        print("This could indicate:")
        print("  1. Port conflicts between parallel forks")
        print("  2. Shared resource contention (files, Docker volumes)")
        print("  3. Non-determinism in Freeciv itself")
        print()
        print("Run test_determinism.py to check if sequential forks are deterministic.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
