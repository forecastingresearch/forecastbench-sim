#!/usr/bin/env python3
"""Check progress of running fork simulations.

Usage:
    uv run python scripts/check_forks.py
    uv run python scripts/check_forks.py --target-turn 271
"""

import argparse
import subprocess
import re
import sys


def get_docker_savegame_counts(container="freeciv-web"):
    """Get savegame counts for all fork users in Docker."""
    result = subprocess.run(
        ["docker", "exec", container, "ls", "/var/lib/tomcat10/webapps/data/savegames/"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error: Could not list Docker savegames: {result.stderr.strip()}")
        sys.exit(1)

    forks = {}
    for user in result.stdout.strip().split("\n"):
        if "fork" not in user:
            continue
        # Count savegames for this user
        count_result = subprocess.run(
            ["docker", "exec", container, "ls",
             f"/var/lib/tomcat10/webapps/data/savegames/{user}/"],
            capture_output=True, text=True
        )
        if count_result.returncode == 0:
            files = [f for f in count_result.stdout.strip().split("\n") if f]
            # Find max turn number
            max_turn = 0
            for f in files:
                m = re.search(r'_T(\d+)_', f)
                if m:
                    max_turn = max(max_turn, int(m.group(1)))
            forks[user] = {"count": len(files), "max_turn": max_turn}

    return forks


def get_local_savegame_counts(recordings_dir="logs/recordings"):
    """Get savegame counts for fork directories already downloaded locally."""
    from pathlib import Path
    recordings = Path(recordings_dir)
    forks = {}
    for d in sorted(recordings.iterdir()):
        if "fork" not in d.name or not d.is_dir():
            continue
        savegames = d / "savegames"
        if not savegames.exists():
            continue
        files = list(savegames.glob("*.sav.*"))
        max_turn = 0
        for f in files:
            m = re.search(r'_T(\d+)_', f.name)
            if m:
                max_turn = max(max_turn, int(m.group(1)))
        forks[d.name] = {"count": len(files), "max_turn": max_turn}

    return forks


def check_running_processes():
    """Check if run_fork.py processes are currently running."""
    result = subprocess.run(
        ["pgrep", "-f", "run_fork.py"],
        capture_output=True, text=True
    )
    pids = [p for p in result.stdout.strip().split("\n") if p]
    return len(pids)


def main():
    parser = argparse.ArgumentParser(description="Check fork simulation progress")
    parser.add_argument("--target-turn", type=int, default=271,
                        help="Target end turn (default: 271)")
    parser.add_argument("--checkpoint-turn", type=int, default=60,
                        help="Checkpoint turn forks started from (default: 60)")
    args = parser.parse_args()

    total_turns = args.target_turn - args.checkpoint_turn
    num_procs = check_running_processes()

    print(f"Running fork processes: {num_procs}")
    print(f"Target: turn {args.checkpoint_turn} → {args.target_turn} ({total_turns} turns)\n")

    # Check Docker (in-progress)
    docker_forks = get_docker_savegame_counts()
    # Check local (completed)
    local_forks = get_local_savegame_counts()

    # Merge: prefer Docker if still in progress, else show local
    all_forks = {}
    for name, info in docker_forks.items():
        all_forks[name] = {**info, "source": "docker"}
    for name, info in local_forks.items():
        if name not in all_forks:
            all_forks[name] = {**info, "source": "local"}

    if not all_forks:
        print("No forks found.")
        return

    # Parse seed and condition from fork name
    def parse_fork(name):
        m = re.match(r'seed(\d+)fork(.+)', name)
        if m:
            return int(m.group(1)), m.group(2)
        return None, name

    # Group by condition
    by_condition = {}
    for name, info in sorted(all_forks.items(), key=lambda x: parse_fork(x[0])):
        seed, condition = parse_fork(name)
        by_condition.setdefault(condition, []).append((seed, info))

    for condition, entries in sorted(by_condition.items()):
        print(f"  {condition}:")
        for seed, info in entries:
            pct = min(100, int(100 * info["max_turn"] / args.target_turn)) if args.target_turn > 0 else 0
            turns_done = info["max_turn"] - args.checkpoint_turn
            bar_len = 20
            filled = int(bar_len * turns_done / total_turns)
            bar = "█" * filled + "░" * (bar_len - filled)
            status = "done" if info["source"] == "local" else f"T{info['max_turn']}"
            print(f"    seed {seed:>2}: {bar} {turns_done:>3}/{total_turns} turns  ({status})")
        print()


if __name__ == "__main__":
    main()
