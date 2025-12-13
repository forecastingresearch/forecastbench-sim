#!/usr/bin/env python3
"""Run multiple Civilization games in parallel with progress tracking

Usage:
    python run_worlds.py --start-seed 0 --end-seed 100
    python run_worlds.py --start-seed 0 --end-seed 100 --batch-size 10
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def run_single_game(seed: int, max_turns: int, num_ai_players: int) -> tuple[int, bool, str]:
    """Run a single game via subprocess."""
    cmd = [
        sys.executable, 'run_world.py',
        '--seed', str(seed),
        '--max_turns', str(max_turns),
        '--num_ai_players', str(num_ai_players),
        '--quiet'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        if result.returncode == 0:
            return (seed, True, "")
        else:
            error = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            return (seed, False, error[:200])
    except Exception as e:
        return (seed, False, str(e)[:200])


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple Civilization games in parallel'
    )
    parser.add_argument('--start-seed', type=int, default=0, help='Starting seed (inclusive)')
    parser.add_argument('--end-seed', type=int, default=100, help='Ending seed (exclusive)')
    parser.add_argument('--max-turns', type=int, default=50, help='Turns per game')
    parser.add_argument('--batch-size', type=int, default=5, help='Concurrent games')
    parser.add_argument('--num-ai-players', type=int, default=5, help='AI players per game')

    args = parser.parse_args()
    seeds = list(range(args.start_seed, args.end_seed))
    total = len(seeds)

    if total == 0:
        print("No games to run")
        return 1

    print(f"Running {total} games (seeds {args.start_seed}-{args.end_seed - 1}, batch size {args.batch_size})")
    print()

    succeeded, failed = [], []

    with ProcessPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {
            executor.submit(run_single_game, seed, args.max_turns, args.num_ai_players): seed
            for seed in seeds
        }

        with tqdm(total=total, desc="Games", unit="game") as pbar:
            for future in as_completed(futures):
                seed, success, error = future.result()
                (succeeded if success else failed).append((seed, error) if not success else seed)
                pbar.update(1)

    print()
    print(f"Complete! {len(succeeded)}/{total} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed seeds: {[s for s, _ in failed]}")

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
