#!/usr/bin/env python3
"""Batch extract JSON data from game recordings.

This script processes multiple game recordings in parallel to extract
the JSON data files needed for question generation and base rate computation.

Usage:
    python generate_data_batch.py
    python generate_data_batch.py --input-dir logs/recordings --output-dir data/games
    python generate_data_batch.py --workers 8 --force
"""

import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def process_single_recording(args: tuple) -> tuple[str, bool, str]:
    """Process a single recording directory.

    Args:
        args: Tuple of (recording_dir, output_file, force)

    Returns:
        Tuple of (game_id, success, error_message)
    """
    recording_dir, output_file, force = args
    game_id = Path(recording_dir).name

    try:
        # Skip if already processed (unless force)
        if Path(output_file).exists() and not force:
            return (game_id, True, "skipped")

        # Import here to avoid issues with multiprocessing
        from civrealm.world_reports import ReportConfig
        from civrealm.world_reports.data_loader import DataLoader
        from civrealm.world_reports.extractors import MetricsCollector, write_world_data

        # Create data loader to check available turns
        data_loader = DataLoader(str(recording_dir))
        summary = data_loader.get_turn_summary()
        max_turn = summary['turn_range'][1]

        if max_turn < 50:
            return (game_id, False, f"insufficient data (max_turn={max_turn})")

        # Create minimal config
        config = ReportConfig(
            recording_dir=str(recording_dir),
            output_dir=str(Path(output_file).parent),
            report_turns=[max_turn],
            enabled_sections=['overview'],
            formats=[],
        )

        # Load states and collect metrics
        states = data_loader.get_states_range(0, max_turn)
        if not states:
            return (game_id, False, "no states loaded")

        collector = MetricsCollector()
        data = collector.collect_all(
            states=states,
            config=config,
            data_loader=data_loader
        )

        # Write to output file
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        write_world_data(data, output_file)

        return (game_id, True, "")

    except Exception as e:
        return (game_id, False, str(e)[:200])


def main():
    parser = argparse.ArgumentParser(
        description='Batch extract JSON data from game recordings'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='logs/recordings',
        help='Directory containing game recordings (default: logs/recordings)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/games',
        help='Directory for output JSON files (default: data/games)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess even if output file exists'
    )
    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        help='Only process recordings matching this pattern (e.g., "s1*")'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    # Find all recording directories
    if args.filter:
        recording_dirs = sorted(input_dir.glob(args.filter))
    else:
        recording_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    if not recording_dirs:
        print(f"No recording directories found in {input_dir}")
        return 1

    print(f"Found {len(recording_dirs)} recording directories")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    print()

    # Prepare tasks
    tasks = []
    for rec_dir in recording_dirs:
        game_id = rec_dir.name
        output_file = output_dir / f"{game_id}_data.json"
        tasks.append((str(rec_dir), str(output_file), args.force))

    # Process in parallel
    succeeded = []
    skipped = []
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_recording, task): task[0] for task in tasks}

        with tqdm(total=len(tasks), desc="Processing", unit="game") as pbar:
            for future in as_completed(futures):
                game_id, success, error = future.result()

                if success:
                    if error == "skipped":
                        skipped.append(game_id)
                    else:
                        succeeded.append(game_id)
                else:
                    failed.append((game_id, error))

                pbar.update(1)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total:     {len(tasks)}")
    print(f"Processed: {len(succeeded)}")
    print(f"Skipped:   {len(skipped)}")
    print(f"Failed:    {len(failed)}")

    if failed:
        print()
        print("Failed recordings:")
        for game_id, error in failed[:10]:
            print(f"  {game_id}: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    print()
    print(f"Output files in: {output_dir}/")

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
