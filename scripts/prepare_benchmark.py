#!/usr/bin/env python3
"""
Prepare benchmark data for CivBench evaluation.

Generates world reports and questions from game data files.
Combines functionality of regenerate_questions.py and generate_txt_report.py.

Usage:
    # Single game
    uv run python scripts/prepare_benchmark.py --game-id seed0

    # With explicit snapshot turn (default: 50)
    uv run python scripts/prepare_benchmark.py --game-id seed0 --snapshot-turn 50

    # Batch mode: all games
    uv run python scripts/prepare_benchmark.py --all

    # Batch mode with filter
    uv run python scripts/prepare_benchmark.py --all --filter "seed1*"

    # Skip components
    uv run python scripts/prepare_benchmark.py --game-id seed0 --skip-report
    uv run python scripts/prepare_benchmark.py --game-id seed0 --skip-questions
    uv run python scripts/prepare_benchmark.py --game-id seed0 --only-report

    # Dry run
    uv run python scripts/prepare_benchmark.py --game-id seed0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm


def process_single_game(
    game_data_file: Path,
    output_dir: Path,
    recordings_dir: Path,
    snapshot_turn: int,
    skip_report: bool,
    skip_questions: bool,
    sample_interval: int,
    map_interval: int,
    force: bool,
    verbose: bool,
) -> dict[str, Any]:
    """
    Process a single game: generate world report and questions.

    Returns:
        Dict with game_id, success, error (if any), and stats
    """
    game_id = game_data_file.stem.replace('_data', '')
    result = {
        'game_id': game_id,
        'success': True,
        'error': '',
        'report_path': None,
        'questions_path': None,
        'conditional_questions_path': None,
        'stats': {},
    }

    try:
        sys.path.insert(0, 'src')

        # Load game data
        with open(game_data_file) as f:
            game_data = json.load(f)

        max_turn = game_data.get('metadata', {}).get('turn', 0)
        if snapshot_turn >= max_turn:
            result['success'] = False
            result['error'] = f"snapshot_turn {snapshot_turn} >= max_turn {max_turn}"
            return result

        questions_dir = output_dir / game_id
        questions_dir.mkdir(parents=True, exist_ok=True)

        # Generate world report (requires recordings)
        if not skip_report:
            recording_dir = recordings_dir / game_id
            if not recording_dir.exists():
                result['success'] = False
                result['error'] = f"Recording directory not found: {recording_dir}"
                return result

            report_dir = questions_dir / 'world_report'
            report_path = report_dir / f'turn_{snapshot_turn:03d}_report.txt'

            if force or not report_path.exists():
                from civrealm.world_reports import generate_txt_report

                txt_path, map_dir = generate_txt_report(
                    game_data_path=game_data_file,
                    recording_dir=recording_dir,
                    output_dir=report_dir,
                    turn=snapshot_turn,
                    sample_interval=sample_interval,
                    map_interval=map_interval,
                )
                result['report_path'] = str(txt_path)
            else:
                result['report_path'] = str(report_path)
                if verbose:
                    print(f"  Skipping report (exists): {report_path}")

        # Generate unconditional questions
        if not skip_questions:
            questions_file = questions_dir / 'questions.json'

            if force or not questions_file.exists():
                from civrealm.world_reports.questions import (
                    QuestionGenerator,
                    QuestionResolver,
                    question_bank_to_dict,
                )

                generator = QuestionGenerator()
                bank = generator.generate_question_bank(
                    game_id=game_id,
                    game_data=game_data,
                    snapshot_turn=snapshot_turn,
                )

                resolver = QuestionResolver()
                resolved_bank = resolver.resolve_batch(bank, game_data)
                bank_dict = question_bank_to_dict(resolved_bank)

                with open(questions_file, 'w') as f:
                    json.dump(bank_dict, f, indent=2)

                # Collect stats
                num_civs = len(bank_dict.get('civilizations', {}))
                num_questions = len(bank_dict.get('questions', []))
                true_count = sum(
                    1 for q in bank_dict.get('questions', [])
                    if q.get('resolution', {}).get('answer', False)
                )

                result['stats'] = {
                    'num_civs': num_civs,
                    'num_questions': num_questions,
                    'true_count': true_count,
                    'false_count': num_questions - true_count,
                }
                result['questions_path'] = str(questions_file)
            else:
                result['questions_path'] = str(questions_file)
                if verbose:
                    print(f"  Skipping questions (exists): {questions_file}")

        # Generate conditional questions if results exist
        if not skip_questions:
            conditional_results_file = questions_dir / 'conditional_results.json'
            conditional_questions_file = questions_dir / 'conditional_questions.json'

            if conditional_results_file.exists():
                if force or not conditional_questions_file.exists():
                    from civrealm.world_reports.questions import (
                        load_conditional_bank,
                        save_as_question_bank,
                    )
                    from civrealm.world_reports.questions.schema import CivilizationInfo

                    cond_bank = load_conditional_bank(conditional_results_file)

                    # Extract civilizations
                    civs = {}
                    for pid_str, civ_data in game_data.get('civilizations', {}).items():
                        civs[int(pid_str)] = CivilizationInfo(
                            name=civ_data.get('name', f'Player {pid_str}'),
                            nation_id=civ_data.get('nation_id', 0),
                        )

                    save_as_question_bank(cond_bank, conditional_questions_file, civs)
                    result['conditional_questions_path'] = str(conditional_questions_file)
                else:
                    result['conditional_questions_path'] = str(conditional_questions_file)
                    if verbose:
                        print(f"  Skipping conditional questions (exists): {conditional_questions_file}")

        return result

    except Exception as e:
        import traceback
        result['success'] = False
        result['error'] = f"{e}\n{traceback.format_exc()[:500]}"
        return result


def _process_single_game_wrapper(args: tuple) -> dict[str, Any]:
    """Wrapper for multiprocessing."""
    return process_single_game(*args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Prepare benchmark data (world reports + questions) from game data'
    )

    # Game selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--game-id', type=str, help='Single game ID (e.g., seed0)')
    group.add_argument('--all', action='store_true', help='Process all games in data/games/')

    # Filtering
    parser.add_argument('--filter', type=str, default=None,
                        help='Glob pattern for --all mode (e.g., "seed1*")')

    # Parameters
    parser.add_argument('--snapshot-turn', type=int, default=50,
                        help='Turn where forecasters see the world (default: 50)')
    parser.add_argument('--sample-interval', type=int, default=5,
                        help='Sampling interval for time series in report (default: 5)')
    parser.add_argument('--map-interval', type=int, default=10,
                        help='Map generation interval in turns (default: 10)')

    # Component selection
    parser.add_argument('--skip-report', action='store_true',
                        help='Skip world report generation')
    parser.add_argument('--skip-questions', action='store_true',
                        help='Skip question generation')
    parser.add_argument('--only-report', action='store_true',
                        help='Only generate world report (skip questions)')

    # Execution options
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel workers for batch mode (default: 4)')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if outputs exist')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be generated without doing it')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Detailed output')

    # Paths
    parser.add_argument('--games-dir', type=Path, default=Path('data/games'),
                        help='Directory containing game data JSON files')
    parser.add_argument('--output-dir', type=Path, default=Path('data/questions'),
                        help='Output directory for questions and reports')
    parser.add_argument('--recordings-dir', type=Path, default=Path('logs/recordings'),
                        help='Recordings base directory')

    args = parser.parse_args()

    # Handle --only-report
    if args.only_report:
        args.skip_questions = True

    # Find game files
    games_dir = args.games_dir
    if not games_dir.exists():
        print(f"Error: Games directory not found: {games_dir}")
        return 1

    if args.game_id:
        game_files = [games_dir / f'{args.game_id}_data.json']
        if not game_files[0].exists():
            print(f"Error: Game data file not found: {game_files[0]}")
            return 1
    else:
        # --all mode
        if args.filter:
            game_files = list(games_dir.glob(f'{args.filter}*_data.json'))
        else:
            game_files = list(games_dir.glob('*_data.json'))
        game_files = sorted(game_files)

    if not game_files:
        print(f"No game data files found in {games_dir}")
        return 1

    # Report what will be done
    print(f"Preparing benchmark for {len(game_files)} game(s)")
    print(f"  Snapshot turn: {args.snapshot_turn}")
    print(f"  Output directory: {args.output_dir}")
    if args.skip_report:
        print("  Skipping: world report")
    if args.skip_questions:
        print("  Skipping: questions")
    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for f in game_files:
            game_id = f.stem.replace('_data', '')
            print(f"  - {game_id}")
        return 0

    print()

    # Process games
    task_args = [
        (
            f,
            args.output_dir,
            args.recordings_dir,
            args.snapshot_turn,
            args.skip_report,
            args.skip_questions,
            args.sample_interval,
            args.map_interval,
            args.force,
            args.verbose,
        )
        for f in game_files
    ]

    succeeded = []
    failed = []
    total_questions = 0
    total_true = 0
    total_false = 0
    civ_counts = []

    if len(game_files) == 1:
        # Single game: run directly
        result = _process_single_game_wrapper(task_args[0])
        if result['success']:
            succeeded.append(result['game_id'])
            if result['stats']:
                total_questions += result['stats']['num_questions']
                total_true += result['stats']['true_count']
                total_false += result['stats']['false_count']
                civ_counts.append(result['stats']['num_civs'])
            if result['report_path']:
                print(f"Report: {result['report_path']}")
            if result['questions_path']:
                print(f"Questions: {result['questions_path']}")
            if result['conditional_questions_path']:
                print(f"Conditional questions: {result['conditional_questions_path']}")
        else:
            failed.append((result['game_id'], result['error']))
    else:
        # Batch mode: parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_single_game_wrapper, arg): arg[0]
                for arg in task_args
            }

            with tqdm(total=len(task_args), desc="Processing", unit="game") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    if result['success']:
                        succeeded.append(result['game_id'])
                        if result['stats']:
                            total_questions += result['stats']['num_questions']
                            total_true += result['stats']['true_count']
                            total_false += result['stats']['false_count']
                            civ_counts.append(result['stats']['num_civs'])
                    else:
                        failed.append((result['game_id'], result['error']))

                    pbar.update(1)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Games processed: {len(succeeded)}")
    print(f"Games failed: {len(failed)}")

    if total_questions > 0:
        print()
        print(f"Total questions: {total_questions}")
        print(f"  True answers: {total_true} ({100*total_true/total_questions:.1f}%)")
        print(f"  False answers: {total_false} ({100*total_false/total_questions:.1f}%)")

    if civ_counts:
        print(f"  Civs per game: min={min(civ_counts)}, max={max(civ_counts)}, avg={sum(civ_counts)/len(civ_counts):.1f}")

    if failed:
        print()
        print("Failed games:")
        for game_id, error in failed[:10]:
            print(f"  {game_id}: {error[:100]}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
