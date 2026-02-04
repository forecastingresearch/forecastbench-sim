#!/usr/bin/env python3
"""
Generate tech_comparative questions and world reports for all games.

This script generates questions of the form:
"Will [Civ A] have more technologies than [Civ B] at turn T?"

at all 7 time horizons (H1-H7) and produces world reports for each.

Usage:
    python scripts/generate_tech_comparative.py
    python scripts/generate_tech_comparative.py --snapshot-turn 50
    python scripts/generate_tech_comparative.py --max-games 10
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, 'src')

from civrealm.world_reports.questions.schema import (
    QuestionBank,
    QuestionInstance,
    QuestionTemplate,
    WorldReportConfig,
    CivilizationInfo,
    classify_horizon,
    calculate_difficulty,
)
from civrealm.world_reports.questions.templates import TECH_COMPARATIVE
from civrealm.world_reports.questions.generator import QuestionGenerator
from civrealm.world_reports.questions.resolver import QuestionResolver
from civrealm.world_reports.questions.io import question_bank_to_json, save_question_bank
from civrealm.world_reports import ReportGenerator, ReportConfig


class TechComparativeGenerator(QuestionGenerator):
    """Generator that only produces tech_comparative questions."""

    def __init__(self):
        # Only use tech_comparative template
        super().__init__(templates=[TECH_COMPARATIVE])


def generate_for_game(
    game_data_file: Path,
    recording_dir: Path,
    output_dir: Path,
    snapshot_turn: int,
) -> dict | None:
    """Generate tech_comparative questions and world report for a single game."""

    try:
        # Load game data
        with open(game_data_file) as f:
            game_data = json.load(f)

        game_id = game_data_file.stem.replace('_data', '')
        max_turn = game_data.get('metadata', {}).get('turn', 0)

        # Skip if game doesn't have enough turns
        if max_turn < snapshot_turn + 20:
            print(f"  Skipping {game_id}: max_turn ({max_turn}) < snapshot_turn + 20")
            return None

        # Create output directory for this game
        game_output_dir = output_dir / game_id
        game_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate questions using only tech_comparative
        generator = TechComparativeGenerator()
        bank = generator.generate_question_bank(
            game_id=game_id,
            game_data=game_data,
            snapshot_turn=snapshot_turn,
            info_availability_levels=['I1'],  # tech_comparative is I1
        )

        if not bank.questions:
            print(f"  Skipping {game_id}: no questions generated")
            return None

        # Resolve questions
        resolver = QuestionResolver()
        resolved_bank = resolver.resolve_batch(bank, game_data)

        # Save questions
        questions_file = game_output_dir / 'questions.json'
        save_question_bank(resolved_bank, questions_file)

        # Generate world report if recording exists
        game_recording_dir = recording_dir / game_id
        if game_recording_dir.exists():
            try:
                report_config = ReportConfig(
                    recording_dir=str(game_recording_dir),
                    output_dir=str(game_output_dir / 'world_report'),
                    report_turns=[snapshot_turn],
                    formats=['html'],
                    plot_style='seaborn',
                    dpi=100,
                )

                report_generator = ReportGenerator(report_config)
                if report_generator.validate_config():
                    report_generator.generate_reports()
                else:
                    print(f"  Warning: Could not validate config for {game_id}")
            except Exception as e:
                print(f"  Warning: Could not generate world report for {game_id}: {e}")
        else:
            print(f"  Warning: No recordings found for {game_id}")

        # Return summary
        by_horizon = {'H1': 0, 'H2': 0, 'H3': 0, 'H4': 0, 'H5': 0, 'H6': 0, 'H7': 0}
        true_count = 0
        for q in resolved_bank.questions:
            by_horizon[q.horizon] = by_horizon.get(q.horizon, 0) + 1
            if q.resolution and q.resolution.answer:
                true_count += 1

        return {
            'game_id': game_id,
            'num_questions': len(resolved_bank.questions),
            'by_horizon': by_horizon,
            'true_count': true_count,
            'false_count': len(resolved_bank.questions) - true_count,
        }

    except Exception as e:
        print(f"  Error processing {game_data_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Generate tech_comparative questions and world reports'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/games',
        help='Directory containing game data JSON files'
    )
    parser.add_argument(
        '--recording-dir', type=str, default='logs/recordings',
        help='Directory containing game recordings'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/questions/tech_comparative',
        help='Output directory for questions and reports'
    )
    parser.add_argument(
        '--snapshot-turn', type=int, default=60,
        help='Turn at which forecasters see data'
    )
    parser.add_argument(
        '--max-games', type=int, default=None,
        help='Maximum number of games to process (for testing)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    recording_dir = Path(args.recording_dir)
    output_dir = Path(args.output_dir)

    # Find all game data files
    data_files = sorted(data_dir.glob('*_data.json'))
    if args.max_games:
        data_files = data_files[:args.max_games]

    print("=" * 70)
    print("TECH COMPARATIVE QUESTION GENERATOR")
    print("=" * 70)
    print()
    print(f"Data directory: {data_dir}")
    print(f"Recording directory: {recording_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Snapshot turn: {args.snapshot_turn}")
    print(f"Games to process: {len(data_files)}")
    print()

    # Process each game
    results = []
    for i, data_file in enumerate(data_files, 1):
        game_id = data_file.stem.replace('_data', '')
        print(f"[{i}/{len(data_files)}] Processing {game_id}...")

        result = generate_for_game(
            game_data_file=data_file,
            recording_dir=recording_dir,
            output_dir=output_dir,
            snapshot_turn=args.snapshot_turn,
        )

        if result:
            results.append(result)
            print(f"  Generated {result['num_questions']} questions "
                  f"(H1: {result['by_horizon'].get('H1', 0)}, "
                  f"H2: {result['by_horizon'].get('H2', 0)}, "
                  f"H3: {result['by_horizon'].get('H3', 0)})")

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    if not results:
        print("No questions generated!")
        return 1

    total_questions = sum(r['num_questions'] for r in results)
    total_true = sum(r['true_count'] for r in results)
    total_h1 = sum(r['by_horizon'].get('H1', 0) for r in results)
    total_h2 = sum(r['by_horizon'].get('H2', 0) for r in results)
    total_h3 = sum(r['by_horizon'].get('H3', 0) for r in results)

    print(f"Games processed: {len(results)}")
    print(f"Total questions: {total_questions}")
    print(f"  H1 (short horizon): {total_h1}")
    print(f"  H2 (medium horizon): {total_h2}")
    print(f"  H3 (long horizon): {total_h3}")
    print()
    print(f"Answer distribution: {total_true} True ({100*total_true/total_questions:.1f}%), "
          f"{total_questions - total_true} False ({100*(total_questions-total_true)/total_questions:.1f}%)")
    print()
    print(f"Output saved to: {output_dir}")

    # Save summary
    summary = {
        'generated_at': datetime.now().isoformat() + 'Z',
        'snapshot_turn': args.snapshot_turn,
        'template': 'tech_comparative',
        'games_processed': len(results),
        'total_questions': total_questions,
        'by_horizon': {
            'H1': total_h1,
            'H2': total_h2,
            'H3': total_h3,
        },
        'true_rate': total_true / total_questions if total_questions else 0,
        'games': results,
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {output_dir / 'summary.json'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
