#!/usr/bin/env python3
"""Compute base rates for question templates across game data.

This script processes game data JSON files to compute base rates for each
(template_id, horizon) combination. These rates can be used to:
1. Calibrate thresholds for balanced True/False distributions
2. Create a BaseRateForecaster baseline
3. Analyze question predictability by template

Usage:
    python compute_base_rates.py
    python compute_base_rates.py --data-dir data/games --snapshot-turn 50
    python compute_base_rates.py --output base_rates.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from civrealm.world_reports.questions import (
    QuestionGenerator,
    QuestionResolver,
)


def compute_base_rates(
    data_files: list[Path],
    snapshot_turn: int,
) -> dict:
    """Compute base rates across all game data files.

    Args:
        data_files: List of paths to game data JSON files
        snapshot_turn: Turn at which forecasters see data

    Returns:
        Dictionary with base rate statistics
    """
    generator = QuestionGenerator()
    resolver = QuestionResolver()

    # Aggregate results: {template_id: {horizon: {true: N, false: N}}}
    results = defaultdict(lambda: defaultdict(lambda: {'true': 0, 'false': 0}))

    # Also track by threshold for threshold-based questions
    threshold_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'true': 0, 'false': 0})))

    games_processed = 0
    questions_total = 0

    for data_file in tqdm(data_files, desc="Processing games"):
        try:
            with open(data_file) as f:
                game_data = json.load(f)

            game_id = game_data.get('metadata', {}).get('username', data_file.stem)
            max_turn = game_data.get('metadata', {}).get('turn', 0)

            # Skip if game doesn't have enough turns
            if max_turn < snapshot_turn + 20:
                continue

            # Generate questions
            bank = generator.generate_question_bank(
                game_id=game_id,
                game_data=game_data,
                snapshot_turn=snapshot_turn,
            )

            if not bank.questions:
                continue

            # Resolve questions
            resolved_bank = resolver.resolve_batch(bank, game_data)

            # Aggregate results
            for q in resolved_bank.questions:
                template_id = q.template_id
                horizon = q.horizon
                answer = q.resolution.answer if q.resolution else False

                if answer:
                    results[template_id][horizon]['true'] += 1
                else:
                    results[template_id][horizon]['false'] += 1

                # Track by threshold for threshold questions
                threshold = q.parameters.get('threshold')
                if threshold is not None:
                    if answer:
                        threshold_results[template_id][threshold][horizon]['true'] += 1
                    else:
                        threshold_results[template_id][threshold][horizon]['false'] += 1

                questions_total += 1

            games_processed += 1

        except Exception as e:
            print(f"Warning: Error processing {data_file}: {e}")
            continue

    # Compute rates
    base_rates = {}
    for template_id, horizons in results.items():
        base_rates[template_id] = {}
        for horizon, counts in horizons.items():
            total = counts['true'] + counts['false']
            rate = counts['true'] / total if total > 0 else 0
            base_rates[template_id][horizon] = {
                'true': counts['true'],
                'false': counts['false'],
                'total': total,
                'rate': round(rate, 4),
            }

    # Compute threshold-specific rates
    by_threshold = {}
    for template_id, thresholds in threshold_results.items():
        by_threshold[template_id] = {}
        for threshold, horizons in sorted(thresholds.items()):
            by_threshold[template_id][threshold] = {}
            for horizon, counts in horizons.items():
                total = counts['true'] + counts['false']
                rate = counts['true'] / total if total > 0 else 0
                by_threshold[template_id][threshold][horizon] = round(rate, 4)

    return {
        'metadata': {
            'num_games': games_processed,
            'num_questions': questions_total,
            'snapshot_turn': snapshot_turn,
            'generated_at': datetime.now().isoformat() + 'Z',
        },
        'base_rates': base_rates,
        'by_threshold': by_threshold,
    }


def print_summary(data: dict):
    """Print a summary of base rate results."""
    print()
    print("=" * 70)
    print("BASE RATE SUMMARY")
    print("=" * 70)

    metadata = data['metadata']
    print(f"\nGames processed: {metadata['num_games']}")
    print(f"Questions total: {metadata['num_questions']}")
    print(f"Snapshot turn: {metadata['snapshot_turn']}")

    print("\n" + "-" * 70)
    print("BASE RATES BY TEMPLATE AND HORIZON")
    print("-" * 70)
    print(f"{'Template':<25} {'H0':>10} {'H1':>10} {'H2':>10} {'H3':>10} {'Overall':>10}")
    print("-" * 70)

    base_rates = data['base_rates']
    for template_id in sorted(base_rates.keys()):
        horizons = base_rates[template_id]

        # Compute rates for each horizon
        h0_rate = horizons.get('H0', {}).get('rate', '-')
        h1_rate = horizons.get('H1', {}).get('rate', '-')
        h2_rate = horizons.get('H2', {}).get('rate', '-')
        h3_rate = horizons.get('H3', {}).get('rate', '-')

        # Compute overall rate
        total_true = sum(h.get('true', 0) for h in horizons.values())
        total_all = sum(h.get('total', 0) for h in horizons.values())
        overall = total_true / total_all if total_all > 0 else 0

        # Format rates
        h0_str = f"{h0_rate:.1%}" if isinstance(h0_rate, float) else str(h0_rate)
        h1_str = f"{h1_rate:.1%}" if isinstance(h1_rate, float) else str(h1_rate)
        h2_str = f"{h2_rate:.1%}" if isinstance(h2_rate, float) else str(h2_rate)
        h3_str = f"{h3_rate:.1%}" if isinstance(h3_rate, float) else str(h3_rate)

        print(f"{template_id:<25} {h0_str:>10} {h1_str:>10} {h2_str:>10} {h3_str:>10} {overall:>10.1%}")

    # Print threshold analysis for key templates
    by_threshold = data.get('by_threshold', {})
    if by_threshold:
        print("\n" + "-" * 70)
        print("SAMPLE THRESHOLD ANALYSIS (tech_count_gte)")
        print("-" * 70)

        if 'tech_count_gte' in by_threshold:
            thresholds = by_threshold['tech_count_gte']
            print(f"{'Threshold':<15} {'H1':>10} {'H2':>10} {'H3':>10}")
            print("-" * 45)
            for threshold in sorted(thresholds.keys())[:10]:
                horizons = thresholds[threshold]
                h1 = horizons.get('H1', '-')
                h2 = horizons.get('H2', '-')
                h3 = horizons.get('H3', '-')
                h1_str = f"{h1:.1%}" if isinstance(h1, float) else str(h1)
                h2_str = f"{h2:.1%}" if isinstance(h2, float) else str(h2)
                h3_str = f"{h3:.1%}" if isinstance(h3, float) else str(h3)
                print(f"{threshold:<15} {h1_str:>10} {h2_str:>10} {h3_str:>10}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute base rates for question templates'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/games',
        help='Directory containing game data JSON files (default: data/games)'
    )
    parser.add_argument(
        '--snapshot-turn',
        type=int,
        default=50,
        help='Turn at which forecasters see data (default: 50)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='base_rates.json',
        help='Output JSON file (default: base_rates.json)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return 1

    # Find all JSON files
    data_files = sorted(data_dir.glob('*_data.json'))
    if not data_files:
        print(f"No data files found in {data_dir}")
        return 1

    print(f"Found {len(data_files)} data files in {data_dir}")
    print(f"Snapshot turn: {args.snapshot_turn}")
    print()

    # Compute base rates
    results = compute_base_rates(
        data_files=data_files,
        snapshot_turn=args.snapshot_turn,
    )

    # Print summary
    print_summary(results)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
