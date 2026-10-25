#!/usr/bin/env python3
"""
Batch generate questions from all game data files.

Usage:
    python scripts/generate_questions_batch.py --data-dir data/games --output questions_all.json
"""

import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, 'src')

from civrealm.world_reports.questions import (
    QuestionGenerator,
    QuestionResolver,
    QuestionBank,
    question_bank_to_dict,
)


def process_single_game(args: tuple) -> dict | None:
    """Process a single game file and return resolved question bank as dict.

    Generates H0 (comprehension), H1-H7 binary (forecasting), and
    H1-H7 continuous (single-civ absolute value) questions.
    """
    data_file, snapshot_turn = args

    try:
        # Import inside function for multiprocessing
        sys.path.insert(0, 'src')
        from civrealm.world_reports.questions import (
            QuestionGenerator,
            QuestionResolver,
            question_bank_to_dict,
        )

        with open(data_file) as f:
            game_data = json.load(f)

        game_id = game_data.get('metadata', {}).get('username', Path(data_file).stem)
        max_turn = game_data.get('metadata', {}).get('turn', 0)

        if snapshot_turn >= max_turn:
            return None

        generator = QuestionGenerator()
        resolver = QuestionResolver()

        # Generate H0 (comprehension) questions
        h0_bank = generator.generate_zero_turn_questions(
            game_id=game_id,
            game_data=game_data,
            snapshot_turn=snapshot_turn,
        )
        resolved_h0 = resolver.resolve_batch(h0_bank, game_data)

        # Generate H1-H7 binary (forecasting) questions
        forecast_bank = generator.generate_question_bank(
            game_id=game_id,
            game_data=game_data,
            snapshot_turn=snapshot_turn,
        )
        resolved_forecast = resolver.resolve_batch(forecast_bank, game_data)

        # Generate H1-H7 continuous questions
        continuous_bank = generator.generate_continuous_questions(
            game_id=game_id,
            game_data=game_data,
            snapshot_turn=snapshot_turn,
        )
        resolved_continuous = resolver.resolve_batch(continuous_bank, game_data)

        # Combine questions from all banks
        h0_dict = question_bank_to_dict(resolved_h0)
        forecast_dict = question_bank_to_dict(resolved_forecast)
        continuous_dict = question_bank_to_dict(resolved_continuous)

        # Merge questions into forecast_dict (use it as base)
        all_questions = (
            h0_dict.get('questions', [])
            + forecast_dict.get('questions', [])
            + continuous_dict.get('questions', [])
        )

        # Add game_id to each question's parameters for later retrieval
        for q in all_questions:
            q['parameters']['game_id'] = game_id

        forecast_dict['questions'] = all_questions

        return forecast_dict

    except Exception as e:
        print(f"Error processing {data_file}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Batch generate questions from game data')
    parser.add_argument('--data-dir', type=str, default='data/games',
                        help='Directory containing game data JSON files')
    parser.add_argument('--snapshot-turn', type=int, default=60,
                        help='Turn at which forecasters see data (default: 60)')
    parser.add_argument('--output', '-o', type=str, default='data/questions/questions_all.json',
                        help='Output JSON file')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    args = parser.parse_args()

    # Find all data files
    data_dir = Path(args.data_dir)
    data_files = sorted(data_dir.glob('*_data.json'))
    print(f"Found {len(data_files)} data files in {data_dir}")
    print(f"Snapshot turn: {args.snapshot_turn}")

    # Prepare arguments for each game
    task_args = [(str(f), args.snapshot_turn) for f in data_files]

    # Process in parallel
    all_questions = []
    all_banks = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_game, arg): arg[0] for arg in task_args}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing games"):
            result = future.result()
            if result:
                all_banks.append(result)
                all_questions.extend(result.get('questions', []))

    # Split by question type
    binary_questions = [q for q in all_questions if q.get('question_type') != 'continuous']
    continuous_questions = [q for q in all_questions if q.get('question_type') == 'continuous']

    # Compute binary statistics
    true_count = sum(1 for q in binary_questions if q.get('resolution', {}).get('answer', False))
    false_count = len(binary_questions) - true_count

    # Count by template and horizon (binary)
    by_template = {}
    by_horizon = {}

    for q in binary_questions:
        t = q.get('template_id', 'unknown')
        h = q.get('horizon')
        if h is None:
            h = q.get('difficulty', {}).get('horizon', 'H1')
        answer = q.get('resolution', {}).get('answer', False)

        if t not in by_template:
            by_template[t] = {'total': 0, 'true': 0, 'false': 0}
        by_template[t]['total'] += 1

        if h not in by_horizon:
            by_horizon[h] = {'true': 0, 'false': 0}

        if answer:
            by_template[t]['true'] += 1
            by_horizon[h]['true'] += 1
        else:
            by_template[t]['false'] += 1
            by_horizon[h]['false'] += 1

    # Count continuous by template and horizon
    continuous_by_template = {}
    continuous_by_horizon = {}

    for q in continuous_questions:
        t = q.get('template_id', 'unknown')
        h = q.get('horizon', 'H1')

        if t not in continuous_by_template:
            continuous_by_template[t] = 0
        continuous_by_template[t] += 1

        if h not in continuous_by_horizon:
            continuous_by_horizon[h] = 0
        continuous_by_horizon[h] += 1

    # Print summary
    print("\n" + "=" * 70)
    print("BATCH QUESTION GENERATION SUMMARY")
    print("=" * 70)
    print(f"\nGames processed: {len(all_banks)}")
    print(f"Total questions: {len(all_questions)}")
    print(f"  Binary: {len(binary_questions)}")
    print(f"  Continuous: {len(continuous_questions)}")
    if binary_questions:
        print(f"Binary answers: {true_count} True ({100*true_count/len(binary_questions):.1f}%), {false_count} False")

    print("\n" + "-" * 70)
    print("BINARY QUESTIONS BY HORIZON")
    print("-" * 70)
    for h in ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']:
        total = by_horizon.get(h, {}).get('true', 0) + by_horizon.get(h, {}).get('false', 0)
        if total > 0:
            rate = 100 * by_horizon[h]['true'] / total
            print(f"  {h}: {total:5d} questions, {rate:5.1f}% True")

    print("\n" + "-" * 70)
    print("BINARY QUESTIONS BY TEMPLATE")
    print("-" * 70)
    print(f"{'Template':<25} {'Total':>8} {'True':>8} {'Rate':>8}")
    print("-" * 70)
    for t, counts in sorted(by_template.items()):
        rate = 100 * counts['true'] / counts['total'] if counts['total'] > 0 else 0
        print(f"{t:<25} {counts['total']:>8} {counts['true']:>8} {rate:>7.1f}%")

    print("\n" + "-" * 70)
    print("CONTINUOUS QUESTIONS BY HORIZON")
    print("-" * 70)
    for h in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']:
        count = continuous_by_horizon.get(h, 0)
        if count > 0:
            print(f"  {h}: {count:5d} questions")

    print("\n" + "-" * 70)
    print("CONTINUOUS QUESTIONS BY TEMPLATE")
    print("-" * 70)
    print(f"{'Template':<25} {'Total':>8}")
    print("-" * 70)
    for t, count in sorted(continuous_by_template.items()):
        print(f"{t:<25} {count:>8}")

    # Save combined output
    output_data = {
        'metadata': {
            'num_games': len(all_banks),
            'num_questions': len(all_questions),
            'num_binary': len(binary_questions),
            'num_continuous': len(continuous_questions),
            'snapshot_turn': args.snapshot_turn,
            'true_rate': true_count / len(binary_questions) if binary_questions else 0,
        },
        'questions': all_questions,
    }

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(all_questions)} questions to {args.output}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print("Done!")


if __name__ == '__main__':
    main()
