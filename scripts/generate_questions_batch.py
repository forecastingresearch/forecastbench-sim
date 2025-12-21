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
    """Process a single game file and return resolved question bank as dict."""
    data_file, snapshot_turn, info_availability_levels = args

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

        # Generate questions
        generator = QuestionGenerator()
        bank = generator.generate_question_bank(
            game_id=game_id,
            game_data=game_data,
            snapshot_turn=snapshot_turn,
            info_availability_levels=info_availability_levels,
        )

        # Resolve questions
        resolver = QuestionResolver()
        resolved_bank = resolver.resolve_batch(bank, game_data)

        bank_dict = question_bank_to_dict(resolved_bank)
        bank_dict["game_id"] = game_id
        return bank_dict

    except Exception as e:
        print(f"Error processing {data_file}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Batch generate questions from game data')
    parser.add_argument('--data-dir', type=str, default='data/games',
                        help='Directory containing game data JSON files')
    parser.add_argument('--snapshot-turn', type=int, default=50,
                        help='Turn at which forecasters see data (default: 50)')
    parser.add_argument('--output', '-o', type=str, default='data/questions/questions_all.json',
                        help='Output JSON file')
    parser.add_argument('--info-availability', nargs='+', default=['I1', 'I2', 'I3'],
                        help='Information availability levels to include (default: I1 I2 I3)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    args = parser.parse_args()

    # Find all data files
    data_dir = Path(args.data_dir)
    data_files = sorted(data_dir.glob('*_data.json'))
    print(f"Found {len(data_files)} data files in {data_dir}")
    print(f"Snapshot turn: {args.snapshot_turn}")
    print(f"Info availability levels: {args.info_availability}")

    # Prepare arguments for each game
    task_args = [(str(f), args.snapshot_turn, args.info_availability) for f in data_files]

    # Process in parallel
    all_questions = []
    all_banks = []
    per_game_dir = Path(args.output).parent

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_game, arg): arg[0] for arg in task_args}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing games"):
            result = future.result()
            if result:
                all_banks.append(result)
                all_questions.extend(result.get('questions', []))
                # Write per-game questions.json in evaluator-friendly layout
                game_id = result.get("game_id", "unknown_game")
                game_dir = per_game_dir / game_id
                game_dir.mkdir(parents=True, exist_ok=True)
                with open(game_dir / "questions.json", "w") as f:
                    json.dump(result, f, indent=2)

    # Compute statistics
    true_count = sum(1 for q in all_questions if q.get('resolution', {}).get('answer', False))
    false_count = len(all_questions) - true_count

    # Count by template and horizon (horizon is nested under difficulty)
    by_template = {}
    by_horizon = {'H1': {'true': 0, 'false': 0}, 'H2': {'true': 0, 'false': 0}, 'H3': {'true': 0, 'false': 0}}

    for q in all_questions:
        t = q.get('template_id', 'unknown')
        h = q.get('difficulty', {}).get('horizon', 'H1')
        answer = q.get('resolution', {}).get('answer', False)

        if t not in by_template:
            by_template[t] = {'total': 0, 'true': 0, 'false': 0}
        by_template[t]['total'] += 1

        if answer:
            by_template[t]['true'] += 1
            by_horizon[h]['true'] += 1
        else:
            by_template[t]['false'] += 1
            by_horizon[h]['false'] += 1

    # Print summary
    print("\n" + "=" * 70)
    print("BATCH QUESTION GENERATION SUMMARY")
    print("=" * 70)
    print(f"\nGames processed: {len(all_banks)}")
    print(f"Total questions: {len(all_questions)}")
    print(f"Answers: {true_count} True ({100*true_count/len(all_questions):.1f}%), {false_count} False")

    print("\n" + "-" * 70)
    print("BY HORIZON")
    print("-" * 70)
    for h in ['H1', 'H2', 'H3']:
        total = by_horizon[h]['true'] + by_horizon[h]['false']
        if total > 0:
            rate = 100 * by_horizon[h]['true'] / total
            print(f"  {h}: {total:5d} questions, {rate:5.1f}% True")

    print("\n" + "-" * 70)
    print("BY TEMPLATE")
    print("-" * 70)
    print(f"{'Template':<25} {'Total':>8} {'True':>8} {'Rate':>8}")
    print("-" * 70)
    for t, counts in sorted(by_template.items()):
        rate = 100 * counts['true'] / counts['total'] if counts['total'] > 0 else 0
        print(f"{t:<25} {counts['total']:>8} {counts['true']:>8} {rate:>7.1f}%")

    # Save combined output
    output_data = {
        'metadata': {
            'num_games': len(all_banks),
            'num_questions': len(all_questions),
            'snapshot_turn': args.snapshot_turn,
            'info_availability_levels': args.info_availability,
            'true_rate': true_count / len(all_questions) if all_questions else 0,
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
