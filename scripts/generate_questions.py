#!/usr/bin/env python3
"""
Generate example questions from game data.

Usage:
    python generate_questions.py reports/s42/turn_299_data.json --snapshot-turn 50 --output questions_s42.json
"""

import argparse
import json
import sys
sys.path.insert(0, 'src')

from civrealm.world_reports.questions import (
    QuestionGenerator,
    QuestionResolver,
    question_bank_to_json,
    save_question_bank,
)


def main():
    parser = argparse.ArgumentParser(description='Generate questions from game data')
    parser.add_argument('data_file', help='Path to game data JSON file')
    parser.add_argument('--snapshot-turn', type=int, default=50,
                        help='Turn at which forecasters see data (default: 50)')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--info-availability', nargs='+', default=['I1', 'I2', 'I3'],
                        help='Information availability levels to include (default: I1 I2 I3)')
    parser.add_argument('--resolution-turns', nargs='+', type=int,
                        help='Specific resolution turns (default: auto-select)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary instead of full output')
    args = parser.parse_args()

    # Load game data
    print(f"Loading game data from {args.data_file}...")
    with open(args.data_file) as f:
        game_data = json.load(f)

    game_id = game_data.get('metadata', {}).get('username', 'unknown')
    max_turn = game_data.get('metadata', {}).get('turn', 0)

    print(f"Game: {game_id}, Max turn: {max_turn}")
    print(f"Snapshot turn: {args.snapshot_turn}")

    if args.snapshot_turn >= max_turn:
        print(f"Error: snapshot_turn ({args.snapshot_turn}) must be < max_turn ({max_turn})")
        sys.exit(1)

    # Generate questions
    print("\nGenerating questions...")
    generator = QuestionGenerator()
    bank = generator.generate_question_bank(
        game_id=game_id,
        game_data=game_data,
        snapshot_turn=args.snapshot_turn,
        resolution_turns=args.resolution_turns,
        info_availability_levels=args.info_availability,
    )

    print(f"Generated {len(bank.questions)} questions")

    # Resolve questions
    print("\nResolving questions...")
    resolver = QuestionResolver()
    resolved_bank = resolver.resolve_batch(bank, game_data)

    # Count by template and answer
    by_template = {}
    true_count = 0
    false_count = 0
    for q in resolved_bank.questions:
        t = q.template_id
        if t not in by_template:
            by_template[t] = {'total': 0, 'true': 0, 'false': 0}
        by_template[t]['total'] += 1
        if q.resolution and q.resolution.answer:
            by_template[t]['true'] += 1
            true_count += 1
        else:
            by_template[t]['false'] += 1
            false_count += 1

    # Print summary
    print("\n" + "=" * 60)
    print("QUESTION SUMMARY")
    print("=" * 60)
    print(f"\nTotal: {len(resolved_bank.questions)} questions")
    print(f"Answers: {true_count} True, {false_count} False")
    print(f"\nBy template:")
    for t, counts in sorted(by_template.items()):
        print(f"  {t}: {counts['total']} ({counts['true']} True, {counts['false']} False)")

    # Print sample questions
    print("\n" + "=" * 60)
    print("SAMPLE QUESTIONS")
    print("=" * 60)

    # Group by difficulty
    by_difficulty = {}
    for q in resolved_bank.questions:
        d = q.difficulty
        if d not in by_difficulty:
            by_difficulty[d] = []
        by_difficulty[d].append(q)

    for d in sorted(by_difficulty.keys()):
        print(f"\n--- Difficulty {d} ({len(by_difficulty[d])} questions) ---")
        for q in by_difficulty[d][:3]:
            answer = q.resolution.answer if q.resolution else "?"
            print(f"  [{q.horizon}/{q.info_availability}] {q.question_text}")
            print(f"           Answer: {answer}")

    # Save output
    if args.output:
        print(f"\nSaving to {args.output}...")
        save_question_bank(resolved_bank, args.output)
        print("Done!")
    elif not args.summary:
        print("\n" + "=" * 60)
        print("JSON OUTPUT (use --output to save to file)")
        print("=" * 60)
        print(question_bank_to_json(resolved_bank))


if __name__ == '__main__':
    main()
